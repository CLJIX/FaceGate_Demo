"""门禁状态机与事件日志。

判定输出严格限定在用户关心的几种状态,不做权限管理:

    ============  ==========  ==========================================
    verdict       中文        触发条件
    ============  ==========  ==========================================
    no_face       无人脸      当前帧没有检出人脸
    observing     观察中      活体证据还没攒够(轨迹太短/头没动)
    spoof         非活体      活体融合分低于阈值,或某条线索一票否决
    unenrolled    未录入      人脸库为空,或最高相似度低于陌生人线
    deny          不是本人    库里有相近的人,但相似度没到开门线
    grant         是本人      相似度达标 → 开门
    ============  ==========  ==========================================
"""

from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from .config import Config
from .gallery import Gallery, Match
from .liveness.analyzer import LivenessResult

VERDICT_CN: Dict[str, str] = {
    "no_face": "无人脸",
    "observing": "观察中",
    "spoof": "非活体",
    "unenrolled": "未录入",
    "deny": "不是本人",
    "grant": "是本人",
}

VERDICT_COLOR: Dict[str, tuple] = {
    "no_face": (150, 150, 150),
    "observing": (0, 190, 255),
    "spoof": (0, 80, 255),
    "unenrolled": (0, 160, 255),
    "deny": (60, 60, 240),
    "grant": (60, 220, 60),
}


@dataclass
class GateState:
    verdict: str = "no_face"
    person: Optional[Match] = None
    similarity: float = 0.0
    door_open: bool = False
    message: str = "等待人脸"
    liveness: Optional[LivenessResult] = None
    embedding_age_ms: float = 0.0
    just_opened: bool = False

    @property
    def verdict_cn(self) -> str:
        return VERDICT_CN.get(self.verdict, self.verdict)


class Gate:
    def __init__(self, cfg: Config, gallery: Gallery):
        self.cfg = cfg
        self.gallery = gallery
        self.state = GateState()
        self._open_until = 0.0
        self._cooldown_until = 0.0
        self._last_grant_id: Optional[str] = None
        self.events: List[dict] = []

    def update(
        self,
        has_face: bool,
        liveness: Optional[LivenessResult],
        embedding: Optional[object],
        embedding_age_ms: float,
    ) -> GateState:
        now = time.time()
        st = self.state
        st.liveness = liveness
        st.embedding_age_ms = embedding_age_ms
        st.just_opened = False
        st.door_open = now < self._open_until

        if not has_face:
            self._open_until = 0.0
            st.verdict, st.person, st.similarity = "no_face", None, 0.0
            st.message = "画面中没有人脸"
            return st

        live_verdict = liveness.verdict if liveness else "observing"
        if live_verdict == "observing":
            st.verdict = "observing"
            st.message = (liveness.reason if liveness else "积累证据中") + " — 请正对摄像头并缓慢转头"
            return st

        if live_verdict == "fake":
            st.verdict = "spoof"
            st.person, st.similarity = None, 0.0
            st.message = f"拒绝:判定为攻击载体。{liveness.reason if liveness else ''}"
            self._log("spoof", None, 0.0, liveness)
            return st

        # 活体通过,进入身份判定
        if embedding is None:
            st.verdict = "observing"
            st.message = "正在提取人脸特征…"
            return st

        matches = self.gallery.match(embedding, top_k=5)  # type: ignore[arg-type]
        if not matches:
            st.verdict, st.person, st.similarity = "unenrolled", None, 0.0
            st.message = "人脸库为空,请先录入(python main.py enroll --name 姓名)"
            return st

        best = matches[0]
        st.person, st.similarity = best, best.similarity
        g = self.cfg.gate

        if best.similarity >= g.accept_threshold:
            st.verdict = "grant"
            st.message = f"开门 — {best.person.name}(相似度 {best.similarity:.3f})"
            if now >= self._cooldown_until or self._last_grant_id != best.person.id:
                st.just_opened = True
                self._open_until = now + g.open_hold_sec
                self._cooldown_until = now + g.cooldown_sec
                self._last_grant_id = best.person.id
                self._log("grant", best, best.similarity, liveness)
                print(f"\n\033[92m>>> 开门 <<< {best.person.name} (id={best.person.id}) "
                      f"相似度={best.similarity:.3f} 活体分={liveness.score:.2f}\033[0m\n")
        elif best.similarity >= g.stranger_threshold:
            st.verdict = "deny"
            st.message = (
                f"拒绝 — 最接近 {best.person.name},相似度 {best.similarity:.3f} "
                f"< 开门线 {g.accept_threshold:.2f}"
            )
            self._log("deny", best, best.similarity, liveness)
        else:
            st.verdict = "unenrolled"
            st.message = f"未录入 — 最高相似度仅 {best.similarity:.3f},库中没有这个人"
            self._log("unenrolled", best, best.similarity, liveness)
        return st

    def _log(self, verdict: str, match: Optional[Match], sim: float, lv: Optional[LivenessResult]) -> None:
        # 同一状态在连续帧里会反复命中,只在状态发生变化时记一条事件
        if self.events and self.events[-1]["verdict"] == verdict and time.time() - self.events[-1]["t"] < 2.0:
            return
        rec = {
            "t": time.time(),
            "verdict": verdict,
            "person": match.person.id if match else None,
            "similarity": round(sim, 4),
            "liveness": round(lv.score, 4) if lv else None,
            "veto_by": lv.veto_by if lv else None,
        }
        self.events.append(rec)
        if len(self.events) > 5000:
            del self.events[:1000]


class MetricsWriter:
    """把每帧的全部原始指标写成 CSV,便于离线画分布、重新标定阈值。"""

    HEADER = [
        "t", "frame", "verdict", "live_score_fused", "live_instant", "static", "temporal", "similarity",
        "deep", "blink", "micro_motion", "planarity", "texture", "moire", "color",
        "ear", "ear_baseline", "motion", "deform", "jitter_deg",
        "homo_resid", "homo_p90", "homo_motion", "inlier",
        "lap", "band_ratio", "peak_ratio", "peak_count",
        "skin", "sat", "pitch", "yaw", "roll", "iod", "det", "addon_live",
    ]

    def __init__(self, path: Path, enabled: bool = True):
        self.path = path
        self.enabled = enabled
        self._fh = None
        self._w = None
        if enabled:
            path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = open(path, "w", newline="", encoding="utf-8-sig")
            self._w = csv.writer(self._fh)
            self._w.writerow(self.HEADER)

    def write(self, row: Dict[str, object]) -> None:
        if self._w is None:
            return
        self._w.writerow([row.get(k, "") for k in self.HEADER])

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None


class EventJournal:
    """门禁事件落盘为 jsonl,一行一条。"""

    def __init__(self, path: Path, enabled: bool = True):
        self.path = path
        self.enabled = enabled
        if enabled:
            path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, records: List[dict]) -> None:
        if not self.enabled or not records:
            return
        with open(self.path, "a", encoding="utf-8") as fh:
            for r in records:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
