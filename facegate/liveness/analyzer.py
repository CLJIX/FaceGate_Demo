"""单目 RGB 活体检测:七路线索 + 加权融合 + 时序迟滞判决。

设计取向是「可解释优先」。每一路线索都:
  * 只输出一两个归一化到 [0,1] 的分数(0 = 确定是攻击载体,1 = 确定是真人);
  * 同时输出它依赖的全部原始物理量,直接显示在调试面板上;
  * 证据不足时返回 ``score=None``,不参与融合,而不是硬给一个分数。

七路线索分别针对不同的攻击方式:

    ============  ==============  ==========================================
    线索           主要拦          物理依据
    ============  ==============  ==========================================
    deep_model    照片/屏幕/面具   insightface 官方 80x80 CNN,单帧纹理判别
    blink         静置照片        真人 3~5 秒会自然眨一次眼,打印件不会
    micro_motion  静置照片/暂停    冻结画面的全局位移与非刚性形变都趋近于 0
    planarity     照片/屏幕(核心)  平面载体的帧间映射严格是单应;三维脸不是
    texture       屏幕翻拍        二次采样 + 压缩抹掉高频细节
    moire         屏幕翻拍        像素栅格在频域留下孤立尖峰
    color         屏幕/纸张       背光色偏、饱和度异常、肤色占比偏低
    ============  ==============  ==========================================

其中 ``planarity`` 是最有信息量的一路,也是「单镜头做深度判别」的关键:
它不需要深度相机,而是利用射影几何 —— 平面在针孔相机前的任意刚性运动,其图像
变换严格是一个 8 自由度单应矩阵;真实人脸是非平面三维体,转动时关键点位移场
带有透视视差,单应无法完全解释。残差(以瞳距为单位)就是三维结构的直接证据。

融合不是简单加权平均,而是**两层结构**:

    单帧子分 static   = 官方CNN / 纹理 / 摩尔纹 / 色彩   的加权平均
    时间子分 temporal = 眨眼 / 运动形变 / 平面性          的加权平均
    融合分 instant    = static × (0.35 + 0.65 × temporal)

之所以这样设计:官方 80x80 CNN 是单帧纹理判别器,实测会被一张光照良好的高清照片
骗到 0.98 的「真人」分;而「6 秒内一次眨眼都没有」「完全没有非刚性形变」这类证据
是单帧模型结构上拿不到的。让时间证据去门控单帧证据,照片攻击才会被压下去。
temporal 证据不足(轨迹太短/头没动)时退化为只用 static,此时判定仍受 min_track_frames
与 confirm_frames 约束,不会立刻放行。
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional

import cv2
import numpy as np

from ..config import LivenessCfg
from ..engine import FaceObs
from ..util import clamp, hsv_skin_ratio, ramp, ramp_inv
from .geometry import (
    IDX_EYE_L,
    IDX_EYE_R,
    HomographyFit,
    analyze_spectrum,
    eye_geometry,
    fit_homography,
    fit_similarity,
    highpass,
    laplacian_var,
    Spectrum,
)

CUE_LABELS: Dict[str, str] = {
    "deep_model": "官方活体CNN",
    "blink": "自然眨眼",
    "micro_motion": "运动/形变",
    "planarity": "平面性视差",
    "texture": "高频纹理",
    "moire": "屏幕摩尔纹",
    "color": "色彩偏置",
}

# 频谱分析的最大边长,见 frame_metrics 内的说明
MAX_SPEC_SIDE = 256


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------


@dataclass
class CueResult:
    key: str
    label: str
    score: Optional[float]       # None = 证据不足,不计入融合
    confidence: float            # 0..1
    weight: float
    veto: bool
    raw: Dict[str, object] = field(default_factory=dict)    # 面板显示用,键为中文
    extra: Dict[str, float] = field(default_factory=dict)   # CSV 落盘用,键为 ASCII
    hint: str = ""

    @property
    def contributes(self) -> bool:
        return self.score is not None and self.confidence > 0.05 and self.weight > 0


@dataclass
class Sample:
    """一帧的全部观测与派生标量。"""

    t: float                     # 相对轨迹起点,秒
    frame_id: int
    bbox: np.ndarray
    kps: np.ndarray
    lm106: Optional[np.ndarray]
    pose: Optional[np.ndarray]
    iod: float                   # 瞳距,所有几何量的归一化尺度
    det_score: float
    live_score: Optional[float]
    live_status: str
    aligned: np.ndarray          # 112x112 BGR
    spectrum: Optional[Spectrum]
    roi_gray: Optional[np.ndarray]
    m: Dict[str, float] = field(default_factory=dict)


@dataclass
class LivenessResult:
    verdict: str                 # live / fake / observing
    score: float                 # 时序平滑后的融合分
    instant: float               # 当帧融合分
    static: float                # 单帧线索子分(官方CNN/纹理/摩尔纹/色彩)
    temporal: Optional[float]    # 时间线索子分(眨眼/运动/平面性),None = 证据不足
    cues: List[CueResult]
    frames: int
    elapsed: float
    reason: str
    veto_by: Optional[str]
    history: np.ndarray


# 时间维度线索:它们能看到「一段时间内发生了什么」,是单帧纹理模型结构上拿不到的证据
TEMPORAL_CUES = ("blink", "micro_motion", "planarity")
# 时间证据缺失时融合分的下限系数:temporal=0 时最终分最多是 static 的 35%
TEMPORAL_GATE_FLOOR = 0.35


# ---------------------------------------------------------------------------
# 单帧指标
# ---------------------------------------------------------------------------


def frame_metrics(
    obs: FaceObs,
    frame: np.ndarray,
    cfg: LivenessCfg,
    compute_spectrum: bool = True,
    prev: Optional[Sample] = None,
) -> tuple[Dict[str, float], Optional[Spectrum], np.ndarray, Optional[np.ndarray]]:
    """算出所有只依赖当前帧的指标。

    频域分析用**原始分辨率**的人脸 ROI 而不是对齐裁剪:对齐会做一次双线性重采样,
    既衰减摩尔纹又引入新的混叠,而摩尔纹的证据恰恰存在于传感器原生频率上。
    纹理清晰度则用 112x112 对齐裁剪,这样人脸远近不影响读数。

    FFT 约占单帧指标的 7ms,而摩尔纹是场景级静态属性,帧间几乎不变,因此允许隔帧
    复用(``compute_spectrum=False`` 时直接沿用上一帧的谱与峰值指标)。
    """
    aligned = obs.aligned
    gray = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY)
    m: Dict[str, float] = {"lap": laplacian_var(gray)}

    skin, sat, val = hsv_skin_ratio(aligned, cfg.color.skin_hue_lo, cfg.color.skin_hue_hi)
    m["skin"], m["sat"], m["val"] = skin, sat, val

    roi_gray = _face_roi_gray(frame, obs.bbox, margin=0.12)
    if roi_gray is not None and max(roi_gray.shape) > MAX_SPEC_SIDE:
        # pocketfft 在非 2 幂的大尺寸上很慢(370x265 要 18ms);限到 256 边长后约 3ms。
        # 摩尔纹关心的是「相对人脸尺寸的栅格周期」,限尺寸反而让指标与人脸远近无关。
        s = MAX_SPEC_SIDE / max(roi_gray.shape)
        roi_gray = cv2.resize(roi_gray, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    spec: Optional[Spectrum] = None
    if compute_spectrum and roi_gray is not None:
        hp = highpass(roi_gray, ksize=7)
        spec = analyze_spectrum(hp, low_keep=cfg.moire.low_freq_keep)
        m["peak_ratio"] = spec.peak_ratio
        m["peak_count"] = float(spec.peak_count)
        m["band_ratio"] = spec.band_ratio
    elif prev is not None and prev.spectrum is not None:
        spec = prev.spectrum
        m["peak_ratio"] = prev.m.get("peak_ratio", 0.0)
        m["peak_count"] = prev.m.get("peak_count", 0.0)
        m["band_ratio"] = prev.m.get("band_ratio", 0.0)
    else:
        m.update(peak_ratio=0.0, peak_count=0.0, band_ratio=0.0)

    ear_l = ear_r = None
    if obs.lm106 is not None:
        gl = eye_geometry(obs.lm106[list(IDX_EYE_L)])
        gr = eye_geometry(obs.lm106[list(IDX_EYE_R)])
        if gl:
            ear_l = gl.openness
        if gr:
            ear_r = gr.openness
        m["ear_l"] = float(ear_l) if ear_l is not None else -1.0
        m["ear_r"] = float(ear_r) if ear_r is not None else -1.0
        m["ear"] = float(np.mean([x for x in (ear_l, ear_r) if x is not None])) if (ear_l or ear_r) else -1.0
        m["mouth_open"] = _mouth_openness(obs.lm106)
    else:
        m.update(ear_l=-1.0, ear_r=-1.0, ear=-1.0, mouth_open=-1.0)

    if obs.pose is not None:
        m["pitch"], m["yaw"], m["roll"] = (float(x) for x in obs.pose[:3])
    else:
        m.update(pitch=0.0, yaw=0.0, roll=0.0)

    m["live"] = float(obs.live_score) if obs.live_score is not None else -1.0
    m["det"] = obs.det_score
    m["iod"] = obs.iod
    return m, spec, gray, roi_gray


def _face_roi_gray(frame: np.ndarray, bbox: np.ndarray, margin: float) -> Optional[np.ndarray]:
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = (float(v) for v in bbox[:4])
    bw, bh = x2 - x1, y2 - y1
    if bw < 24 or bh < 24:
        return None
    x1 = int(max(0, x1 - bw * margin))
    y1 = int(max(0, y1 - bh * margin))
    x2 = int(min(w, x2 + bw * margin))
    y2 = int(min(h, y2 + bh * margin))
    if x2 - x1 < 24 or y2 - y1 < 24:
        return None
    return cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)


def _mouth_openness(lm: np.ndarray) -> float:
    from .geometry import IDX_MOUTH

    pts = np.asarray(lm, dtype=np.float64)[list(IDX_MOUTH)]
    d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=2)
    i, j = np.unravel_index(int(np.argmax(d)), d.shape)
    span = float(d[i, j])
    return span if span > 0 else 0.0


# ---------------------------------------------------------------------------
# 分析器
# ---------------------------------------------------------------------------


class LivenessAnalyzer:
    def __init__(self, cfg: LivenessCfg):
        self.cfg = cfg
        self.samples: Deque[Sample] = deque()
        self._lost = 0
        self._verdict = "observing"
        self._pending: Optional[str] = None
        self._pending_n = 0
        self._history: Deque[float] = deque(maxlen=600)
        self.last_homography: Optional[HomographyFit] = None
        self.last_roi_gray = None
        self._t_start = time.perf_counter()
        self.reset_reason = ""

    # -- 轨迹管理 ------------------------------------------------------------

    def reset(self, why: str = "") -> None:
        self.samples.clear()
        self._lost = 0
        self._verdict = "observing"
        self._pending = None
        self._pending_n = 0
        self._history.clear()
        self.last_homography = None
        self._t_start = time.perf_counter()
        self.reset_reason = why

    def mark_lost(self) -> None:
        self._lost += 1
        if self._lost >= self.cfg.track_reset_frames and self.samples:
            self.reset(f"连续 {self._lost} 帧丢失人脸")

    def push(self, obs: FaceObs, frame: np.ndarray, frame_id: int, spec_every: int = 2) -> None:
        self._lost = 0
        prev = self.samples[-1] if self.samples else None
        do_spec = prev is None or prev.spectrum is None or (frame_id % max(1, spec_every)) == 0
        m, spec, _, roi_gray = frame_metrics(obs, frame, self.cfg, compute_spectrum=do_spec, prev=prev)
        now = time.perf_counter()
        if not self.samples:
            self._t_start = now
        self.samples.append(
            Sample(
                t=now - self._t_start,
                frame_id=frame_id,
                bbox=obs.bbox.copy(),
                kps=obs.kps.copy(),
                lm106=None if obs.lm106 is None else obs.lm106.copy(),
                pose=None if obs.pose is None else obs.pose.copy(),
                iod=max(obs.iod, 1e-3),
                det_score=obs.det_score,
                live_score=obs.live_score,
                live_status=obs.live_status,
                aligned=obs.aligned,
                spectrum=spec,
                roi_gray=roi_gray,
                m=m,
            )
        )
        self.last_roi_gray = roi_gray
        cutoff = self.samples[-1].t - max(self.cfg.window_sec, self.cfg.blink.window_sec)
        while len(self.samples) > 2 and self.samples[0].t < cutoff:
            self.samples.popleft()

    # -- 判决 ----------------------------------------------------------------

    def evaluate(self) -> LivenessResult:
        cfg = self.cfg
        samples = list(self.samples)
        elapsed = (samples[-1].t - samples[0].t) if len(samples) > 1 else 0.0
        cues = [
            self._cue_deep(samples),
            self._cue_blink(samples, elapsed),
            self._cue_micro_motion(samples),
            self._cue_planarity(samples),
            self._cue_texture(samples),
            self._cue_moire(samples),
            self._cue_color(samples),
        ]

        total_w = sum(c.weight * c.confidence for c in cues if c.contributes)
        static_w = sum(c.weight * c.confidence for c in cues if c.contributes and c.key not in TEMPORAL_CUES)
        static = (
            sum(c.score * c.weight * c.confidence for c in cues if c.contributes and c.key not in TEMPORAL_CUES)
            / static_w
            if static_w > 1e-6
            else 0.5
        )
        t_cues = [c for c in cues if c.key in TEMPORAL_CUES and c.contributes]
        t_w = sum(c.weight * c.confidence for c in t_cues)
        temporal: Optional[float] = (
            sum(c.score * c.weight * c.confidence for c in t_cues) / t_w if t_w > 1e-6 else None
        )

        # 时间证据对单帧证据做门控:官方 CNN 这类单帧纹理模型会被一张清晰的照片骗过,
        # 而「6 秒内一次眨眼都没有」「完全没有非刚性形变」是它结构上看不到的证据。
        # 因此 temporal=0 时融合分被压到 static 的 35%,temporal=1 时完全信任 static。
        if temporal is None:
            instant = static
        else:
            instant = static * (TEMPORAL_GATE_FLOOR + (1.0 - TEMPORAL_GATE_FLOOR) * temporal)
        if total_w < 1e-6:
            instant = 0.5

        self._history.append(instant)
        smooth = float(np.median(list(self._history)[-max(3, cfg.confirm_frames) :]))

        veto = next((c for c in cues if c.veto), None)
        if len(samples) < cfg.min_track_frames or total_w < 1e-6:
            raw_verdict = "observing"
        elif veto is not None:
            raw_verdict = "fake"
        elif smooth >= cfg.decision_threshold:
            raw_verdict = "live"
        else:
            raw_verdict = "fake"

        self._verdict = self._apply_hysteresis(raw_verdict)
        return LivenessResult(
            verdict=self._verdict,
            score=smooth,
            instant=instant,
            static=static,
            temporal=temporal,
            cues=cues,
            frames=len(samples),
            elapsed=elapsed,
            reason=_explain(self._verdict, cues, veto, len(samples), cfg, static, temporal),
            veto_by=veto.key if veto else None,
            history=np.asarray(list(self._history), dtype=np.float32),
        )

    def _apply_hysteresis(self, raw: str) -> str:
        """连续 confirm_frames 帧同向才切换,避免单帧抖动导致门禁反复开合。"""
        if raw == self._verdict:
            self._pending, self._pending_n = None, 0
            return self._verdict
        if raw == self._pending:
            self._pending_n += 1
        else:
            self._pending, self._pending_n = raw, 1
        if self._pending_n >= max(1, self.cfg.confirm_frames):
            self._verdict, self._pending, self._pending_n = raw, None, 0
        return self._verdict

    def _spec(self, key: str):
        return self.cfg.cues.get(key)

    def _disabled(self, key: str) -> Optional[CueResult]:
        s = self._spec(key)
        if s is None or not s.enabled or s.weight <= 0:
            return CueResult(
                key=key,
                label=CUE_LABELS.get(key, key),
                score=None,
                confidence=0.0,
                weight=0.0,
                veto=False,
                hint="已关闭",
            )
        return None

    def _mk(
        self,
        key: str,
        score: Optional[float],
        conf: float,
        raw: Dict[str, object],
        hint: str = "",
        extra: Optional[Dict[str, float]] = None,
    ) -> CueResult:
        spec = self._spec(key) or self.cfg.cues["deep_model"]
        veto = False
        if score is not None and spec.veto_below is not None:
            veto = score < spec.veto_below and conf >= 0.6
        return CueResult(
            key, CUE_LABELS.get(key, key), score, clamp(conf), spec.weight, veto, raw,
            extra or {}, hint,
        )

    # -- 各路线索 ------------------------------------------------------------

    def _cue_deep(self, samples: List[Sample]) -> CueResult:
        key = "deep_model"
        if (off := self._disabled(key)) is not None:
            return off
        vals = [s.live_score for s in samples if s.live_score is not None]
        status = samples[-1].live_status if samples else "none"
        if not vals:
            return self._mk(key, None, 0.0, {"状态": status}, "官方模型未给出分数")
        med = float(np.median(vals[-30:]))
        return self._mk(
            key,
            med,
            clamp(len(vals) / 5.0),
            {
                "中位分": f"{med:.3f}",
                "最小": f"{min(vals[-30:]):.3f}",
                "最大": f"{max(vals[-30:]):.3f}",
                "样本": len(vals),
                "状态": status,
            },
            "insightface 80x80 RGB CNN 直出真人概率",
            extra={"mid": med, "lo": float(min(vals[-30:])), "hi": float(max(vals[-30:]))},
        )

    def _cue_blink(self, samples: List[Sample], elapsed: float) -> CueResult:
        key = "blink"
        if (off := self._disabled(key)) is not None:
            return off
        b = self.cfg.blink
        win = [s for s in samples if s.t >= samples[-1].t - b.window_sec] if samples else []
        ear = np.asarray([s.m.get("ear", -1.0) for s in win], dtype=np.float64)
        ear = ear[ear > 0]
        if len(ear) < 8:
            return self._mk(key, None, 0.0, {"帧数": len(ear)}, "采样不足,继续观察")

        baseline = float(np.percentile(ear, 85))
        if baseline < 1e-4:
            return self._mk(key, None, 0.0, {"基线": f"{baseline:.4f}"}, "眼部关键点不可靠")
        rel = ear / baseline
        closed = rel < b.ear_close_ratio
        open_ = rel > 0.82
        blinks = int(np.sum(np.diff(closed.astype(np.int8)) == 1))
        # 一次完整眨眼要求「闭合前有睁眼、闭合后有睁眼」,避免把长时间闭眼算成眨眼
        valid = 0
        for i in range(1, len(closed) - 1):
            if closed[i] and not closed[i - 1] and open_[i:].any():
                valid += 1
        blinks = max(blinks, valid) if valid else blinks
        depth = 1.0 - float(rel.min())

        s_depth = ramp(depth, 0.05, 1.0 - b.ear_close_ratio)
        s_count = 1.0 if blinks >= b.min_blinks else ramp(blinks, 0, b.min_blinks) * 0.5
        score = 0.6 * s_depth + 0.4 * s_count
        conf = clamp(elapsed / max(b.window_sec, 0.1)) * clamp(len(ear) / 20.0)
        hint = (
            f"窗口 {elapsed:.1f}s 内眨眼 {blinks} 次"
            if elapsed >= b.window_sec
            else f"观察中({elapsed:.1f}/{b.window_sec:.0f}s),已见 {blinks} 次眨眼"
        )
        return self._mk(
            key,
            score,
            conf,
            {
                "眨眼次数": blinks,
                "EAR基线": f"{baseline:.3f}",
                "EAR当前": f"{float(ear[-1]):.3f}",
                "最小相对值": f"{float(rel.min()):.2f}",
                "闭合深度": f"{depth:.2f}",
                "窗口": f"{elapsed:.1f}s",
            },
            hint,
            extra={
                "ear_baseline": baseline,
                "blinks": float(blinks),
                "depth": depth,
                "min_rel": float(rel.min()),
            },
        )

    def _cue_micro_motion(self, samples: List[Sample]) -> CueResult:
        key = "micro_motion"
        if (off := self._disabled(key)) is not None:
            return off
        c = self.cfg.micro_motion
        if len(samples) < 6:
            return self._mk(key, None, 0.0, {"帧数": len(samples)}, "采样不足")

        motion, deform, jitter = [], [], []
        for a, b in zip(samples[:-1], samples[1:]):
            iod = max(a.iod, 1e-3)
            ca = (a.bbox[:2] + a.bbox[2:]) / 2.0
            cb = (b.bbox[:2] + b.bbox[2:]) / 2.0
            motion.append(float(np.linalg.norm(cb - ca)) / iod)
            if a.lm106 is not None and b.lm106 is not None:
                sim = fit_similarity(a.lm106, b.lm106, iod)
                deform.append(sim.residual)
            if a.pose is not None and b.pose is not None:
                jitter.append(float(np.linalg.norm(b.pose[:3] - a.pose[:3])))

        mv = float(np.median(motion)) if motion else 0.0
        df = float(np.median(deform)) if deform else 0.0
        jt = float(np.median(jitter)) if jitter else 0.0

        s_motion = ramp(mv, c.motion_static_below, c.motion_live_above)
        s_shake = ramp_inv(jt, c.jitter_calm_above, c.jitter_violent_above)
        s_deform = ramp(df, c.deform_fake_below, c.deform_live_above)
        score = 0.45 * s_motion + 0.25 * s_shake + 0.30 * s_deform

        if s_motion < 0.15:
            hint = "画面几乎冻结:静置照片 / 暂停的视频"
        elif s_shake < 0.3:
            hint = "抖动过猛:手持屏幕或视频快进"
        elif s_deform < 0.3:
            hint = "有整体运动但几乎无非刚性形变"
        else:
            hint = "运动量与自然形变都在正常区间"
        return self._mk(
            key,
            score,
            clamp(len(motion) / 10.0),
            {
                "帧间位移": f"{mv:.4f}",
                "非刚性形变": f"{df:.4f}",
                "姿态抖动°": f"{jt:.2f}",
                "运动分": f"{s_motion:.2f}",
                "平稳分": f"{s_shake:.2f}",
                "形变分": f"{s_deform:.2f}",
            },
            hint,
            extra={"motion": mv, "deform": df, "jitter_deg": jt},
        )

    def _cue_planarity(self, samples: List[Sample]) -> CueResult:
        key = "planarity"
        if (off := self._disabled(key)) is not None:
            return off
        p = self.cfg.planarity
        self.last_homography = None
        if len(samples) < 4:
            return self._mk(key, None, 0.0, {"帧数": len(samples)}, "采样不足")

        cur = samples[-1]
        if cur.lm106 is None:
            return self._mk(key, None, 0.0, {}, "缺少关键点")

        # 取「大约 reference_age_sec 之前」的那一帧作参考,太近则位移被噪声淹没
        target = cur.t - p.reference_age_sec
        ref = min(
            (s for s in samples[:-1] if s.lm106 is not None),
            key=lambda s: abs(s.t - target),
            default=None,
        )
        if ref is None or (cur.t - ref.t) < p.reference_age_sec * 0.45:
            return self._mk(
                key, None, 0.0,
                {"基线": f"{(cur.t - ref.t):.2f}s" if ref else "-"},
                "轨迹太短,还没攒够时间基线",
            )

        fit = fit_homography(ref.lm106, cur.lm106, ref.iod, p.ransac_reproj_px)
        self.last_homography = fit
        raw = {
            "单应残差": f"{fit.residual_median:.4f}",
            "残差P90": f"{fit.residual_p90:.4f}",
            "位移量": f"{fit.motion:.4f}",
            "内点率": f"{fit.inlier_ratio:.2f}",
            "时间基线": f"{cur.t - ref.t:.2f}s",
        }
        if fit.motion < p.require_motion_norm:
            return self._mk(
                key, None, 0.15, raw,
                f"头部转动不足({fit.motion:.4f} < {p.require_motion_norm}),线索暂不可用",
                extra=_homo_extra(fit),
            )

        score = ramp(fit.residual_median, p.residual_fake_below, p.residual_live_above)
        # 转动越大,平面与非平面的差异越明显,置信度越高
        conf = clamp(fit.motion / max(p.require_motion_norm * 3.0, 1e-6)) * clamp(len(samples) / 15.0)
        hint = (
            "残差极小:位移场可被单应完全解释 → 平面载体"
            if score < 0.25
            else "残差显著:存在单应无法解释的透视视差 → 三维实体"
            if score > 0.75
            else "残差处于过渡带,证据不够强"
        )
        return self._mk(key, score, conf, raw, hint, extra=_homo_extra(fit))

    def _cue_texture(self, samples: List[Sample]) -> CueResult:
        key = "texture"
        if (off := self._disabled(key)) is not None:
            return off
        c = self.cfg.texture
        recent = samples[-15:]
        if len(recent) < 3:
            return self._mk(key, None, 0.0, {"帧数": len(recent)}, "采样不足")
        lap = float(np.median([s.m["lap"] for s in recent if "lap" in s.m]))
        band = float(np.median([s.m.get("band_ratio", 0.0) for s in recent]))
        s1 = ramp(lap, c.laplacian_fake_below, c.laplacian_live_above)
        s2 = ramp(band, c.band_ratio_fake_below, c.band_ratio_live_above)
        return self._mk(
            key,
            0.6 * s1 + 0.4 * s2,
            clamp(len(recent) / 8.0),
            {"Laplacian方差": f"{lap:.1f}", "高低频比": f"{band:.3f}", "纹理分": f"{s1:.2f}", "频带分": f"{s2:.2f}"},
            "对齐到 112x112 后统计,消除人脸远近的影响",
            extra={"lap": lap, "band_ratio": band},
        )

    def _cue_moire(self, samples: List[Sample]) -> CueResult:
        key = "moire"
        if (off := self._disabled(key)) is not None:
            return off
        c = self.cfg.moire
        recent = [s for s in samples[-20:] if s.spectrum is not None]
        if len(recent) < 3:
            return self._mk(key, None, 0.0, {"帧数": len(recent)}, "采样不足")
        pr = float(np.median([s.m["peak_ratio"] for s in recent]))
        pc = float(np.median([s.m["peak_count"] for s in recent]))
        s_ratio = ramp_inv(pr, c.peak_ratio_live_below, c.peak_ratio_fake_above)
        s_count = 1.0 if pc < c.min_peaks else ramp_inv(pc, c.min_peaks, c.min_peaks * 3.0)
        score = 0.65 * s_ratio + 0.35 * s_count
        hint = (
            f"检出 {int(pc)} 个孤立频谱尖峰(峰噪比 {pr:.1f}) → 疑似像素栅格"
            if score < 0.35
            else "频谱无周期性尖峰,高频能量平滑衰减"
        )
        return self._mk(
            key,
            score,
            clamp(len(recent) / 8.0),
            {"峰噪比": f"{pr:.2f}", "尖峰数": int(pc), "高频占比": f"{recent[-1].m.get('band_ratio', 0):.3f}"},
            hint,
            extra={"peak_ratio": pr, "peak_count": pc},
        )

    def _cue_color(self, samples: List[Sample]) -> CueResult:
        key = "color"
        if (off := self._disabled(key)) is not None:
            return off
        c = self.cfg.color
        recent = samples[-15:]
        if len(recent) < 3:
            return self._mk(key, None, 0.0, {"帧数": len(recent)}, "采样不足")
        sat = float(np.median([s.m["sat"] for s in recent]))
        skin = float(np.median([s.m["skin"] for s in recent]))
        s_sat = ramp_inv(sat, c.sat_live_below, c.sat_fake_above)
        s_skin = ramp(skin, c.skin_ratio_fake_below, 0.70)
        return self._mk(
            key,
            0.5 * s_sat + 0.5 * s_skin,
            clamp(len(recent) / 8.0) * 0.8,   # 这一路受白平衡影响大,整体压一点置信度
            {"平均饱和度": f"{sat:.3f}", "肤色占比": f"{skin:.3f}", "饱和分": f"{s_sat:.2f}", "肤色分": f"{s_skin:.2f}"},
            "屏幕背光偏冷、饱和度高;打印件偏灰、饱和度低",
            extra={"sat": sat, "skin": skin},
        )


def _homo_extra(fit) -> Dict[str, float]:
    return {
        "homo_resid": fit.residual_median,
        "homo_p90": fit.residual_p90,
        "homo_motion": fit.motion,
        "inlier": fit.inlier_ratio,
    }


def _explain(
    verdict: str,
    cues: List[CueResult],
    veto: Optional[CueResult],
    n: int,
    cfg: LivenessCfg,
    static: float,
    temporal: Optional[float],
) -> str:
    if verdict == "observing":
        return f"证据积累中({n}/{cfg.min_track_frames} 帧),请稍微转动头部"
    if veto is not None:
        return f"「{veto.label}」触发一票否决:{veto.hint}"
    gate_txt = (
        f"单帧 {static:.2f} × 时间门控 {temporal:.2f}"
        if temporal is not None
        else f"单帧 {static:.2f}(时间证据不足)"
    )
    active = sorted((c for c in cues if c.contributes), key=lambda c: c.score or 0)
    if not active:
        return "没有可用线索"
    worst, best = active[0], active[-1]
    if verdict == "live":
        return f"通过({gate_txt})。最弱线索「{worst.label}」={worst.score:.2f}"
    return f"判为攻击载体({gate_txt})。主要证据:「{worst.label}」={worst.score:.2f} — {worst.hint}"
