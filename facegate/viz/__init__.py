"""调试面板合成器。

布局(基准 1320x976,运行时按屏幕等比缩放):

    ┌─────────────────────────┬──────────────────────────┐
    │                         │ P1 FFT 幅度谱 + 摩尔纹峰    │
    │   主画面 720x540         ├──────────────────────────┤
    │   (HUD 叠加)             │ P2 对齐脸/纹理/分组残差     │
    │                         ├──────────────────────────┤
    │                         │ P3 时序波形               │
    ├─────────────────────────┴──────────────────────────┤
    │ 底栏一:七路线索得分条 · 融合仪表 · 库内相似度            │
    ├────────────────────────────────────────────────────┤
    │ 底栏二:逐线索判据(实测值/阈值/结论) · 载体类型匹配       │
    └────────────────────────────────────────────────────┘
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from insightface.utils import face_align

from ..config import Config
from ..engine import EngineStats, FaceObs
from ..gate import GateState
from ..gallery import Match
from ..liveness.analyzer import LivenessAnalyzer, LivenessResult
from ..liveness.attack import classify_attack
from .hud import draw_main_view, hud_texts
from .panels import (
    BOTTOM_H,
    MAIN_H,
    MAIN_W,
    PANEL_W,
    REASON_H,
    ROW_H,
    TOTAL_H,
    TOTAL_W,
    Canvas,
    draw_cue_bars,
    draw_face_tiles,
    draw_gallery,
    draw_gauge,
    draw_reason_band,
    draw_spectrum,
    draw_toolbar,
    draw_waveform,
)

__all__ = ["Dashboard", "BASE_W", "BASE_H"]

BASE_W, BASE_H = TOTAL_W, TOTAL_H


def aligned_landmarks(kps: np.ndarray, lm106: np.ndarray, size: int = 112) -> Optional[np.ndarray]:
    """把原图坐标下的 106 点映射到 ArcFace 对齐裁剪坐标系,便于叠在对齐图上显示。"""
    try:
        M = face_align.estimate_norm(np.asarray(kps, dtype=np.float32), size)
    except Exception:  # noqa: BLE001
        return None
    pts = np.asarray(lm106, dtype=np.float32).reshape(-1, 1, 2)
    return cv2.transform(pts, M).reshape(-1, 2)


class Dashboard:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        from ..util import TextPainter

        self.painter = TextPainter(cfg.viz.font, cfg.viz.font_size)
        self.hits: List[tuple] = []
        self._transform = (1.0, 0, 0)
        self.base_h = TOTAL_H

    # -- 交互 --------------------------------------------------------------

    def display(self, base: np.ndarray, win_w: int, win_h: int) -> np.ndarray:
        """把基准画布等比缩放并居中放进当前窗口尺寸(letterbox),记录变换供鼠标反解。"""
        bh = base.shape[0]
        scale = min(win_w / BASE_W, win_h / bh)
        dw, dh = max(1, int(BASE_W * scale)), max(1, int(bh * scale))
        ox, oy = (win_w - dw) // 2, (win_h - dh) // 2
        out = np.zeros((win_h, win_w, 3), dtype=np.uint8)
        interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
        out[oy: oy + dh, ox: ox + dw] = cv2.resize(base, (dw, dh), interpolation=interp)
        self._transform = (scale, ox, oy)
        return out

    def to_base(self, mx: int, my: int) -> Tuple[float, float]:
        scale, ox, oy = self._transform
        return (mx - ox) / scale, (my - oy) / scale

    def hit_test(self, bx: float, by: float):
        for x, y, w, h, action, payload in reversed(self.hits):
            if x <= bx <= x + w and y <= by <= y + h:
                return action, payload
        return None, None

    # -- 渲染 --------------------------------------------------------------

    def render(
        self,
        frame: np.ndarray,
        obs: Optional[FaceObs],
        gate: GateState,
        lv: Optional[LivenessResult],
        analyzer: LivenessAnalyzer,
        matches: Sequence[Match],
        fps: float,
        dfps: float,
        stats: EngineStats,
        source: str,
        similarity: float,
        gallery_total: int = 0,
        notice: str = "",
        ui_state: Optional[Dict[str, object]] = None,
    ) -> np.ndarray:
        minimal = bool(self.cfg.viz.minimal)
        height = (MAIN_H + BOTTOM_H) if minimal else TOTAL_H
        cv = Canvas(self.painter, TOTAL_W, height)
        self.hits = cv.hits
        self.base_h = height

        main = draw_main_view(
            frame, obs, gate, lv, analyzer, self.cfg.viz, fps, stats, MAIN_W, MAIN_H
        )
        cv.blit(0, 0, main)
        draw_toolbar(cv, 8, 8, ui_state or {})
        hud_texts(cv, obs, gate, lv, fps, dfps, stats, source, self.cfg.viz)

        if not minimal:
            px = MAIN_W
            draw_spectrum(
                cv, px, 0, PANEL_W, ROW_H[0],
                analyzer.samples[-1].spectrum if analyzer.samples else None,
                self.cfg.liveness,
            )

            alm = None
            if obs is not None and obs.lm106 is not None:
                alm = aligned_landmarks(obs.kps, obs.lm106)
            draw_face_tiles(
                cv, px, ROW_H[0], PANEL_W, ROW_H[1],
                obs.aligned if obs is not None else None,
                alm,
                analyzer.last_homography,
                obs.lm106 if obs is not None else None,
                None,
                obs.iod if obs is not None else 0.0,
            )

            draw_waveform(
                cv, px, ROW_H[0] + ROW_H[1], PANEL_W, ROW_H[2],
                list(analyzer.samples), self.cfg.viz.waveform_seconds,
                self.cfg.liveness.blink,
                lv.history if lv is not None else None,
                similarity,
            )

        by = MAIN_H
        cues: List = lv.cues if lv is not None else []
        total_w = sum(c.weight * c.confidence for c in cues if c.contributes)
        draw_cue_bars(cv, 0, by, 660, BOTTOM_H, cues, total_w)
        draw_gauge(
            cv, 660, by, 300, BOTTOM_H, lv, gate.verdict, gate.verdict_cn,
            self.cfg.liveness.decision_threshold,
        )
        draw_gallery(cv, 960, by, TOTAL_W - 960, BOTTOM_H, list(matches), self.cfg.gate, gallery_total)

        if not minimal:
            ry = MAIN_H + BOTTOM_H
            attack = classify_attack(cues) if lv is not None else None
            draw_reason_band(cv, 0, ry, TOTAL_W, REASON_H, cues, self.cfg.liveness, attack)

        if notice:
            cv.text(TOTAL_W // 2 - 220, 4, notice, (60, 220, 255), 16)
        return cv.finish()
