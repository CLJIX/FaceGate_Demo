"""调试面板:把活体检测的中间量画出来。

布局(总 1210x680):

    ┌──────────────────────┬────────────────────────┐
    │                      │ P1 FFT 幅度谱 + 摩尔纹峰 │
    │   主画面 640x480      ├────────────────────────┤
    │   (HUD 叠加)          │ P2 对齐脸/纹理/分组残差  │
    │                      ├────────────────────────┤
    │                      │ P3 时序波形             │
    ├──────────────────────┴────────────────────────┤
    │ 底栏:七路线索得分条 · 融合仪表 · 库内相似度        │
    └───────────────────────────────────────────────┘

所有中文文本先收集起来,最后由 ``TextPainter`` 一次性光栅化,避免每帧几十次
PIL 往返。
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from ..config import VizCfg
from ..gallery import Match
from ..liveness.analyzer import CueResult, LivenessResult, Sample
from ..liveness.attack import DISCRIMINATORS, AttackAnalysis, explain_cue
from ..liveness.geometry import (
    IDX_BROW_L,
    IDX_BROW_R,
    IDX_CONTOUR,
    IDX_EYE_L,
    IDX_EYE_R,
    IDX_MOUTH,
    IDX_NOSE,
    HomographyFit,
    Spectrum,
)
from ..util import TextPainter, clamp

BG = (24, 24, 28)
PANEL_BG = (34, 34, 40)
GRID = (62, 62, 72)
FG = (222, 222, 226)
DIM = (150, 150, 158)
ACCENT = (0, 200, 255)
GOOD = (80, 220, 120)
BAD = (70, 90, 250)
WARN = (60, 200, 255)

MAIN_W, MAIN_H = 720, 540
PANEL_W = 600
ROW_H = (150, 160, 230)
BOTTOM_H = 200
REASON_H = 258
TOTAL_W = MAIN_W + PANEL_W
TOTAL_H = MAIN_H + BOTTOM_H + REASON_H

LANDMARK_GROUPS: Tuple[Tuple[str, Tuple[int, ...]], ...] = (
    ("轮廓", IDX_CONTOUR),
    ("左眉", IDX_BROW_L),
    ("右眉", IDX_BROW_R),
    ("左眼", IDX_EYE_L),
    ("右眼", IDX_EYE_R),
    ("鼻", IDX_NOSE),
    ("嘴", IDX_MOUTH),
)


_BLANK_CACHE: Dict[Tuple[int, int], np.ndarray] = {}


class Canvas:
    """带延迟文本渲染与点击命中区域的画布。

    ``hits`` 收集 ``(x, y, w, h, action, payload)``;主循环把鼠标坐标反解到基准画布
    坐标后做命中测试,从而让面板上的按钮、线索条、列表行都可以点击。
    """

    def __init__(self, painter: TextPainter, w: int = TOTAL_W, h: int = TOTAL_H):
        blank = _BLANK_CACHE.get((w, h))
        if blank is None:
            blank = np.full((h, w, 3), BG, dtype=np.uint8)
            _BLANK_CACHE[(w, h)] = blank
        # 整屏 np.full 约 7ms,拷贝缓存模板只要 ~1ms
        self.img = blank.copy()
        self.painter = painter
        self._texts: List[Tuple[int, int, str, Tuple[int, int, int], int]] = []
        self.hits: List[Tuple[int, int, int, int, str, object]] = []

    # -- 交互区域 ----------------------------------------------------------

    def hit(self, x: int, y: int, w: int, h: int, action: str, payload: object = None) -> None:
        self.hits.append((int(x), int(y), int(w), int(h), action, payload))

    def button(
        self,
        x: int,
        y: int,
        w: int,
        h: int,
        label: str,
        action: str,
        active: bool = False,
        payload: object = None,
        color=ACCENT,
    ) -> int:
        bg = (58, 58, 70) if active else (44, 44, 52)
        self.filled(x, y, w, h, bg)
        self.rect(x, y, w, h, color if active else (86, 86, 98))
        tw = self.painter.measure(label, 12)[0]
        self.text(x + max(4, (w - tw) // 2), y + h - 6, label, FG if active else (200, 200, 208), 12)
        self.hit(x, y, w, h, action, payload)
        return w

    # -- 基础绘制 ----------------------------------------------------------

    def text(self, x: int, y: int, s: str, color=FG, size: Optional[int] = None) -> int:
        size = size or self.painter.size
        self._texts.append((int(x), int(y), str(s), color, size))
        return int(self.painter.measure(str(s), size)[0])

    def rtext(self, right: int, y: int, s: str, color=FG, size: Optional[int] = None) -> None:
        size = size or self.painter.size
        w = self.painter.measure(str(s), size)[0]
        self.text(right - w, y, s, color, size)

    def rect(self, x: int, y: int, w: int, h: int, color=GRID, thickness: int = 1) -> None:
        cv2.rectangle(self.img, (x, y), (x + w - 1, y + h - 1), color, thickness)

    def filled(self, x: int, y: int, w: int, h: int, color) -> None:
        if w <= 0 or h <= 0:
            return
        cv2.rectangle(self.img, (x, y), (x + w - 1, y + h - 1), color, -1)

    def line(self, p1, p2, color=GRID, thickness=1) -> None:
        cv2.line(self.img, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), color, thickness, cv2.LINE_AA)

    def blit(self, x: int, y: int, tile: np.ndarray) -> None:
        h, w = tile.shape[:2]
        H, W = self.img.shape[:2]
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(W, x + w), min(H, y + h)
        if x2 <= x1 or y2 <= y1:
            return
        self.img[y1:y2, x1:x2] = tile[y1 - y: y2 - y, x1 - x: x2 - x]

    def panel(self, x: int, y: int, w: int, h: int, title: str) -> Tuple[int, int, int, int]:
        """画一个带标题的面板,返回内容区 (cx, cy, cw, ch)。"""
        self.filled(x, y, w, h, PANEL_BG)
        self.rect(x, y, w, h, GRID)
        self.filled(x + 1, y + 1, w - 2, 19, (44, 44, 52))
        self.text(x + 8, y + 15, title, ACCENT, self.painter.size)
        return x + 8, y + 26, w - 16, h - 32

    def finish(self) -> np.ndarray:
        return self.painter.render(self.img, self._texts)


# ---------------------------------------------------------------------------
# P1 频谱
# ---------------------------------------------------------------------------


_SPEC_TILE_CACHE: Dict[int, tuple] = {}


def draw_spectrum(cv: Canvas, x: int, y: int, w: int, h: int, spec: Optional[Spectrum], cfg) -> None:
    cx, cy, cw, ch = cv.panel(x, y, w, h, "P1  FFT 幅度谱 — 屏幕摩尔纹检测")
    if spec is None or spec.mag.size < 64:
        cv.text(cx, cy + ch // 2, "无频谱数据(人脸 ROI 太小)", DIM)
        return

    tile_h = ch - 34
    cached = _SPEC_TILE_CACHE.get(id(spec))
    if cached is not None and cached[0] is spec:
        vis, scale = cached[1], cached[2]
    else:
        mag = spec.mag
        lo, hi = np.percentile(mag, 1), np.percentile(mag, 99.9)
        norm = np.clip((mag - lo) / max(hi - lo, 1e-6), 0, 1)
        vis = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
        scale = min(cw / vis.shape[1], tile_h / vis.shape[0])
        nw, nh = max(1, int(vis.shape[1] * scale)), max(1, int(vis.shape[0] * scale))
        vis = cv2.resize(vis, (nw, nh), interpolation=cv2.INTER_NEAREST)
        if len(_SPEC_TILE_CACHE) > 4:
            _SPEC_TILE_CACHE.clear()
        _SPEC_TILE_CACHE[id(spec)] = (spec, vis, scale)
    nw, nh = vis.shape[1], vis.shape[0]
    cv.blit(cx, cy, vis)

    if spec.peaks is not None and len(spec.peaks):
        for px, py in spec.peaks[:24]:
            pxx = int(cx + px * scale)
            pyy = int(cy + py * scale)
            cv2.circle(cv.img, (pxx, pyy), 5, (0, 60, 255), 1, cv2.LINE_AA)
            cv2.circle(cv.img, (pxx, pyy), 1, (255, 255, 255), -1)
    cv2.circle(cv.img, (cx + nw // 2, cy + nh // 2), 3, (255, 255, 255), 1)

    tx = cx + nw + 10
    if tx < x + w - 120:
        cv.text(tx, cy + 12, "红圈 = 检出的孤立尖峰", DIM, 12)
        cv.text(tx, cy + 30, "白点 = 直流分量", DIM, 12)

    yy = cy + tile_h + 6
    verdict = "疑似屏幕" if spec.peak_count >= cfg.moire.min_peaks and spec.peak_ratio > cfg.moire.peak_ratio_fake_above else "无周期性尖峰"
    col = BAD if verdict == "疑似屏幕" else GOOD
    cv.text(cx, yy + 12, f"峰噪比 {spec.peak_ratio:6.2f}   尖峰数 {spec.peak_count:3d}   高频占比 {spec.band_ratio:.3f}", FG, 13)
    cv.rtext(x + w - 8, yy + 12, verdict, col, 13)


# ---------------------------------------------------------------------------
# P2 对齐人脸 / 纹理 / 分组残差
# ---------------------------------------------------------------------------


def draw_face_tiles(
    cv: Canvas,
    x: int,
    y: int,
    w: int,
    h: int,
    aligned: Optional[np.ndarray],
    aligned_lm: Optional[np.ndarray],
    homography: Optional[HomographyFit],
    cur_lm: Optional[np.ndarray],
    bbox_origin: Optional[Tuple[float, float]],
    iod: float,
) -> None:
    cx, cy, cw, ch = cv.panel(x, y, w, h, "P2  对齐人脸(112²) · Laplacian 纹理 · 单应残差分组")
    tile = min(ch, 118)
    gap = (cw - tile * 3) // 4

    # 1) 对齐人脸
    if aligned is not None:
        a = cv2.resize(aligned, (tile, tile), interpolation=cv2.INTER_NEAREST)
        if aligned_lm is not None:
            k = tile / 112.0
            for p in aligned_lm:
                cv2.circle(a, (int(p[0] * k), int(p[1] * k)), 1, (0, 255, 200), -1)
        cv.blit(cx + gap, cy, a)
        cv.text(cx + gap, cy + tile + 14, "对齐裁剪 + 106点", DIM, 11)
    else:
        cv.text(cx + gap, cy + tile // 2, "无对齐图", DIM, 11)

    # 2) Laplacian 响应
    x2 = cx + gap * 2 + tile
    if aligned is not None:
        g = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY)
        lap = np.abs(cv2.Laplacian(g, cv2.CV_32F))
        lap = cv2.resize(lap, (tile, tile), interpolation=cv2.INTER_NEAREST)
        n = np.clip(lap / max(float(np.percentile(lap, 99)), 1e-6), 0, 1)
        heat = cv2.applyColorMap((n * 255).astype(np.uint8), cv2.COLORMAP_JET)
        cv.blit(x2, cy, heat)
        cv.text(x2, cy + tile + 14, f"Laplacian 方差 {float(lap.var()):.0f}", DIM, 11)
    else:
        cv.text(x2, cy + tile // 2, "无纹理图", DIM, 11)

    # 3) 分组残差柱状图
    x3 = cx + gap * 3 + tile * 2
    bw = (cw - gap * 3 - tile * 2)
    cv.text(x3, cy + 10, "各部位单应残差(×1000/瞳距)", DIM, 11)
    if homography is not None and homography.ok and homography.residual is not None and len(homography.residual) == 106:
        base = cy + 18
        avail = tile - 18
        r = homography.residual
        rmax = max(float(np.percentile(r, 98)), 1e-4)
        for i, (name, idx) in enumerate(LANDMARK_GROUPS):
            val = float(np.median(r[list(idx)]))
            bh = int(clamp(val / rmax, 0, 1) * avail)
            bx = x3 + i * (bw // len(LANDMARK_GROUPS))
            bwidth = max(6, bw // len(LANDMARK_GROUPS) - 6)
            col = (80, 200, 255) if val < rmax * 0.7 else (60, 120, 255)
            cv.filled(bx, base + avail - bh, bwidth, bh, col)
            cv.rect(bx, base, bwidth, avail, GRID)
            cv.text(bx - 2, base + avail + 13, name, DIM, 11)
        cv.text(x3, base + avail + 28, f"中位 {homography.residual_median:.4f} · P90 {homography.residual_p90:.4f}", FG, 12)
    else:
        cv.text(x3, cy + tile // 2, "轨迹太短或头未转动", DIM, 11)


# ---------------------------------------------------------------------------
# P3 时序波形
# ---------------------------------------------------------------------------


def _polyline(cv: Canvas, xs: np.ndarray, ys: np.ndarray, color, thickness=1) -> None:
    if len(xs) < 2:
        return
    pts = np.stack([xs, ys], axis=1).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(cv.img, [pts], False, color, thickness, cv2.LINE_AA)


def draw_waveform(
    cv: Canvas,
    x: int,
    y: int,
    w: int,
    h: int,
    samples: Sequence[Sample],
    seconds: float,
    blink_cfg,
    history: Optional[np.ndarray],
    similarity: float,
) -> None:
    cx, cy, cw, ch = cv.panel(x, y, w, h, "P3  时序波形 — 每帧原始指标")
    if len(samples) < 2:
        cv.text(cx, cy + ch // 2, "等待轨迹积累…", DIM)
        return

    t1 = samples[-1].t
    t0 = max(t1 - seconds, samples[0].t)
    win = [s for s in samples if s.t >= t0]
    if len(win) < 2:
        win = list(samples)[-2:]
    ts = np.asarray([s.t for s in win], dtype=np.float64)

    rows = [
        ("EAR 眼睛开合", [s.m.get("ear", -1) for s in win], 0.0, 0.6, (120, 230, 120)),
        ("姿态角 pitch/yaw/roll", None, -45.0, 45.0, None),
        ("分数 活体/官方/相似度", None, 0.0, 1.0, None),
    ]
    rh = (ch - 8) // len(rows)

    for ri, (label, series, lo, hi, color) in enumerate(rows):
        ry = cy + ri * rh
        cv.rect(cx, ry, cw, rh - 4, (46, 46, 54))
        cv.text(cx + 4, ry + 12, label, DIM, 11)
        px = lambda t: cx + (t - t0) / max(t1 - t0, 1e-6) * cw  # noqa: E731
        py = lambda v: ry + rh - 6 - (np.clip((np.asarray(v, dtype=np.float64) - lo) / (hi - lo), 0, 1) * (rh - 14))  # noqa: E731

        if ri == 0:
            vals = np.asarray(series, dtype=np.float64)
            valid = vals >= 0
            base = float(np.percentile(vals[valid], 85)) if valid.sum() > 3 else 0.3
            _polyline(cv, px(ts[valid]), py(vals[valid]), color, 1)
            thr = base * blink_cfg.ear_close_ratio
            cv.line((cx, py(thr)), (cx + cw, py(thr)), BAD, 1)
            cv.line((cx, py(base)), (cx + cw, py(base)), GRID, 1)
            cv.rtext(cx + cw - 4, ry + 12, f"基线 {base:.3f} / 闭眼线 {thr:.3f}", DIM, 11)

        elif ri == 1:
            for key, col in (("pitch", (90, 190, 255)), ("yaw", (255, 190, 90)), ("roll", (220, 120, 255))):
                vals = np.asarray([s.m.get(key, 0.0) for s in win], dtype=np.float64)
                _polyline(cv, px(ts), py(vals), col, 1)
            cv.rtext(cx + cw - 4, ry + 12, "pitch  yaw  roll", DIM, 11)

        else:
            live = np.asarray([max(s.m.get("live", -1.0), 0.0) for s in win], dtype=np.float64)
            _polyline(cv, px(ts), py(live), (80, 120, 255), 1)
            if history is not None and len(history) > 1:
                hh = np.asarray(history[-len(win):], dtype=np.float64)
                if len(hh) > 1:
                    _polyline(cv, px(ts[-len(hh):]), py(hh), (255, 255, 255), 2)
            if similarity > 0:
                cv.line((cx, py(similarity)), (cx + cw, py(similarity)), (60, 200, 60), 1)
            cv.rtext(cx + cw - 4, ry + 12, "白=融合 蓝=官方CNN 绿线=相似度", DIM, 11)

    for gx in range(1, 8):
        gxx = cx + int(cw * gx / 8)
        cv.line((gxx, cy), (gxx, cy + ch - 4), (40, 40, 46), 1)


# ---------------------------------------------------------------------------
# 底栏
# ---------------------------------------------------------------------------


def draw_cue_bars(cv: Canvas, x: int, y: int, w: int, h: int, cues: List[CueResult], total_w: float) -> None:
    cx, cy, cw, ch = cv.panel(x, y, w, h, "七路线索得分  (0=攻击载体  1=真人  ▒=证据不足  ·  点击某行可开关该线索)")
    rowh = (ch - 6) // max(len(cues), 1)
    bar_w = int(cw * 0.30)
    for i, c in enumerate(cues):
        ry = cy + 2 + i * rowh
        cv.hit(cx, ry, cw, rowh, "toggle_cue", c.key)
        share = (c.weight * c.confidence / total_w * 100) if total_w > 1e-6 else 0.0
        name = f"{c.label}"
        cv.text(cx, ry + rowh - 5, name, FG, 13)
        bx = cx + 108
        cv.filled(bx, ry + 2, bar_w, rowh - 6, (48, 48, 56))
        if c.score is None:
            for k in range(0, bar_w, 6):
                cv.filled(bx + k, ry + 2, 3, rowh - 6, (70, 70, 80))
        else:
            fillw = int(clamp(c.score) * bar_w)
            col = GOOD if c.score >= 0.6 else (WARN if c.score >= 0.35 else BAD)
            cv.filled(bx, ry + 2, fillw, rowh - 6, col)
            conf_w = int(clamp(c.confidence) * bar_w)
            cv.line((bx + conf_w, ry + 2), (bx + conf_w, ry + rowh - 5), (255, 255, 255), 1)
        cv.rect(bx, ry + 2, bar_w, rowh - 6, GRID)
        sv = "  n/a " if c.score is None else f"{c.score:.3f}"
        cv.text(bx + bar_w + 8, ry + rowh - 5, sv, FG, 13)
        cv.text(bx + bar_w + 62, ry + rowh - 5, f"权重占比 {share:4.1f}%", DIM, 11)
        if c.veto:
            cv.text(bx + bar_w + 160, ry + rowh - 5, "一票否决", BAD, 12)
        else:
            hint = c.hint if len(c.hint) <= 34 else c.hint[:33] + "…"
            cv.text(bx + bar_w + 160, ry + rowh - 5, hint, DIM, 11)


def draw_gauge(
    cv: Canvas,
    x: int,
    y: int,
    w: int,
    h: int,
    lv: Optional[LivenessResult],
    verdict: str,
    verdict_cn: str,
    threshold: float = 0.5,
) -> None:
    cx, cy, cw, ch = cv.panel(x, y, w, h, "融合判决")
    colors = {"grant": GOOD, "deny": BAD, "spoof": BAD, "unenrolled": WARN, "observing": ACCENT, "no_face": DIM}
    col = colors.get(verdict, FG)

    score = lv.score if lv else 0.0
    # 半圆仪表
    cxc, cyc = cx + cw // 2, cy + 78
    r = 62
    cv2.ellipse(cv.img, (cxc, cyc), (r, r), 180, 0, 180, (48, 48, 56), 12, cv2.LINE_AA)
    cv2.ellipse(cv.img, (cxc, cyc), (r, r), 180, 180, 180 + 180 * clamp(score), col, 12, cv2.LINE_AA)
    # 阈值刻度线
    ta = np.pi * (1.0 - clamp(threshold))
    cv.line(
        (cxc + int(np.cos(ta) * (r - 16)), cyc - int(np.sin(ta) * (r - 16))),
        (cxc + int(np.cos(ta) * (r + 8)), cyc - int(np.sin(ta) * (r + 8))),
        (230, 230, 235), 2,
    )
    ang = np.pi * (1.0 - clamp(score))
    cv.line((cxc, cyc), (cxc + int(np.cos(ang) * (r - 14)), cyc - int(np.sin(ang) * (r - 14))), FG, 2)
    cv.text(cxc - 26, cyc + 22, f"{score:.2f}", FG, 20)
    cv.text(cx + 4, cy + 14, f"判决阈值 {threshold:.2f}", DIM, 11)

    ty = cyc + 34
    cv.text(cx + 4, ty + 18, f"判定:{verdict_cn}", col, 18)
    if lv is not None:
        tstr = "n/a" if lv.temporal is None else f"{lv.temporal:.2f}"
        cv.text(cx + 4, ty + 38, f"单帧 {lv.static:.2f}  ×  时间门控 {tstr}", (210, 210, 215), 13)
        cv.text(cx + 4, ty + 56, f"当帧融合 {lv.instant:.3f}   平滑 {lv.score:.3f}", DIM, 12)
        cv.text(cx + 4, ty + 74, f"轨迹 {lv.frames} 帧 / {lv.elapsed:.1f}s", DIM, 12)
    else:
        cv.text(cx + 4, ty + 38, "尚无活体结果", DIM, 12)


def draw_gallery(cv: Canvas, x: int, y: int, w: int, h: int, matches: List[Match], gate_cfg, total: int) -> None:
    cx, cy, cw, ch = cv.panel(x, y, w, h, f"库内相似度  (共 {total} 人 · 点击行进入管理)")
    bw = cv.button(x + w - 74, y + 2, 66, 17, "管理人脸库", "manage", color=ACCENT)
    if not matches:
        cv.text(cx, cy + 30, "人脸库为空", DIM, 13)
        cv.button(cx, cy + 44, 96, 22, "录入新人脸", "enroll", color=GOOD)
        return

    bar_w = cw - 150
    rowh = min(22, (ch - 18) // max(len(matches), 1))
    for i, m in enumerate(matches[:8]):
        ry = cy + 4 + i * rowh
        cv.hit(cx - 4, ry, cw, rowh, "gallery_select", m.person.id)
        cv.text(cx, ry + rowh - 6, m.person.name[:10], FG, 12)
        bx = cx + 92
        cv.filled(bx, ry + 2, bar_w, rowh - 6, (48, 48, 56))
        sim = clamp(m.similarity, -0.2, 1.0)
        fillw = int(clamp(sim, 0, 1) * bar_w)
        col = GOOD if m.similarity >= gate_cfg.accept_threshold else (
            WARN if m.similarity >= gate_cfg.stranger_threshold else BAD
        )
        cv.filled(bx, ry + 2, fillw, rowh - 6, col)
        cv.rect(bx, ry + 2, bar_w, rowh - 6, GRID)
        cv.text(bx + bar_w + 6, ry + rowh - 6, f"{m.similarity:+.3f}", FG, 12)

    for thr, name, c in (
        (gate_cfg.accept_threshold, "开门", GOOD),
        (gate_cfg.stranger_threshold, "陌生人", WARN),
    ):
        tx = cx + 92 + int(clamp(thr) * bar_w)
        cv.line((tx, cy), (tx, cy + ch - 4), c, 1)
        cv.text(tx - 12, cy + ch - 2, name, c, 10)


# ---------------------------------------------------------------------------
# 底栏二:逐线索判据 + 攻击类型判别
# ---------------------------------------------------------------------------


def _wrap(cv: Canvas, text: str, width_px: int, size: int, max_lines: int = 3) -> List[str]:
    out: List[str] = []
    cur = ""
    i = 0
    for i, ch in enumerate(text):
        if cv.painter.measure(cur + ch, size)[0] > width_px:
            out.append(cur)
            cur = ch
            if len(out) >= max_lines:
                break
        else:
            cur += ch
    else:
        if cur:
            out.append(cur)
        return out or [""]
    # 触发行数上限:剩余文本并进末行,超宽则截断加省略号
    last = out[-1]
    for ch in text[i:]:
        if cv.painter.measure(last + ch, size)[0] > width_px - 14:
            last += "…"
            break
        last += ch
    out[-1] = last
    return out


def _trunc(cv: Canvas, text: str, width_px: int, size: int) -> str:
    if cv.painter.measure(text, size)[0] <= width_px:
        return text
    while len(text) > 1 and cv.painter.measure(text + "…", size)[0] > width_px:
        text = text[:-1]
    return text + "…"


def draw_reason_band(
    cv: Canvas,
    x: int,
    y: int,
    w: int,
    h: int,
    cues: List[CueResult],
    liveness_cfg,
    attack: Optional[AttackAnalysis],
) -> None:
    cx, cy, cw, ch = cv.panel(x, y, w, h, "判决推理 — 每路线索的实测值 / 判据 / 结论,以及载体类型匹配")
    col_w = (cw - 24) // 3
    cols_x = [cx + 4, cx + 8 + col_w, cx + 16 + col_w * 2]
    line = 13
    size = 11

    groups = (cues[:4], cues[4:], None)
    for gi, (gx, group) in enumerate(zip(cols_x, groups[:2])):
        yy = cy + 2
        for c in group:
            sc = "n/a" if c.score is None else f"{c.score:.3f}"
            col = GOOD if (c.score or 0) >= 0.6 else (WARN if (c.score or 0) >= 0.35 else BAD)
            if c.score is None:
                col = DIM
            cv.text(gx, yy + line, f"{c.label}  {sc}", col, 12)
            cv.text(gx + 118, yy + line, f"权重占比见左栏", DIM, 10)
            yy += line + 2
            for ln in explain_cue(c, liveness_cfg):
                for piece in _wrap(cv, ln, col_w - 8, size):
                    cv.text(gx + (0 if not ln.startswith("  ") else 0), yy + line, piece,
                            FG if not ln.startswith("  ") else (185, 185, 192), size)
                    yy += line
            yy += 5

    # 第三列:攻击类型匹配
    gx = cols_x[2]
    yy = cy + 2
    cv.text(gx, yy + line, "载体类型匹配(与期望签名的吻合度)", ACCENT, 12)
    yy += line + 6
    if attack is None or not attack.enough:
        cv.text(gx, yy + line, attack.note if attack else "线索不足,暂不分类", DIM, size)
    else:
        bar_w = col_w - 150
        for s in attack.scores:
            is_win = attack.winner is not None and s.key == attack.winner.key
            cv.hit(gx, yy, col_w, line + 3, "attack_class", s.key)
            cv.text(gx, yy + line - 2, s.label, FG if is_win else DIM, size)
            bx = gx + 108
            cv.filled(bx, yy + 2, bar_w, line - 4, (48, 48, 56))
            cv.filled(bx, yy + 2, int(clamp(s.consistency) * bar_w), line - 4,
                      GOOD if is_win else (90, 90, 100))
            cv.rect(bx, yy + 2, bar_w, line - 4, GRID)
            cv.text(bx + bar_w + 6, yy + line - 2, f"{s.consistency:.2f}", FG if is_win else DIM, size)
            yy += line + 3
        yy += 2
        win = attack.winner
        cv.text(gx, yy + line, f"判定载体:{win.label}", (60, 200, 255), 12)
        yy += line + 3
        cv.text(gx, yy + line, _trunc(cv, f"支持:{'、'.join(win.support) or '—'}", col_w - 8, size), (185, 185, 192), size)
        yy += line
        cv.text(gx, yy + line, _trunc(cv, f"反对:{'、'.join(win.against) or '—'}", col_w - 8, size), (185, 185, 192), size)
        yy += line + 5
        cv.text(gx, yy + line, "关键判别点", ACCENT, 12)
        yy += line + 3
        for pair, keys, why in DISCRIMINATORS:
            cv.text(gx, yy + line, _trunc(cv, f"{pair}:{why}", col_w - 8, size), DIM, size)
            yy += line + 1


# ---------------------------------------------------------------------------
# 工具栏(可点击)
# ---------------------------------------------------------------------------


def draw_toolbar(cv: Canvas, x: int, y: int, state: Dict[str, object]) -> None:
    bx = x
    specs = [
        ("暂停" if state.get("paused") else "继续", "pause", bool(state.get("paused"))),
        ("录入人脸", "enroll", False),
        ("人脸库管理", "manage", False),
        ("截图", "snapshot", False),
        ("全屏" if not state.get("fullscreen") else "退出全屏", "fullscreen", bool(state.get("fullscreen"))),
        (f"性能:{state.get('profile', 'balanced')}", "perf_next", False),
        (f"叠加:{state.get('overlay', '全开')}", "overlay_next", False),
        ("重置轨迹", "reset_track", False),
    ]
    if state.get("camera") is not None:
        specs.append((f"摄像头:{state.get('camera')}", "camera_next", False))
    for label, action, active in specs:
        tw = cv.painter.measure(label, 12)[0] + 16
        bx += cv.button(bx, y, tw, 20, label, action, active=active) + 6


# ---------------------------------------------------------------------------
# 人脸库管理界面
# ---------------------------------------------------------------------------


def draw_manage(
    cv: Canvas,
    gallery,
    sel: int,
    pending: str,
    notice: str,
) -> None:
    cx, cy, cw, ch = cv.panel(0, 0, TOTAL_W, TOTAL_H, "人脸库管理 — 点击行选择,或用 ↑/↓;按钮与快捷键等效")
    people = sorted(gallery.people.values(), key=lambda p: p.created_at)

    # 操作按钮条(与快捷键等效,全部可点击)
    by = cy + 2
    bx = cx
    for label, action, danger in (
        ("删除选中", "mg_delete", True),
        ("改名", "mg_rename", False),
        ("清空全库", "mg_clear", True),
        ("录入新人", "mg_enroll", False),
        ("返回识别", "mg_back", False),
    ):
        tw = cv.painter.measure(label, 12)[0] + 18
        bx += cv.button(bx, by, tw, 22, label, action, color=BAD if danger else ACCENT) + 8
    bx += 12
    bx += cv.button(bx, by, 34, 22, "↑", "mg_up") + 6
    bx += cv.button(bx, by, 34, 22, "↓", "mg_down") + 6

    if notice:
        cv.text(cx + 4, by + 40, notice, (60, 220, 255), 14)

    if not people:
        cv.text(cx + 8, cy + 96, "人脸库为空。点「录入新人」或按 e 录入第一个人。", DIM, 16)
        cv.text(cx + 8, cy + 126, f"存储位置:{gallery.root}", DIM, 13)
        return

    row_h = 74
    list_w = int(cw * 0.62)
    top = cy + 62
    for i, p in enumerate(people):
        ry = top + i * row_h
        if ry + row_h > cy + ch:
            cv.text(cx + 8, ry + 10, f"… 还有 {len(people) - i} 人未显示", DIM, 12)
            break
        selected = i == sel
        if selected:
            cv.filled(cx + 2, ry, list_w - 8, row_h - 6, (52, 52, 62))
        cv.rect(cx + 2, ry, list_w - 8, row_h - 6, ACCENT if selected else GRID)
        cv.hit(cx + 2, ry, list_w - 8, row_h - 6, "mg_select", i)

        thumb = gallery.thumbnail(p.id)
        if thumb is not None:
            cv.blit(cx + 8, ry + 5, cv2.resize(thumb, (row_h - 16, row_h - 16)))
        tx = cx + row_h + 2
        cv.text(tx, ry + 20, p.name, FG if selected else (200, 200, 205), 16)
        cv.text(tx, ry + 44, f"id={p.id}   样本 {p.samples} 条   平均检测分 {p.mean_det_score:.3f}", DIM, 12)
        cv.text(tx, ry + 60, f"录入于 {time.strftime('%Y-%m-%d %H:%M', time.localtime(p.created_at))}", DIM, 11)

    # 右侧:选中者详情
    dx = cx + list_w + 8
    dw = cw - list_w - 12
    p = people[min(sel, len(people) - 1)]
    cv.text(dx, cy + 30, f"当前选中:{p.name}", ACCENT, 15)
    meta = gallery._meta.get(p.id, {})
    yaw = meta.get("yaw")
    pitch = meta.get("pitch")
    if yaw is not None and len(yaw):
        cv.text(dx, cy + 56, f"样本姿态覆盖", FG, 13)
        for i, (yv, pv) in enumerate(zip(yaw, pitch)):
            cv.text(dx + 8, cy + 76 + i * 17, f"#{i + 1}  yaw {float(yv):+6.1f}°   pitch {float(pv):+6.1f}°", DIM, 12)
    cv.text(dx, cy + 76 + len(meta.get('yaw', [])) * 17 + 14, f"文件:{p.id}.npz / {p.id}_thumb.jpg", DIM, 12)
    cv.text(dx, cy + 76 + len(meta.get('yaw', [])) * 17 + 34, f"目录:{gallery.root}", DIM, 12)

    if pending:
        cv.filled(TOTAL_W // 2 - 300, TOTAL_H // 2 - 46, 600, 92, (40, 40, 120))
        cv.rect(TOTAL_W // 2 - 300, TOTAL_H // 2 - 46, 600, 92, BAD, 2)
        cv.text(TOTAL_W // 2 - 280, TOTAL_H // 2 - 20, _trunc(cv, notice or pending, 560, 16), (255, 255, 255), 16)
        cv.text(
            TOTAL_W // 2 - 280, TOTAL_H // 2 + 12,
            f"再按一次 {pending[0]} 确认,其它任意键取消",
            (220, 220, 225), 13,
        )
