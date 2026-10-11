"""主画面 HUD:把检测结果与活体线索直接叠加在摄像头画面上。

叠加内容(每项都可在 config.yaml 的 viz 段单独关闭,运行时也能用快捷键切换):
  * 人脸框,颜色随门禁判定变化
  * 106 点关键点,按部位着色
  * 眼睛长轴与上下眼睑张口量 —— 眨眼线索的原始依据
  * 三维姿态轴(pitch/yaw/roll)
  * 单应预测点与实际关键点的连线 —— 平面性线索的残差,线越长说明越不像平面
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from ..config import VizCfg
from ..engine import FaceObs, draw_pose_axis
from ..gate import VERDICT_COLOR, GateState
from ..liveness.analyzer import LivenessAnalyzer, LivenessResult
from ..liveness.geometry import (
    IDX_BROW_L,
    IDX_BROW_R,
    IDX_CONTOUR,
    IDX_EYE_L,
    IDX_EYE_R,
    IDX_MOUTH,
    IDX_NOSE,
    eye_geometry,
)
from .panels import Canvas, MAIN_H, MAIN_W

MAIN_X, MAIN_Y = 0, 0

GROUP_COLORS = {
    IDX_CONTOUR: (140, 140, 160),
    IDX_BROW_L: (200, 160, 80),
    IDX_BROW_R: (200, 160, 80),
    IDX_EYE_L: (80, 230, 120),
    IDX_EYE_R: (80, 230, 120),
    IDX_NOSE: (80, 180, 255),
    IDX_MOUTH: (255, 130, 180),
}


def draw_landmarks(img: np.ndarray, lm: np.ndarray) -> None:
    """向量化打点:106 次 cv2.circle 约 1.5ms,索引赋值约 0.05ms。"""
    H, W = img.shape[:2]
    for idx, color in GROUP_COLORS.items():
        pts = np.round(lm[list(idx)]).astype(np.int64)
        xs = np.clip(pts[:, 0], 1, W - 2)
        ys = np.clip(pts[:, 1], 1, H - 2)
        img[ys, xs] = color
        img[ys - 1, xs] = color
        img[ys + 1, xs] = color
        img[ys, xs - 1] = color
        img[ys, xs + 1] = color


def draw_eyes(img: np.ndarray, lm: np.ndarray, scale: float = 1.0) -> None:
    for idx in (IDX_EYE_L, IDX_EYE_R):
        g = eye_geometry(lm[list(idx)])
        if g is None:
            continue
        a, b = tuple(np.round(g.corner_a).astype(int)), tuple(np.round(g.corner_b).astype(int))
        cv2.line(img, a, b, (255, 220, 80), 1, cv2.LINE_AA)
        up, dn = tuple(np.round(g.apex_up).astype(int)), tuple(np.round(g.apex_down).astype(int))
        cv2.line(img, up, dn, (80, 230, 120), 1, cv2.LINE_AA)
        cv2.putText(img, f"{g.openness:.2f}", (int((a[0] + b[0]) / 2) - 14, min(a[1], b[1]) - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40 * scale, (80, 230, 120), 1, cv2.LINE_AA)


def draw_homography_residual(
    img: np.ndarray,
    cur_lm: np.ndarray,
    predicted: np.ndarray,
    residual: np.ndarray,
    thresh: float,
) -> None:
    """单应预测点 → 实际关键点的残差连线,颜色按残差分三档;线越长越不像平面。"""
    rmax = max(float(residual.max()), 1e-6)
    t = residual / rmax
    for i in range(len(cur_lm)):
        p = (int(round(cur_lm[i, 0])), int(round(cur_lm[i, 1])))
        q = (int(round(predicted[i, 0])), int(round(predicted[i, 1])))
        if t[i] < 0.34:
            color = (80, 220, 120)
        elif t[i] < 0.7:
            color = (80, 180, 255)
        else:
            color = (60, 60, 255)
        cv2.line(img, q, p, color, 1, cv2.LINE_AA)
    over = np.nonzero(residual > thresh)[0]
    for i in over:
        p = (int(round(cur_lm[i, 0])), int(round(cur_lm[i, 1])))
        cv2.circle(img, p, 3, (60, 60, 255), 1, cv2.LINE_AA)


def draw_main_view(
    frame: np.ndarray,
    obs: Optional[FaceObs],
    gate: GateState,
    lv: Optional[LivenessResult],
    analyzer: LivenessAnalyzer,
    viz: VizCfg,
    fps: float,
    stats,
    w: int,
    h: int,
) -> np.ndarray:
    """把一帧原始画面渲染成 640x480 的 HUD 主视图(不含中文文本,文本由 Canvas 统一画)。"""
    fh, fw = frame.shape[:2]
    scale = min(w / fw, h / fh)
    nw, nh = int(fw * scale), int(fh * scale)
    view = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    ox, oy = (w - nw) // 2, (h - nh) // 2
    canvas[oy: oy + nh, ox: ox + nw] = view

    if obs is None:
        return canvas

    def T(p):
        return (int(p[0] * scale) + ox, int(p[1] * scale) + oy)

    color = VERDICT_COLOR.get(gate.verdict, (200, 200, 200))
    x1, y1, x2, y2 = (float(v) for v in obs.bbox)
    p1, p2 = T((x1, y1)), T((x2, y2))

    # 四角括号式人脸框,比整框更不遮挡五官
    L = 22
    for cx_, cy_, dx, dy in ((p1[0], p1[1], 1, 1), (p2[0], p1[1], -1, 1), (p1[0], p2[1], 1, -1), (p2[0], p2[1], -1, -1)):
        cv2.line(canvas, (cx_, cy_), (cx_ + dx * L, cy_), color, 2, cv2.LINE_AA)
        cv2.line(canvas, (cx_, cy_), (cx_, cy_ + dy * L), color, 2, cv2.LINE_AA)

    if viz.show_homography and analyzer.last_homography is not None and obs.lm106 is not None:
        fit = analyzer.last_homography
        if fit.ok and fit.predicted is not None and len(fit.predicted) == 106:
            pred = np.stack([fit.predicted[:, 0] * scale + ox, fit.predicted[:, 1] * scale + oy], axis=1)
            cur = np.stack([obs.lm106[:, 0] * scale + ox, obs.lm106[:, 1] * scale + oy], axis=1)
            draw_homography_residual(canvas, cur, pred, fit.residual, 0.012)

    if viz.show_landmarks and obs.lm106 is not None:
        lm = np.stack([obs.lm106[:, 0] * scale + ox, obs.lm106[:, 1] * scale + oy], axis=1)
        draw_landmarks(canvas, lm)
        if viz.show_eye_axis:
            draw_eyes(canvas, lm, scale=scale)

    if viz.show_pose_axis and obs.pose is not None:
        kps = np.stack([obs.kps[:, 0] * scale + ox, obs.kps[:, 1] * scale + oy], axis=1)
        draw_pose_axis(canvas, kps, obs.pose, length=max(24.0, obs.iod * scale * 0.55))

    for i, k in enumerate(obs.kps):
        cv2.circle(canvas, T(k), 2, (255, 255, 255), -1, cv2.LINE_AA)

    # 判定色带
    cv2.rectangle(canvas, (0, 0), (w - 1, 5), color, -1)
    cv2.rectangle(canvas, (0, h - 6), (w - 1, h - 1), color, -1)
    if gate.door_open:
        cv2.rectangle(canvas, (0, 0), (w - 1, h - 1), (60, 220, 60), 4)
    return canvas


def hud_texts(
    cv: Canvas,
    obs: Optional[FaceObs],
    gate: GateState,
    lv: Optional[LivenessResult],
    fps: float,
    dfps: float,
    stats,
    source: str,
    viz: VizCfg,
) -> None:
    """主画面上的中文文本与状态读数(顶部让出工具栏高度)。"""
    x, y = 10, 58
    cv.text(x, y, f"输入 {source}", (200, 200, 200), 13)
    cv.text(x, y + 18, f"分析 {fps:4.1f} fps   显示 {dfps:4.1f} fps", (200, 200, 200), 13)
    cv.text(x, y + 36, f"检测 {stats.det_ms:5.1f}ms  快路径 {stats.fast_ms:5.1f}ms  特征 {stats.embed_ms:5.1f}ms",
            (170, 170, 178), 12)

    if obs is not None:
        pose = obs.pose
        pstr = (
            f"pitch {pose[0]:+6.1f}°  yaw {pose[1]:+6.1f}°  roll {pose[2]:+6.1f}°"
            if pose is not None else "姿态 n/a"
        )
        cv.text(x, y + 58, pstr, (200, 200, 200), 13)
        cv.text(
            x, y + 76,
            f"瞳距 {obs.iod:5.1f}px   det {obs.det_score:.3f}   框 "
            f"{int(obs.bbox[2] - obs.bbox[0])}x{int(obs.bbox[3] - obs.bbox[1])}",
            (170, 170, 178), 12,
        )
        live = "n/a" if obs.live_score is None else f"{obs.live_score:.3f}"
        cv.text(x, y + 94, f"官方活体CNN {live}  ({obs.live_status})", (170, 170, 178), 12)
        if gate.embedding_age_ms > 0:
            cv.text(x, y + 112, f"特征新鲜度 {gate.embedding_age_ms:.0f}ms 前", (140, 140, 150), 12)

    color = VERDICT_COLOR.get(gate.verdict, (200, 200, 200))
    if lv is not None:
        lv_cn = {"live": "活体", "fake": "非活体"}.get(lv.verdict, "证据积累中")
        lv_col = (80, 220, 120) if lv.verdict == "live" else (70, 90, 250) if lv.verdict == "fake" else (200, 200, 200)
        tstr = "n/a" if lv.temporal is None else f"{lv.temporal:.2f}"
        cv.text(
            MAIN_X + 10, MAIN_Y + MAIN_H - 84,
            f"活体:{lv_cn} {lv.score:.2f}   (单帧 {lv.static:.2f} × 时间门 {tstr})",
            lv_col, 18,
        )
    else:
        cv.text(MAIN_X + 10, MAIN_Y + MAIN_H - 84, "活体:—  (尚无人脸轨迹)", (150, 150, 158), 18)
    cv.text(MAIN_X + 10, MAIN_Y + MAIN_H - 58, f"判定:{gate.verdict_cn}", color, 24)
    msg = gate.message
    for i in range(2):
        seg = msg[i * 30: (i + 1) * 30]
        if seg:
            cv.text(MAIN_X + 10, MAIN_Y + MAIN_H - 30 + i * 15, seg, (230, 230, 235), 13)

    if lv is not None and lv.veto_by:
        cv.text(MAIN_X + MAIN_W - 190, MAIN_Y + 22, f"一票否决:{lv.veto_by}", (70, 90, 250), 14)

    keys = "q 退出 · e 录入 · space 暂停 · r 重置轨迹 · s 截图 · v 切换叠加 · h 帮助"
    cv.text(MAIN_X + 10, MAIN_Y + MAIN_H - 8, keys, (120, 120, 130), 11)
