"""活体检测用到的几何与频域原语。

这一层只做数学,不含任何阈值判断,便于单独验证。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# insightface 106 点关键点分区
#
# 实测(用 t1.jpg 把 106 点按 bbox 归一化后聚类)得到的索引布局:
#   0..32    面部轮廓(33 点)
#   33..42   图像左侧的眼睛(10 点)
#   43..51   图像左侧眉毛(9 点)
#   52..71   嘴部(20 点)
#   72..86   鼻部(15 点)
#   87..96   图像右侧的眼睛(10 点)
#   97..105  图像右侧眉毛(9 点)
# ---------------------------------------------------------------------------

IDX_CONTOUR = tuple(range(0, 33))
IDX_EYE_L = tuple(range(33, 43))
IDX_BROW_L = tuple(range(43, 52))
IDX_MOUTH = tuple(range(52, 72))
IDX_NOSE = tuple(range(72, 87))
IDX_EYE_R = tuple(range(87, 97))
IDX_BROW_R = tuple(range(97, 106))

# 平面性线索最敏感的区域:鼻部有最大的深度起伏,轮廓受下颌转动影响最大
IDX_DEPTH_SENSITIVE = IDX_NOSE + IDX_CONTOUR


@dataclass
class EyeGeometry:
    openness: float              # EAR:垂直张口 / 水平长度,闭眼时骤降
    corner_a: np.ndarray         # 长轴两端(内外眼角),取点集中相距最远的一对
    corner_b: np.ndarray
    apex_up: np.ndarray          # 长轴法向上极值点(上眼睑)
    apex_down: np.ndarray        # 下极值点(下眼睑)
    axis: np.ndarray             # 单位长轴方向


def eye_geometry(pts: np.ndarray) -> Optional[EyeGeometry]:
    """用「最远点对 = 内外眼角」定义眼睛长轴,不依赖具体索引语义。

    眨眼时眼角位置基本不变,长轴稳定;上下眼睑沿法向收拢,``openness`` 随之下降。
    这种定义比硬编码上下眼睑索引更抗关键点抖动。
    """
    pts = np.asarray(pts, dtype=np.float64)
    if pts.shape[0] < 4:
        return None
    d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=2)
    i, j = np.unravel_index(int(np.argmax(d)), d.shape)
    axis = pts[j] - pts[i]
    length = float(np.linalg.norm(axis))
    if length < 1e-6:
        return None
    u = axis / length
    n = np.array([-u[1], u[0]])
    rel = pts - pts[i]
    perp = rel @ n
    hi, lo = int(np.argmax(perp)), int(np.argmin(perp))
    opening = float(perp[hi] - perp[lo])
    return EyeGeometry(
        openness=opening / length,
        corner_a=pts[i].copy(),
        corner_b=pts[j].copy(),
        apex_up=pts[hi].copy(),
        apex_down=pts[lo].copy(),
        axis=u,
    )


def interocular(kps: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(kps, dtype=np.float64)[0] - np.asarray(kps, dtype=np.float64)[1]))


# ---------------------------------------------------------------------------
# 二维变换拟合
# ---------------------------------------------------------------------------


@dataclass
class SimilarityFit:
    """4 自由度:尺度 s、平面内旋转 theta、平移 t。

    一张平面照片在镜头前做平移 + 平面内旋转 + 前后移动时,其图像变换严格落在
    这个模型里;真实人脸的眨眼、张嘴、以及三维转动带来的透视变化则落在残差里。
    """

    s: float
    theta_deg: float
    t: np.ndarray
    residual: float          # 拟合后残差中位数(已归一化)
    explained: float         # 被模型解释掉的位移中位数(已归一化)


def fit_similarity(src: np.ndarray, dst: np.ndarray, norm: float) -> SimilarityFit:
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    cs, cd = src.mean(0), dst.mean(0)
    a = src - cs
    b = dst - cd
    num_a = float((a * b).sum())
    num_b = float((a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]).sum())
    den = float((a * a).sum())
    if den < 1e-9:
        return SimilarityFit(1.0, 0.0, cd - cs, 0.0, 0.0)
    sc, ss = num_a / den, num_b / den
    s = float(np.hypot(sc, ss))
    theta = float(np.degrees(np.arctan2(ss, sc)))
    R = np.array([[sc, -ss], [ss, sc]])
    pred = a @ R.T + cd
    resid = float(np.median(np.linalg.norm(pred - dst, axis=1))) / max(norm, 1e-6)
    explained = float(np.median(np.linalg.norm(pred - src, axis=1))) / max(norm, 1e-6)
    return SimilarityFit(s=s, theta_deg=theta, t=cd - cs, residual=resid, explained=explained)


@dataclass
class HomographyFit:
    """8 自由度单应拟合,用于平面性(单目三维结构)线索。

    打印照片、手机/平板屏幕本质上是**平面**。平面在针孔相机前做任意刚性运动时,
    两帧图像之间的映射严格是一个单应矩阵 —— 这是射影几何的基本结论。因此:

        平面载体  → 单应能把关键点位移场几乎完全解释掉,残差 ≈ 关键点噪声
        真实人脸  → 非平面三维体转动时位移场带透视视差,单应无法解释,残差显著更大
                    (鼻尖、眼窝、下颌这些深度起伏大的位置残差最大)

    残差以瞳距为单位,消除人脸远近的影响。
    """

    residual_median: float
    residual_p90: float
    residual: np.ndarray      # (N,) 每个关键点的残差,单位 = 瞳距
    predicted: np.ndarray     # (N,2) 单应预测出的当前帧点位,原图坐标,用于叠加显示
    inlier_ratio: float
    motion: float             # 参考帧到当前帧的位移中位数,单位 = 瞳距
    ok: bool


def fit_homography(
    src: np.ndarray,
    dst: np.ndarray,
    norm: float,
    ransac_px: float = 3.0,
) -> HomographyFit:
    src = np.asarray(src, dtype=np.float32)
    dst = np.asarray(dst, dtype=np.float32)
    n = max(float(norm), 1e-6)
    # 全局缩放不改变单应拟合的优劣,但能把残差直接换算成「瞳距的倍数」
    s, d = src / n, dst / n
    disp = float(np.median(np.linalg.norm((d - s).astype(np.float64), axis=1)))

    H, mask = cv2.findHomography(s, d, cv2.RANSAC, ransac_px / n, maxIters=2000, confidence=0.999)
    if H is None:
        return HomographyFit(0.0, 0.0, np.zeros(len(s)), src.copy(), 0.0, disp, False)

    homo = np.hstack([s, np.ones((len(s), 1), dtype=np.float64)])
    proj = homo @ H.T
    w = np.where(np.abs(proj[:, 2:3]) < 1e-9, 1e-9, proj[:, 2:3])
    pred = (proj[:, :2] / w).astype(np.float64)
    resid = np.linalg.norm(pred - d.astype(np.float64), axis=1)
    inlier_ratio = float(mask.mean()) if mask is not None else 1.0

    return HomographyFit(
        residual_median=float(np.median(resid)),
        residual_p90=float(np.percentile(resid, 90)),
        residual=resid,
        predicted=(pred * n).astype(np.float32),
        inlier_ratio=inlier_ratio,
        motion=disp,
        ok=True,
    )


# ---------------------------------------------------------------------------
# 频域分析
# ---------------------------------------------------------------------------


@dataclass
class Spectrum:
    mag: np.ndarray           # log 幅度谱(已 fftshift),用于面板显示
    peak_ratio: float         # 最强孤立峰 / 中频带中位数
    peak_count: int           # 超过 peak_thresh 的孤立峰数量
    peaks: np.ndarray         # (K,2) 峰值坐标,频谱图坐标系
    band_ratio: float         # 高频带能量 / 总能量
    peak_thresh: float


def analyze_spectrum(gray_hp: np.ndarray, low_keep: int = 6, peak_factor: float = 4.0) -> Spectrum:
    """对高通滤波后的灰度图做二维 FFT,找周期性尖峰。

    屏幕由规则像素栅格发光,翻拍时栅格周期会原样进入图像,在频域表现为
    **孤立、尖锐、能量远高于邻域**的峰值;真实皮肤的高频能量则随半径平滑衰减,
    不会出现这种离散尖峰。
    """
    img = np.asarray(gray_hp, dtype=np.float32)
    if img.size < 64:
        return Spectrum(np.zeros((8, 8), np.float32), 0.0, 0, np.zeros((0, 2), np.int32), 0.0, 0.0)

    f = np.fft.fftshift(np.fft.fft2(img))
    mag = np.log1p(np.abs(f))
    h, w = mag.shape
    cy, cx = h // 2, w // 2

    yy, xx = np.ogrid[:h, :w]
    r = np.sqrt(((yy - cy) * (w / max(h, 1))) ** 2 + (xx - cx) ** 2)
    rmax = float(r.max())
    low_r = low_keep * (rmax / max(min(h, w) // 2, 1))

    band = (r > low_r) & (r < rmax * 0.92)
    if band.sum() < 16:
        band = r > low_r
    baseline = float(np.median(mag[band]))
    sigma = float(np.std(mag[band])) + 1e-6

    # 局部极大值:与 5x5 膨胀结果相等者即为孤立峰
    local_max = cv2.dilate(mag, np.ones((5, 5), np.uint8))
    thresh = baseline + peak_factor * sigma
    cand = band & (mag >= local_max - 1e-9) & (mag > thresh)
    ys, xs = np.nonzero(cand)
    if len(ys):
        order = np.argsort(-mag[ys, xs])
        ys, xs = ys[order], xs[order]
        keep = _non_max_suppress(ys, xs, mag, min_dist=3)
        ys, xs = ys[keep], xs[keep]
    else:
        keep = np.zeros(0, dtype=int)

    peak_ratio = float((mag[ys, xs].max() - baseline) / sigma) if len(ys) else 0.0
    peak_count = int(np.sum(mag[ys, xs] > thresh)) if len(ys) else 0

    energy = np.abs(f) ** 2
    total = float(energy[r > low_r].sum()) + 1e-9
    hi = float(energy[r > rmax * 0.45].sum())
    return Spectrum(
        mag=mag.astype(np.float32),
        peak_ratio=peak_ratio,
        peak_count=peak_count,
        peaks=np.stack([xs, ys], axis=1).astype(np.int32) if len(ys) else np.zeros((0, 2), np.int32),
        band_ratio=hi / total,
        peak_thresh=thresh,
    )


def _non_max_suppress(ys: np.ndarray, xs: np.ndarray, mag: np.ndarray, min_dist: int) -> np.ndarray:
    """抑制相邻重复峰,返回保留的下标。"""
    kept: list[int] = []
    for i in range(len(ys)):
        ok = True
        for j in kept:
            if abs(int(ys[i]) - int(ys[j])) <= min_dist and abs(int(xs[i]) - int(xs[j])) <= min_dist:
                ok = False
                break
        if ok:
            kept.append(i)
    return np.asarray(kept, dtype=int)


def highpass(gray: np.ndarray, ksize: int = 7) -> np.ndarray:
    """减去高斯低频,保留纹理与栅格成分。"""
    g = np.asarray(gray, dtype=np.float32)
    blur = cv2.GaussianBlur(g, (ksize, ksize), 0)
    return g - blur


def laplacian_var(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(np.asarray(gray, dtype=np.uint8), cv2.CV_32F).var())


def radial_profile(mag: np.ndarray, bins: int = 24) -> Tuple[np.ndarray, np.ndarray]:
    """log 幅度谱的径向平均轮廓,用于观察高频衰减斜率。"""
    h, w = mag.shape
    cy, cx = h // 2, w // 2
    yy, xx = np.ogrid[:h, :w]
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2).astype(np.int32)
    rmax = int(r.max())
    edges = np.linspace(0, rmax, bins + 1).astype(int)
    centers, values = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (r >= a) & (r < b)
        if m.sum() == 0:
            continue
        centers.append((a + b) / 2 / max(rmax, 1))
        values.append(float(mag[m].mean()))
    return np.asarray(centers, dtype=np.float32), np.asarray(values, dtype=np.float32)
