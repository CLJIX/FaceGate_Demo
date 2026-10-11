"""自检与阈值标定。

跑 ``python main.py selftest`` 会做四件事:

1. 检查依赖、模型包、摄像头、人脸库目录是否就绪。
2. **几何验证**:取 insightface 3D 关键点模型自带的规范人脸形状 ``mean_lmk``
   (带真实 z),在「保留 z」和「把 z 压成 0」两种情况下做同一组三维旋转 + 针孔
   投影,并对两组二维轨迹注入同量级的高斯关键点噪声,然后跑单应拟合。前者是真实
   三维脸的位移场,后者是平面载体的位移场。加了噪声之后两者的残差才可比 ——
   这一步回答的是「在真实关键点噪声下,单应残差还能不能区分平面和三维」。
3. **噪声底标定**:对一张真实人脸施加已知单应变换序列(等价于把一张打印照片在
   镜头前转动),用 106 点关键点模型跟踪后再跑单应拟合。此时真值残差为 0,测到的
   残差全部来自关键点噪声,这就是 ``residual_fake_below`` 该设的位置。
4. 抽检 EAR、姿态、活体分等原语是否输出合理量级。

最后给出 ``liveness.planarity`` 两个阈值的建议值。
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .config import Config, load_config
from .liveness.geometry import IDX_EYE_L, IDX_EYE_R, eye_geometry, fit_homography, interocular
from .util import imread_u


@dataclass
class Report:
    name: str
    ok: bool
    lines: List[str]


def _hr(title: str) -> str:
    return f"\n{'=' * 76}\n{title}\n{'=' * 76}"


def _sample_image() -> Optional[np.ndarray]:
    import insightface

    for name in ("t1.jpg", "Tom_Hanks_54745.png"):
        p = os.path.join(os.path.dirname(insightface.__file__), "data", "images", name)
        img = imread_u(p)
        if img is not None:
            return img
    return None


# ---------------------------------------------------------------------------
# 1. 环境与模型
# ---------------------------------------------------------------------------


def check_environment(cfg: Config) -> Tuple[Report, Optional[object]]:
    lines: List[str] = []
    ok = True
    try:
        import insightface
        import onnxruntime as ort

        lines.append(
            f"insightface {insightface.__version__} / onnxruntime {ort.__version__} / OpenCV {cv2.__version__}"
        )
        lines.append(f"可用 ExecutionProvider: {ort.get_available_providers()}")
    except Exception as exc:  # noqa: BLE001
        return Report("环境", False, [f"依赖缺失: {exc}"]), None

    from .engine import FaceEngine

    t0 = time.time()
    try:
        engine = FaceEngine(cfg.engine)
    except Exception as exc:  # noqa: BLE001
        return Report("环境", False, lines + [f"模型加载失败: {exc}"]), None
    lines.append(
        f"模型加载 {time.time() - t0:.1f}s;检测输入 {tuple(cfg.engine.det_size)};"
        f" intra_op={cfg.engine.intra_op_threads};EP={engine.stats.providers}"
    )
    lines.append(f"官方活体 addon: {'已加载' if engine.live is not None else '未加载'}")

    cap = cv2.VideoCapture(cfg.camera.index, cv2.CAP_DSHOW)
    if cap.isOpened():
        got, fr = cap.read()
        lines.append(f"摄像头 {cfg.camera.index}: {'可用 ' + str(fr.shape) if got else '已打开但读不到画面'}")
        ok = ok and got
    else:
        lines.append(f"摄像头 {cfg.camera.index}: 不可用(可在 config.yaml 用 camera.source 指定视频文件)")
    cap.release()

    gdir = cfg.gallery_dir
    n = len(list(gdir.glob("*.npz"))) if gdir.exists() else 0
    lines.append(f"人脸库目录: {gdir} ({'存在' if gdir.exists() else '尚未创建'}, 已有 {n} 人)")
    return Report("环境", ok, lines), engine


# ---------------------------------------------------------------------------
# 挑选一张适合做标定的脸:够大、够正
# ---------------------------------------------------------------------------


def _probe_face(engine, target_iod: float = 130.0) -> Tuple[Optional[np.ndarray], Optional[object], str]:
    img = _sample_image()
    if img is None:
        return None, None, "找不到 insightface 自带样图"

    cands = engine.detect(img, max_faces=32)
    usable = [o for o in cands if o.lm106 is not None and o.iod > 40 and abs(float(o.pose[1])) < 28]
    if not usable:
        return None, None, f"样图中没有足够大且接近正面的人脸(共检出 {len(cands)} 张)"
    best = max(usable, key=lambda o: o.iod)

    x1, y1, x2, y2 = (float(v) for v in best.bbox)
    bw, bh = x2 - x1, y2 - y1
    pad = 0.9
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    hw, hh = bw * (0.5 + pad), bh * (0.5 + pad)
    H, W = img.shape[:2]
    ax1, ay1 = int(max(0, cx - hw)), int(max(0, cy - hh))
    ax2, ay2 = int(min(W, cx + hw)), int(min(H, cy + hh))
    crop = img[ay1:ay2, ax1:ax2]
    if crop.size == 0:
        return None, None, "裁剪失败"

    scale = target_iod / max(best.iod, 1e-6)
    crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    obs = engine.detect(crop)
    if not obs or obs[0].lm106 is None:
        return None, None, "放大后重新检测失败"
    return crop, obs[0], f"瞳距 {obs[0].iod:.1f}px,姿态 {np.round(obs[0].pose, 1).tolist()}"


# ---------------------------------------------------------------------------
# 2. 几何验证:三维脸 vs 平面脸的位移场(含关键点噪声)
# ---------------------------------------------------------------------------


def _rotation(axis: str, deg: float) -> np.ndarray:
    t = np.deg2rad(deg)
    c, s = np.cos(t), np.sin(t)
    if axis == "x":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    if axis == "y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def _project(shape3d: np.ndarray, R: np.ndarray, focal: float = 4.0) -> np.ndarray:
    """规范坐标下的三维形状 → 旋转 → 针孔投影到二维。"""
    pts = shape3d @ R.T
    z = pts[:, 2] + focal
    return pts[:, :2] / np.maximum(z[:, None], 1e-6) * focal


def geometric_discrimination(
    mean_lmk: np.ndarray,
    angles: Tuple[float, ...] = (8.0, 15.0, 25.0, 35.0),
    noise_levels: Tuple[float, ...] = (0.0, 0.003, 0.010),
) -> Report:
    shape = np.asarray(mean_lmk, dtype=np.float64)
    if shape.ndim != 2 or shape.shape[1] != 3:
        return Report("几何验证", False, [f"mean_lmk 形状异常: {shape.shape},需要 (N,3)"])

    flat = shape.copy()
    flat[:, 2] = 0.0
    front = _project(shape, _rotation("y", 0.0))
    iod = float(np.linalg.norm(front[36] - front[45])) or 1.0

    lines = [
        f"规范人脸 {shape.shape[0]} 点;深度起伏/瞳距 = {np.ptp(shape[:, 2]) / (iod or 1):.2f};"
        f"瞳距(投影单位) = {iod:.3f}",
        "残差单位 = 瞳距。σ 为注入的高斯关键点噪声(占瞳距比例)。",
    ]
    verdicts: List[str] = []
    ok = True
    for sigma in noise_levels:
        lines.append("")
        lines.append(f"  σ = {sigma * 100:.1f}% 瞳距")
        lines.append(f"  {'绕Y转角':>8} {'平面脸残差':>12} {'三维脸残差':>12} {'判别比':>8}")
        ratios: List[float] = []
        for deg in angles:
            rng = np.random.default_rng(12345)
            res = {}
            for label, shp in (("planar", flat), ("3d", shape)):
                base = _project(shp, _rotation("y", 0.0))
                cur = _project(shp, _rotation("y", deg))
                if sigma > 0:
                    base = base + rng.normal(0, sigma * iod, base.shape)
                    cur = cur + rng.normal(0, sigma * iod, cur.shape)
                fit = fit_homography(base, cur, iod, ransac_px=3.0 * sigma * iod + 1e-3)
                res[label] = fit.residual_median
            r = res["3d"] / max(res["planar"], 1e-12)
            ratios.append(r)
            lines.append(
                f"  {deg:>7.0f}° {res['planar']:>12.5f} {res['3d']:>12.5f} {r:>8.1f}x"
            )
        med = float(np.median(ratios))
        verdicts.append(f"σ={sigma * 100:.1f}% 瞳距 → 中位判别比 {med:.1f}x,最大 {max(ratios):.1f}x")
        if abs(sigma - 0.003) < 1e-9 and max(ratios) < 3.0:
            ok = False

    lines += [
        "",
        "平面脸的位移场严格落在 8 自由度单应里,残差只来自噪声;三维脸的位移场带",
        "透视视差,单应无法完全解释,残差随转角增大。判别比就是这条线索的可用裕度。",
        "",
        "σ=0.3% 对应本项目实测的关键点噪声底(见下一节),是真正有参考意义的一档。",
        "可以看到裕度随转角迅速拉开:这也是 planarity 线索要求头部必须转动一定角度",
        "才生效(require_motion_norm)的原因 —— 头不动时它没有信息量。",
        "",
    ] + [f"  {v}" for v in verdicts]
    return Report("几何验证:平面 vs 三维位移场", ok, lines)


# ---------------------------------------------------------------------------
# 3. 噪声底标定
# ---------------------------------------------------------------------------


def noise_floor(engine, angles: Tuple[float, ...] = (-12.0, -7.0, 7.0, 12.0)) -> Report:
    img, o0, info = _probe_face(engine)
    if img is None or o0 is None:
        return Report("噪声底标定", False, [info])

    # 先加一圈边,免得透视变换把脸推出画面导致检不到
    h0, w0 = img.shape[:2]
    pad = int(0.35 * max(h0, w0))
    canvas = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    base = engine.detect(canvas)
    if not base or base[0].lm106 is None:
        return Report("噪声底标定", False, [info, "加边后重新检测失败"])
    o0 = base[0]

    h, w = canvas.shape[:2]
    focal = float(max(w, h)) * 2.2
    K = np.array([[focal, 0, w / 2], [0, focal, h / 2], [0, 0, 1]], dtype=np.float64)
    iod = interocular(o0.kps)
    c0 = (o0.bbox[:2] + o0.bbox[2:]) / 2.0

    lines = [
        f"标定用脸: {info};加边后画布 {w}x{h},瞳距 {iod:.1f}px,等效焦距 {focal:.0f}px",
        "真值残差为 0(变换本身就是一个单应),所以测到的一切都是关键点模型的不一致性。",
        "",
    ]
    lines.append(
        f"  {'透视旋转':>9} {'位移量':>9} {'单应残差':>10} {'残差P90':>10} {'内点率':>7} {'瞳距比':>7}  备注"
    )
    resid: List[float] = []
    for deg in angles:
        # K·Ry·K⁻¹ 描述的是「相机绕自身光心转动」,会把整幅图平移 f·tan(θ),
        # 人脸会被甩出画面。这里先把它平移到画布中心再转,最后补一个纯平移把
        # 人脸送回原位 —— 平移与单应复合仍是单应,标定所需的「真值残差为 0」不变。
        dx, dy = w / 2 - c0[0], h / 2 - c0[1]
        T = np.array([[1, 0, dx], [0, 1, dy], [0, 0, 1]], dtype=np.float64)
        Tinv = np.array([[1, 0, -dx], [0, 1, -dy], [0, 0, 1]], dtype=np.float64)
        M = Tinv @ K @ _rotation("y", deg) @ np.linalg.inv(K) @ T
        c0h = np.append(c0, 1.0)
        c1 = M @ c0h
        c1 = c1[:2] / max(abs(c1[2]), 1e-9)
        shift = np.array([[1, 0, c0[0] - c1[0]], [0, 1, c0[1] - c1[1]], [0, 0, 1]], dtype=np.float64)
        M = shift @ M
        warped = cv2.warpPerspective(canvas, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
        obs = engine.detect(warped)
        if not obs or obs[0].lm106 is None:
            lines.append(f"  {deg:>8.0f}° {'—':>9} {'—':>10} {'—':>10} {'—':>7} {'—':>7}  变换后未检出人脸,丢弃")
            continue
        o1 = obs[0]
        iod_ratio = o1.iod / max(iod, 1e-6)
        drift = float(np.linalg.norm((o1.bbox[:2] + o1.bbox[2:]) / 2.0 - c0)) / iod
        fit = fit_homography(o0.lm106, o1.lm106, iod, 3.0)
        bad = []
        if fit.inlier_ratio < 0.85:
            bad.append(f"内点率 {fit.inlier_ratio:.2f}")
        if not 0.70 <= iod_ratio <= 1.40:
            bad.append(f"瞳距比 {iod_ratio:.2f}")
        if drift > 3.0:
            bad.append(f"漂移 {drift:.1f}")
        if bad:
            note = "跟踪不可靠(" + ", ".join(bad) + "),丢弃"
        else:
            note = "有效"
            resid.append(fit.residual_median)
        lines.append(
            f"  {deg:>8.0f}° {fit.motion:>9.4f} {fit.residual_median:>10.4f} "
            f"{fit.residual_p90:>10.4f} {fit.inlier_ratio:>7.2f} {iod_ratio:>7.2f}  {note}"
        )

    if len(resid) < 2:
        return Report(
            "噪声底标定",
            False,
            lines + ["有效样本不足 2 个,无法标定。可把 angles 调小或换一张更正的样图。"],
        )

    floor = float(np.median(resid))
    hi = float(np.max(resid))
    lines += [
        "",
        f"实测关键点噪声底: 中位 {floor:.4f}、最大 {hi:.4f} 瞳距。",
        f"建议 liveness.planarity.residual_fake_below = {max(hi * 1.8, 0.010):.4f}",
        f"建议 liveness.planarity.residual_live_above = {max(hi * 5.0, 0.040):.4f}",
        "",
        "以上是「平面载体」这一侧的标定。真人一侧请在 Demo 运行时读面板上的",
        "「单应残差」:缓慢左右转头 20~30 度,读数应当稳定高于 fake_below。",
    ]
    return Report("噪声底标定(真实关键点 + 已知单应)", True, lines)


# ---------------------------------------------------------------------------
# 4. 原语抽检
# ---------------------------------------------------------------------------


def primitive_checks(engine) -> Report:
    img, o, info = _probe_face(engine)
    if img is None or o is None:
        return Report("原语抽检", False, [info])

    lines = [f"抽检用脸: {info}", f"det_score = {o.det_score:.3f}"]
    for name, idx in (("左眼", IDX_EYE_L), ("右眼", IDX_EYE_R)):
        g = eye_geometry(o.lm106[list(idx)])
        if g is None:
            lines.append(f"  {name}: 关键点不足")
            continue
        axis = float(np.linalg.norm(g.corner_b - g.corner_a))
        lines.append(
            f"  {name} EAR = {g.openness:.3f},长轴 = {axis:.1f}px"
            f"(睁眼常见 0.18~0.35,闭合 <0.10)"
        )
    lines.append(f"  官方活体分 = {o.live_score}  status = {o.live_status}")

    from .liveness.analyzer import frame_metrics
    from .config import LivenessCfg

    m, spec, _, _ = frame_metrics(o, img, LivenessCfg())
    lines.append(
        f"  Laplacian 方差 = {m['lap']:.1f},高低频比 = {m.get('band_ratio', 0):.3f},"
        f"峰噪比 = {m.get('peak_ratio', 0):.2f},尖峰数 = {int(m.get('peak_count', 0))}"
    )
    lines.append(f"  肤色占比 = {m['skin']:.3f},平均饱和度 = {m['sat']:.3f}")
    if spec is not None:
        lines.append(f"  频谱图尺寸 = {spec.mag.shape},可正常绘制")

    ok = o.live_score is not None and m["lap"] > 0
    return Report("原语抽检(EAR / 姿态 / 纹理 / 频谱)", ok, lines)


def run_all(cfg: Optional[Config] = None) -> int:
    cfg = cfg or load_config()
    print(_hr("人脸门禁 Demo 自检"))
    env, engine = check_environment(cfg)
    reports = [env]

    if engine is not None:
        reports.append(primitive_checks(engine))
        mean = getattr(engine.lm3d, "mean_lmk", None)
        if mean is not None:
            reports.append(geometric_discrimination(np.asarray(mean)))
        else:
            reports.append(Report("几何验证", False, ["3D 关键点模型未提供 mean_lmk"]))
        reports.append(noise_floor(engine))

    for r in reports:
        print(_hr(f"[{'通过' if r.ok else '失败'}] {r.name}"))
        for line in r.lines:
            print("  " + line)

    failed = [r.name for r in reports if not r.ok]
    print(_hr("结论"))
    if failed:
        print(f"  未通过: {failed}")
        print("  「环境」项失败会阻塞 Demo;其余项失败只影响阈值标定精度。")
        return 1
    print("  全部通过。下一步: python main.py enroll --name 张三")
    return 0


if __name__ == "__main__":
    sys.exit(run_all())
