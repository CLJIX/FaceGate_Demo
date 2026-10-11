"""insightface 封装:把管线拆成「快路径」与「慢路径」。

实测(20 核 CPU / ONNX Runtime 1.31 / buffalo_l):

    检测 det_10g @480      63 ms   ┐
    官方活体 CNN 80x80      2.5 ms  ├ 快路径 ≈ 95 ms → 10.5 fps
    106 点关键点            3.2 ms  │  (每帧都跑,眨眼/微动/平面性线索需要连续采样)
    3D 68 点 + 姿态        27 ms   ┘
    ArcFace R50 512维      100 ms  ← 慢路径,按 recognize_interval_ms 节流

直接调 ``FaceAnalysis.get()`` 会把上面五步全跑一遍,而且默认 SessionOptions 下
四个模型各自抢满 20 线程,实测 1735 ms/帧。这里显式设定 ``intra_op_num_threads``
并手动驱动子模型,把识别从每帧解耦出来。
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import cv2
import numpy as np
import onnxruntime as ort
from insightface.app import FaceAnalysis
from insightface.app.common import Face
from insightface.utils import face_align

from .config import EngineCfg

_MODULES = ["detection", "recognition", "landmark_2d_106", "landmark_3d_68"]

# 性能档位。SCRFD 的耗时几乎只由 det_size(网络输入边长)决定,与源图大小无关,
# 所以提速的正道是:锁定人脸后只在人脸周围的 ROI 里检测 —— 此时人脸占满输入,
# 可以用小得多的 det_size 而保持甚至提高有效分辨率。
PERF_PROFILES: Dict[str, Dict[str, object]] = {
    "quality": dict(det_size=480, roi_det_size=320, pose_every=1, spec_every=1, rec_ms=400, roi=False, refresh=10, eval_every=1),
    "balanced": dict(det_size=320, roi_det_size=192, pose_every=2, spec_every=2, rec_ms=400, roi=True, refresh=15, eval_every=2),
    "fast": dict(det_size=256, roi_det_size=160, pose_every=4, spec_every=4, rec_ms=700, roi=True, refresh=20, eval_every=3),
    "ultra": dict(det_size=256, roi_det_size=160, pose_every=4, spec_every=6, rec_ms=800, roi=True, refresh=20, eval_every=3),
}
PROFILE_ORDER = ("quality", "balanced", "fast", "ultra")
# ultra 档额外关闭侧栏面板,只保留主画面与得分条,把渲染从 ~21ms 压到 ~6ms
PROFILE_MINIMAL_VIZ = {"ultra": True}


@dataclass
class FaceObs:
    """单帧单人的全部中间结果。所有字段都会显示在调试面板上。"""

    bbox: np.ndarray                 # (4,) x1,y1,x2,y2
    det_score: float
    kps: np.ndarray                  # (5,2) 双眼/鼻尖/嘴角
    lm106: Optional[np.ndarray]      # (106,2)
    pose: Optional[np.ndarray]       # (3,) 见 POSE_LABELS
    live_status: str                 # ok / input_rejected / error / disabled
    live_score: Optional[float]      # 官方 CNN 输出的真人概率
    aligned: np.ndarray              # (112,112,3) BGR, ArcFace 标准对齐裁剪
    embedding: Optional[np.ndarray] = None
    iod: float = 0.0                 # 瞳距,所有几何量的归一化尺度
    age_ms: float = 0.0              # 该帧特征距今多久,用于面板提示


POSE_LABELS = ("pitch", "yaw", "roll")


@dataclass
class EngineStats:
    det_ms: float = 0.0
    fast_ms: float = 0.0
    embed_ms: float = 0.0
    n_faces: int = 0
    providers: List[str] = field(default_factory=list)


class FaceEngine:
    def __init__(self, cfg: EngineCfg):
        self.cfg = cfg
        so = ort.SessionOptions()
        so.intra_op_num_threads = int(cfg.intra_op_threads)
        so.inter_op_num_threads = 1
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self._sess_options = so

        addons = ["liveness"] if cfg.use_liveness_addon else None
        self.app = FaceAnalysis(
            name=cfg.model_pack,
            allowed_modules=_MODULES,
            addons=addons,
            # observe 模式:即使判定为非活体也照常输出特征,便于对照观察
            liveness_mode="observe",
            liveness_threshold=cfg.liveness_threshold,
            sess_options=so,
        )
        self.app.prepare(ctx_id=-1, det_thresh=cfg.det_thresh, det_size=tuple(cfg.det_size))

        self.det = self.app.models["detection"]
        self.rec = self.app.models["recognition"]
        self.lm106 = self.app.models["landmark_2d_106"]
        self.lm3d = self.app.models["landmark_3d_68"]
        self.live = self.app.addons.get("liveness")
        self.stats = EngineStats(providers=list(self.det.session.get_providers()))
        self._last_pose: Optional[np.ndarray] = None
        self._roi: Optional[tuple] = None
        self._last_iod = 1.0
        self._embed_lock = threading.Lock()
        self._embed_busy = False
        self._embed_result: Optional[np.ndarray] = None
        self.profile: Dict[str, object] = {}
        self.profile_name = "balanced"
        self.set_profile(cfg.perf_profile if hasattr(cfg, "perf_profile") else "balanced")
        # ROI 检测用独立会话:运行期切换 det_size 会触发 SCRFD 重新 prepare(每次约 63ms),
        # 宁可多占 17MB 内存也不要在帧循环里重建会话
        self.det_roi = self._clone_detector(int(self.profile["roi_det_size"]))

    # -- 快路径 --------------------------------------------------------------

    def detect(
        self,
        image: np.ndarray,
        max_faces: Optional[int] = None,
        with_pose: bool = True,
        frame_id: int = 0,
    ) -> List[FaceObs]:
        """检测 + 活体 + 关键点(+姿态)。不含 ArcFace 特征提取。

        ``with_pose=False`` 时跳过 3D 68 点模型(约 27ms),复用上一帧的姿态角。

        开启 ROI 跟踪后,锁定人脸的帧只在人脸周围 1.8 倍框内做检测,并用更小的
        det_size(人脸占满输入,有效分辨率反而更高),检测耗时从 31ms 降到 ~12ms;
        每 ``refresh`` 帧或 ROI 校验失败时回退全帧检测,避免跟丢或漏掉新进入的人。
        """
        t0 = time.perf_counter()
        prof = self.profile
        use_roi = bool(self.cfg.use_roi_tracking) and bool(prof["roi"])

        bboxes = kpss = None
        if use_roi and self._roi is not None and frame_id % int(prof["refresh"]) != 0:
            bboxes, kpss = self._detect_roi(image)
        if bboxes is None or bboxes.shape[0] == 0:
            self._set_det_size(int(prof["det_size"]))
            bboxes, kpss = self.det.detect(image, max_num=0, metric="default")
        t_det = time.perf_counter()
        self.stats.det_ms = (t_det - t0) * 1000.0

        if bboxes.shape[0] == 0:
            self.stats.n_faces = 0
            self.stats.fast_ms = (time.perf_counter() - t0) * 1000.0
            self._roi = None
            self._last_pose = None
            return []

        limit = self.cfg.max_faces if max_faces is None else max_faces
        order = np.argsort(-_areas(bboxes))[: max(1, limit)]
        out: List[FaceObs] = []
        for i in order:
            face = Face(bbox=bboxes[i, 0:4], kps=kpss[i], det_score=bboxes[i, 4])

            status, score = self._run_liveness(image, face)
            lm106 = self._safe(self.lm106.get, image, face, "landmark_2d_106")
            if with_pose:
                self._safe(self.lm3d.get, image, face, "landmark_3d_68")
                self._last_pose = None if face.get("pose") is None else np.asarray(face.pose, dtype=np.float32)

            kps = np.asarray(face.kps, dtype=np.float32)
            iod = float(np.linalg.norm(kps[0] - kps[1]))
            self._roi = tuple(float(v) for v in face.bbox[:4])
            self._last_iod = iod
            out.append(
                FaceObs(
                    bbox=np.asarray(face.bbox, dtype=np.float32),
                    det_score=float(face.det_score),
                    kps=kps,
                    lm106=None if lm106 is None else np.asarray(lm106, dtype=np.float32),
                    pose=None if self._last_pose is None else self._last_pose.copy(),
                    live_status=status,
                    live_score=score,
                    aligned=self.align(image, kps),
                    iod=iod,
                )
            )
        self.stats.n_faces = len(out)
        self.stats.fast_ms = (time.perf_counter() - t0) * 1000.0
        return out

    # -- 性能档与 ROI --------------------------------------------------------

    def set_profile(self, name: str) -> str:
        if name not in PERF_PROFILES:
            name = "balanced"
        self.profile = dict(PERF_PROFILES[name])
        self.profile_name = name
        self.cfg.det_size = [int(self.profile["det_size"])] * 2
        self.cfg.pose_every_n_frames = int(self.profile["pose_every"])
        self.cfg.spectrum_every_n_frames = int(self.profile["spec_every"])
        self.cfg.recognize_interval_ms = int(self.profile["rec_ms"])
        self.cfg.use_roi_tracking = bool(self.profile["roi"])
        self._set_det_size(int(self.profile["det_size"]))
        if getattr(self, "det_roi", None) is not None:
            self.det_roi = self._clone_detector(int(self.profile["roi_det_size"]))
        self._roi = None
        return name

    def _clone_detector(self, size: int):
        """再加载一个 SCRFD 会话,固定跑 ROI 的小输入尺寸。"""
        from insightface.model_zoo import model_zoo

        path = getattr(self.det.session, "model_path", None)
        if not path:
            from pathlib import Path as _P

            path = str(_P(self.app.model_dir) / "det_10g.onnx")
        m = model_zoo.get_model(
            str(path), providers=["CPUExecutionProvider"], sess_options=self._sess_options
        )
        m.prepare(-1, input_size=(size, size), det_thresh=self.cfg.det_thresh)
        return m

    def reset_roi(self) -> None:
        """丢弃 ROI 跟踪状态,下一帧回到全帧检测。"""
        self._roi = None

    def _set_det_size(self, size: int) -> None:
        # SCRFD.prepare 之后 input_size 是标量 int,这里按标量比较,避免每帧重复 prepare
        cur = getattr(self.det, "input_size", None)
        if cur is None:
            return
        cur_list = [int(v) for v in np.atleast_1d(cur).tolist()]
        if cur_list and cur_list[0] == int(size):
            return
        self.det.prepare(-1, input_size=(int(size), int(size)), det_thresh=self.cfg.det_thresh)

    def _detect_roi(self, image: np.ndarray):
        """在上一帧人脸框外扩 0.9 倍的区域内检测;校验不通过则返回 None 触发全帧回退。"""
        x1, y1, x2, y2 = self._roi
        H, W = image.shape[:2]
        bw, bh = x2 - x1, y2 - y1
        rx1, ry1 = int(max(0, x1 - bw * 0.9)), int(max(0, y1 - bh * 0.9))
        rx2, ry2 = int(min(W, x2 + bw * 0.9)), int(min(H, y2 + bh * 0.9))
        if rx2 - rx1 < 48 or ry2 - ry1 < 48:
            return None, None
        crop = image[ry1:ry2, rx1:rx2]
        b, k = self.det_roi.detect(crop, max_num=1, metric="default")
        if b is None or b.shape[0] == 0:
            return None, None
        i = int(np.argmax(_areas(b)))
        bb = b[i].astype(np.float64).copy()      # 保留第 5 列检测分
        bb[0] += rx1
        bb[2] += rx1
        bb[1] += ry1
        bb[3] += ry1
        kk = k[i].astype(np.float64).copy()
        kk[:, 0] += rx1
        kk[:, 1] += ry1

        iod_prev = max(self._last_iod, 1e-3)
        pc = np.array([(bb[0] + bb[2]) / 2, (bb[1] + bb[3]) / 2])
        oc = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
        if float(np.linalg.norm(pc - oc)) > 0.7 * iod_prev:
            return None, None
        iod_new = float(np.linalg.norm(kk[0] - kk[1]))
        if not 0.65 * iod_prev <= iod_new <= 1.5 * iod_prev:
            return None, None
        return bb[None, :], kk[None, :]

    def _run_liveness(self, image: np.ndarray, face: Face) -> tuple[str, Optional[float]]:
        if self.live is None:
            return "disabled", None
        try:
            res = self.live.get(image, face)
        except Exception as exc:  # 对齐失败/裁剪越界等,降级为「无信息」而不是崩溃
            return f"error:{type(exc).__name__}", None
        return str(res.get("status", "unknown")), (
            float(res["live_score"]) if res.get("live_score") is not None else None
        )

    @staticmethod
    def _safe(fn, image, face, key):
        try:
            return fn(image, face)
        except Exception:
            return face.get(key)

    # -- 慢路径 --------------------------------------------------------------

    def embed(self, image: np.ndarray, obs: FaceObs) -> np.ndarray:
        t0 = time.perf_counter()
        feat = self.rec.get_feat(obs.aligned).flatten()
        self.stats.embed_ms = (time.perf_counter() - t0) * 1000.0
        norm = float(np.linalg.norm(feat))
        obs.embedding = feat / norm if norm > 0 else feat
        return obs.embedding

    def request_embed(self, aligned: np.ndarray) -> bool:
        """把 512 维特征提取丢到后台线程。

        ArcFace 在六个 ORT 会话共存时单次可达 260ms(线程自旋竞争),放在主循环里
        每 8 帧就吃掉 30ms 均摊。身份比对只需要「最近一次」的结果,天然适合异步。
        """
        with self._embed_lock:
            if self._embed_busy or self._embed_result is not None:
                return False
            self._embed_busy = True
        t = threading.Thread(target=self._embed_worker, args=(aligned.copy(),), daemon=True)
        t.start()
        return True

    def _embed_worker(self, aligned: np.ndarray) -> None:
        t0 = time.perf_counter()
        try:
            feat = self.rec.get_feat(aligned).flatten()
            norm = float(np.linalg.norm(feat))
            feat = feat / norm if norm > 0 else feat
        except Exception:  # noqa: BLE001
            feat = None
        with self._embed_lock:
            self._embed_result = feat
            self._embed_busy = False
        self.stats.embed_ms = (time.perf_counter() - t0) * 1000.0

    def take_embedding(self) -> Optional[np.ndarray]:
        with self._embed_lock:
            res = self._embed_result
            self._embed_result = None
        return res

    def embed_crop(self, aligned: np.ndarray) -> np.ndarray:
        """直接对已对齐的 112x112 裁剪提特征,录入时复用。"""
        feat = self.rec.get_feat(aligned).flatten()
        norm = float(np.linalg.norm(feat))
        return feat / norm if norm > 0 else feat

    # -- 工具 ----------------------------------------------------------------

    @staticmethod
    def align(image: np.ndarray, kps: np.ndarray, size: int = 112) -> np.ndarray:
        return face_align.norm_crop(image, landmark=np.asarray(kps, dtype=np.float32), image_size=size)

    @staticmethod
    def cosine(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b) / max(1e-9, np.linalg.norm(a) * np.linalg.norm(b)))


def _areas(bboxes: np.ndarray) -> np.ndarray:
    w = np.maximum(0.0, bboxes[:, 2] - bboxes[:, 0])
    h = np.maximum(0.0, bboxes[:, 3] - bboxes[:, 1])
    return w * h


def draw_pose_axis(
    image: np.ndarray,
    kps: np.ndarray,
    pose: np.ndarray,
    length: float = 45.0,
) -> None:
    """把 pitch/yaw/roll 画成从鼻尖出发的三维坐标轴,原地修改 image。"""
    pitch, yaw, roll = (np.deg2rad(float(p)) for p in pose[:3])
    Rx = np.array([[1, 0, 0], [0, np.cos(pitch), -np.sin(pitch)], [0, np.sin(pitch), np.cos(pitch)]])
    Ry = np.array([[np.cos(yaw), 0, np.sin(yaw)], [0, 1, 0], [-np.sin(yaw), 0, np.cos(yaw)]])
    Rz = np.array([[np.cos(roll), -np.sin(roll), 0], [np.sin(roll), np.cos(roll), 0], [0, 0, 1]])
    R = Rz @ Ry @ Rx
    origin = np.array([0.0, 0.0, 0.0])
    axes = np.array([[length, 0, 0], [0, length, 0], [0, 0, length]], dtype=np.float64)
    proj = (R @ axes.T).T
    nose = tuple(np.round(kps[2]).astype(int))
    colors = ((0, 0, 255), (0, 200, 0), (255, 120, 0))  # X 红 / Y 绿 / Z 蓝
    for i, c in enumerate(colors):
        end = (int(nose[0] + proj[i, 0]), int(nose[1] + proj[i, 1]))
        cv2.arrowedLine(image, nose, end, c, 2, cv2.LINE_AA, tipLength=0.22)
