"""摄像头 / 视频文件采集。

支持两种输入:
  * 摄像头序号(默认),实时门禁场景
  * 视频文件路径,用于做「屏幕翻拍 / 视频回放」攻击测试,可反复重放同一段素材
"""

from __future__ import annotations

import os
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .config import CameraCfg

_BACKENDS = {
    "dshow": cv2.CAP_DSHOW,
    "msmf": cv2.CAP_MSMF,
    "any": cv2.CAP_ANY,
    "v4l2": cv2.CAP_V4L2,
    "avfoundation": cv2.CAP_AVFOUNDATION,
}


def list_cameras(max_index: int = 6) -> List[Tuple[int, Tuple[int, int]]]:
    """按序号探测可用摄像头,返回 ``[(序号, (宽, 高)), ...]``。

    打不开的序号直接跳过;DSHOW 失败时退回默认后端再试一次,与 VideoSource 一致。
    探测必然产生大量「打不开序号 N」的后端日志,期间把 stdout/stderr 的 fd 指向
    devnull 静音(C 层日志不走 Python 的 sys.stdout,只能动 fd)。
    """
    out: List[Tuple[int, Tuple[int, int]]] = []
    sys.stdout.flush()
    sys.stderr.flush()
    keep = [os.dup(1), os.dup(2)]
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 1)
    os.dup2(devnull, 2)
    try:
        for i in range(max_index):
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
            if not cap.isOpened():
                cap = cv2.VideoCapture(i)
            if not cap.isOpened():
                continue
            ok, fr = cap.read()
            if ok and fr is not None:
                out.append((i, (int(fr.shape[1]), int(fr.shape[0]))))
            cap.release()
    finally:
        os.dup2(keep[0], 1)
        os.dup2(keep[1], 2)
        os.close(devnull)
        os.close(keep[0])
        os.close(keep[1])
        sys.stdout.flush()
        sys.stderr.flush()
    return out


@dataclass
class Frame:
    image: np.ndarray      # BGR, 已按需镜像
    index: int             # 自增帧号
    stamp: float           # time.perf_counter() 秒
    source: str            # "camera:0" 或文件路径


class VideoSource:
    """摄像头或视频文件源。

    摄像头默认走**后台抓取线程**:DirectShow 的 ``read()`` 是阻塞的,处理一帧 60ms 时
    会额外等下一个帧边界(33ms),实测把帧率从处理能力的 17fps 拖到 10fps。后台线程持续
    把最新帧放进单槽缓冲,主循环只取「当前最新」,帧率就完全由处理速度决定。
    视频文件源保持顺序阻塞读取,保证攻击素材可复现地逐帧重放。
    """

    def __init__(self, cfg: CameraCfg):
        self.cfg = cfg
        self.is_file = bool(cfg.source)
        self._loop_count = 0

        if self.is_file:
            path = Path(cfg.source)  # type: ignore[arg-type]
            if not path.exists():
                raise FileNotFoundError(f"视频源不存在: {path}")
            self.cap = cv2.VideoCapture(str(path))
            self.label = str(path)
            self.index = -1
            if not self.cap.isOpened():
                raise RuntimeError(f"无法打开视频源 {self.label}。")
            ok, probe = self.cap.read()
            if not ok or probe is None:
                raise RuntimeError(f"视频源 {self.label} 已打开但读不到画面。")
            self.size = (probe.shape[1], probe.shape[0])
        else:
            probe = self._open_camera(int(cfg.index))

        self._index = 0
        self._t0 = time.perf_counter()
        self._thread: Optional[threading.Thread] = None
        self._running = False
        if not self.is_file and cfg.async_grab:
            self._start_grab(probe)
        else:
            self._pending: Optional[np.ndarray] = probe

    def _open_camera(self, index: int) -> np.ndarray:
        """打开摄像头并读一帧验证,返回首帧;失败抛 RuntimeError。"""
        backend = _BACKENDS.get(str(self.cfg.backend).lower(), cv2.CAP_ANY)
        cap = cv2.VideoCapture(index, backend)
        if not cap.isOpened():
            # DSHOW 打不开时退回默认后端再试一次
            cap = cv2.VideoCapture(index)
        if not cap.isOpened():
            raise RuntimeError(
                f"无法打开摄像头 {index}。请检查是否被其他程序占用,"
                f"或在 config.yaml 中把 camera.backend 改成 msmf。"
            )
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cfg.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg.height)
        if self.cfg.fps:
            cap.set(cv2.CAP_PROP_FPS, self.cfg.fps)
        ok, probe = cap.read()
        if not ok or probe is None:
            cap.release()
            raise RuntimeError(f"摄像头 {index} 已打开但读不到画面。")
        self.cap = cap
        self.label = f"camera:{index}"
        self.index = index
        self.size = (probe.shape[1], probe.shape[0])
        return probe

    def _start_grab(self, probe: np.ndarray) -> None:
        self._lock = threading.Lock()
        self._latest: Optional[np.ndarray] = probe
        self._new = True
        self._running = True
        self._thread = threading.Thread(target=self._grab_loop, daemon=True, name="camera-grab")
        self._thread.start()

    def _stop_grab(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

    def switch_camera(self, index: int) -> str:
        """运行中换到另一个摄像头,返回新 label;视频文件源不支持。"""
        if self.is_file:
            raise RuntimeError("当前是视频文件源,不能切换摄像头")
        self._stop_grab()
        self.cap.release()
        probe = self._open_camera(index)
        self._index = 0
        self._t0 = time.perf_counter()
        if self.cfg.async_grab:
            self._start_grab(probe)
        else:
            self._pending = probe
        return self.label

    def _grab_loop(self) -> None:
        while self._running:
            ok, fr = self.cap.read()
            if not ok or fr is None:
                self._running = False
                break
            with self._lock:
                self._latest = fr
                self._new = True

    def read(self) -> Optional[Frame]:
        """取一帧;文件源读到结尾时自动循环。摄像头断开则返回 None。"""
        if self._thread is not None:
            image = self._wait_latest()
            if image is None:
                return None
        elif getattr(self, "_pending", None) is not None:
            image, self._pending = self._pending, None
        else:
            ok, image = self.cap.read()
            if not ok or image is None:
                if self.is_file:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    self._loop_count += 1
                    ok, image = self.cap.read()
                if not ok or image is None:
                    return None
        if self.cfg.mirror and not self.is_file:
            image = cv2.flip(image, 1)
        self._index += 1
        return Frame(image=image, index=self._index, stamp=time.perf_counter() - self._t0, source=self.label)

    def _wait_latest(self, timeout: float = 0.5) -> Optional[np.ndarray]:
        deadline = time.perf_counter() + timeout
        while self._running or self._peek_new():
            with self._lock:
                if self._new:
                    self._new = False
                    return self._latest
            if time.perf_counter() > deadline:
                return None
            time.sleep(0.0008)
        with self._lock:
            if self._new:
                self._new = False
                return self._latest
        return None

    def _peek_new(self) -> bool:
        with self._lock:
            return self._new

    @property
    def loops(self) -> int:
        return self._loop_count

    def release(self) -> None:
        self._stop_grab()
        self.cap.release()

    def __enter__(self) -> "VideoSource":
        return self

    def __exit__(self, *_exc) -> None:
        self.release()
