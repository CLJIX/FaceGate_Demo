"""生成两份可复现的测试素材,用于验证活体检测的两条路径。

    samples/attack_photo.mp4   完全静止的照片视频 → 期望判定「非活体」
    samples/live_proxy.mp4     带眨眼/自然晃动/嘴部开合的活体代理 → 期望通过活体

这不是真实攻击样本,而是 ground-truth 受控的机械验证素材:没有打印机和屏幕也能
复现地测试时间维度线索(眨眼、运动形变)是否真的在工作。

    python tools/make_live_proxy.py
"""

import os
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import insightface

from facegate.config import load_config
from facegate.engine import FaceEngine
from facegate.util import imread_u

FPS = 15
SECONDS = 10
SIZE = (640, 480)


def load_face() -> tuple[np.ndarray, object]:
    cfg = load_config()
    eng = FaceEngine(cfg.engine)
    img = imread_u(Path(insightface.__file__).parent / "data" / "images" / "t1.jpg")
    base = cv2.resize(img[230:470, 430:750], SIZE)
    return base, eng.detect(base)[0]


def make_photo_attack(base: np.ndarray, path: Path) -> None:
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, SIZE)
    for _ in range(FPS * 6):
        w.write(base)
    w.release()
    print(f"written {path} ({path.stat().st_size} bytes) — 静止照片,期望:非活体")


def blink_env(t: float) -> float:
    ph = (t % 3.2) / 3.2
    if ph > 0.09:
        return 0.0
    return float(np.sin(np.pi * (ph / 0.09)) ** 2)


def squash(frame: np.ndarray, center, radius: int, fy: float) -> np.ndarray:
    cx, cy = center
    x1, y1 = max(0, cx - radius), max(0, cy - radius)
    x2, y2 = min(frame.shape[1], cx + radius), min(frame.shape[0], cy + radius)
    if x2 <= x1 or y2 <= y1 or fy >= 0.999:
        return frame
    roi = frame[y1:y2, x1:x2]
    h, w = roi.shape[:2]
    M = np.float32([[1, 0, 0], [0, fy, (cy - y1) * (1 - fy)]])
    frame[y1:y2, x1:x2] = cv2.warpAffine(roi, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    return frame


def make_live_proxy(base: np.ndarray, obs, path: Path) -> None:
    eyes = [tuple(np.round(p).astype(int)) for p in obs.kps[:2]]
    iod = obs.iod
    mouth_c = (int((obs.kps[3][0] + obs.kps[4][0]) / 2), int((obs.kps[3][1] + obs.kps[4][1]) / 2))
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, SIZE)
    for i in range(FPS * SECONDS):
        t = i / FPS
        dx = 3.0 * np.sin(2 * np.pi * t / 2.1)
        dy = 2.0 * np.sin(2 * np.pi * t / 1.7)
        ang = 1.2 * np.sin(2 * np.pi * t / 3.3)
        M = cv2.getRotationMatrix2D((SIZE[0] / 2, SIZE[1] / 2), ang, 1.0)
        M[0, 2] += dx
        M[1, 2] += dy
        fr = cv2.warpAffine(base, M, SIZE, borderMode=cv2.BORDER_REPLICATE)

        b = blink_env(t)
        if b > 0.01:
            for c in eyes:
                fr = squash(fr, (int(c[0] + dx), int(c[1] + dy)), int(iod * 0.42), 1.0 - 0.85 * b)
        mo = 0.06 * np.sin(2 * np.pi * t / 1.1)
        if abs(mo) > 0.01:
            fr = squash(fr, (int(mouth_c[0] + dx), int(mouth_c[1] + dy)), int(iod * 0.5), 1.0 + mo)
        w.write(fr)
    w.release()
    print(f"written {path} ({path.stat().st_size} bytes) — 眨眼+晃动代理,期望:通过活体")


def main() -> int:
    out = ROOT / "samples"
    out.mkdir(exist_ok=True)
    base, obs = load_face()
    make_photo_attack(base, out / "attack_photo.mp4")
    make_live_proxy(base, obs, out / "live_proxy.mp4")
    print("\n测试: python main.py --source samples/attack_photo.mp4 recognize")
    return 0


if __name__ == "__main__":
    sys.exit(main())
