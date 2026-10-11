"""通用工具:Unicode 安全的图像 IO、中文文本渲染、帧率统计。

Windows 上两个坑,这里统一处理:

1. ``cv2.imread`` / ``imwrite`` 用系统 ANSI 编码解释路径,项目目录名带中文时
   会静默返回 None。改用 ``np.fromfile`` + ``cv2.imdecode``。
2. ``cv2.putText`` 只支持 ASCII,中文会变成 "???"。改用 Pillow 光栅化。
"""

from __future__ import annotations

import time
from collections import deque
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

BGR = Tuple[int, int, int]

_FONT_CANDIDATES = (
    "msyh.ttc",      # 微软雅黑
    "msyhbd.ttc",
    "simhei.ttf",    # 黑体
    "simsun.ttc",    # 宋体
    "Deng.ttf",
)


def imread_u(path: str | Path) -> Optional[np.ndarray]:
    p = Path(path)
    if not p.exists():
        return None
    return cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)


def imwrite_u(path: str | Path, image: np.ndarray) -> bool:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    ext = p.suffix or ".jpg"
    ok, buf = cv2.imencode(ext, image)
    if not ok:
        return False
    buf.tofile(str(p))
    return True


def find_cjk_font(explicit: Optional[str] = None) -> Optional[str]:
    if explicit and Path(explicit).exists():
        return explicit
    roots = [Path("C:/Windows/Fonts"), Path("/usr/share/fonts"), Path.home() / ".fonts"]
    for root in roots:
        if not root.is_dir():
            continue
        for name in _FONT_CANDIDATES:
            cand = root / name
            if cand.exists():
                return str(cand)
        for cand in root.glob("**/*.tt[cf]"):
            return str(cand)
    return None


class TextPainter:
    """把一批中文文本一次性绘制到 BGR 画布上。

    每帧只做一次 numpy → PIL → numpy 往返(约 5ms),比逐条文本贴图快得多。
    找不到中文字体时退化为 ``cv2.putText``,至少 ASCII 部分仍可显示。
    """

    def __init__(self, font_path: Optional[str] = None, size: int = 15):
        self.size = size
        self.font_path = find_cjk_font(font_path)
        self._fonts: dict[int, object] = {}
        self._pil = None
        # 文本宽度缓存:换行/截断要逐字符测量,不缓存的话每帧上千次 getlength 会吃掉 100ms+
        self._char_w: dict[tuple, float] = {}
        self._str_w: dict[tuple, float] = {}
        self._glyphs: dict[tuple, tuple] = {}
        if self.font_path:
            try:
                from PIL import Image, ImageDraw, ImageFont

                self._pil = (Image, ImageDraw, ImageFont)
            except ImportError:
                self.font_path = None

    def _font(self, size: int):
        if size not in self._fonts:
            _, _, ImageFont = self._pil
            self._fonts[size] = ImageFont.truetype(self.font_path, size)
        return self._fonts[size]

    def _cw(self, ch: str, size: int) -> float:
        k = (ch, size)
        w = self._char_w.get(k)
        if w is None:
            w = float(self._font(size).getlength(ch))
            if len(self._char_w) > 8192:
                self._char_w.clear()
            self._char_w[k] = w
        return w

    def render(
        self,
        canvas: np.ndarray,
        items: Iterable[Sequence],
    ) -> np.ndarray:
        """items 中每项为 ``(x, y, text, color_bgr)`` 或 ``(x, y, text, color_bgr, size)``。

        中文文本用「字形缓存」绘制:每个 (字符, 字号, 颜色) 的 alpha 位图只光栅化一次,
        之后按 advance 宽度逐字 blit 并做 alpha 混合。整屏约两千字时比每帧调用
        ``ImageDraw.text`` 快一个数量级(80ms → 约 8ms)。
        """
        items = list(items)
        if not items:
            return canvas
        if self._pil is None:
            for it in items:
                x, y, text, color = it[0], it[1], it[2], it[3]
                cv2.putText(canvas, str(text), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
            return canvas
        for it in items:
            x, y, text, color = it[0], it[1], it[2], it[3]
            size = int(it[4]) if len(it) > 4 else self.size
            self._draw_string(canvas, int(x), int(y), str(text), tuple(int(c) for c in color), size)
        return canvas

    def _glyph(self, ch: str, size: int, color: Tuple[int, int, int]):
        key = (ch, size, color)
        g = self._glyphs.get(key)
        if g is None:
            Image, ImageDraw, _ = self._pil
            font = self._font(size)
            # Pillow ≥12 的 getmask 返回的 ImagingCore 无法直接转 numpy,
            # 改为把单字绘制到临时 Image 上再取 alpha。
            bbox = font.getbbox(ch)
            w = max(1, int(bbox[2] - bbox[0]))
            h = max(1, int(bbox[3] - bbox[1]))
            im = Image.new("L", (w, h), 0)
            ImageDraw.Draw(im).text((-bbox[0], -bbox[1]), ch, font=font, fill=255)
            alpha = np.asarray(im, dtype=np.float32) / 255.0
            g = (alpha, int(bbox[0]), int(bbox[1]))
            if len(self._glyphs) > 6000:
                self._glyphs.clear()
            self._glyphs[key] = g
        return g

    def _draw_string(self, canvas, x: int, y: int, text: str, color, size: int) -> None:
        pen = float(x)
        top = y - size
        col = np.array(color, dtype=np.float32)
        H, W = canvas.shape[:2]
        for ch in text:
            adv = self._cw(ch, size)
            if ch.isspace():
                pen += adv
                continue
            alpha, ox, oy = self._glyph(ch, size, color)
            h, w = alpha.shape
            if h and w:
                x0, y0 = int(round(pen + ox)), int(round(top + oy))
                cx0, cy0 = max(0, x0), max(0, y0)
                cx1, cy1 = min(W, x0 + w), min(H, y0 + h)
                if cx1 > cx0 and cy1 > cy0:
                    aa = alpha[cy0 - y0: cy1 - y0, cx0 - x0: cx1 - x0]
                    region = canvas[cy0:cy1, cx0:cx1]
                    solid = aa >= 0.999
                    if solid.any():
                        region[solid] = color
                    mid = (aa > 0.001) & ~solid
                    if mid.any():
                        am = aa[mid][..., None]
                        region[mid] = (region[mid] * (1.0 - am) + col * am).astype(np.uint8)
            pen += adv

    def measure(self, text: str, size: Optional[int] = None) -> Tuple[int, int]:
        size = size or self.size
        if self._pil is None:
            return int(len(text) * size * 0.55), size
        key = (text, size)
        w = self._str_w.get(key)
        if w is None:
            w = sum(self._cw(c, size) for c in text)
            if len(self._str_w) > 20000:
                self._str_w.clear()
            self._str_w[key] = w
        return int(w), size


class FpsMeter:
    """滑动窗口帧率,同时记录单帧耗时以便观察推理开销。"""

    def __init__(self, window: int = 30):
        self._stamps: deque[float] = deque(maxlen=window)
        self.last_ms: float = 0.0

    def tick(self) -> float:
        now = time.perf_counter()
        if self._stamps:
            self.last_ms = (now - self._stamps[-1]) * 1000.0
        self._stamps.append(now)
        return self.value

    @property
    def value(self) -> float:
        if len(self._stamps) < 2:
            return 0.0
        span = self._stamps[-1] - self._stamps[0]
        return (len(self._stamps) - 1) / span if span > 0 else 0.0


class StopWatch:
    """统计各阶段耗时,用于面板上的性能读数。"""

    def __init__(self):
        self._marks: dict[str, List[float]] = {}
        self._t0 = time.perf_counter()

    def mark(self, name: str, dt: float) -> None:
        buf = self._marks.setdefault(name, [])
        buf.append(dt * 1000.0)
        if len(buf) > 30:
            del buf[0]

    def avg(self, name: str) -> float:
        buf = self._marks.get(name)
        return float(np.mean(buf)) if buf else 0.0

    def names(self) -> List[str]:
        return list(self._marks)


def sanitize_id(raw: str) -> str:
    """把用户输入的姓名/ID 变成安全的文件名片段。"""
    keep = [c if (c.isalnum() or c in "-_") else "_" for c in raw.strip()]
    out = "".join(keep).strip("_")
    return out[:48] or "person"


def clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return float(max(lo, min(hi, v)))


def ramp(v: Optional[float], lo: float, hi: float) -> float:
    """把原始指标线性映射到 [0,1];lo 对应 0,hi 对应 1。None 透传。"""
    if v is None:
        return None  # type: ignore[return-value]
    if hi == lo:
        return 1.0 if v >= hi else 0.0
    return clamp((v - lo) / (hi - lo))


def ramp_inv(v: Optional[float], lo: float, hi: float) -> float:
    """反向映射:lo 对应 1,hi 对应 0(用于「越大越假」的指标)。"""
    if v is None:
        return None  # type: ignore[return-value]
    return ramp(v, hi, lo)


def hsv_skin_ratio(bgr: np.ndarray, hue_lo: int, hue_hi: int) -> Tuple[float, float, float]:
    """返回 (肤色像素占比, 平均饱和度, 平均明度)。"""
    if bgr.size == 0:
        return 0.0, 0.0, 0.0
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h = hsv[:, :, 0].astype(np.float32) / 179.0
    s = hsv[:, :, 1].astype(np.float32) / 255.0
    v = hsv[:, :, 2].astype(np.float32) / 255.0
    hue = h * 179.0
    if hue_lo <= hue_hi:
        mask = (hue >= hue_lo) & (hue <= hue_hi)
    else:
        mask = (hue >= hue_lo) | (hue <= hue_hi)
    mask &= s > 0.12
    mask &= v > 0.15
    return float(mask.mean()), float(s.mean()), float(v.mean())
