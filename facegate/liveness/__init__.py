"""单目 RGB 活体检测:几何原语 + 七路线索 + 融合判决。"""

from .analyzer import (
    CUE_LABELS,
    CueResult,
    LivenessAnalyzer,
    LivenessResult,
    Sample,
    frame_metrics,
)

__all__ = [
    "CUE_LABELS",
    "CueResult",
    "LivenessAnalyzer",
    "LivenessResult",
    "Sample",
    "frame_metrics",
]
