"""配置定义与加载。

所有可调参数集中在项目根目录的 ``config.yaml``;本模块把它解析成带类型的
dataclass,便于 IDE 补全、拼写检查,以及启动时打印一份完整参数快照。
"""

from __future__ import annotations

import dataclasses as dc
import typing
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, get_args, get_origin

import yaml

# ---------------------------------------------------------------------------
# 分节定义
# ---------------------------------------------------------------------------


@dc.dataclass
class CameraCfg:
    index: int = 0
    width: int = 640
    height: int = 480
    fps: int = 30
    backend: str = "dshow"
    source: Optional[str] = None
    mirror: bool = True
    async_grab: bool = True


@dc.dataclass
class EngineCfg:
    model_pack: str = "buffalo_l"
    det_size: List[int] = dc.field(default_factory=lambda: [320, 320])
    det_thresh: float = 0.5
    intra_op_threads: int = 4
    use_liveness_addon: bool = True
    liveness_threshold: float = 0.80
    max_faces: int = 1
    recognize_interval_ms: int = 400
    pose_every_n_frames: int = 2
    spectrum_every_n_frames: int = 2
    perf_profile: str = "balanced"
    use_roi_tracking: bool = True


@dc.dataclass
class CueSpec:
    """单路活体线索的开关、融合权重与一票否决线。"""

    enabled: bool = True
    weight: float = 1.0
    veto_below: Optional[float] = None


@dc.dataclass
class BlinkCfg:
    ear_close_ratio: float = 0.55
    min_blinks: int = 1
    window_sec: float = 6.0


@dc.dataclass
class MicroMotionCfg:
    motion_static_below: float = 0.0008
    motion_live_above: float = 0.0040
    jitter_calm_above: float = 2.0
    jitter_violent_above: float = 8.0
    deform_fake_below: float = 0.006
    deform_live_above: float = 0.030


@dc.dataclass
class PlanarityCfg:
    reference_age_sec: float = 0.6
    require_motion_norm: float = 0.08
    residual_fake_below: float = 0.012
    residual_live_above: float = 0.042
    ransac_reproj_px: float = 3.0


@dc.dataclass
class TextureCfg:
    laplacian_fake_below: float = 35.0
    laplacian_live_above: float = 90.0
    band_ratio_fake_below: float = 0.10
    band_ratio_live_above: float = 0.28


@dc.dataclass
class MoireCfg:
    peak_ratio_fake_above: float = 6.0
    peak_ratio_live_below: float = 3.0
    min_peaks: int = 3
    low_freq_keep: int = 6


@dc.dataclass
class ColorCfg:
    sat_fake_above: float = 0.42
    sat_live_below: float = 0.30
    skin_hue_lo: int = 3
    skin_hue_hi: int = 25
    skin_ratio_fake_below: float = 0.35


@dc.dataclass
class LivenessCfg:
    window_sec: float = 3.0
    min_track_frames: int = 12
    decision_threshold: float = 0.50
    confirm_frames: int = 5
    track_reset_frames: int = 10
    cues: Dict[str, CueSpec] = dc.field(
        default_factory=lambda: {
            "deep_model": CueSpec(weight=1.6, veto_below=0.08),
            "blink": CueSpec(weight=1.2),
            "micro_motion": CueSpec(weight=0.9),
            "planarity": CueSpec(weight=1.4),
            "texture": CueSpec(weight=0.6),
            "moire": CueSpec(weight=1.0, veto_below=0.20),
            "color": CueSpec(weight=0.4),
        }
    )
    blink: BlinkCfg = dc.field(default_factory=BlinkCfg)
    micro_motion: MicroMotionCfg = dc.field(default_factory=MicroMotionCfg)
    planarity: PlanarityCfg = dc.field(default_factory=PlanarityCfg)
    texture: TextureCfg = dc.field(default_factory=TextureCfg)
    moire: MoireCfg = dc.field(default_factory=MoireCfg)
    color: ColorCfg = dc.field(default_factory=ColorCfg)


@dc.dataclass
class GateCfg:
    accept_threshold: float = 0.42
    stranger_threshold: float = 0.16
    cooldown_sec: float = 3.0
    open_hold_sec: float = 2.0


@dc.dataclass
class GalleryCfg:
    dir: str = "data/gallery"
    samples_per_person: int = 5
    min_det_score: float = 0.65
    require_live: bool = True
    max_pairwise_sim: float = 0.92
    min_face_px: int = 90


@dc.dataclass
class VizCfg:
    enabled: bool = True
    panel_width: int = 570
    show_landmarks: bool = True
    show_pose_axis: bool = True
    show_homography: bool = True
    show_eye_axis: bool = True
    waveform_seconds: float = 8.0
    font: Optional[str] = None
    font_size: int = 15
    snapshot_dir: str = "runs/snapshots"
    minimal: bool = False


@dc.dataclass
class LogCfg:
    dir: str = "runs"
    level: str = "INFO"
    metrics_csv: bool = True


@dc.dataclass
class Config:
    camera: CameraCfg = dc.field(default_factory=CameraCfg)
    engine: EngineCfg = dc.field(default_factory=EngineCfg)
    liveness: LivenessCfg = dc.field(default_factory=LivenessCfg)
    gate: GateCfg = dc.field(default_factory=GateCfg)
    gallery: GalleryCfg = dc.field(default_factory=GalleryCfg)
    viz: VizCfg = dc.field(default_factory=VizCfg)
    log: LogCfg = dc.field(default_factory=LogCfg)

    root: Path = dc.field(default=Path("."), compare=False, repr=False)

    def resolve(self, rel: str) -> Path:
        p = Path(rel)
        return p if p.is_absolute() else (self.root / p).resolve()

    @property
    def gallery_dir(self) -> Path:
        return self.resolve(self.gallery.dir)

    @property
    def log_dir(self) -> Path:
        return self.resolve(self.log.dir)


# ---------------------------------------------------------------------------
# 通用 dict -> dataclass 加载
# ---------------------------------------------------------------------------


def _strip_optional(tp: Any) -> tuple[Any, bool]:
    if get_origin(tp) is Union:
        args = [a for a in get_args(tp) if a is not type(None)]
        return (args[0] if len(args) == 1 else Union[tuple(args)]), True
    return tp, False


def _coerce(tp: Any, value: Any, path: str) -> Any:
    tp, optional = _strip_optional(tp)
    if value is None:
        if optional or tp is type(None):
            return None
        raise ValueError(f"{path}: 不允许为空")

    origin = get_origin(tp)
    if dc.is_dataclass(tp) and isinstance(tp, type):
        return from_dict(tp, value, path)
    if origin in (list, List):
        inner = get_args(tp)[0] if get_args(tp) else Any
        return [_coerce(inner, v, f"{path}[{i}]") for i, v in enumerate(value)]
    if origin in (dict, Dict):
        kt, vt = get_args(tp) if get_args(tp) else (Any, Any)
        return {_coerce(kt, k, path): _coerce(vt, v, f"{path}.{k}") for k, v in value.items()}
    if tp is Any or tp is None:
        return value
    if tp is Path:
        return Path(value)
    if tp in (int, float, str, bool):
        if tp is bool and not isinstance(value, bool):
            raise ValueError(f"{path}: 需要布尔值,得到 {value!r}")
        return tp(value)
    return value


def from_dict(cls: type, data: Any, path: str = "") -> Any:
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ValueError(f"{path or cls.__name__}: 需要一个映射,得到 {type(data).__name__}")
    # 本模块启用了 PEP 563 延迟注解,f.type 是字符串,统一用 get_type_hints 解析成真实类型
    hints = typing.get_type_hints(cls)
    unknown = sorted(set(data) - set(hints))
    if unknown:
        raise ValueError(f"{path or cls.__name__}: 未知配置项 {unknown};可用项 {sorted(hints)}")
    kwargs = {}
    for name, tp in hints.items():
        if name in data:
            kwargs[name] = _coerce(tp, data[name], f"{path}.{name}" if path else name)
    return cls(**kwargs)


def load_config(path: str | Path | None = None, root: str | Path | None = None) -> Config:
    """读取 config.yaml;文件不存在时返回全默认配置。"""
    root = Path(root) if root is not None else Path(__file__).resolve().parent.parent
    cfg_path = Path(path) if path else root / "config.yaml"
    data: Dict[str, Any] = {}
    if cfg_path.exists():
        with open(cfg_path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
    cfg = from_dict(Config, data, "config")
    cfg.root = Path(root)
    return cfg


def dump_config(cfg: Config) -> str:
    """输出参数快照,启动时打印到日志。"""
    lines: List[str] = []

    def walk(obj: Any, indent: int) -> None:
        pad = "  " * indent
        for f in dc.fields(obj):
            if f.name in ("root",):
                continue
            val = getattr(obj, f.name)
            if dc.is_dataclass(val):
                lines.append(f"{pad}{f.name}:")
                walk(val, indent + 1)
            elif isinstance(val, dict) and val and all(dc.is_dataclass(v) for v in val.values()):
                lines.append(f"{pad}{f.name}:")
                for k, v in val.items():
                    lines.append(f"{pad}  {k}:")
                    walk(v, indent + 2)
            else:
                lines.append(f"{pad}{f.name}: {val}")

    walk(cfg, 0)
    return "\n".join(lines)
