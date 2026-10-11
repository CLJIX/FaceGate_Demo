"""攻击类型判别:把七路线索的读数翻译成「这是哪一类载体」。

用户最常问的问题是「它凭什么知道是照片还是视频」。答案不在某一路线索里,而在
**线索签名**上:四类载体在七路线索上的表现各不相同,把当前读数与每类的期望签名
做匹配,最吻合的那类就是判定结果。

期望签名(1 = 该线索应呈现「活体特征」,0 = 应呈现「攻击特征」,None = 该类不关心):

    ============  ==========  ==========  ==========  ==================
    线索           真人        静置照片     晃动照片     屏幕翻拍/视频回放
    ============  ==========  ==========  ==========  ==================
    官方活体CNN    不关心      不关心      不关心      不关心
    自然眨眼       1          0          0          1   ← 回放里本来就有眨眼
    运动/形变      1          0          1          1   ← 回放里本来就有运动
    平面性视差     1          不关心      0          0   ← 所有平面载体的死穴
    高频纹理       1          1          1          0   ← 屏幕二次采样丢高频
    屏幕摩尔纹     1          1          1          0   ← 像素栅格的频域指纹
    色彩偏置       不关心      不关心      不关心      0
    ============  ==========  ==========  ==========  ==================

这张表里藏着三个关键事实,也是面板上「关键判别点」要讲清的:

1. **眨眼和运动拦不住视频回放**。回放视频里有真人的眨眼和动作,这两路会被骗。
   拦回放靠的是摩尔纹、纹理、色彩(屏幕的物理指纹)加上平面性视差(屏幕是平面)。
2. **静置照片靠眨眼和运动就能拦**,因为它既不会眨眼也没有任何形变。
3. **晃动的照片是最难的一类**:它有运动、纹理也清晰、还没有摩尔纹,唯一露馅的地方
   是「平面」——转动时位移场可被单应完全解释,以及「没有非刚性形变」(不会眨眼张嘴)。

官方 CNN 对四类都不关心:它是单帧纹理判别器,实测对高清照片会给 0.98 的真人分,
在「分类是哪一种攻击」这个问题上不提供信息,只参与活体/非活体的门控。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from ..config import LivenessCfg
from .analyzer import CueResult

ATTACK_CLASSES: Tuple[Tuple[str, str], ...] = (
    ("live", "真人"),
    ("print_static", "静置打印照片"),
    ("print_moved", "晃动的打印照片"),
    ("screen", "屏幕翻拍/视频回放"),
)

SIGNATURE: Dict[str, Dict[str, Optional[int]]] = {
    "live": {"deep_model": None, "blink": 1, "micro_motion": 1, "planarity": 1, "texture": 1, "moire": 1, "color": None},
    "print_static": {"deep_model": None, "blink": 0, "micro_motion": 0, "planarity": None, "texture": 1, "moire": 1, "color": None},
    "print_moved": {"deep_model": None, "blink": 0, "micro_motion": 1, "planarity": 0, "texture": 1, "moire": 1, "color": None},
    "screen": {"deep_model": None, "blink": 1, "micro_motion": 1, "planarity": 0, "texture": 0, "moire": 0, "color": 0},
}

# 每一对类别之间,真正起区分作用的线索
DISCRIMINATORS: Tuple[Tuple[str, str, str], ...] = (
    ("静置照片 vs 真人", "blink,micro_motion", "照片不会眨眼、也没有任何非刚性形变"),
    ("晃动照片 vs 真人", "planarity,micro_motion", "照片是平面且不会眨眼张嘴,转动时位移场可被单应完全解释"),
    ("屏幕/回放 vs 真人", "moire,texture,planarity", "回放自带眨眼和运动,只能靠屏幕的物理指纹和平面性"),
    ("照片 vs 屏幕", "moire,texture,color", "屏幕有像素栅格摩尔纹、二次采样丢高频、背光色偏;打印件没有"),
)


@dataclass
class ClassScore:
    key: str
    label: str
    consistency: float          # 0..1,与期望签名的吻合度
    support: List[str] = field(default_factory=list)    # 支持该类的线索
    against: List[str] = field(default_factory=list)    # 反对该类的线索
    n_used: int = 0


@dataclass
class AttackAnalysis:
    scores: List[ClassScore]
    winner: Optional[ClassScore]
    margin: float
    enough: bool
    note: str


def classify_attack(cues: List[CueResult]) -> AttackAnalysis:
    by_key = {c.key: c for c in cues}
    out: List[ClassScore] = []
    for key, label in ATTACK_CLASSES:
        sig = SIGNATURE[key]
        num = 0.0
        den = 0.0
        support: List[str] = []
        against: List[str] = []
        for cue_key, expect in sig.items():
            c = by_key.get(cue_key)
            if expect is None or c is None or c.score is None or c.confidence < 0.15:
                continue
            agree = c.score if expect == 1 else 1.0 - c.score
            w = max(c.weight, 0.5) * c.confidence
            num += agree * w
            den += w
            (support if agree >= 0.5 else against).append(c.label)
        out.append(
            ClassScore(
                key=key,
                label=label,
                consistency=(num / den) if den > 1e-6 else 0.0,
                support=support,
                against=against,
                n_used=len(support) + len(against),
            )
        )

    out.sort(key=lambda s: -s.consistency)
    used = max((s.n_used for s in out), default=0)
    enough = used >= 3
    winner = out[0] if enough else None
    margin = (out[0].consistency - out[1].consistency) if len(out) > 1 else 0.0
    if not enough:
        note = f"参与匹配的线索只有 {used} 路,还不足以区分载体类型"
    elif margin < 0.08:
        note = f"「{out[0].label}」与「{out[1].label}」吻合度接近,证据还不够分明"
    else:
        note = f"最吻合「{winner.label}」,领先次优 {margin:.2f}"
    return AttackAnalysis(scores=out, winner=winner, margin=margin, enough=enough, note=note)


# ---------------------------------------------------------------------------
# 逐线索的详细判据说明
# ---------------------------------------------------------------------------


def explain_cue(cue: CueResult, cfg: LivenessCfg) -> List[str]:
    """给出一路线索的「实测值 / 阈值 / 结论」,用于界面与文档。"""
    e = cue.extra
    if cue.score is None:
        return [f"{cue.label}:证据不足 — {cue.hint}"]

    if cue.key == "deep_model":
        v = e.get("mid", cue.score)
        return [
            f"{cue.label} = {cue.score:.3f}(官方 80×80 CNN 单帧直出)",
            "  单帧纹理判别器,对高清照片会被骗到 0.9+,只参与门控、不参与攻击分类",
        ]

    if cue.key == "blink":
        base = e.get("ear_baseline", 0.0)
        depth = e.get("depth", 0.0)
        b = cfg.blink
        return [
            f"{cue.label} = {cue.score:.3f}|窗口内眨眼 {int(e.get('blinks', 0))} 次,"
            f"EAR 基线 {base:.3f},闭合深度 {depth:.2f}",
            f"  判据:相对基线跌到 {b.ear_close_ratio:.2f} 以下记一次闭眼;≥{b.min_blinks} 次给高分",
        ]

    if cue.key == "micro_motion":
        return [
            f"{cue.label} = {cue.score:.3f}|帧间位移 {e.get('motion', 0):.4f},"
            f"非刚性形变 {e.get('deform', 0):.4f},姿态抖动 {e.get('jitter_deg', 0):.2f}°",
            "  判据:完全冻结→静置照片;有运动但无非刚性形变→平面载体;形变充分→真人",
        ]

    if cue.key == "planarity":
        p = cfg.planarity
        return [
            f"{cue.label} = {cue.score:.3f}|单应残差 {e.get('homo_resid', 0):.4f} 瞳距"
            f"(P90 {e.get('homo_p90', 0):.4f}),位移 {e.get('homo_motion', 0):.4f},内点率 {e.get('inlier', 0):.2f}",
            f"  判据:残差 <{p.residual_fake_below:.3f}→平面载体;>{p.residual_live_above:.3f}→三维实体;"
            f"位移 <{p.require_motion_norm:.2f} 时不出结论",
        ]

    if cue.key == "texture":
        t = cfg.texture
        return [
            f"{cue.label} = {cue.score:.3f}|Laplacian 方差与高低频比(对齐到 112² 后统计)",
            f"  判据:方差 <{t.laplacian_fake_below:.0f} 或高频比 <{t.band_ratio_fake_below:.2f}→二次采样/压缩",
        ]

    if cue.key == "moire":
        m = cfg.moire
        return [
            f"{cue.label} = {cue.score:.3f}|峰噪比 {e.get('peak_ratio', 0):.2f},孤立尖峰 {int(e.get('peak_count', 0))} 个",
            f"  判据:峰噪比 >{m.peak_ratio_fake_above:.1f} 且尖峰 ≥{m.min_peaks}→像素栅格;真人频谱平滑衰减",
        ]

    if cue.key == "color":
        c = cfg.color
        return [
            f"{cue.label} = {cue.score:.3f}|平均饱和度 {e.get('sat', 0):.3f},肤色占比 {e.get('skin', 0):.3f}",
            f"  判据:饱和度 >{c.sat_fake_above:.2f} 或肤色占比 <{c.skin_ratio_fake_below:.2f}→背光/纸张",
        ]

    return [f"{cue.label} = {cue.score:.3f}"]
