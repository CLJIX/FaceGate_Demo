"""人脸库:每人一个 ``.npz`` 存特征,``index.json`` 存索引。

存储位置由 ``config.yaml`` 的 ``gallery.dir`` 决定,默认 ``<项目根>/data/gallery``:

    data/gallery/
      index.json          人员索引:{id, 姓名, 创建时间, 样本数, 平均检测分}
      <id>.npz            embeddings (N,512) float32 + 每条样本的质量指标
      <id>_thumb.jpg      缩略图,命令行列表和管理界面用

之所以按人分文件而不是全库一个大 npz:删除单人时不需要重写整个库,
也不会因为一次写坏而丢掉全部数据。
"""

from __future__ import annotations

import json
import shutil
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .config import GalleryCfg
from .util import imread_u, imwrite_u, sanitize_id

EMBED_DIM = 512


@dataclass
class Person:
    id: str
    name: str
    created_at: float
    samples: int = 0
    mean_det_score: float = 0.0
    note: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Match:
    person: Person
    similarity: float        # 与该人全部样本的最高余弦相似度
    mean_similarity: float   # 与该人全部样本的平均余弦相似度
    best_index: int          # 命中的是第几条样本


@dataclass
class Gallery:
    cfg: GalleryCfg
    root: Path
    people: Dict[str, Person] = field(default_factory=dict)
    _feats: Dict[str, np.ndarray] = field(default_factory=dict, repr=False)
    _meta: Dict[str, Dict[str, np.ndarray]] = field(default_factory=dict, repr=False)

    # -- 载入与落盘 ----------------------------------------------------------

    @classmethod
    def open(cls, cfg: GalleryCfg, root: Path) -> "Gallery":
        g = cls(cfg=cfg, root=root)
        root.mkdir(parents=True, exist_ok=True)
        g.reload()
        return g

    def reload(self) -> None:
        self.people.clear()
        self._feats.clear()
        self._meta.clear()
        idx_path = self.root / "index.json"
        entries: List[dict] = []
        if idx_path.exists():
            try:
                with open(idx_path, "r", encoding="utf-8") as fh:
                    entries = json.load(fh).get("people", [])
            except (json.JSONDecodeError, OSError):
                entries = []
        for e in entries:
            pid = str(e.get("id", ""))
            npz = self.root / f"{pid}.npz"
            if not pid or not npz.exists():
                continue
            try:
                data = np.load(npz, allow_pickle=False)
            except (OSError, ValueError):
                continue
            emb = np.asarray(data["embeddings"], dtype=np.float32)
            if emb.ndim != 2 or emb.shape[1] != EMBED_DIM or emb.shape[0] == 0:
                continue
            self._feats[pid] = emb
            self._meta[pid] = {k: np.asarray(data[k]) for k in data.files if k != "embeddings"}
            self.people[pid] = Person(
                id=pid,
                name=str(e.get("name", pid)),
                created_at=float(e.get("created_at", 0.0)),
                samples=int(e.get("samples", emb.shape[0])),
                mean_det_score=float(e.get("mean_det_score", 0.0)),
                note=str(e.get("note", "")),
            )
        # index.json 可能落后于实际文件,补录孤儿 npz
        for npz in sorted(self.root.glob("*.npz")):
            pid = npz.stem
            if pid in self.people:
                continue
            try:
                data = np.load(npz, allow_pickle=False)
                emb = np.asarray(data["embeddings"], dtype=np.float32)
            except (OSError, ValueError, KeyError):
                continue
            if emb.ndim != 2 or emb.shape[1] != EMBED_DIM:
                continue
            self._feats[pid] = emb
            self._meta[pid] = {k: np.asarray(data[k]) for k in data.files if k != "embeddings"}
            self.people[pid] = Person(pid, pid, time.time(), emb.shape[0])
        self._save_index()

    def _save_index(self) -> None:
        payload = {
            "version": 1,
            "updated_at": time.time(),
            "people": [p.to_dict() for p in sorted(self.people.values(), key=lambda x: x.created_at)],
        }
        tmp = self.root / "index.json.tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
        tmp.replace(self.root / "index.json")

    # -- 查询 ----------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.people)

    @property
    def total_samples(self) -> int:
        return int(sum(f.shape[0] for f in self._feats.values()))

    def get(self, pid: str) -> Optional[Person]:
        return self.people.get(pid)

    def find(self, key: str) -> List[Person]:
        """按 ID 精确匹配,或按姓名子串匹配。"""
        k = key.strip()
        if not k:
            return []
        if k in self.people:
            return [self.people[k]]
        return [p for p in self.people.values() if k.lower() in p.name.lower() or k.lower() in p.id.lower()]

    def match(self, embedding: np.ndarray, top_k: int = 5) -> List[Match]:
        """返回余弦相似度最高的若干人,按相似度降序。特征已归一化时即点积。"""
        q = np.asarray(embedding, dtype=np.float32).ravel()
        nq = float(np.linalg.norm(q))
        if nq < 1e-9 or not self._feats:
            return []
        q = q / nq
        out: List[Match] = []
        for pid, feats in self._feats.items():
            norms = np.linalg.norm(feats, axis=1)
            norms[norms < 1e-9] = 1e-9
            sims = feats @ q / norms
            best = int(np.argmax(sims))
            out.append(
                Match(
                    person=self.people[pid],
                    similarity=float(sims[best]),
                    mean_similarity=float(sims.mean()),
                    best_index=best,
                )
            )
        out.sort(key=lambda m: -m.similarity)
        return out[:top_k]

    # -- 写入 ----------------------------------------------------------------

    def add_sample(
        self,
        key: str,
        name: Optional[str],
        embedding: np.ndarray,
        det_score: float,
        thumbnail: Optional[np.ndarray] = None,
        extra: Optional[Dict[str, float]] = None,
    ) -> Tuple[Person, int]:
        """给某人追加一条样本;人不存在则创建。返回 (person, 该人当前样本数)。"""
        pid = sanitize_id(key)
        emb = np.asarray(embedding, dtype=np.float32).reshape(1, EMBED_DIM)

        if pid in self._feats:
            old = self._feats[pid]
            norms = np.linalg.norm(old, axis=1)
            norms[norms < 1e-9] = 1e-9
            if float(np.max(old @ emb[0] / norms)) > self.cfg.max_pairwise_sim:
                raise DuplicateSample(f"与已有样本过于相似(>{self.cfg.max_pairwise_sim}),请换个姿态")
            feats = np.vstack([old, emb])
        else:
            feats = emb

        meta: Dict[str, np.ndarray] = {}
        prev = self._meta.get(pid, {})
        for k in ("det_score", "live_score", "yaw", "pitch", "roll", "laplacian"):
            arr = prev.get(k, np.zeros((feats.shape[0] - 1,), dtype=np.float32))
            val = float(det_score) if k == "det_score" else float((extra or {}).get(k, 0.0))
            meta[k] = np.append(np.asarray(arr, dtype=np.float32), np.float32(val))
        for k in ("det_score", "live_score", "yaw", "pitch", "roll", "laplacian"):
            if meta[k].shape[0] < feats.shape[0]:
                meta[k] = np.pad(meta[k], (0, feats.shape[0] - meta[k].shape[0]))

        person = self.people.get(pid)
        if person is None:
            person = Person(id=pid, name=(name or pid).strip() or pid, created_at=time.time())
        else:
            if name:
                person.name = name.strip()

        np.savez_compressed(self.root / f"{pid}.npz", embeddings=feats, **meta)
        self._feats[pid] = feats
        self._meta[pid] = meta
        person.samples = int(feats.shape[0])
        person.mean_det_score = float(np.mean(meta["det_score"][: person.samples]))
        self.people[pid] = person

        if thumbnail is not None:
            imwrite_u(self.root / f"{pid}_thumb.jpg", thumbnail)
        self._save_index()
        return person, person.samples

    def rename(self, pid: str, new_name: str) -> bool:
        p = self.people.get(pid)
        if p is None or not new_name.strip():
            return False
        p.name = new_name.strip()
        self._save_index()
        return True

    def remove(self, key: str) -> List[Person]:
        """按 ID 或姓名删除,返回被删掉的人。"""
        targets = self.find(key)
        for p in targets:
            for suffix in (".npz", "_thumb.jpg"):
                f = self.root / f"{p.id}{suffix}"
                if f.exists():
                    f.unlink()
            self.people.pop(p.id, None)
            self._feats.pop(p.id, None)
            self._meta.pop(p.id, None)
        if targets:
            self._save_index()
        return targets

    def clear(self) -> int:
        n = len(self.people)
        for f in self.root.iterdir():
            if f.suffix in (".npz", ".jpg") or f.name == "index.json":
                f.unlink()
        self.people.clear()
        self._feats.clear()
        self._meta.clear()
        return n

    def thumbnail(self, pid: str):
        return imread_u(self.root / f"{pid}_thumb.jpg")

    def backup(self) -> Optional[Path]:
        if not self.root.exists() or not any(self.root.iterdir()):
            return None
        dst = self.root.parent / f"gallery_backup_{time.strftime('%Y%m%d_%H%M%S')}"
        shutil.copytree(self.root, dst)
        return dst


class DuplicateSample(Exception):
    """新样本与已有样本姿态过于接近,没有信息量。"""
