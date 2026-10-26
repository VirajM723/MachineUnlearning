"""SISA training (Sharded, Isolated, Sliced, Aggregated) with exact unlearning.

Reference: Bourtoule et al., "Machine Unlearning", IEEE S&P 2021.

Training data is partitioned into ``S`` disjoint shards and every shard into
``R`` disjoint slices. One model is trained per shard, incrementally: on slice
1, then slices 1..2, and so on, checkpointing model *and optimizer* state after
each step. Predictions are the mean (or median) over shard models.

To unlearn a record we drop it from its slice, restore the shard's checkpoint
from just before that slice was introduced, and replay the remaining slices.
No other shard is touched. Since every slice is trained from a fixed seed, the
result is bit-for-bit what training without the record would have produced.
"""

from __future__ import annotations

import copy
import time
from dataclasses import asdict, dataclass, field
from typing import Iterable

import numpy as np
import torch

from .model import MLPRegressor, predict, train_epochs


@dataclass
class SISAConfig:
    num_shards: int = 10
    num_slices: int = 5
    epochs_per_slice: int = 5
    batch_size: int = 64
    lr: float = 1e-3
    hidden: tuple[int, ...] = (512, 512, 64)
    dropout: float = 0.3
    aggregation: str = "mean"  # "mean" or "median"
    seed: int = 0
    device: str = "cpu"

    def __post_init__(self):
        if self.aggregation not in ("mean", "median"):
            raise ValueError("aggregation must be 'mean' or 'median'")
        if self.num_shards < 1 or self.num_slices < 1:
            raise ValueError("num_shards and num_slices must be >= 1")
        self.hidden = tuple(self.hidden)


@dataclass
class SlicePart:
    """The rows of one (shard, slice) cell. Deleted rows are physically removed."""

    X: torch.Tensor
    y: torch.Tensor
    ids: np.ndarray


@dataclass
class UnlearnReport:
    requested: int
    removed: int
    not_found: list[int] = field(default_factory=list)
    # shard index -> first slice that had to be retrained
    retrained_from: dict[int, int] = field(default_factory=dict)
    slices_retrained: int = 0
    seconds: float = 0.0

    def summary(self) -> str:
        shards = ", ".join(f"shard {s} from slice {r}" for s, r in sorted(self.retrained_from.items()))
        return (
            f"removed {self.removed}/{self.requested} records in {self.seconds:.2f}s; "
            f"retrained {self.slices_retrained} slice steps ({shards or 'nothing'})"
        )


def _to_cpu(obj):
    if torch.is_tensor(obj):
        return obj.detach().cpu()
    if isinstance(obj, dict):
        return {k: _to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_cpu(v) for v in obj]
    return obj


class SISAEnsemble:
    def __init__(self, config: SISAConfig, in_features: int):
        self.config = config
        self.in_features = in_features
        self.parts: list[list[SlicePart]] = []
        self.models: list[MLPRegressor] = []
        # checkpoints[s][r] = state after training shard s on slices 0..r
        self.checkpoints: list[list[dict]] = []
        self.location: dict[int, tuple[int, int]] = {}

    # ------------------------------------------------------------------ setup

    def _seed(self, *keys: int) -> int:
        return int(np.random.SeedSequence([self.config.seed, *keys]).generate_state(1)[0])

    def _partition(self, X: np.ndarray, y: np.ndarray, ids: np.ndarray) -> None:
        cfg = self.config
        order = np.random.default_rng(self._seed(0xDA7A)).permutation(len(X))
        # array_split keeps every row (the original notebook silently dropped the
        # remainder of len(X) // num_shards).
        self.parts = []
        for s, shard_rows in enumerate(np.array_split(order, cfg.num_shards)):
            shard_parts = []
            for r, rows in enumerate(np.array_split(shard_rows, cfg.num_slices)):
                shard_parts.append(
                    SlicePart(torch.as_tensor(X[rows]), torch.as_tensor(y[rows]), ids[rows].copy())
                )
                for rid in ids[rows]:
                    self.location[int(rid)] = (s, r)
            self.parts.append(shard_parts)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        ids: np.ndarray,
        exclude: Iterable[int] = (),
        verbose: bool = False,
    ) -> "SISAEnsemble":
        """Partition and train. ``exclude`` drops records *after* partitioning,
        i.e. trains exactly what fitting on all of ``ids`` and then unlearning
        ``exclude`` should produce — the reference for verifying unlearning.
        """
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        ids = np.asarray(ids, dtype=np.int64)
        if len(np.unique(ids)) != len(ids):
            raise ValueError("ids must be unique")
        if X.shape[1] != self.in_features:
            raise ValueError(f"expected {self.in_features} features, got {X.shape[1]}")

        self.location = {}
        self._partition(X, y, ids)
        self._remove(exclude)
        self.models = [None] * self.config.num_shards  # type: ignore[list-item]
        self.checkpoints = [[] for _ in range(self.config.num_shards)]
        for s in range(self.config.num_shards):
            t0 = time.perf_counter()
            self._train_shard(s, start_slice=0)
            if verbose:
                print(f"shard {s + 1}/{self.config.num_shards} trained in {time.perf_counter() - t0:.1f}s")
        return self

    # --------------------------------------------------------------- training

    def _new_model(self, shard: int) -> tuple[MLPRegressor, torch.optim.Optimizer]:
        cfg = self.config
        torch.manual_seed(self._seed(shard, 0x1A17))
        model = MLPRegressor(self.in_features, cfg.hidden, cfg.dropout).to(cfg.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
        return model, optimizer

    @staticmethod
    def _snapshot(model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> dict:
        # Deep copies on the CPU. Storing model.state_dict() directly (as the
        # original notebook did) keeps live references to the parameters, so
        # every "checkpoint" silently becomes the final model.
        return {
            "model": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            "optimizer": _to_cpu(copy.deepcopy(optimizer.state_dict())),
        }

    def _restore(self, shard: int, slice_idx: int) -> tuple[MLPRegressor, torch.optim.Optimizer]:
        model, optimizer = self._new_model(shard)
        ckpt = self.checkpoints[shard][slice_idx]
        model.load_state_dict(ckpt["model"])
        # deepcopy: load_state_dict may alias the stored tensors, and Adam
        # updates its moments in place.
        optimizer.load_state_dict(copy.deepcopy(ckpt["optimizer"]))
        return model, optimizer

    def _train_shard(self, shard: int, start_slice: int) -> int:
        """(Re)train ``shard`` from ``start_slice`` onward; returns slice steps run."""
        cfg = self.config
        if start_slice == 0:
            model, optimizer = self._new_model(shard)
        else:
            model, optimizer = self._restore(shard, start_slice - 1)
        del self.checkpoints[shard][start_slice:]

        for r in range(start_slice, cfg.num_slices):
            seen = self.parts[shard][: r + 1]
            X = torch.cat([p.X for p in seen])
            y = torch.cat([p.y for p in seen])
            if len(X):
                train_epochs(model, optimizer, X, y, cfg.epochs_per_slice, cfg.batch_size, self._seed(shard, r + 1))
            self.checkpoints[shard].append(self._snapshot(model, optimizer))
        self.models[shard] = model
        return cfg.num_slices - start_slice

    # -------------------------------------------------------------- unlearning

    def _remove(self, ids: Iterable[int]) -> dict[tuple[int, int], set[int]]:
        """Physically delete rows; returns the removed ids grouped by (shard, slice)."""
        removed: dict[tuple[int, int], set[int]] = {}
        for rid in ids:
            loc = self.location.pop(int(rid), None)
            if loc is not None:
                removed.setdefault(loc, set()).add(int(rid))
        for (s, r), rids in removed.items():
            part = self.parts[s][r]
            keep = ~np.isin(part.ids, list(rids))
            self.parts[s][r] = SlicePart(part.X[torch.as_tensor(keep)], part.y[torch.as_tensor(keep)], part.ids[keep])
        return removed

    def unlearn(self, ids: Iterable[int]) -> UnlearnReport:
        """Exactly remove the given record ids from the ensemble.

        Requests are batched: each affected shard is retrained once, starting
        from the earliest slice that contained one of its deleted records.
        """
        t0 = time.perf_counter()
        ids = [int(i) for i in ids]
        report = UnlearnReport(requested=len(ids), removed=0)

        removed = self._remove(ids)
        report.removed = sum(len(v) for v in removed.values())
        report.not_found = [i for i in dict.fromkeys(ids) if not any(i in v for v in removed.values())]
        for s, r in removed:
            report.retrained_from[s] = min(r, report.retrained_from.get(s, r))

        for s, r in sorted(report.retrained_from.items()):
            report.slices_retrained += self._train_shard(s, r)

        report.seconds = time.perf_counter() - t0
        return report

    # --------------------------------------------------------------- inference

    def predict_shards(self, X: np.ndarray | torch.Tensor) -> torch.Tensor:
        X = torch.as_tensor(np.asarray(X, dtype=np.float32))
        return torch.stack([predict(m, X) for m in self.models])

    def predict(self, X: np.ndarray | torch.Tensor) -> np.ndarray:
        per_shard = self.predict_shards(X)
        if self.config.aggregation == "median":
            return per_shard.median(dim=0).values.numpy()
        return per_shard.mean(dim=0).numpy()

    # ------------------------------------------------------------- inspection

    def __contains__(self, record_id: int) -> bool:
        return int(record_id) in self.location

    def __len__(self) -> int:
        return len(self.location)

    def shard_sizes(self) -> list[int]:
        return [sum(len(p.ids) for p in shard) for shard in self.parts]

    # ------------------------------------------------------------ persistence

    def save(self, path: str) -> None:
        """Persist everything needed to keep serving and unlearning.

        Training rows are stored too: retraining a shard needs them. Deleted
        rows are gone from ``parts`` and from every surviving checkpoint.
        """
        torch.save(
            {
                "config": asdict(self.config),
                "in_features": self.in_features,
                "parts": [
                    [{"X": p.X, "y": p.y, "ids": torch.as_tensor(p.ids)} for p in shard] for shard in self.parts
                ],
                "checkpoints": self.checkpoints,
            },
            path,
        )

    @classmethod
    def load(cls, path: str, device: str | None = None) -> "SISAEnsemble":
        state = torch.load(path, map_location="cpu", weights_only=True)
        config = SISAConfig(**state["config"])
        if device is not None:
            config.device = device
        ens = cls(config, state["in_features"])
        ens.parts = [
            [SlicePart(p["X"], p["y"], p["ids"].numpy()) for p in shard] for shard in state["parts"]
        ]
        ens.location = {
            int(rid): (s, r) for s, shard in enumerate(ens.parts) for r, p in enumerate(shard) for rid in p.ids
        }
        ens.checkpoints = state["checkpoints"]
        ens.models = []
        for s in range(config.num_shards):
            model, _ = ens._new_model(s)
            model.load_state_dict(ens.checkpoints[s][-1]["model"])
            ens.models.append(model)
        return ens
