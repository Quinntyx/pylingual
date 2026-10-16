"""Small, unmerged LoRA adapters for PyLingual's shared model families.

Uses only PyTorch; the frozen backbone is held once, not copied per version.
Adapter checkpoints are bound to an immutable backbone ID and explicit targets.
"""
from __future__ import annotations

import json
import math
from contextlib import contextmanager
from pathlib import Path
from threading import RLock

import torch
from torch import nn
from torch.nn import functional as F


class LowRank(nn.Module):
    def __init__(self, linear: nn.Linear, rank: int, alpha: float, dropout: float):
        super().__init__()
        if rank < 1 or alpha <= 0 or not 0 <= dropout < 1:
            raise ValueError("Invalid LoRA rank, alpha or dropout")
        self.a = nn.Parameter(linear.weight.new_empty(rank, linear.in_features))
        self.b = nn.Parameter(linear.weight.new_zeros(linear.out_features, rank))
        nn.init.kaiming_uniform_(self.a, a=math.sqrt(5))
        self.scale = alpha / rank
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return F.linear(F.linear(self.dropout(x), self.a), self.b) * self.scale


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear):
        super().__init__()
        self.base = base
        self.adapters = nn.ModuleDict()
        self.active = None

    @property
    def weight(self):
        return self.base.weight

    @property
    def bias(self):
        return self.base.bias

    def forward(self, x):
        y = self.base(x)
        if self.active is not None:
            y = y + self.adapters[self.active](x).to(y.dtype)
        return y


def adapter_key(version: str) -> str:
    return "py" + version.replace(".", "_")


class AdapterBank:
    """Owns a shared backbone and serializes adapter selection plus inference."""
    def __init__(self, model: nn.Module, backbone_id: str, targets: tuple[str, ...]):
        if not backbone_id or not targets:
            raise ValueError("A backbone ID and LoRA targets are required")
        self.model = model
        self.backbone_id = backbone_id
        self.targets = tuple(targets)
        self.lock = RLock()
        self.layers = {}
        self.specs = {}
        for name, module in list(model.named_modules()):
            if isinstance(module, nn.Linear) and name.rsplit(".", 1)[-1] in targets:
                parent_name, _, attr = name.rpartition(".")
                parent = model.get_submodule(parent_name) if parent_name else model
                wrapped = LoRALinear(module)
                setattr(parent, attr, wrapped)
                self.layers[name] = wrapped
        if not self.layers:
            raise ValueError(f"No linear layers matched LoRA targets {targets!r}")
        model.requires_grad_(False)

    def add(self, version: str, rank=8, alpha=16.0, dropout=0.0):
        with self.lock:
            if version in self.specs:
                raise ValueError(f"Adapter {version} already exists")
            key = adapter_key(version)
            for layer in self.layers.values():
                adapter = LowRank(layer.base, rank, alpha, dropout)
                adapter.train(self.model.training)
                layer.adapters[key] = adapter
            self.specs[version] = {"rank": rank, "alpha": alpha, "dropout": dropout}

    def select(self, version: str | None):
        if version is not None and version not in self.specs:
            raise ValueError(f"No adapter registered for Python {version}")
        key = adapter_key(version) if version is not None else None
        for layer in self.layers.values():
            layer.active = key

    @contextmanager
    def use(self, version: str | None):
        with self.lock:
            previous = next(iter(self.layers.values())).active
            self.select(version)
            try:
                yield self.model
            finally:
                for layer in self.layers.values():
                    layer.active = previous

    def trainable(self, version: str):
        self.model.requires_grad_(False)
        key = adapter_key(version)
        self.select(version)
        parameters = []
        for layer in self.layers.values():
            adapter = layer.adapters[key]
            adapter.requires_grad_(True)
            parameters.extend(adapter.parameters())
        return parameters

    def save(self, version: str, directory: Path):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        key = adapter_key(version)
        state = {f"{name}.{part}": tensor.detach().cpu() for name, layer in self.layers.items() for part, tensor in layer.adapters[key].state_dict().items()}
        torch.save(state, directory / "adapter.pt")
        metadata = {"format": 1, "backbone_id": self.backbone_id, "version": version, "targets": list(self.targets), "layers": list(self.layers), **self.specs[version]}
        (directory / "adapter.json").write_text(json.dumps(metadata, indent=2) + "\n")

    def load(self, version: str, directory: Path):
        with self.lock:
            directory = Path(directory)
            meta = json.loads((directory / "adapter.json").read_text())
            if meta.get("format") != 1 or meta.get("backbone_id") != self.backbone_id or meta.get("version") != version:
                raise ValueError("Adapter format, version or backbone ID mismatch")
            if tuple(meta["targets"]) != self.targets or meta["layers"] != list(self.layers):
                raise ValueError("Adapter target layers do not match the backbone")
            state = torch.load(directory / "adapter.pt", map_location="cpu", weights_only=True)
            expected = {f"{name}.{part}" for name in self.layers for part in ("a", "b")}
            if set(state) != expected:
                raise ValueError("Adapter checkpoint contains missing or unexpected tensors")
            # Validate all shapes before mutating the shared model.
            for name, layer in self.layers.items():
                if tuple(state[f"{name}.a"].shape) != (meta["rank"], layer.base.in_features) or tuple(state[f"{name}.b"].shape) != (layer.base.out_features, meta["rank"]):
                    raise ValueError(f"Adapter shape mismatch at {name}")
            self.add(version, meta["rank"], meta["alpha"], meta["dropout"])
            key = adapter_key(version)
            for name, layer in self.layers.items():
                layer.adapters[key].load_state_dict({part: state[f"{name}.{part}"] for part in ("a", "b")})
            self.model.eval()


class VersionedPipeline:
    def __init__(self, pipeline, bank: AdapterBank, version: str):
        self.pipeline = pipeline
        self.bank = bank
        self.version = version

    def __getattr__(self, name):
        return getattr(self.pipeline, name)

    def __call__(self, *args, **kwargs):
        with self.bank.use(self.version):
            # Materialize lazy pipeline iterators while the adapter lock is held.
            result = self.pipeline(*args, **kwargs)
            if not isinstance(result, (list, dict, str)) and hasattr(result, "__next__"):
                result = list(result)
            return result
