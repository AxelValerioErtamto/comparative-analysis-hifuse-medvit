"""
Loads HiFuse-Small and MedViT-Small straight from the official repositories,
without copying or modifying their code.

Expected layout (clone with the commands in README.md):
    third_party/HiFuse/main_model.py   -> HiFuse_Small(num_classes)
    third_party/MedViT/MedViT.py       -> MedViT_small(num_classes=...)

Both repos ship a file called `utils.py`. MedViT.py does `from utils import ...`,
so a naive sys.path import can pick up the wrong one (or your own utils.py).
`_load_file` imports each repo file under a unique module name and swaps in
that repo's own `utils` only for the duration of the import.
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import torch.nn as nn

THIRD_PARTY = Path(os.environ.get("THIRD_PARTY_DIR", Path(__file__).resolve().parent / "third_party"))
_CACHE: dict[str, object] = {}

MODELS = {
    # name: (repo folder, file, module alias, sibling modules the file imports, factory)
    "hifuse_small": ("HiFuse", "main_model.py", "official_hifuse_main_model", (), "HiFuse_Small"),
    "medvit_small": ("MedViT", "MedViT.py", "official_medvit", ("utils",), "MedViT_small"),
}


def _exec_module(alias: str, file: Path):
    spec = importlib.util.spec_from_file_location(alias, file)
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    spec.loader.exec_module(module)
    return module


def _load_file(alias: str, file: Path, siblings: tuple[str, ...]):
    if alias in _CACHE:
        return _CACHE[alias]
    if not file.exists():
        raise FileNotFoundError(
            f"{file} not found. Clone the official repo into {file.parent} "
            f"(see README.md) or set THIRD_PARTY_DIR.")
    saved = {name: sys.modules.get(name) for name in siblings}
    try:
        for name in siblings:
            _exec_module(name, file.parent / f"{name}.py")
        module = _exec_module(alias, file)
    finally:
        for name, old in saved.items():
            if old is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old
    _CACHE[alias] = module
    return module


def build_model(name: str, num_classes: int = 2, drop_path: float | None = None,
                dropout: float | None = None) -> nn.Module:
    """Official architecture, official default init, only the class count changed.

    drop_path / dropout = None keeps each repo's own defaults (exactly the official factory).
    Setting them only changes the regularisation rates the official constructors already
    expose; layers, widths and depths are identical to the official -Small variants:
      HiFuse-Small: main_model(depths=(2,2,6,2), conv_depths=(2,2,6,2)); stochastic depth in the
                    global, local and fusion paths; dropout in the Swin MLP/projection layers.
                    Official default: all 0.
      MedViT-Small: MedViT(stem_chs=[64,32,64], depths=[3,4,10,3]); stochastic depth path_dropout,
                    dropout in attention projection / MLP. Official default: path_dropout 0.1, drop 0.
    """
    if name not in MODELS:
        raise ValueError(f"unknown model {name!r}; choose from {list(MODELS)}")
    repo, fname, alias, siblings, factory = MODELS[name]
    module = _load_file(alias, THIRD_PARTY / repo / fname, siblings)
    if drop_path is None and dropout is None:
        return getattr(module, factory)(num_classes=num_classes)
    if name == "hifuse_small":
        dp = 0.0 if drop_path is None else drop_path
        do = 0.0 if dropout is None else dropout
        return module.main_model(num_classes=num_classes, depths=(2, 2, 6, 2), conv_depths=(2, 2, 6, 2),
                                 drop_rate=do, drop_path_rate=dp, conv_drop_path_rate=dp, HFF_dp=dp)
    if name == "medvit_small":
        dp = 0.1 if drop_path is None else drop_path
        do = 0.0 if dropout is None else dropout
        return module.MedViT(stem_chs=[64, 32, 64], depths=[3, 4, 10, 3], path_dropout=dp, drop=do,
                             num_classes=num_classes)
    raise ValueError(name)


def count_parameters(model: nn.Module) -> float:
    return sum(p.numel() for p in model.parameters()) / 1e6
