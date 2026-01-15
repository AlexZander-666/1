"""
Thin wrapper entrypoint for PhyCL-Net experiments.

Canonical CLI name per `docs/main.tex`: `phycl_net`.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_impl():
    impl_path = Path(__file__).with_name("PhyCL-Net_experiments.py")
    spec = importlib.util.spec_from_file_location("phycl_net_experiments_impl", impl_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load experiments module from: {impl_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_impl = _load_impl()

LiteAMSNet = _impl.LiteAMSNet
SisFallDataset = _impl.SisFallDataset
_resolve_sisfall_root = _impl._resolve_sisfall_root
parse_ablation_config = _impl.parse_ablation_config
main = _impl.main

__all__ = ["LiteAMSNet", "SisFallDataset", "_resolve_sisfall_root", "parse_ablation_config", "main"]

if __name__ == "__main__":
    main()
