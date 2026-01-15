"""
Thin wrapper entrypoint for PhyCL-Net experiments.

Canonical CLI name per `paper/jec/last2.tex`: `phycl_net`.
"""

from DMC_Net_experiments import (  # noqa: F401
    LiteAMSNet,
    SisFallDataset,
    _resolve_sisfall_root,
    main,
    parse_ablation_config,
)

__all__ = [
    "LiteAMSNet",
    "SisFallDataset",
    "_resolve_sisfall_root",
    "parse_ablation_config",
    "main",
]


if __name__ == "__main__":
    main()
