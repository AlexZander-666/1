#!/usr/bin/env python
"""
CPU efficiency benchmark for `docs/main.tex` (p50/p95 latency, params, FLOPs).

Notes
- Activate the required env before running (project rule): `conda activate SCI666`
- This script is intended to be run on the target CPU environment used for paper reporting.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import torch

# Ensure `code/` is on sys.path when running via `python code/scripts/...`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.PhyCL_Net import PhyCL_Net


@dataclass(frozen=True)
class BenchResult:
    model: str
    device: str
    threads: int
    batch_size: int
    input_shape: list[int]
    iters: int
    warmup: int
    params: int
    params_m: float
    flops_g: Optional[float]
    latency_ms_p50: float
    latency_ms_p95: float


def _set_single_thread(single_thread: bool) -> int:
    if not single_thread:
        return int(torch.get_num_threads())
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    return 1


def _build_model(model_name: str, proj_dim: int) -> torch.nn.Module:
    model_key = (model_name or "").lower()
    if model_key not in {"phycl_net", "mspa_faa_pdk"}:
        raise ValueError(f"Unsupported --model: {model_name}")

    ablation: Dict[str, bool] = {"mspa": True, "dks": True, "faa": True}
    if model_key == "phycl_net":
        ablation["mspa"] = False

    model = PhyCL_Net(in_channels=3, num_classes=2, proj_dim=proj_dim, ablation=ablation)
    if hasattr(model, "set_deploy"):
        model.set_deploy(True)
    model.eval()
    return model


def _count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def _try_flops(model: torch.nn.Module, x: torch.Tensor) -> Optional[float]:
    # Prefer fvcore if available; fall back to None.
    try:
        from fvcore.nn import FlopCountAnalysis  # type: ignore
    except Exception:
        return None

    try:
        flops = FlopCountAnalysis(model, x).total()
        return float(flops) / 1e9
    except Exception:
        return None


def _percentile(values: list[float], p: float) -> float:
    if not values:
        raise ValueError("empty values")
    values_sorted = sorted(values)
    k = int(round((len(values_sorted) - 1) * p))
    k = max(0, min(k, len(values_sorted) - 1))
    return float(values_sorted[k])


def benchmark(
    *,
    model_name: str,
    batch_size: int,
    length: int,
    iters: int,
    warmup: int,
    single_thread: bool,
    proj_dim: int,
    ckpt: Optional[Path],
) -> BenchResult:
    threads = _set_single_thread(single_thread)
    device = torch.device("cpu")

    model = _build_model(model_name, proj_dim=proj_dim).to(device)
    if ckpt is not None:
        state = torch.load(str(ckpt), map_location="cpu")
        if isinstance(state, dict) and "model_state" in state:
            state = state["model_state"]
        model.load_state_dict(state, strict=False)

    x = torch.randn(batch_size, 3, length, device=device)

    params = _count_params(model)
    flops_g = _try_flops(model, x[:1])

    # Warmup
    with torch.inference_mode():
        for _ in range(warmup):
            _ = model(x)

        times: list[float] = []
        for _ in range(iters):
            t0 = time.perf_counter()
            _ = model(x)
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000.0)

    return BenchResult(
        model=model_name,
        device=str(device),
        threads=threads,
        batch_size=batch_size,
        input_shape=[batch_size, 3, length],
        iters=iters,
        warmup=warmup,
        params=params,
        params_m=float(params) / 1e6,
        flops_g=flops_g,
        latency_ms_p50=_percentile(times, 0.50),
        latency_ms_p95=_percentile(times, 0.95),
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=["phycl_net", "mspa_faa_pdk"])
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--length", type=int, default=512)
    p.add_argument("--iters", type=int, default=300)
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--single-thread", action="store_true")
    p.add_argument("--proj-dim", type=int, default=128)
    p.add_argument("--ckpt", type=str, default=None, help="Optional checkpoint path (.pth).")
    p.add_argument("--out", type=str, default=None, help="Optional output JSON path.")
    args = p.parse_args()

    ckpt = Path(args.ckpt) if args.ckpt else None
    res = benchmark(
        model_name=args.model,
        batch_size=args.batch_size,
        length=args.length,
        iters=args.iters,
        warmup=args.warmup,
        single_thread=args.single_thread,
        proj_dim=args.proj_dim,
        ckpt=ckpt,
    )

    payload: Dict[str, Any] = asdict(res)
    print(json.dumps(payload, indent=2))

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
