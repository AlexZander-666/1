"""Model loading and inference helpers for the PhyCL-Net demo."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torch.serialization as ts

from models import Phycl_Net


def load_phycl_net(
    ckpt_path: Path,
    *,
    mspa_enabled: bool,
    device: str = "cpu",
    sample_rate: float = 50.0,
) -> torch.nn.Module:
    """Load PhyCL-Net with or without MSPA branch."""
    ckpt_path = Path(ckpt_path)
    model = Phycl_Net(
        in_channels=3,
        num_classes=2,
        ablation={"mspa": mspa_enabled, "dks": True, "faa": True},
        sample_rate=sample_rate,
    )
    model.to(device)

    ckpt = _safe_load_checkpoint(ckpt_path, map_location=device)
    state_dict = _extract_state_dict(ckpt)
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    try:
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
    except RuntimeError as e:
        raise RuntimeError(
            "Checkpoint parameters do not match the model architecture. "
            "Check that `in_channels`, `num_classes`, and the MSPA ablation flag match the checkpoint."
        ) from e
    if unexpected:
        print(f"[warn] Unexpected keys in checkpoint: {unexpected}")
    if missing:
        print(f"[warn] Missing keys in checkpoint (likely OK for projection heads): {missing}")

    model.eval()
    return model


def _extract_state_dict(ckpt) -> dict:
    """
    Support multiple checkpoint formats used across scripts:
    - {'model_state_dict': ...} (common)
    - {'state_dict': ...} (common)
    - {'model_state': ...} (used by some training runs)
    - raw state_dict itself
    """
    if isinstance(ckpt, dict):
        for key in ("model_state_dict", "state_dict", "model_state", "model"):
            v = ckpt.get(key)
            if isinstance(v, dict):
                return v
        # If this dict looks like a raw state_dict already, accept it.
        if all(isinstance(k, str) for k in ckpt.keys()):
            if all(hasattr(v, "shape") or isinstance(v, (int, float, str, bytes, list, tuple, dict)) for v in ckpt.values()):
                # Heuristic fallback: only keep tensor-like entries
                tensor_items = {k: v for k, v in ckpt.items() if hasattr(v, "shape")}
                if tensor_items:
                    return tensor_items
    raise ValueError(
        "Unsupported checkpoint format: cannot find model weights. "
        "Expected keys like 'model_state_dict', 'state_dict', or 'model_state'."
    )


def _safe_load_checkpoint(path: Path, map_location: str):
    """
    Torch 2.6 defaults weights_only=True; older checkpoints may need full pickle.
    Also allow NumPy reconstruct used by many legacy checkpoints.
    """
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except Exception:
        # Allow numpy reconstruct globals if required
        try:
            import numpy as np
            ts.add_safe_globals([np.core.multiarray._reconstruct])
        except Exception:
            pass
        return torch.load(path, map_location=map_location, weights_only=False)


def predict_probabilities(
    model: torch.nn.Module,
    windows: Iterable[np.ndarray],
    *,
    batch_size: int = 32,
    device: str = "cpu",
) -> np.ndarray:
    """Run batched inference, returning fall probabilities (class-1 softmax)."""
    model.to(device)
    probs: List[np.ndarray] = []
    batch: List[np.ndarray] = []
    with torch.inference_mode():
        for w in windows:
            batch.append(w.astype(np.float32))
            if len(batch) >= batch_size:
                probs.append(_infer_batch(model, batch, device))
                batch = []
        if batch:
            probs.append(_infer_batch(model, batch, device))
    if not probs:
        return np.zeros((0,), dtype=np.float32)
    return np.concatenate(probs, axis=0)


def _infer_batch(model: torch.nn.Module, batch: List[np.ndarray], device: str) -> np.ndarray:
    x = torch.from_numpy(np.stack(batch)).to(device)
    logits, _, _ = model(x, return_embeddings=False)
    # Binary: class-1 = fall
    prob = F.softmax(logits, dim=1)[:, 1]
    return prob.detach().cpu().numpy()


def measure_latency_ms(
    model: torch.nn.Module,
    *,
    window_size: int = 512,
    runs: int = 200,
    warmup: int = 20,
    device: str = "cpu",
) -> Tuple[float, float]:
    """
    Measure p50/p95 latency on single-thread CPU by default.
    Returns (p50_ms, p95_ms).
    """
    torch.set_grad_enabled(False)
    try:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except RuntimeError:
        # PyTorch forbids changing interop threads after work has started.
        pass

    dummy = torch.zeros((1, 3, window_size), device=device)
    times: List[float] = []
    import time

    with torch.inference_mode():
        for i in range(warmup + runs):
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(dummy, return_embeddings=False)
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            dt_ms = (time.perf_counter() - t0) * 1000.0
            if i >= warmup:
                times.append(dt_ms)

    if not times:
        return 0.0, 0.0
    p50 = float(np.percentile(times, 50))
    p95 = float(np.percentile(times, 95))
    return p50, p95
