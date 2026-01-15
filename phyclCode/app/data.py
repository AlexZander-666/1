"""Data loaders for SisFall txt and generic CSV inputs."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


def _select_sisfall_channels(raw_data: np.ndarray, mode: str = "accel3") -> np.ndarray:
    """
    Slice raw SisFall arrays to requested channels.
    Follows the training loader convention in PhyCL_Net_experiments.py.
    """
    mode_l = (mode or "accel3").lower()
    c = raw_data.shape[0]
    if mode_l == "accel3" or c < 4:
        return raw_data[: min(3, c)]
    if mode_l == "accel6":
        if c >= 6:
            return raw_data[:6]
    if mode_l in ("accel6+gyro", "accel9", "full"):
        if c >= 9:
            return raw_data[:9]
    return raw_data[: min(c, 9)]


def _window_signal(x: np.ndarray, window_size: int, stride: int) -> List[np.ndarray]:
    """Create sliding windows (C, Lw) from (C, L)."""
    c, l = x.shape
    if l < window_size:
        pad = np.zeros((c, window_size), dtype=np.float32)
        pad[:, :l] = x
        return [pad]
    out: List[np.ndarray] = []
    start = 0
    while start + window_size <= l:
        out.append(x[:, start : start + window_size].astype(np.float32))
        start += stride
    return out


def load_sisfall_windows(
    root: Path,
    *,
    window_size: int = 512,
    stride: int = 256,
    channels_used: str = "accel3",
) -> List[Tuple[np.ndarray, int, str]]:
    """
    Load SisFall ADL/FALL txt files and return standardized windows.

    Returns:
        List of (window, label, subject_id)
    """
    root = Path(root)
    candidates = [root, root / "SisFall"]
    sisfall_root = None
    for cand in candidates:
        if (cand / "ADL").is_dir() and (cand / "FALL").is_dir():
            sisfall_root = cand
            break
    if sisfall_root is None:
        raise FileNotFoundError(f"Cannot find SisFall ADL/FALL under {root}")

    items: List[Tuple[np.ndarray, int, str]] = []

    def add_file(path: Path, label: int):
        rows = []
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip().rstrip(";")
                if not line:
                    continue
                parts = [p.strip() for p in line.replace(";", ",").split(",") if p.strip()]
                if len(parts) < 3:
                    continue
                try:
                    nums = [float(v) for v in parts]
                except ValueError:
                    continue
                rows.append(nums)
        if not rows:
            return
        raw = np.array(rows, dtype=np.float32).T
        raw = _select_sisfall_channels(raw, channels_used)
        mean = raw.mean(axis=1, keepdims=True)
        std = raw.std(axis=1, keepdims=True) + 1e-6
        norm = (raw - mean) / std
        subject = _parse_subject(path.name)
        for w in _window_signal(norm, window_size, stride):
            items.append((w, label, subject))

    for folder, label in (("ADL", 0), ("FALL", 1)):
        for txt in (sisfall_root / folder).glob("*.txt"):
            add_file(txt, label)
    return items


def _parse_subject(fname: str) -> str:
    stem = Path(fname).stem
    for token in stem.split("_"):
        if token.upper().startswith("SA") and token[2:].isdigit():
            return token.upper()
    return stem.upper()


def load_csv_windows(
    path: Path,
    *,
    columns: Sequence[str] | None = None,
    sample_rate_hz: float = 50.0,
    target_rate_hz: float = 50.0,
    window_size: int = 512,
    stride: int = 256,
) -> List[np.ndarray]:
    """
    Load a CSV with at least 3 numeric columns and produce standardized windows.
    """
    path = Path(path)
    if columns is None:
        with path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader, None)
        if header and len(header) >= 3:
            columns = header[:3]
        else:
            columns = [0, 1, 2]

    df = pd.read_csv(path)
    if len(df) == 0:
        return []
    data = df.loc[:, columns].to_numpy(dtype=np.float32)
    data = data.T  # (C, Lsrc)
    data = _resample(data, sample_rate_hz, target_rate_hz)
    mean = data.mean(axis=1, keepdims=True)
    std = data.std(axis=1, keepdims=True) + 1e-6
    norm = (data - mean) / std
    return _window_signal(norm, window_size, stride)


def _resample(x: np.ndarray, src_rate: float, tgt_rate: float) -> np.ndarray:
    """Linear resample CxL to match target sampling rate."""
    if abs(src_rate - tgt_rate) < 1e-6 or x.shape[1] < 2:
        return x
    c, l = x.shape
    duration = (l - 1) / src_rate
    tgt_len = int(round(duration * tgt_rate)) + 1
    src_t = np.linspace(0, duration, num=l, dtype=np.float32)
    tgt_t = np.linspace(0, duration, num=tgt_len, dtype=np.float32)
    out = np.empty((c, tgt_len), dtype=np.float32)
    for i in range(c):
        out[i] = np.interp(tgt_t, src_t, x[i])
    return out
