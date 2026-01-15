"""Shared configuration defaults for the PhyCL-Net demo."""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class DemoConfig:
    window_size: int = 512
    stride: int = 256
    sample_rate_hz: float = 50.0
    channels_used: str = "accel3"
    num_threads: int = 1
    batch_size: int = 32
    threshold: float = 0.5
    consecutive_required: int = 2
    cooldown_seconds: float = 5.0
    device: str = "cpu"
    # Paths are optional; the GUI will fill them in.
    mspa_off_ckpt: Path | None = None
    mspa_on_ckpt: Path | None = None


DEFAULT_CONFIG = DemoConfig()
