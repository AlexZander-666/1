"""
Paper-facing model name wrapper.

The paper refers to the main model as PhyCL-Net and the CLI uses `--model phycl_net`.
The implementation lives in `code/models/PhyCL_Net.py`.
"""

from __future__ import annotations

from .PhyCL_Net import PhyCL_Net

# Alias kept for readability in paper-facing code and configs.
PhyCLNet = PhyCL_Net

__all__ = ["PhyCL_Net", "PhyCLNet"]

