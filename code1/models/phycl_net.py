"""
Paper-facing model name wrappers.

`paper/jec/last2.tex` refers to the main model as PhyCL-Net. Internally the
implementation is shared with `AMSNetV2` (with `mspa=False` enforced via CLI
when using `--model phycl_net`).
"""

from .ams_net_v2 import AMSNetV2

# Alias for clarity in paper-facing code.
PhyCLNet = AMSNetV2

__all__ = ["PhyCLNet", "AMSNetV2"]
