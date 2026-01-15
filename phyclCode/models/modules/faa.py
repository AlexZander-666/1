import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import CoordinateAttention1d


class FallAwareAttention(nn.Module):
    """
    Physics-inspired fall-aware attention emphasizing multiple physical cues:
    - SVM (magnitude) for overall intensity
    - Jerk (first derivative) for impacts
    - Jerk rate (second derivative) to separate transient vs sustained
    """

    def __init__(self, channels: int, kernel_size: int = 3, reduction: int = 4, use_axis_attention: bool = True):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.jerk_conv = nn.Conv1d(
            channels,
            channels,
            kernel_size,
            padding=padding,
            groups=channels,
            bias=False,
        )
        self.jerk_rate_conv = nn.Conv1d(
            channels,
            channels,
            kernel_size,
            padding=padding,
            groups=channels,
            bias=False,
        )
        hidden = max(1, channels // reduction)
        self.context = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(hidden, channels, kernel_size=1),
        )
        self.gate = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=1),
            nn.Sigmoid(),
        )
        self.axis_attn = CoordinateAttention1d(channels, reduction=reduction, kernel_size=kernel_size) if use_axis_attention else None
        self.norm = nn.GroupNorm(1, channels)
        self.res_scale = nn.Parameter(torch.tensor(0.0))
        self.last_attention = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError("FallAwareAttention expects (B, C, L).")

        x_axis = self.axis_attn(x) if self.axis_attn is not None else x

        svm = torch.sqrt((x ** 2).sum(dim=1, keepdim=True) + 1e-8)
        jerk = x[..., 1:] - x[..., :-1]
        jerk = F.pad(jerk, (1, 0))
        jerk_rate = jerk[..., 1:] - jerk[..., :-1]
        jerk_rate = F.pad(jerk_rate, (1, 0))

        local = self.jerk_conv(torch.abs(jerk)) + self.jerk_rate_conv(torch.abs(jerk_rate))
        context = self.context(x_axis) + self.context(svm.expand_as(x))
        attn = self.gate(local + context)
        self.last_attention = attn.detach()
        out = self.norm(x_axis * attn)
        return x + self.res_scale * out
