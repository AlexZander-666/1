from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from .contrastive import InfoNCELoss, SupervisedContrastiveLoss
from .center_loss import CenterLoss
from models.modules.tfcl import TimeFreqContrastiveLoss, HierarchicalTFContrastiveLoss

__all__ = [
    "InfoNCELoss",
    "SupervisedContrastiveLoss",
    "CenterLoss",
    "Phycl_NetLoss",
]


class Phycl_NetLoss(nn.Module):
    """
    Combined loss for Phycl_Net.
    L = L_ce + alpha * L_tfcl + beta * L_center
    """

    def __init__(
        self,
        num_classes: int = 2,
        feat_dim: int = 128,
        alpha: float = 0.1,
        beta: float = 0.01,
        use_tfcl: bool = True,
        hierarchical_tfcl: bool = True,
        tf_cross_weight: float = 0.5,
        temperature: float = 0.1,
        supervised_weight: float = 0.0,
        label_smoothing: float = 0.0,
        class_weights: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.ce = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
        self.hierarchical_tfcl = hierarchical_tfcl
        if use_tfcl:
            if hierarchical_tfcl:
                self.tfcl = HierarchicalTFContrastiveLoss(
                    temperature=temperature,
                    cross_layer_weight=tf_cross_weight,
                    supervised_weight=supervised_weight,
                )
            else:
                self.tfcl = TimeFreqContrastiveLoss(temperature=temperature, supervised_weight=supervised_weight)
        else:
            self.tfcl = None
        self.center = CenterLoss(num_classes, feat_dim) if beta > 0 else None

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        z_time: Optional[torch.Tensor] = None,
        z_freq: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        loss_ce = self.ce(logits, labels)
        loss_tfcl = torch.tensor(0.0, device=logits.device, dtype=logits.dtype)
        loss_center = torch.tensor(0.0, device=logits.device, dtype=logits.dtype)
        stats: Dict[str, float] = {"ce": float(loss_ce.detach().cpu())}

        if self.tfcl is not None and z_time is not None and z_freq is not None:
            if self.hierarchical_tfcl and isinstance(z_time, (list, tuple)) and isinstance(z_freq, (list, tuple)):
                tfcl_val, tfcl_stats = self.tfcl(list(z_time), list(z_freq), labels)
            else:
                tfcl_val, tfcl_stats = self.tfcl(z_time, z_freq, labels)
            loss_tfcl = tfcl_val
            stats.update(tfcl_stats)

        def _select_last(feat):
            if feat is None:
                return None
            if isinstance(feat, (list, tuple)):
                return feat[-1]
            return feat

        center_time = _select_last(z_time)
        center_freq = _select_last(z_freq)
        if self.center is not None and center_time is not None:
            center_feats = center_time if center_freq is None else 0.5 * (center_time + center_freq)
            loss_center = self.center(center_feats, labels)
            stats["center"] = float(loss_center.detach().cpu())

        total = loss_ce + self.alpha * loss_tfcl + self.beta * loss_center
        stats["total"] = float(total.detach().cpu())
        return total, stats
