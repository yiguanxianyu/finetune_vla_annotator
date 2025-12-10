# -*- coding: utf-8 -*-
"""
模型与损失模块
==============
- 将原有的 KHead 与 BoundaryHead 作为子模块集成到 ActionSegmentationModel。
- 在 forward 内部计算文本/边界/段数等多任务损失，并返回总损失与各项子损失。
"""

from typing import Dict, Any, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

# from transformers import Qwen2_5_VLForConditionalGeneration
from transformers import Qwen2_5_VLForConditionalGeneration

from .head import BoundaryHead, KHead


class ActionSegmentationModel(nn.Module):
    """Compose the multimodal base model with action-specific heads."""

    def __init__(
        self,
        base_model: Qwen2_5_VLForConditionalGeneration,
        embed_dim: int = 2048,
        k_max: int = 10,
        alpha_text: float = 1.0,
    ) -> None:
        super().__init__()
        self.base_model = base_model
        self.khead = KHead(embed_dim=embed_dim, k_max=k_max)
        self.bdhead = BoundaryHead(embed_dim=embed_dim)
        self.alpha_text = float(alpha_text)

    def forward(
        self,
        inputs_lm: Dict[str, torch.Tensor],
        labels: Optional[torch.Tensor],
        num_frames: torch.Tensor,
        video_mask: torch.Tensor,
    ) -> Dict[str, Any]:
        """Run the base model and action heads; no loss aggregation happens here."""
        base_out = self.base_model(**inputs_lm, labels=labels, output_hidden_states=True)
        hidden_states = base_out.hidden_states[-1]
        k_logits = self.khead(hidden_states, video_mask)
        start_logits, end_logits, valid_mask = self.bdhead(hidden_states, video_mask, num_frames)
        return {
            "model_outputs": base_out,
            "hidden_states": hidden_states,
            "k_logits": k_logits,
            "start_logits": start_logits,
            "end_logits": end_logits,
            "valid_mask": valid_mask,
            "loss_text": base_out.loss,
        }

    def compute_loss(
        self,
        forward_outputs: Dict[str, Any],
        probs_start: torch.Tensor,
        probs_end: torch.Tensor,
        K_label: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Aggregate training losses based on cached forward outputs and supervision."""
        k_logits = forward_outputs["k_logits"]
        start_logits = forward_outputs["start_logits"]
        end_logits = forward_outputs["end_logits"]
        valid_mask = forward_outputs["valid_mask"]

        target_start = self._broadcast_targets(probs_start, start_logits)
        target_end = self._broadcast_targets(probs_end, end_logits)

        loss_start = self._masked_bce_with_logits(start_logits, target_start, valid_mask)
        loss_end = self._masked_bce_with_logits(end_logits, target_end, valid_mask)
        loss_k = F.cross_entropy(k_logits, K_label)

        loss_text_val = forward_outputs.get("loss_text")

        total_loss = loss_start + loss_end + loss_k + self.alpha_text * loss_text_val

        return total_loss, {
            "loss": total_loss.detach(),
            "loss_text": loss_text_val.detach(),
            "loss_start": loss_start.detach(),
            "loss_end": loss_end.detach(),
            "loss_K": loss_k.detach(),
        }

    @staticmethod
    def _broadcast_targets(targets: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        if targets.dim() == 1:
            targets = targets.unsqueeze(0)
        if targets.size(0) == 1 and reference.size(0) > 1:
            targets = targets.expand(reference.size(0), -1)
        elif targets.size(0) != reference.size(0):
            expected = reference.size(0)
            raise ValueError(f"Target batch size mismatch: expected 1 or {expected}, but got {targets.size(0)}")
        if targets.size(-1) != reference.size(-1):
            if targets.size(-1) < reference.size(-1):
                pad = reference.size(-1) - targets.size(-1)
                targets = F.pad(targets, (0, pad))
            else:
                targets = targets[..., : reference.size(-1)]
        return targets.to(reference.dtype)

    @staticmethod
    def _masked_bce_with_logits(
        logits: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        if mask.shape != logits.shape:
            expected = logits.shape
            raise ValueError(f"Mask shape mismatch: expected {expected}, got {mask.shape}")
        mask_f = mask.to(dtype=logits.dtype)
        loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        loss = loss * mask_f
        denom = mask_f.sum().clamp_min(1.0)
        return loss.sum() / denom
