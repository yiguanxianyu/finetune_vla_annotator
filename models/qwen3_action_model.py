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

from transformers import Qwen3VLForConditionalGeneration

from .head import BoundaryHead, KHead


class ActionSegmentationModel(nn.Module):
    """Compose the multimodal base model with action-specific heads."""

    def __init__(self, base_model: Qwen3VLForConditionalGeneration, embed_dim: int = 2048, k_max: int = 10) -> None:
        super().__init__()
        self.base_model = base_model
        self.khead = KHead(embed_dim=embed_dim, k_max=k_max)
        self.bdhead = BoundaryHead(embed_dim=embed_dim)
        # self.alpha_text = float(alpha_text)
        # self.loss_weights = nn.Parameter(torch.ones(3))  

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
        
        video_hidden_states = []
        for hs, vm in zip(hidden_states, video_mask):
            video_hidden_states.append(hs[vm])

        k_logits = self.khead(video_hidden_states)
        seg_logits = self.bdhead(video_hidden_states, num_frames)
        return {
            "loss_text": base_out["loss"],
            "k_logits": k_logits,
            "seg_logits": seg_logits,
        }

    def compute_loss(
        self,
        forward_outputs: Dict[str, Any],
        segments_label: torch.Tensor,
        actions_count_label: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Aggregate training losses based on cached forward outputs and supervision."""
        k_logits = forward_outputs["k_logits"]
        seg_logits = forward_outputs["seg_logits"]

        loss_bound = self.bdhead.compute_loss(seg_logits, segments_label)
        loss_k = self.khead.compute_loss(k_logits, actions_count_label)
        loss_text = forward_outputs["loss_text"]

        total_loss = loss_bound + loss_k + loss_text

        return total_loss, {
            "loss": total_loss.detach(),
            "loss_text": loss_text.detach(),
            "loss_bound": loss_bound.detach(),
            "loss_K": loss_k.detach(),
        }
