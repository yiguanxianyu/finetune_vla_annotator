# -*- coding: utf-8 -*-
"""
模型与损失模块
==============
- 将原有的 KHead 与 BoundaryHead 作为子模块集成到 ActionSegmentationModel。
- 在 forward 内部计算文本/边界/段数等多任务损失，并返回总损失与各项子损失。
"""

from typing import Any, Dict, Optional

import torch

from transformers import PretrainedConfig, PreTrainedModel, Qwen3VLForConditionalGeneration

from .head import BoundaryHead, KHead


class ActionSegmentationConfig(PretrainedConfig):
    model_type = "action-segmentation"

    def __init__(self, embed_dim=2048, k_max=10, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.k_max = k_max


class ActionSegmentationModel(PreTrainedModel):
    """
    PreTrainedModel-compatible version of your ActionSegmentationModel.
    """

    config_class = ActionSegmentationConfig
    supports_gradient_checkpointing = True

    def __init__(self, config: ActionSegmentationConfig, base_model: Qwen3VLForConditionalGeneration, **kwargs):
        super().__init__(config, **kwargs)
        self.qwen3vlmodel = base_model
        self.khead = KHead(embed_dim=config.embed_dim, k_max=config.k_max)
        self.bdhead = BoundaryHead(embed_dim=config.embed_dim)

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.qwen3vlmodel.gradient_checkpointing_enable(gradient_checkpointing_kwargs)

    def forward(
        self,
        inputs_lm: Dict[str, torch.Tensor],
        text_label: Optional[torch.Tensor] = None,
        segments_label: torch.Tensor = None,
        actions_count_label: torch.Tensor = None,
        video_mask: torch.Tensor = None,
        num_frames: torch.Tensor = None,
        **kwargs,
    ):
        """
        Trainer 将自动调用 forward。

        如果提供 labels（segments_label / actions_count_label），会返回 loss。
        否则返回 logits。
        """
        base_out = self.qwen3vlmodel(
            **inputs_lm,
            labels=text_label,
            output_hidden_states=True,
        )

        hidden_states = base_out.hidden_states[-1]
        video_hidden_states = [hs[vm] for hs, vm in zip(hidden_states, video_mask)]

        k_logits = self.khead(video_hidden_states)
        seg_logits = self.bdhead(video_hidden_states, num_frames)

        # 如果没有标签 → 推理模式
        if segments_label is None or actions_count_label is None:
            return {
                "loss": None,
                "text_loss": base_out.loss,
            }

        # 有监督 → 计算 loss
        loss_bound = self.bdhead.compute_loss(seg_logits, segments_label)
        loss_k = self.khead.compute_loss(k_logits, actions_count_label)
        loss_text = base_out.loss

        total_loss = loss_text + loss_bound + loss_k

        return {
            "loss": total_loss,
            "loss_text": loss_text.detach(),
            "loss_bound": loss_bound.detach(),
            "loss_K": loss_k.detach(),
        }


# class ActionSegmentationModel(nn.Module):
#     """Compose the multimodal base model with action-specific heads."""

#     def __init__(self, base_model: Qwen3VLForConditionalGeneration, embed_dim: int = 2048, k_max: int = 10) -> None:
#         super().__init__()
#         self.base_model = base_model
#         self.khead = KHead(embed_dim=embed_dim, k_max=k_max)
#         self.bdhead = BoundaryHead(embed_dim=embed_dim)
#         # self.alpha_text = float(alpha_text)
#         # self.loss_weights = nn.Parameter(torch.ones(3))

#     def forward(
#         self,
#         inputs_lm: Dict[str, torch.Tensor],
#         labels: Optional[torch.Tensor],
#         num_frames: torch.Tensor,
#         video_mask: torch.Tensor,
#     ) -> Dict[str, Any]:
#         """Run the base model and action heads; no loss aggregation happens here."""
#         base_out = self.base_model(**inputs_lm, labels=labels, output_hidden_states=True)
#         hidden_states = base_out.hidden_states[-1]

#         video_hidden_states = []
#         for hs, vm in zip(hidden_states, video_mask):
#             video_hidden_states.append(hs[vm])

#         k_logits = self.khead(video_hidden_states)
#         seg_logits = self.bdhead(video_hidden_states, num_frames)
#         return {
#             "loss_text": base_out["loss"],
#             "k_logits": k_logits,
#             "seg_logits": seg_logits,
#         }

#     def compute_loss(
#         self,
#         forward_outputs: Dict[str, Any],
#         segments_label: torch.Tensor,
#         actions_count_label: torch.Tensor,
#     ) -> Dict[str, torch.Tensor]:
#         """Aggregate training losses based on cached forward outputs and supervision."""
#         k_logits = forward_outputs["k_logits"]
#         seg_logits = forward_outputs["seg_logits"]

#         loss_bound = self.bdhead.compute_loss(seg_logits, segments_label)
#         loss_k = self.khead.compute_loss(k_logits, actions_count_label)
#         loss_text = forward_outputs["loss_text"]

#         total_loss = loss_bound + loss_k + loss_text

#         return total_loss, {
#             "loss": total_loss.detach(),
#             "loss_text": loss_text.detach(),
#             "loss_bound": loss_bound.detach(),
#             "loss_K": loss_k.detach(),
#         }
