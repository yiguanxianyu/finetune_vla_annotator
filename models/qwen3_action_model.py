# -*- coding: utf-8 -*-
"""
模型与损失模块
==============
- 将原有的 KHead 与 BoundaryHead 作为子模块集成到 ActionSegmentationModel。
- 在 forward 内部计算文本/边界/段数等多任务损失，并返回总损失与各项子损失。
"""

from typing import Any, Dict, Optional

import torch
from peft import LoraConfig, get_peft_model

from transformers import PretrainedConfig, PreTrainedModel, Qwen3VLForConditionalGeneration

from .head import BoundaryHead, KHead


class ActionSegmentationConfig(PretrainedConfig):
    model_type = "action-segmentation"

    def __init__(self, vlm="Qwen/Qwen3-VL-2B-Instruct", k_max=10, embed_dim=None, **kwargs):
        super().__init__(**kwargs)
        self.k_max = k_max
        self.train_lora = True
        self.vlm = vlm
        if embed_dim is None:
            if "2B" in vlm:
                self.embed_dim = 2048
            elif "4B" in vlm:
                self.embed_dim = 2560
            else:
                raise ValueError(f"Unknown vlm model for embed_dim inference: {vlm}")


class ActionSegmentationModel(PreTrainedModel):
    """
    PreTrainedModel-compatible version of your ActionSegmentationModel.
    """

    config_class = ActionSegmentationConfig
    supports_gradient_checkpointing = True

    def __init__(self, config: ActionSegmentationConfig, **kwargs):
        super().__init__(config, **kwargs)
        self.qwen3vlmodel = Qwen3VLForConditionalGeneration.from_pretrained(
            config.vlm,
            attn_implementation="flash_attention_2",
            dtype=torch.bfloat16,
            device_map="auto",
        )
        self.qwen3vlmodel.config.use_cache = False

        if config.train_lora:
            self.qwen3vlmodel = load_lora_model(self.qwen3vlmodel, config)

        self.khead = KHead(embed_dim=config.embed_dim, k_max=config.k_max)
        self.bdhead = BoundaryHead(embed_dim=config.embed_dim)

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.qwen3vlmodel.gradient_checkpointing_enable(gradient_checkpointing_kwargs)

    def forward(
        self,
        inputs_lm: Dict[str, torch.Tensor],
        label_assist: Dict[str, torch.Tensor],
        **kwargs,
    ):
        """
        Trainer 将自动调用 forward。

        如果提供 labels（segments_label / actions_count_label），会返回 loss。
        否则返回 logits。
        """
        text_label = label_assist["text_label"]
        segments_label = label_assist["segments_label"]
        actions_count_label = label_assist["actions_count_label"]
        video_mask = label_assist["video_mask"]
        num_frames = label_assist["num_frames"]

        base_out = self.qwen3vlmodel(
            **inputs_lm,
            labels=text_label,
            output_hidden_states=True,
        )

        hidden_states = base_out.hidden_states[-1]
        video_hidden_states = [hs[vm] for hs, vm in zip(hidden_states, video_mask)]

        k_logits = self.khead(video_hidden_states)  # list (B, k_max)
        seg_logits = self.bdhead(video_hidden_states, num_frames)  # list (B, T)

        # 如果没有标签 → 推理模式
        if segments_label is None or actions_count_label is None:
            k_preds = torch.argmax(torch.stack(k_logits), dim=-1)
            seg_preds = []
            for k, seg in zip(k_preds, seg_logits):
                temp = seg.topk(k.item()).indices.sort().values
                seg_preds.append(temp)

            return {
                "loss_text": base_out.loss,
                "logits": {
                    "k_logits": k_logits,
                    "seg_logits": seg_logits,
                    "text_logits": base_out.logits,
                },
                "predictions": {
                    "k_preds": k_preds,
                    "seg_preds": seg_preds,
                },
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

    def generate(self, *args, **kwargs):
        # First run get K and segments prediction heads if needed

        return self.qwen3vlmodel.generate(*args, **kwargs)


def load_lora_model(base_model: str, args):
    lora_config = LoraConfig(
        r=16,  # 秩
        lora_alpha=16,  # alpha值
        lora_dropout=0.1,
        inference_mode=False,
        target_modules=["q_proj", "v_proj"],
        task_type="CAUSAL_LM",
    )

    peft_model = get_peft_model(base_model, lora_config)
    peft_model.print_trainable_parameters()

    return peft_model


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
