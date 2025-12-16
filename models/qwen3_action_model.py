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
from dataclasses import dataclass
from transformers import PretrainedConfig, PreTrainedModel, Qwen3VLForConditionalGeneration
from transformers.modeling_outputs import ModelOutput
from transformers.cache_utils import Cache

from .head import BoundaryHead, KHead


@dataclass
class Qwen3VLActionCausalLMOutputWithPast(ModelOutput):
    r"""
    loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
        Language modeling loss (for next-token prediction).
    logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
        Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
    past_key_values (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
        It is a [`~cache_utils.Cache`] instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).

        Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
        `past_key_values` input) to speed up sequential decoding.
    rope_deltas (`torch.LongTensor` of shape `(batch_size, )`, *optional*):
        The rope index difference between sequence length and multimodal rope.
    """

    loss: Optional[torch.FloatTensor] = None
    loss_text: Optional[torch.FloatTensor] = None
    loss_bound: Optional[torch.FloatTensor] = None
    loss_K: Optional[torch.FloatTensor] = None
    k_preds: Optional[torch.LongTensor] = None
    seg_preds: Optional[torch.LongTensor] = None
    logits: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Cache] = None
    hidden_states: Optional[tuple[torch.FloatTensor]] = None
    attentions: Optional[tuple[torch.FloatTensor]] = None
    rope_deltas: Optional[torch.LongTensor] = None


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
        self.qwen3vlmodel.config.output_hidden_states = False
        self.qwen3vlmodel.config.output_attentions = False

        if config.train_lora:
            self.qwen3vlmodel = load_lora_model(self.qwen3vlmodel, config)

        self.khead = KHead(embed_dim=config.embed_dim, k_max=config.k_max)
        self.bdhead = BoundaryHead(embed_dim=config.embed_dim)

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.qwen3vlmodel.gradient_checkpointing_enable(gradient_checkpointing_kwargs)

    def forward(
        self,
        inputs_lm: Dict[str, torch.Tensor],
        video_mask: torch.Tensor,
        labels: Dict[str, torch.Tensor] = None,
        **kwargs,
    ):
        """
        Trainer 将自动调用 forward。

        如果提供 labels（segments_label / actions_count_label），会返回 loss。
        否则返回 logits。
        """
        if not labels:
            text_label = None
        else:
            text_label = labels["text_label"]
            segments_label = labels["segments_label"]
            actions_count_label = labels["actions_count_label"]
            num_frames = labels["num_frames"]

        base_out = self.qwen3vlmodel(
            **inputs_lm,
            labels=text_label,
            output_hidden_states=True,
        )

        hidden_states = base_out.hidden_states[-1]
        video_hidden_states = [hs[vm] for hs, vm in zip(hidden_states, video_mask)]

        k_logits = self.khead(video_hidden_states)  # list (B, k_max)
        k_preds = torch.argmax(torch.stack(k_logits), dim=-1)

        if not labels:
            num_frames = k_preds

        seg_logits = self.bdhead(video_hidden_states, num_frames)  # list (B, T)
        seg_preds = self.top_k_mask(seg_logits[0], k_preds[0] + 1)

        if not labels:
            # 无监督 → 只返回预测结果
            total_loss = loss_bound = loss_k = None
            loss_text = base_out.loss.detach() if base_out.loss else None
            return Qwen3VLActionCausalLMOutputWithPast(
                loss=total_loss,
                loss_text=loss_text.detach(),
                loss_bound=loss_bound.detach(),
                loss_K=loss_k.detach(),
                k_preds=k_preds,
                seg_preds=seg_preds,
            )
        else:
            # 有监督 → 计算 loss
            loss_bound = self.bdhead.compute_loss(seg_logits, segments_label)
            loss_k = self.khead.compute_loss(k_logits, actions_count_label)
            loss_text = base_out.loss
            total_loss = loss_text + loss_bound + loss_k
            return Qwen3VLActionCausalLMOutputWithPast(
                loss=total_loss,
                loss_text=loss_text.detach(),
                loss_bound=loss_bound.detach(),
                loss_K=loss_k.detach(),
                k_preds=k_preds,
                seg_preds=seg_preds,
                logits=base_out.logits,
                past_key_values=base_out.past_key_values,
                hidden_states=base_out.hidden_states,
                attentions=base_out.attentions,
                rope_deltas=base_out.rope_deltas,
            )

    def generate(self, *args, **kwargs):
        # First run get K and segments prediction heads if needed

        return self.qwen3vlmodel.generate(*args, **kwargs)

    def top_k_mask(self, s: torch.Tensor, k: int) -> torch.Tensor:
        """
        s: 输入张量 (例如 shape 为 [N] 或 [Batch, N])
        k: 要保留的前 k 大的数
        """
        # 1. 边界处理：如果 k 大于等于序列长度，直接返回全 1
        # s.shape[-1] 获取最后一个维度的大小
        if k >= s.shape[-1]:
            return torch.ones_like(s)

        # 2. 创建全 0 的底板
        mask = torch.zeros_like(s)

        # 3. 找到前 k 大的数值的【索引】 (Indices)
        # topk 返回 (values, indices)，我们只需要 indices
        _, indices = torch.topk(s, k, dim=-1)

        # 4. 使用 scatter 将对应索引位置设为 1
        # src=1.0 会自动广播
        mask.scatter_(dim=-1, index=indices, value=1.0)

        return mask


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
