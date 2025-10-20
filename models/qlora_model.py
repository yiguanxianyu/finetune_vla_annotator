# -*- coding: utf-8 -*-
"""
QLoRA helpers for the action segmentation model.
"""

from typing import List, Optional

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration

from .action_model import ActionSegmentationModel


DEFAULT_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
    "proj",
    "fc1",
    "fc2",
    "w1",
    "w2",
    "w3",
]


class QLoRAActionSegmentationModel(ActionSegmentationModel):
    """ActionSegmentationModel with a built-in QLoRA base loader."""

    @classmethod
    def init(
        cls,
        model_name: str,
        *,
        embed_dim: int = 2048,
        k_max: int = 10,
        alpha_text: float = 1.0,
        lora_r: int = 64,
        lora_alpha: int = 128,
        lora_dropout: float = 0.05,
        target_modules: Optional[List[str]] = None,
        attn_implementation: str = "flash_attention_2",
        gradient_checkpointing: bool = True,
    ) -> "QLoRAActionSegmentationModel":
        """Construct the model and load the quantized LoRA-decorated backbone."""
        base_model = cls._load_base_model(
            model_name=model_name,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            attn_implementation=attn_implementation,
            gradient_checkpointing=gradient_checkpointing,
        )
        return cls(
            base_model=base_model,
            embed_dim=embed_dim,
            k_max=k_max,
            alpha_text=alpha_text,
        )

    @staticmethod
    def _load_base_model(
        model_name: str,
        *,
        lora_r: int,
        lora_alpha: int,
        lora_dropout: float,
        target_modules: Optional[List[str]] = None,
        attn_implementation: str = "flash_attention_2",
        gradient_checkpointing: bool = True,
    ) -> Qwen2_5_VLForConditionalGeneration:
        if target_modules is None:
            target_modules = DEFAULT_TARGET_MODULES

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

        base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            attn_implementation=attn_implementation,
            trust_remote_code=True,
            use_cache=False,
        )

        base_model = prepare_model_for_kbit_training(
            base_model,
            use_gradient_checkpointing=gradient_checkpointing,
        )

        peft_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            inference_mode=False,
            target_modules=target_modules,
            task_type="CAUSAL_LM",
        )
        peft_model = get_peft_model(base_model, peft_config)

        if gradient_checkpointing:
            peft_model.enable_input_require_grads()

        peft_model.print_trainable_parameters()

        return peft_model
