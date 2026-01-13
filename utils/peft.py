# -*- coding: utf-8 -*-
"""PEFT / QLoRA helper utilities."""

from typing import List, Optional

from accelerate import Accelerator

from models.qlora_model import DEFAULT_TARGET_MODULES, QLoRAActionSegmentationModel

__all__ = [
    "DEFAULT_TARGET_MODULES",
    "load_base_model_with_qlora",
]


def load_base_model_with_qlora(
    model_name: str,
    accelerator: Accelerator,
    lora_r: int = 64,
    lora_alpha: int = 128,
    lora_dropout: float = 0.05,
    target_modules: Optional[List[str]] = None,
    attn_implementation: str = "flash_attention_2",
    gradient_checkpointing: bool = True,
):
    """Backwards-compatible shim that delegates to the QLoRA model helper."""
    if target_modules is None:
        target_modules = DEFAULT_TARGET_MODULES
    return QLoRAActionSegmentationModel._load_base_model(
        model_name=model_name,
        accelerator=accelerator,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        attn_implementation=attn_implementation,
        gradient_checkpointing=gradient_checkpointing,
    )
