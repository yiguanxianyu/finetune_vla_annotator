#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Training entry point for video action segmentation with QLoRA (DeepSpeed)."""

import os

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import itertools
import warnings
from typing import Iterable, Tuple

import deepspeed
import torch
from transformers import AutoProcessor, Qwen2_5_VLProcessor

from data.dataset import build_collator, build_dataloader
from models.qlora_model import QLoRAActionSegmentationModel

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def build_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Video Action Segmentation Training (DeepSpeed + QLoRA)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hf_model", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--min_pixels", type=int, default=256 * 28 * 28)
    parser.add_argument("--max_pixels", type=int, default=1280 * 28 * 28)
    parser.add_argument("--data_root", type=str, default=None, help="Dataset root containing task_info/observations.")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--shuffle", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_steps", type=int, default=200)
    parser.add_argument("--log_every", type=int, default=1)
    parser.add_argument("--weight_decay", type=float, default=0.005)
    parser.add_argument("--save_dir", type=str, default="checkpoints_qora")
    parser.add_argument("--embed_dim", type=int, default=2048)
    parser.add_argument("--kmax", type=int, default=10)
    parser.add_argument("--alpha_text", type=float, default=0.1)
    parser.add_argument("--lora_r", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=64)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--attn_implementation", type=str, default="flash_attention_2")
    parser.add_argument("--gradient_checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--tune_lora", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--tune_heads", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--lr", type=float, default=1e-6, help="Base learning rate")
    parser.add_argument("--lora_lr", type=float, default=1e-6, help="Override LR for LoRA adapters")
    parser.add_argument("--head_lr", type=float, default=1e-6, help="Override LR for segmentation heads")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank passed by DeepSpeed")
    parser = deepspeed.add_config_arguments(parser)
    return parser.parse_args()


def configure_processor(processor: AutoProcessor, min_pixels: int, max_pixels: int) -> None:
    """Align processor vision sizing with CLI configuration."""
    image_processor = processor.image_processor
    image_processor.min_pixels = min_pixels
    image_processor.max_pixels = max_pixels
    image_processor.size["longest_edge"] = max_pixels
    image_processor.size["shortest_edge"] = min_pixels


def is_main_process() -> bool:
    """Rank 0 判定（DeepSpeed/torch.distributed 下安全）。"""
    return deepspeed.comm.get_rank() == 0


def print_trainable_breakdown(task_model: QLoRAActionSegmentationModel) -> None:
    def _count(params: Iterable[torch.nn.Parameter]) -> Tuple[int, int]:
        params_trainable = 0
        params_frozen = 0
        for p in params:
            if p.requires_grad:
                params_trainable += p.numel()
            else:
                params_frozen += p.numel()
        return params_trainable + params_frozen, params_trainable

    total_params, trainable_params = _count(task_model.parameters())
    if is_main_process():
        print(
            f"Total params: {total_params / 1e9:.2f}B, trainable: {trainable_params / 1e6:.2f}M, "
            f"ratio: {100 * trainable_params / max(total_params, 1):.2f}%"
        )

    qwen_model = task_model.base_model
    vision_total, vision_trainable = _count(qwen_model.visual.parameters())
    llm_total, llm_trainable = _count(qwen_model.model.parameters())
    head_total, head_trainable = _count(task_model.khead.parameters())
    head_total2, head_trainable2 = _count(task_model.bdhead.parameters())

    if is_main_process():
        print(
            "Trainable breakdown => "
            f"vision {vision_trainable / 1e6:.3f}/{vision_total / 1e6:.3f} M | "
            f"llm {llm_trainable / 1e6:.3f}/{llm_total / 1e6:.3f} M | "
            f"heads {(head_trainable + head_trainable2) / 1e6:.3f}/{(head_total + head_total2) / 1e6:.3f} M"
        )


def main() -> None:
    args = build_args()
    torch.manual_seed(args.seed)

    processor = Qwen2_5_VLProcessor.from_pretrained(
        args.hf_model, min_pixels=args.min_pixels, max_pixels=args.max_pixels, trust_remote_code=True, use_fast=True
    )
    configure_processor(processor, args.min_pixels, args.max_pixels)

    task_model = QLoRAActionSegmentationModel.init(
        model_name=args.hf_model,
        embed_dim=args.embed_dim,
        k_max=args.kmax,
        alpha_text=args.alpha_text,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        attn_implementation=args.attn_implementation,
        gradient_checkpointing=args.gradient_checkpointing,
    )

    # DataLoader 逻辑保持不变
    dataset, dataloader = build_dataloader(
        processor=processor,
        dataset_root=args.data_root,
        batch_size=args.batch_size,
        shuffle=args.shuffle,
        num_workers=args.num_workers,
    )
    collate_fn = build_collator(processor)

    ds_config_path = args.deepspeed_config
    params = list(p for p in task_model.parameters() if p.requires_grad)
    model_engine, optimizer, dataloader, lr_scheduler = deepspeed.initialize(
        model=task_model,
        model_parameters=params,
        training_data=dataset,
        config=ds_config_path,
        collate_fn=collate_fn,
    )

    # DeepSpeed 内部处理梯度累积；此处保留变量以兼容原打印/语义
    data_iter = itertools.cycle(dataloader)
    device = model_engine.local_rank

    # 训练循环：以 DeepSpeed 的 backward/step 替代 accelerate 版本
    for step in range(1, args.max_steps + 1):
        batch = next(data_iter)
        inputs_lm, assist = batch
        inputs_lm = inputs_lm.to(device)
        assist = assist.to(device)
        forward_out = model_engine(
            inputs_lm=inputs_lm,
            labels=assist["text_label"],
            num_frames=assist["num_frames"],
            video_mask=assist["video_mask"],
        )
        out = model_engine.module.compute_loss(
            forward_outputs=forward_out,
            probs_start=assist["probs_start"],
            probs_end=assist["probs_end"],
            K_label=assist["k_label"],
        )

        loss = out["loss"]
        model_engine.backward(loss)
        model_engine.step()

        if (step % args.log_every) == 0 and is_main_process():
            loss_values = {name: float(val) for name, val in out.items() if name.startswith("loss")}
            print(f"[step {step:04d}] " + " ".join([f"{name}={value:.6f}" for name, value in loss_values.items()]))

    # 同步
    try:
        deepspeed.comm.barrier()
    except Exception:
        pass

    # 保存：保持与原有文件结构一致（LoRA 适配器与两个头）
    if is_main_process():
        os.makedirs(args.save_dir, exist_ok=True)
        unwrapped = model_engine.module  # DeepSpeed 包裹下的原始 nn.Module
        qwen_model = unwrapped.base_model.get_model()
        qwen_model.config.use_cache = False
        try:
            unwrapped.base_model.save_pretrained(os.path.join(args.save_dir, "base_model_lora"))
        except Exception as exc:  # noqa: BLE001
            print(f"Failed to save LoRA adapters: {exc}")
        torch.save(unwrapped.khead.state_dict(), os.path.join(args.save_dir, "khead.pt"))
        torch.save(unwrapped.bdhead.state_dict(), os.path.join(args.save_dir, "bdhead.pt"))
        print(f"Weights saved to: {args.save_dir}")

    if is_main_process():
        print("Training complete.")


if __name__ == "__main__":
    main()
