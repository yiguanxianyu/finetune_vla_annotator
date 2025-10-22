#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Training entry point for video action segmentation with QLoRA."""

import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["OMP_NUM_THREADS"] = "4"

import argparse
import time
import warnings
from typing import Dict, List, Tuple

import accelerate
import swanlab
import torch
from tqdm import tqdm
from transformers import Qwen2_5_VLProcessor

from data.dataset import build_dataloader, build_dataloader_for_eval
from models.qlora_model import QLoRAActionSegmentationModel
from utils.eval import eval_lm

warnings.filterwarnings("ignore", category=DeprecationWarning)
# warnings.filterwarnings("ignore", category=UserWarning)


def build_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Video Action Segmentation Training (Accelerate + QLoRA)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hf_model", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--min_pixels", type=int, default=256 * 28 * 28)
    parser.add_argument("--max_pixels", type=int, default=1280 * 28 * 28)
    parser.add_argument("--data_root", type=str, default="test_data", help="Dataset root")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--shuffle", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--num_epochs", type=int, default=80)
    parser.add_argument("--log_every", type=int, default=1)
    parser.add_argument("--eval_every", type=int, default=5)
    parser.add_argument("--weight_decay", type=float, default=0.005)
    parser.add_argument("--save_dir", type=str, default="checkpoints_qora")
    parser.add_argument("--embed_dim", type=int, default=2048)
    parser.add_argument("--kmax", type=int, default=16)
    parser.add_argument("--alpha_text", type=float, default=0.4)
    parser.add_argument("--lora_r", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=64)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--attn_implementation", type=str, default="flash_attention_2")
    parser.add_argument("--gradient_checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--tune_lora", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--tune_heads", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--lr", type=float, default=1e-5, help="Base learning rate")
    parser.add_argument("--lora_lr", type=float, default=1e-5, help="Override LR for LoRA adapters")
    parser.add_argument("--head_lr", type=float, default=1e-5, help="Override LR for segmentation heads")
    return parser.parse_args()


def create_optimizer(
    task_model: QLoRAActionSegmentationModel, args: argparse.Namespace
) -> Tuple[torch.optim.Optimizer, List[str]]:
    lr_overrides = {
        "lora": args.lora_lr,
        "heads": args.head_lr,
    }

    grouped: Dict[str, List[torch.nn.Parameter]] = {
        "lora": [],
        "heads": [],
        "other": [],
    }

    for name, param in task_model.named_parameters():
        if not param.requires_grad:
            continue
        elif "lora_" in name:
            grouped["lora"].append(param)
        elif name.startswith(("khead.", "bdhead.")):
            grouped["heads"].append(param)
        else:
            grouped["other"].append(param)

    param_groups = []
    log_lines: List[str] = []
    for group_name, params in grouped.items():
        if not params:
            continue
        lr = lr_overrides.get(group_name, args.lr)
        param_groups.append({"params": params, "lr": lr, "weight_decay": args.weight_decay})
        count = sum(p.numel() for p in params)
        log_lines.append(f"Optimizer group '{group_name}': {count / 1e6:.3f} M params @ lr={lr}")

    if not param_groups:
        raise ValueError("No trainable parameters found – enable LoRA, heads, or base modules.")

    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, weight_decay=args.weight_decay)
    return optimizer, log_lines


def train_one_epoch(network, dataloader, accelerator, optimizer):
    ep_time_start = time.time()
    network.train()
    data_epoch = {}

    if accelerator.is_main_process:
        dataloader = tqdm(dataloader)

    data_time_start = time.time()
    for batch in dataloader:
        inputs_lm, assist = batch
        data_time_end = time.time()

        forward_out = network(
            inputs_lm=inputs_lm,
            labels=assist["text_label"],
            num_frames=assist["num_frames"],
            video_mask=assist["video_mask"],
        )
        out = network.compute_loss(
            forward_outputs=forward_out,
            probs_start=assist["probs_start"],
            probs_end=assist["probs_end"],
            K_label=assist["k_label"],
        )
        accelerator.backward(out["loss"])
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        out["data_time"] = data_time_end - data_time_start
        out["loss"] = out["loss"].detach()
        data_time_start = time.time()

        for key, value in out.items():
            data_epoch.setdefault(key, [])
            data_epoch[key].append(value)

    data_epoch = {k: torch.tensor(v).cpu().mean() for k, v in data_epoch.items()}
    data_epoch["epoch_time"] = time.time() - ep_time_start
    return data_epoch


def train(model, dataloader, optimizer, accelerator, processor, dataloader_eval, args):
    for epoch in range(1, args.num_epochs + 1):
        data_epoch = train_one_epoch(model, dataloader, accelerator, optimizer)

        if accelerator.is_main_process:
            log_info = {}
            if epoch % args.log_every == 0:
                log_info.update(data_epoch)
            if epoch % args.eval_every == 0:
                metrics = eval_lm(model, processor, dataloader_eval)
                log_info.update(metrics)
            swanlab.log(log_info)

    accelerator.wait_for_everyone()


def main() -> None:
    torch.backends.cudnn.benchmark = True
    args = build_args()

    accelerator = accelerate.Accelerator()
    accelerate.utils.set_seed(args.seed, device_specific=True)

    processor = Qwen2_5_VLProcessor.from_pretrained(
        args.hf_model,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
        trust_remote_code=True,
        use_fast=True,
    )
    # configure_processor(processor, args.min_pixels, args.max_pixels)
    accelerator.print(f"Processor loaded: {args.hf_model}")
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
    task_model.base_model.config.use_cache = False
    accelerator.print("Model loaded.")

    dataloader = build_dataloader(
        processor=processor,
        dataset_root=args.data_root,
        batch_size=args.batch_size,
        shuffle=args.shuffle,
        num_workers=args.num_workers,
    )
    dataloader_eval = build_dataloader_for_eval(
        processor=processor,
        dataset_root=args.data_root,
        batch_size=1,
        shuffle=False,
        num_workers=1,
    )
    accelerator.print(f"Dataset and dataloader built with {len(dataloader.dataset)} samples.")
    optimizer, opt_logs = create_optimizer(task_model, args)
    for line in opt_logs:
        accelerator.print(line)

    if accelerator.is_main_process:
        _ = swanlab.init(
            # 设置项目
            project="my-project",
            # 跟踪超参数与实验元数据
            config=vars(args),
        )

    task_model, optimizer, dataloader, dataloader_eval = accelerator.prepare(
        task_model, optimizer, dataloader, dataloader_eval
    )

    with accelerator.autocast():
        train(task_model, dataloader, optimizer, accelerator, processor, dataloader_eval, args)

    if accelerator.is_main_process:
        os.makedirs(args.save_dir, exist_ok=True)
        unwrapped = accelerator.unwrap_model(task_model)
        try:
            unwrapped.base_model.save_pretrained(os.path.join(args.save_dir, "base_model_lora"))
        except Exception as exc:
            accelerator.print(f"Failed to save LoRA adapters: {exc}")
        torch.save(unwrapped.khead.state_dict(), os.path.join(args.save_dir, "khead.pt"))
        torch.save(unwrapped.bdhead.state_dict(), os.path.join(args.save_dir, "bdhead.pt"))
        accelerator.print(f"Weights saved to: {args.save_dir}")

    accelerator.print("Training complete.")


if __name__ == "__main__":
    main()
