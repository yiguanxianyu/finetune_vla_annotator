#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import time
import warnings
from typing import Dict, List, Tuple

import swanlab
import torch
from data.dataset import build_dataloader
from models.qwen3_action_model import ActionSegmentationModel
from tqdm import tqdm

from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor

# from utils.eval import eval_lm
warnings.filterwarnings("ignore", category=DeprecationWarning)


def build_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Video Action Segmentation Training (AMP)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hf_model", type=str, default="Qwen/Qwen3-VL-2B-Instruct")
    parser.add_argument("--data_root", type=str, default="/mnt/e/AgiBotWorld-sub", help="Dataset root")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--num_epochs", type=int, default=80)
    parser.add_argument("--log_every", type=int, default=1)
    parser.add_argument("--eval_every", type=int, default=5)
    parser.add_argument("--weight_decay", type=float, default=0.005)
    parser.add_argument("--save_dir", type=str, default="checkpoints_qora")
    parser.add_argument("--embed_dim", type=int, default=2048)
    parser.add_argument("--kmax", type=int, default=16)
    parser.add_argument("--attn_implementation", type=str, default="flash_attention_2")
    parser.add_argument("--gradient_checkpointing", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--lr", type=float, default=1e-5, help="Base learning rate")
    return parser.parse_args()


def build_optimizer(model, args: argparse.Namespace) -> Tuple[torch.optim.Optimizer, List[str]]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    return optimizer


def build_model(args):
    processor = Qwen3VLProcessor.from_pretrained(args.hf_model, trust_remote_code=True, use_fast=True)
    base_model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.hf_model,
        attn_implementation=args.attn_implementation,
        dtype=torch.bfloat16,
        device_map="cuda:0",
        trust_remote_code=True,
    )
    base_model.config.use_cache = False
    if args.gradient_checkpointing:
        base_model.gradient_checkpointing_enable()

    model = ActionSegmentationModel(
        base_model=base_model,
        embed_dim=args.embed_dim,
        k_max=args.kmax,
    ).to("cuda:0")

    return model, processor


def train_one_epoch(model, dataloader, optimizer):
    ep_time_start = time.time()
    model.train()
    data_epoch = {}

    data_time_start = time.time()
    for batch in tqdm(dataloader):
        inputs_lm, assist = batch
        inputs_lm = inputs_lm.to("cuda:0")
        assist = assist.to("cuda:0")
        data_time_end = time.time()
        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            forward_out = model(
                inputs_lm=inputs_lm,
                labels=assist["text_label"],
                num_frames=assist["num_frames"],
                video_mask=assist["video_mask"],
            )
            loss, loss_value = model.compute_loss(
                forward_outputs=forward_out,
                segments_label=assist["segments_label"],
                actions_count_label=assist["actions_count_label"],
            )

        loss.backward()
        optimizer.step()
        print(loss_value)

        loss_value["data_time"] = data_time_end - data_time_start
        data_time_start = time.time()

        for key, value in loss_value.items():
            data_epoch.setdefault(key, [])
            data_epoch[key].append(value)

    data_epoch = {k: torch.tensor(v).cpu().mean() for k, v in data_epoch.items()}
    data_epoch["epoch_time (min)"] = (time.time() - ep_time_start) / 60
    return data_epoch


def train(model, dataloader, optimizer, processor, dataloader_eval, args):
    # metrics = eval_lm(model, processor, dataloader_eval, accelerator)
    # swanlab.log(metrics, step=0)
    for epoch in range(1, args.num_epochs + 1):
        data_epoch = train_one_epoch(model, dataloader, optimizer)

        # log_info = {}
        # if epoch % args.log_every == 0:
        #     log_info.update(data_epoch)
        # if epoch % args.eval_every == 0:
        #     metrics = eval_lm(model, processor, dataloader_eval, accelerator)
        #     log_info.update(metrics)
        # swanlab.log(log_info, step=epoch)


def main() -> None:
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = True
    args = build_args()

    # configure_processor(processor, args.min_pixels, args.max_pixels)
    print(f"Processor loaded: {args.hf_model}")
    task_model, processor = build_model(args)

    dataloader = build_dataloader(
        processor=processor,
        dataset_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
    )
    # dataloader_eval = build_dataloader_for_eval(
    #     processor=processor,
    #     dataset_root=args.data_root,
    #     batch_size=1,
    #     shuffle=False,
    #     num_workers=0,
    # )
    print(f"Dataset and dataloader built with {len(dataloader.dataset)} samples.")
    optimizer = build_optimizer(task_model, args)

    _ = swanlab.init(
        # 设置项目
        project="my-project",
        # 跟踪超参数与实验元数据
        config=vars(args),
    )

    train(task_model, dataloader, optimizer, processor, dataloader, args)


if __name__ == "__main__":
    main()
