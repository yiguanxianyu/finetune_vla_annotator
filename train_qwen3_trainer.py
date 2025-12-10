#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import time
import warnings
from typing import Dict, List, Tuple

import torch
from data.dataset import VideoActionDataset, build_collator
from models.qwen3_action_model import ActionSegmentationModel, ActionSegmentationConfig
from tqdm import tqdm

from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor, Trainer, TrainingArguments

# from utils.eval import eval_lm
warnings.filterwarnings("ignore", category=DeprecationWarning)


def build_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Video Action Segmentation Training (AMP)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hf_model", type=str, default="Qwen/Qwen3-VL-2B-Instruct")
    parser.add_argument("--output_dir", type=str, default="output/qwen3_action_segmentation_2B")
    parser.add_argument("--data_root", type=str, default="/mnt/e/AgiBotWorld-sub", help="Dataset root")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--num_epochs", type=int, default=4)
    parser.add_argument("--logging_steps", type=int, default=100)
    parser.add_argument("--eval_every", type=int, default=5)
    parser.add_argument("--weight_decay", type=float, default=0.005)
    parser.add_argument("--embed_dim", type=int, default=2048)
    parser.add_argument("--kmax", type=int, default=16)
    parser.add_argument("--attn_implementation", type=str, default="flash_attention_2")
    parser.add_argument("--gradient_checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-5, help="Base learning rate")
    return parser.parse_args()


def build_model(args):
    processor = Qwen3VLProcessor.from_pretrained(args.hf_model, trust_remote_code=True, use_fast=True)
    base_model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.hf_model,
        attn_implementation=args.attn_implementation,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    base_model.config.use_cache = False
    config = ActionSegmentationConfig()
    model = ActionSegmentationModel(config, base_model=base_model, device_map="auto")

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


def train(model, processor, dataset_train, args):
    training_args = TrainingArguments(
        seed=args.seed,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=args.gradient_checkpointing,
        learning_rate=args.lr,
        num_train_epochs=args.num_epochs,
        weight_decay=args.weight_decay,
        logging_steps=args.logging_steps,
        output_dir=args.output_dir,
        save_total_limit=2,
        save_strategy="epoch",
        bf16=True,
        bf16_full_eval=True,
        remove_unused_columns=False,
        report_to=["swanlab"],
        fsdp="auto_wrap",
        fsdp_config={"fsdp_version": 2},
        dataloader_num_workers=args.num_workers,
        dataloader_persistent_workers=True,
        dataloader_pin_memory=True,
        # torch_compile=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset_train,
        data_collator=build_collator(processor),
    )

    trainer.train()


def main() -> None:
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = True
    args = build_args()
    task_model, processor = build_model(args)
    dataset_train = VideoActionDataset(dataset_root=args.data_root)
    train(task_model, processor, dataset_train, args)


if __name__ == "__main__":
    main()
