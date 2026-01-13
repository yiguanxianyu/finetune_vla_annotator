#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import warnings

import torch
from transformers import Qwen3VLProcessor, Trainer, TrainingArguments, Qwen3VLForConditionalGeneration

from data.dataset import VideoActionDataset, build_collator_text_only

# from utils.eval import eval_lm
warnings.filterwarnings("ignore", category=DeprecationWarning)


def build_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Video Action Segmentation Training (AMP)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hf_model", type=str, default="Qwen/Qwen3-VL-2B-Instruct")
    parser.add_argument("--output_dir", type=str, default="output/qwen3vl_2B")
    parser.add_argument("--data_root", type=str, default="/mnt/e/observations_sub", help="Dataset root")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--eval_every", type=int, default=100)
    parser.add_argument("--weight_decay", type=float, default=0.005)
    parser.add_argument("--gradient_checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-5, help="Base learning rate")
    parser.add_argument("--use_lora", type=bool, default=True, help="Whether to use LoRA")
    return parser.parse_args()


def build_model(args):
    processor = Qwen3VLProcessor.from_pretrained(args.hf_model, trust_remote_code=True, use_fast=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.hf_model,
        attn_implementation="flash_attention_2",
        dtype=torch.bfloat16,
    )

    return model, processor


def train(model, processor, dataset_train, dataset_eval, args):
    training_args = TrainingArguments(
        seed=args.seed,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        learning_rate=args.lr,
        num_train_epochs=args.num_epochs,
        eval_strategy="epoch",
        weight_decay=args.weight_decay,
        logging_steps=args.logging_steps,
        output_dir=args.output_dir,
        save_total_limit=3,
        save_strategy="epoch",
        bf16=True,
        bf16_full_eval=True,
        remove_unused_columns=False,
        report_to=["swanlab"],
        # fsdp="auto_wrap",
        # fsdp_config={"fsdp_version": 2},
        dataloader_num_workers=args.num_workers,
        dataloader_persistent_workers=(args.num_workers > 0),
        dataloader_pin_memory=True,
        # torch_compile=True, torch.compile 不兼容flash attention
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset_train,
        eval_dataset=dataset_eval,
        data_collator=build_collator_text_only(processor),
    )

    trainer.train()

    if args.use_lora:
        model = model.merge_and_unload()

    model.save_pretrained(args.output_dir)
    processor.save_pretrained(args.output_dir)
    torch.distributed.destroy_process_group()


def main() -> None:
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("medium")
    args = build_args()
    task_model, processor = build_model(args)
    dataset_train = VideoActionDataset(dataset_root=args.data_root, split="train")
    dataset_eval = VideoActionDataset(dataset_root=args.data_root, split="val")
    train(task_model, processor, dataset_train, dataset_eval, args)


if __name__ == "__main__":
    main()
