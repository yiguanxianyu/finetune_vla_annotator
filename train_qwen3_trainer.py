#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import warnings

import torch
from data.dataset import VideoActionDataset, build_collator
from models.qwen3_action_model import ActionSegmentationConfig, ActionSegmentationModel

from transformers import Qwen3VLProcessor, Trainer, TrainingArguments

# from utils.eval import eval_lm
warnings.filterwarnings("ignore", category=DeprecationWarning)


def build_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Video Action Segmentation Training (AMP)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hf_model", type=str, default="Qwen/Qwen3-VL-2B-Instruct")
    parser.add_argument("--output_dir", type=str, default="output/qwen3_action_segmentation_2B")
    parser.add_argument("--data_root", type=str, default="/mnt/e/observations_sub", help="Dataset root")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--eval_every", type=int, default=5)
    parser.add_argument("--weight_decay", type=float, default=0.005)
    parser.add_argument("--embed_dim", type=int, default=2048)
    parser.add_argument("--kmax", type=int, default=24)
    parser.add_argument("--gradient_checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-5, help="Base learning rate")
    parser.add_argument("--use_lora", type=bool, default=True, help="Whether to use LoRA")
    return parser.parse_args()


def compute_metrics(preditions):
    print("Computing metrics...")
    print(preditions)
    return {"k_accuracy": 0.9, "seg_accuracy": 0.8}


def build_model(args):
    processor = Qwen3VLProcessor.from_pretrained(args.hf_model, trust_remote_code=True, use_fast=True)
    config = ActionSegmentationConfig(
        args.hf_model,
        args.kmax,
    )
    model = ActionSegmentationModel(config, device_map="auto")

    return model, processor


def train(model, processor, dataset_train, args):
    training_args = TrainingArguments(
        seed=args.seed,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        learning_rate=args.lr,
        num_train_epochs=args.num_epochs,
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
        data_collator=build_collator(processor),
        compute_metrics=compute_metrics,
    )

    trainer.train()

    if args.use_lora:
        model.qwen3vlmodel = model.qwen3vlmodel.merge_and_unload()

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
    dataset_train = VideoActionDataset(dataset_root=args.data_root)
    train(task_model, processor, dataset_train, args)


if __name__ == "__main__":
    main()
