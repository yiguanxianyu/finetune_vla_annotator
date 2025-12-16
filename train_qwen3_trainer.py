#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import warnings

import numpy as np
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
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--eval_every", type=int, default=100)
    parser.add_argument("--weight_decay", type=float, default=0.005)
    parser.add_argument("--embed_dim", type=int, default=2048)
    parser.add_argument("--kmax", type=int, default=24)
    parser.add_argument("--gradient_checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-5, help="Base learning rate")
    parser.add_argument("--use_lora", type=bool, default=True, help="Whether to use LoRA")
    return parser.parse_args()


def compute_metrics(eval_pred):
    preditions, label_ids = eval_pred
    k_pred = preditions[3]
    seg_pred = preditions[4]
    k_label = label_ids["actions_count_label"]
    seg_label = label_ids["segments_label"].astype(int)

    loss_text = preditions[0].mean().item()
    loss_bound = preditions[1].mean().item()
    loss_K = preditions[2].mean().item()

    k_sse = ((k_pred - k_label) ** 2).mean().item()
    seg_acc = np.mean(seg_pred != seg_label)

    return {"loss_text": loss_text, "loss_bound": loss_bound, "loss_K": loss_K, "k_sse": k_sse, "seg_hamming": seg_acc}


def preprocess_logits_for_metrics(logits, labels):
    # Loss, k_preds, seg_preds
    return logits[0:5]  # 只返回 loss_text, loss_bound, loss_K, k_preds, seg_preds


def build_model(args):
    processor = Qwen3VLProcessor.from_pretrained(args.hf_model, trust_remote_code=True, use_fast=True)
    config = ActionSegmentationConfig(
        args.hf_model,
        args.kmax,
    )
    model = ActionSegmentationModel(config, device_map="auto")

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
        data_collator=build_collator(processor),
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
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
    dataset_train = VideoActionDataset(dataset_root=args.data_root, split="train")
    dataset_eval = VideoActionDataset(dataset_root=args.data_root, split="val")
    train(task_model, processor, dataset_train, dataset_eval, args)


if __name__ == "__main__":
    main()
