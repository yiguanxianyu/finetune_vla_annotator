# -*- coding: utf-8 -*-
"""
数据集与数据加载器
==================
- 支持按清单加载多条样本，默认回退到示例样本。
- 与官方 Qwen-VL fine-tune 代码保持相同的消息/视频预处理流程。
"""

import json
import os
import random
import warnings
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d
from torch.utils.data import DataLoader, Dataset
from transformers.feature_extraction_utils import BatchFeature
from transformers.processing_utils import VideosKwargs

from transformers import AutoProcessor


def build_collator(processor, args=None):
    """Merge a list of samples into a single batched dict that the model can consume."""
    # pad_token_id = processor.tokenizer.pad_token_id
    im_start_id = processor.tokenizer.convert_tokens_to_ids("<|im_start|>")
    pad_token_id = processor.tokenizer.pad_token_id
    use_soft_label = True
    guassian_sigma = 1

    def _collator(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        bs = len(batch)
        messages = [i["messages"] for i in batch]
        actions_count = [i["actions_count"] for i in batch]
        actions_segments = np.array([i["actions_segments"] for i in batch])

        inputs_lm, video_metadata = processor.apply_chat_template(
            messages,
            tokenize=True,
            padding=True,
            add_generation_prompt=False,
            return_dict=True,
            return_metadata=True,
            return_tensors="pt",
            videos_kwargs=VideosKwargs(fps=1),
        )
        num_frames = []
        segments_label = []
        for seg, meta in zip(actions_segments, video_metadata):
            num_frames_piece = len(meta.frames_indices)  # 对标签根据采样帧数量进行处理
            sample_rate = len(meta.frames_indices) / meta.total_num_frames
            segment_processed = (seg * sample_rate).round().astype(int)
            label = np.zeros(num_frames_piece + 1)
            label[segment_processed] = 1
            if use_soft_label:
                label = gaussian_filter1d(label, sigma=guassian_sigma)
                label[segment_processed] = 1
            segments_label.append(label)
            num_frames.append(num_frames_piece)

        segments_label = torch.from_numpy(np.concat(segments_label))
        actions_count_label = torch.tensor(actions_count)
        video_mask = inputs_lm["input_ids"].eq(processor.video_token_id)

        # 遮盖输入部分
        text_label = inputs_lm["input_ids"].clone()
        text_label[text_label == pad_token_id] = -100

        for b in range(bs):
            # 找到当前样本中所有 im_start_id 的位置索引
            start_indices = torch.where(inputs_lm["input_ids"][b] == im_start_id)[0]
            if len(start_indices) > 0:
                # Mask 掉从开头到 answer_start 之前的所有内容
                # 注意：这里 +1 是因为通常希望模型从 im_start_id 的下一个词开始预测
                answer_start = start_indices[-1] + 1
                text_label[b, :answer_start] = -100
            else:
                # 如果这条数据里根本没有 im_start_id（脏数据），通常整个 mask 掉，不计算 loss
                text_label[b, :] = -100

        label = BatchFeature(
            data=dict(
                text_label=text_label,
                actions_count_label=actions_count_label,
                segments_label=segments_label,
                video_mask=video_mask,
                num_frames=num_frames,
            ),
            tensor_type="pt",
        )

        # rows, cols = torch.where(inputs["input_ids"] == im_start_id)
        # answer_start = cols.view(bs, -1)[:, -1] + 1
        # for b in range(bs):
        #     text_label[b, : answer_start[b]] = -100
        # 遮盖pad部分

        # num_frames = 1
        # # texts = [item["text"] for item in batch]
        # # video_inputs = [item["video_input"] for item in batch]
        # # image_inputs = [item["image_input"] for item in batch]
        # num_frames = [item["num_frames"] for item in batch]
        # fps_list = [item["fps"] for item in batch]
        # probs_start = torch.stack([item["probs_start"] for item in batch], dim=0)
        # probs_end = torch.stack([item["probs_end"] for item in batch], dim=0)
        # k_label = torch.cat([item["K_label"] for item in batch], dim=0)

        # data = dict(
        #     num_frames=torch.tensor(num_frames),
        #     video_mask=video_mask,
        #     text_label=text_label,
        #     probs_start=probs_start,
        #     probs_end=probs_end,
        #     k_label=k_label,
        # )

        # if "answer" in batch[0]:
        #     # 评估时，保留答案部分用于计算精度
        #     data["answer_texts"] = processor.tokenizer([item["answer"] for item in batch])["input_ids"]

        # return inputs, label
        return dict(
            inputs_lm=inputs_lm,
            text_label=text_label,
            actions_count_label=actions_count_label,
            segments_label=segments_label,
            video_mask=video_mask,
            num_frames=num_frames,
        )

    return _collator


DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant that helps people find the action segments in the video "
    "and answer the question based on the video."
)
DEFAULT_USER_INSTRUCTION = (
    "Describe this video in json format. It is known that a video can be divided into {} segments."
)


class ActionSample:
    """Container describing a single training sample."""

    def __init__(self, video: str, label: dict):
        self.video = video
        self.label = label
        self.system_prompt: str = DEFAULT_SYSTEM_PROMPT
        self.user_instruction: str = DEFAULT_USER_INSTRUCTION
        self.actions_count = self.get_actions_count()
        self.actions_segments = self.get_actions_segments()
        assert self.actions_count + 1 == len(self.actions_segments)

    def build_messages(self) -> List[Dict[str, Any]]:
        video_path = self._resolve_path(self.video)
        user_content: List[Dict[str, Any]] = [
            {"type": "video", "video": video_path},
            {"type": "text", "text": self.user_instruction.format(self.actions_count)},
        ]

        return [
            {"role": "system", "content": [{"type": "text", "text": self.system_prompt}]},
            {"role": "user", "content": user_content},
        ]

    def build_messages_with_gt(self) -> List[Dict[str, Any]]:
        messages = self.build_messages()
        messages.append(
            {
                "role": "assistant",
                "content": [{"type": "text", "text": json.dumps(self.label, indent=2)}],
            }
        )
        return messages

    def get_actions_count(self):
        return len(self.label["label_info"]["action_config"])

    def get_actions_segments(self):
        frame_seq = [None]
        for piece in self.label["label_info"]["action_config"]:
            if piece["start_frame"] != frame_seq[-1]:
                frame_seq.append(piece["start_frame"])
            frame_seq.append(piece["end_frame"])
        return frame_seq[1:]

    def _resolve_path(self, path: str) -> str:
        if os.path.isabs(path) or self.data_root is None:
            return path
        return os.path.join(self.data_root, path)


def _load_samples_from_root(dataset_root: Optional[str]):
    root = Path(dataset_root)
    task_info_dir = root / "task_info"
    observations_dir = root / "observations"

    assert task_info_dir.exists(), f"Dataset root {dataset_root} must contain 'task_info' directories."
    assert observations_dir.exists(), f"Dataset root {dataset_root} must contain 'observations' directories."
    data = {}
    samples = []
    for ep in observations_dir.rglob("**/head_color.mp4"):
        parts = ep.parts
        task_id = parts[-4]
        episode_id = parts[-3]
        data.setdefault(task_id, {})[episode_id] = str(ep.resolve())
    key_order = ["task_name", "init_scene_text", "label_info"]
    for key, val in data.items():
        task_json = task_info_dir / f"task_{key}.json"
        records = json.loads(task_json.read_text(encoding="utf-8"))
        for record in records:
            episode_id = str(record["episode_id"])
            if episode_id not in val or episode_id is None:
                continue

            ordered_record = OrderedDict((k, record[k]) for k in key_order)
            sample = ActionSample(video=val[episode_id], label=ordered_record)
            samples.append(sample)

    if not samples:
        raise ValueError(f"No valid samples discovered under dataset root: {dataset_root}")

    return samples


class VideoActionDataset(Dataset):
    def __init__(
        self,
        dataset_root: Optional[str] = None,
        max_samples=None,
    ):
        super().__init__()
        self.samples = _load_samples_from_root(dataset_root)
        if max_samples:
            self.samples = random.sample(self.samples, k=max_samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:  # noqa: D401
        try:
            return self.load_sample(idx)
        except Exception as exc:  # noqa: BLE001
            warnings.warn(
                f"Failed to load sample {idx}, retrying with a different one: {exc}",
                UserWarning,
            )
            new_idx = random.choice(list(range(len(self))))
            return self[new_idx]

    def load_sample(self, idx: int) -> ActionSample:
        messages = self.samples[idx].build_messages_with_gt()
        actions_count = self.samples[idx].actions_count
        actions_segments = self.samples[idx].actions_segments

        return dict(
            messages=messages,
            actions_count=actions_count,
            actions_segments=actions_segments,
        )
        # full_text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

        # video_inputs, total_frames_list, fps_list = process_video_info_torchcodec(messages, nframes=self.nframes)
        # video_input = video_inputs[0]
        # num_frames = video_input.size(0)

        # probs_start, probs_end, k_label = _build_segment_targets(
        #     assistant_text=self.samples[idx].label,
        #     total_timesteps=num_frames,
        #     n_fullframes=total_frames_list[0],
        # )

        # return dict(
        #     text=full_text,
        #     video_input=video_input,
        #     image_input=None,
        #     num_frames=num_frames,
        #     fps=fps_list[0],
        #     probs_start=probs_start,
        #     probs_end=probs_end,
        #     K_label=k_label,
        # )


# class VideoActionDatasetForEval(Dataset):
#     """Dataset that mirrors官方 Qwen-VL fine-tuning预处理流程.不含目标输出"""

#     def __init__(
#         self,
#         processor: AutoProcessor,
#         dataset_root: Optional[str] = None,
#         max_samples=None,
#     ):
#         super().__init__()
#         self.processor = processor
#         self.video_token_id = self.processor.video_token_id

#         self.samples = _load_samples_from_root(dataset_root)
#         if max_samples:
#             self.samples = random.sample(self.samples, k=max_samples)

#         self.nframes = 30

#     def __len__(self) -> int:
#         return len(self.samples)

#     def __getitem__(self, idx: int) -> Dict[str, Any]:  # noqa: D401
#         try:
#             return self.load_sample(idx)
#         except Exception as exc:  # noqa: BLE001
#             idx = random.choice(range(len(self)))
#             warnings.warn(
#                 f"Failed to load sample {idx}, retrying with a different one: {exc}",
#                 UserWarning,
#             )
#             return self.load_sample(idx)

#     def load_sample(self, idx: int) -> ActionSample:
#         messages = self.samples[idx].build_messages()
#         full_text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
#         answer = self.samples[idx].label

#         video_inputs, total_frames_list, fps_list = process_video_info_torchcodec(messages, nframes=self.nframes)
#         video_input = video_inputs[0]
#         num_frames = video_input.size(0)

#         probs_start, probs_end, k_label = _build_segment_targets(
#             assistant_text=self.samples[idx].label,
#             total_timesteps=num_frames,
#             n_fullframes=total_frames_list[0],
#         )

#         return dict(
#             text=full_text,
#             answer=answer,
#             video_input=video_input,
#             image_input=None,
#             num_frames=num_frames,
#             fps=fps_list[0],
#             probs_start=probs_start,
#             probs_end=probs_end,
#             K_label=k_label,
#         )


# def _build_segment_targets(
#     assistant_text: str,
#     total_timesteps: int,
#     *,
#     n_fullframes: int,
# ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
#     """Generate boundary supervision from assistant JSON annotation."""

#     try:
#         annotation = json.loads(assistant_text)
#     except json.JSONDecodeError as exc:  # noqa: F841
#         raise ValueError("assistant_text must be a JSON string with label_info action_config.")

#     configs: Iterable[Dict[str, Any]] = annotation.get("label_info", {}).get("action_config", [])
#     configs = list(configs)
#     if not configs:
#         raise ValueError("annotation.label_info.action_config is empty – cannot build targets.")

#     probs_start = torch.zeros(total_timesteps, dtype=torch.float32)
#     probs_end = torch.zeros_like(probs_start)

#     for seg in configs:
#         start_frame = seg.get("start_frame")
#         end_frame = seg.get("end_frame")
#         if start_frame is None or end_frame is None:
#             continue
#         start_idx = int(round(start_frame * total_timesteps / max(n_fullframes, 1)))
#         end_idx = int(round(end_frame * total_timesteps / max(n_fullframes, 1)))
#         start_idx = min(max(start_idx, 0), total_timesteps - 1)
#         end_idx = min(max(end_idx, 0), total_timesteps - 1)
#         probs_start[start_idx] = 1.0
#         probs_end[end_idx] = 1.0

#     eps = 1e-8
#     ps = gaussian_filter1d(probs_start.numpy(), sigma=1)
#     pe = gaussian_filter1d(probs_end.numpy(), sigma=1)
#     ps = ps / (ps.max() + eps)
#     pe = pe / (pe.max() + eps)
#     probs_start = torch.from_numpy(ps).float()
#     probs_end = torch.from_numpy(pe).float()

#     num_segments = max(len(configs), 1)
#     k_label = torch.tensor([num_segments - 1], dtype=torch.long)


#     return probs_start, probs_end, k_label


def build_dataloader(
    processor,
    dataset_root,
    batch_size: int = 2,
    shuffle: bool = False,
    num_workers: int = 0,
) -> DataLoader:
    dataset = VideoActionDataset(processor=processor, dataset_root=dataset_root)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=True,
        num_workers=num_workers,
        collate_fn=build_collator(processor),
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )


# def build_dataloader_for_eval(
#     processor: AutoProcessor,
#     *,
#     dataset_root: Optional[str] = None,
#     batch_size: int = 1,
#     shuffle: bool = False,
#     num_workers: int = 0,
# ) -> DataLoader:
#     dataset = VideoActionDatasetForEval(processor=processor, dataset_root=dataset_root, max_samples=8)

#     return DataLoader(
#         dataset,
#         batch_size=batch_size,
#         shuffle=shuffle,
#         num_workers=num_workers,
#         collate_fn=build_collator(processor),
#         pin_memory=True,
#         persistent_workers=(num_workers > 0),
#     )


if __name__ == "__main__":
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-2B-Thinking")
    dataloader = build_dataloader(processor, "/mnt/e/AgiBotWorld-sub")
    dataset = dataloader.dataset

    # sample: ActionSample = dataset.samples[-1]

    input_, label_ = next(iter(dataloader))
    input_
