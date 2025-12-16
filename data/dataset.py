# -*- coding: utf-8 -*-
"""
数据集与数据加载器
==================
- 支持按清单加载多条样本，默认回退到示例样本。
- 与官方 Qwen-VL fine-tune 代码保持相同的消息/视频预处理流程。
"""

import json
import random
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional
from functools import cached_property
import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers.feature_extraction_utils import BatchFeature
from transformers.processing_utils import VideosKwargs
from torchcodec.decoders import VideoDecoder
from transformers import AutoProcessor
import copy

DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant that help people exactly find the action segments in the video."
DEFAULT_USER_INSTRUCTION = "Describe this video in json format. Ensure that the segments are non-overlapping and cover the entire video from start to end."

DEFAULT_LABEL = "It is known that this video can be divided into {} segments. "


class ActionSample:
    """Container describing a single training sample."""

    def __init__(self, video: str, label: dict):
        self.video = video
        self.label = label
        self.system_prompt: str = DEFAULT_SYSTEM_PROMPT
        self.user_instruction: str = DEFAULT_USER_INSTRUCTION
        self.actions_segments = self.get_actions_segments()
        assert self.actions_count + 1 == len(self.actions_segments)

    @cached_property
    def video_meta(self) -> Dict[str, Any]:
        decoder = VideoDecoder(self.video)
        return decoder.metadata

    @cached_property
    def actions_count(self):
        "Get the number of action segments in this sample."
        return len(self.label["label_info"]["action_config"])

    @cached_property
    def _system_messages(self) -> List[Dict[str, Any]]:
        return [
            {"role": "system", "content": [{"type": "text", "text": self.system_prompt}]},
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": self.video},
                    {"type": "text", "text": self.user_instruction},
                ],
            },
        ]

    @cached_property
    def text_label(self):
        return DEFAULT_LABEL.format(self.actions_count)

    def build_messages_with_gt(self, frame_sample_rate=1, use_text_label=True) -> List[Dict[str, Any]]:
        messages = copy.deepcopy(self._system_messages)
        if use_text_label:
            messages[1]["content"].append({"type": "text", "text": self.text_label})

        label = copy.deepcopy(self.label)
        for piece in label["label_info"]["action_config"]:
            piece["start_frame"] = round(piece["start_frame"] * frame_sample_rate)
            piece["end_frame"] = round(piece["end_frame"] * frame_sample_rate)

        messages.append(
            {
                "role": "assistant",
                "content": [{"type": "text", "text": json.dumps(label, indent=2)}],
            }
        )
        return messages

    def get_actions_segments(self):
        frame_seq = [None]
        for piece in self.label["label_info"]["action_config"]:
            if piece["start_frame"] != frame_seq[-1]:
                frame_seq.append(piece["start_frame"])
            frame_seq.append(piece["end_frame"])
        return frame_seq[1:]


def build_collator_text_only(processor, args=None):
    im_start_id = processor.tokenizer.convert_tokens_to_ids("<|im_start|>")
    fps = None
    _sample_frames = 8

    # text only collator
    def collator(batch: List[ActionSample]):
        messages = []
        for b in batch:
            meta = b.video_meta
            local_sample_frames = _sample_frames
            if fps is not None:
                local_sample_frames = round(meta.end_stream_seconds_from_content * fps)
            # Video sampling start from 0 and end at num_frames,so frame_sample_rate segments to num_frames - 1
            frame_sample_rate = (local_sample_frames - 1) / meta.num_frames
            message = b.build_messages_with_gt(frame_sample_rate=frame_sample_rate, use_text_label=False)
            messages.append(message)

        if fps:
            video_kwargs = VideosKwargs(fps=fps)
        else:
            video_kwargs = VideosKwargs(num_frames=_sample_frames, fps=None)

        # Make batched inputs for VLM
        inputs_lm = processor.apply_chat_template(
            messages,
            tokenize=True,
            padding=True,
            add_generation_prompt=False,
            return_dict=True,
            return_tensors="pt",
            videos_kwargs=video_kwargs,
        )
        # 遮盖输入部分
        text_label = inputs_lm["input_ids"].clone()

        for b in range(len(batch)):
            # 找到当前样本中所有 im_start_id 的位置索引
            start_indices = torch.where(inputs_lm["input_ids"][b] == im_start_id)[0]

            # Mask 掉从开头到 answer_start 之前的所有内容
            # 注意：这里 +1 是因为通常希望模型从 im_start_id 的下一个词开始预测
            answer_start = start_indices[-1] + 1
            text_label[b, :answer_start] = -100

        return inputs_lm, text_label

    return collator


def build_collator(processor, args=None):
    """Merge a list of samples into a single batched dict that the model can consume."""
    im_start_id = processor.tokenizer.convert_tokens_to_ids("<|im_start|>")
    pad_token_id = processor.tokenizer.pad_token_id
    use_soft_label = True
    guassian_sigma = 1
    fps = None
    _sample_frames = 8

    # new version collator
    def new_collator(batch: List[ActionSample]):
        segments_label = []
        num_frames = []
        messages = []
        actions_count = []
        for b in batch:
            meta = b.video_meta
            local_sample_frames = _sample_frames
            if fps is not None:
                local_sample_frames = round(meta.end_stream_seconds_from_content * fps)
            # Video sampling start from 0 and end at num_frames,so frame_sample_rate segments to num_frames - 1
            frame_sample_rate = (local_sample_frames - 1) / meta.num_frames
            segment_processed = np.round(np.array(b.actions_segments) * frame_sample_rate).astype(int)
            message = b.build_messages_with_gt(frame_sample_rate=frame_sample_rate)
            # frame is 0-indexed, so +1 for the last frame
            label = np.zeros(local_sample_frames)
            label[segment_processed] = 1
            if use_soft_label:
                label = gaussian_filter1d(label, sigma=guassian_sigma)
                label[segment_processed] = 1

            segments_label.append(label)
            actions_count.append(b.actions_count)
            num_frames.append(local_sample_frames)
            messages.append(message)
        if fps:
            video_kwargs = VideosKwargs(fps=fps)
        else:
            video_kwargs = VideosKwargs(num_frames=_sample_frames, fps=None)

        # Make batched inputs for VLM
        inputs_lm = processor.apply_chat_template(
            messages,
            tokenize=True,
            padding=True,
            add_generation_prompt=False,
            return_dict=True,
            return_tensors="pt",
            videos_kwargs=video_kwargs,
        )

        segments_label = torch.from_numpy(np.concat(segments_label))
        actions_count_label = torch.tensor(actions_count)
        video_mask = inputs_lm["input_ids"].eq(processor.video_token_id)

        # 遮盖输入部分
        text_label = inputs_lm["input_ids"].clone()
        text_label[text_label == pad_token_id] = -100

        for b in range(len(batch)):
            # 找到当前样本中所有 im_start_id 的位置索引
            start_indices = torch.where(inputs_lm["input_ids"][b] == im_start_id)[0]

            # Mask 掉从开头到 answer_start 之前的所有内容
            # 注意：这里 +1 是因为通常希望模型从 im_start_id 的下一个词开始预测
            answer_start = start_indices[-1] + 1
            text_label[b, :answer_start] = -100

        labels = BatchFeature(
            data=dict(
                text_label=text_label,
                actions_count_label=actions_count_label,
                segments_label=segments_label,
                num_frames=num_frames,
            ),
            tensor_type="pt",
        )

        return dict(
            inputs_lm=inputs_lm,
            video_mask=video_mask,
            labels=labels,
        )

    def _collator(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        messages = [i["messages"] for i in batch]
        actions_count = [i["actions_count"] for i in batch]
        actions_segments = [i["actions_segments"] for i in batch]

        try:
            inputs_lm, video_metadata = processor.apply_chat_template(
                messages,
                tokenize=True,
                padding=True,
                add_generation_prompt=False,
                return_dict=True,
                return_metadata=True,
                return_tensors="pt",
                videos_kwargs=VideosKwargs(num_frames=_sample_frames, fps=None),
                # videos_kwargs=VideosKwargs(fps=1),
            )
        except Exception as e:
            videos = [i["video_path"] for i in batch]
            raise RuntimeError(f"Processor failed to process batch: {e}, videos: {videos}")

        num_frames = []
        segments_label = []
        for seg, meta in zip(actions_segments, video_metadata):
            num_frames_piece = len(meta.frames_indices)  # 对标签根据采样帧数量进行处理
            sample_rate = len(meta.frames_indices) / meta.total_num_frames
            segment_processed = np.round(np.array(seg) * sample_rate).astype(int)
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

        for b in range(len(batch)):
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

        return dict(
            inputs_lm=inputs_lm,
            text_label=text_label,
            actions_count_label=actions_count_label,
            segments_label=segments_label,
            video_mask=video_mask,
            num_frames=num_frames,
        )

    return new_collator


class VideoActionDataset(Dataset):
    def __init__(self, dataset_root: Path | str, split: str, max_samples=None):
        super().__init__()
        assert split in ["train", "val", "test"], f"Unsupported split: {split}"
        self.split = split
        self.samples = self.load_samples_from_root(dataset_root)
        if max_samples:
            self.samples = random.sample(self.samples, k=max_samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:  # noqa: D401
        return self.samples[idx]
        # return self.load_sample(idx)

    def load_sample(self, idx: int) -> ActionSample:
        messages = self.samples[idx].build_messages_with_gt()
        actions_count = self.samples[idx].actions_count
        actions_segments = self.samples[idx].actions_segments

        return dict(
            messages=messages,
            actions_count=actions_count,
            actions_segments=actions_segments,
            video_path=self.samples[idx].video,
        )

    def load_samples_from_root(self, dataset_root: Path | str) -> List[ActionSample]:
        root = Path(dataset_root)
        task_info = root / "task_info/task_episode_key.json"
        observations_dir = root / self.split

        assert task_info.exists(), FileNotFoundError(f"Dataset root {dataset_root} must contain task_info")
        assert observations_dir.exists(), FileNotFoundError(
            f"Dataset root {dataset_root} must contain 'observations' directories."
        )
        task_info = json.loads(task_info.read_text(encoding="utf-8"))

        key_order = ["task_name", "init_scene_text", "label_info"]
        samples = []
        for ep in tqdm(list(observations_dir.rglob("**/head_color.mp4")), desc="Loading dataset..."):
            episode_id = ep.parts[-3]
            label = task_info.get(episode_id)
            ordered_record = OrderedDict((k, label[k]) for k in key_order)
            sample = ActionSample(video=str(ep.resolve()), label=ordered_record)
            samples.append(sample)

        if not samples:
            raise ValueError(f"No valid samples discovered under dataset root: {dataset_root}")

        return samples


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


def build_dataloader(
    processor,
    dataset_root,
    batch_size: int = 2,
    shuffle: bool = False,
    num_workers: int = 0,
) -> DataLoader:
    dataset = VideoActionDataset(dataset_root=dataset_root, max_samples=64)

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
    dataloader = build_dataloader(processor, "/mnt/e/observations_sub", num_workers=8)

    all_num_tokens = 0
    for i in range(3):
        for batch in tqdm(dataloader):
            all_num_tokens += batch["inputs_lm"]["input_ids"].size(1)
        print("Average num tokens:", all_num_tokens / len(dataloader.dataset))
        all_num_tokens = 0
