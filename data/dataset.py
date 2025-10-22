# -*- coding: utf-8 -*-
"""
数据集与数据加载器
==================
- 支持按清单加载多条样本，默认回退到示例样本。
- 与官方 Qwen-VL fine-tune 代码保持相同的消息/视频预处理流程。
"""

import warnings
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
import random
import torch
from scipy.ndimage import gaussian_filter1d
from torch.utils.data import DataLoader, Dataset
from transformers import AutoProcessor
from transformers.feature_extraction_utils import BatchFeature

try:
    from utils.vision_process import process_video_info_torchcodec
except ImportError:
    pass


def build_collator(processor):
    """Merge a list of samples into a single batched dict that the model can consume."""
    # pad_token_id = processor.tokenizer.pad_token_id
    im_start_id = processor.tokenizer.convert_tokens_to_ids("<|im_start|>")

    def _coll(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        bs = len(batch)
        texts = [item["text"] for item in batch]
        video_inputs = [item["video_input"] for item in batch]
        image_inputs = [item["image_input"] for item in batch]
        num_frames = [item["num_frames"] for item in batch]
        fps_list = [item["fps"] for item in batch]
        probs_start = torch.stack([item["probs_start"] for item in batch], dim=0)
        probs_end = torch.stack([item["probs_end"] for item in batch], dim=0)
        k_label = torch.cat([item["K_label"] for item in batch], dim=0)

        inputs = processor(
            text=texts,
            images=None if all(x is None for x in image_inputs) else image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            fps=fps_list,
        )

        video_mask = inputs["input_ids"].eq(processor.video_token_id)
        rows, cols = torch.where(inputs["input_ids"] == im_start_id)
        answer_start = cols.view(bs, -1)[:, -1] + 1
        text_label = inputs["input_ids"].clone()
        for b in range(bs):
            text_label[b, : answer_start[b]] = -100

        data = dict(
            num_frames=torch.tensor(num_frames),
            video_mask=video_mask,
            text_label=text_label,
            probs_start=probs_start,
            probs_end=probs_end,
            k_label=k_label,
        )

        if "answer" in batch[0]:
            # 评估时，保留答案部分用于计算精度
            data["answer_texts"] = processor.tokenizer([item["answer"] for item in batch])["input_ids"]

        return inputs, BatchFeature(data=data, tensor_type="pt")

    return _coll


DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant that helps people find the action segments in the video and answer the question "
    "based on the video."
)
DEFAULT_USER_INSTRUCTION = "Describe this video in json format."


@dataclass
class ActionSample:
    """Container describing a single training sample."""

    video: str
    assistant_text: str
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    user_instruction: str = DEFAULT_USER_INSTRUCTION
    extra_user_content: Optional[List[Dict[str, Any]]] = None
    data_root: Optional[str] = None

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "ActionSample":
        data_root = raw.get("data_path") or raw.get("data_root")
        video = raw.get("video") or raw.get("video_path")
        if video is None:
            # 兼容 messages 形式，自动解析首个 video 内容
            messages = raw.get("messages") or []
            for message in messages:
                if not isinstance(message.get("content"), list):
                    continue
                for element in message["content"]:
                    if element.get("type") == "video" or "video" in element:
                        video = element.get("video") or element.get("video_path")
                        break
                if video:
                    break
        if video is None:
            raise ValueError("Sample must provide a video path via 'video', 'video_path' or messages content.")

        assistant_text = raw.get("assistant_text") or raw.get("json_text")
        if assistant_text is None and "messages" in raw:
            for message in raw["messages"]:
                if message.get("role") == "assistant":
                    blocks = message.get("content") or []
                    for block in blocks:
                        if block.get("type") == "text" and isinstance(block.get("text"), str):
                            assistant_text = block["text"]
                            break
                    if assistant_text:
                        break
        if assistant_text is None:
            raise ValueError("Sample must include assistant supervision via 'assistant_text' or messages content.")

        system_prompt = raw.get("system_prompt") or raw.get("system") or DEFAULT_SYSTEM_PROMPT
        user_instruction = raw.get("user_instruction") or raw.get("instruction") or DEFAULT_USER_INSTRUCTION
        extra_user_content = raw.get("user_content")

        return cls(
            video=video,
            assistant_text=assistant_text,
            system_prompt=system_prompt,
            user_instruction=user_instruction,
            extra_user_content=extra_user_content,
            data_root=data_root,
        )

    def build_messages(self) -> List[Dict[str, Any]]:
        video_path = self._resolve_path(self.video)
        user_content: List[Dict[str, Any]] = [
            {"type": "video", "video": video_path},
            {"type": "text", "text": self.user_instruction},
        ]
        if self.extra_user_content:
            user_content.extend(self.extra_user_content)

        return [
            {
                "role": "system",
                "content": [{"type": "text", "text": self.system_prompt}],
            },
            {"role": "user", "content": user_content},
        ]

    def build_messages_with_gt(self) -> List[Dict[str, Any]]:
        messages = self.build_messages()
        messages.append(
            {
                "role": "assistant",
                "content": [{"type": "text", "text": self.assistant_text}],
            }
        )
        return messages

    def _resolve_path(self, path: str) -> str:
        if os.path.isabs(path) or self.data_root is None:
            return path
        return os.path.join(self.data_root, path)


def _load_samples_from_root(
    dataset_root: Optional[str],
) -> Optional[List[ActionSample]]:
    root = Path(dataset_root)
    task_info_dir = root / "task_info"
    observations_dir = root / "observations"

    if not task_info_dir.exists() or not observations_dir.exists():
        raise FileNotFoundError(f"Dataset root {dataset_root} must contain 'task_info' and 'observations' directories.")

    data = {}
    samples = []
    for ep in observations_dir.rglob("**/head_color.mp4"):
        parts = ep.parts
        task_id = parts[-4]
        episode_id = parts[-3]
        data.setdefault(task_id, {})[episode_id] = str(ep.resolve())

    for key, val in data.items():
        task_json = task_info_dir / f"task_{key}.json"
        records = json.loads(task_json.read_text(encoding="utf-8"))
        for record in records:
            episode_id = str(record.get("episode_id"))
            if (episode_id) not in val or episode_id is None:
                continue

            assistant_text = json.dumps(record, ensure_ascii=False)
            samples.append(
                ActionSample(
                    video=val[episode_id],
                    assistant_text=assistant_text,
                )
            )

    if not samples:
        raise ValueError(f"No valid samples discovered under dataset root: {dataset_root}")

    return samples


def _get_sample_test():
    warnings.warn(
        "Using fallback example samples – please provide a valid dataset_root for training.",
        UserWarning,
    )
    # 示例样本，回退用
    default_path = "test_data/observations/616/843991/videos/head_color.mp4"
    default_json = """
{
    "episode_id": 843991,
    "label_info": {
        "action_config": [
            {
                "start_frame": 37,
                "end_frame": 469,
                "action_text": "Fold the lower quarter of the green short-sleeve garment to the middle with both arms.",
                "skill": "Fold"
            },
            {
                "start_frame": 469,
                "end_frame": 888,
                "action_text": "Fold the upper quarter of the green short-sleeve garment to the middle with both arms.",
                "skill": "Fold"
            },
            {
                "start_frame": 888,
                "end_frame": 1746,
                "action_text": "Grasp the green short-sleeve collar with both hands and fold it down to the hem.",
                "skill": "Fold"
            }
        ]
    },
    "task_name": "Fold T-shirts",
    "init_scene_text": "Place the T-shirt face down on the bed/table with the neckline facing to the left."
}
"""
    return [
        ActionSample(
            video=default_path,
            assistant_text=default_json.strip(),
        )
    ] * 16


class VideoActionDataset(Dataset):
    """Dataset that mirrors官方 Qwen-VL fine-tuning预处理流程."""

    def __init__(
        self,
        processor: AutoProcessor,
        dataset_root: Optional[str] = None,
        max_samples=None,
    ):
        super().__init__()
        self.processor = processor
        self.video_token_id = self.processor.video_token_id

        self.samples = _load_samples_from_root(dataset_root)
        if self.samples is None:
            # 回退到兼容旧脚本的示例样本
            self.samples = _get_sample_test()
        if max_samples:
            self.samples = random.sample(self.samples, k=max_samples)

        self.nframes = 30

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:  # noqa: D401
        try:
            return self.load_sample(idx)
        except Exception as exc:  # noqa: BLE001
            idx = random.choice(range(len(self)))
            warnings.warn(
                f"Failed to load sample {idx}, retrying with a different one: {exc}",
                UserWarning,
            )
            return self.load_sample(idx)

    def load_sample(self, idx: int) -> ActionSample:
        messages = self.samples[idx].build_messages_with_gt()
        full_text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

        video_inputs, total_frames_list, fps_list = process_video_info_torchcodec(messages, nframes=self.nframes)
        video_input = video_inputs[0]
        num_frames = video_input.size(0)

        probs_start, probs_end, k_label = _build_segment_targets(
            assistant_text=self.samples[idx].assistant_text,
            total_timesteps=num_frames,
            n_fullframes=total_frames_list[0],
        )

        return dict(
            text=full_text,
            video_input=video_input,
            image_input=None,
            num_frames=num_frames,
            fps=fps_list[0],
            probs_start=probs_start,
            probs_end=probs_end,
            K_label=k_label,
        )


class VideoActionDatasetForEval(Dataset):
    """Dataset that mirrors官方 Qwen-VL fine-tuning预处理流程.不含目标输出"""

    def __init__(
        self,
        processor: AutoProcessor,
        dataset_root: Optional[str] = None,
        max_samples=None,
    ):
        super().__init__()
        self.processor = processor
        self.video_token_id = self.processor.video_token_id

        self.samples = _load_samples_from_root(dataset_root)
        if self.samples is None:
            # 回退到兼容旧脚本的示例样本
            self.samples = _get_sample_test()
        if max_samples:
            self.samples = random.sample(self.samples, k=max_samples)

        self.nframes = 30

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:  # noqa: D401
        try:
            return self.load_sample(idx)
        except Exception as exc:  # noqa: BLE001
            idx = random.choice(range(len(self)))
            warnings.warn(
                f"Failed to load sample {idx}, retrying with a different one: {exc}",
                UserWarning,
            )
            return self.load_sample(idx)

    def load_sample(self, idx: int) -> ActionSample:
        messages = self.samples[idx].build_messages()
        full_text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        answer = self.samples[idx].assistant_text

        video_inputs, total_frames_list, fps_list = process_video_info_torchcodec(messages, nframes=self.nframes)
        video_input = video_inputs[0]
        num_frames = video_input.size(0)

        probs_start, probs_end, k_label = _build_segment_targets(
            assistant_text=self.samples[idx].assistant_text,
            total_timesteps=num_frames,
            n_fullframes=total_frames_list[0],
        )

        return dict(
            text=full_text,
            answer=answer,
            video_input=video_input,
            image_input=None,
            num_frames=num_frames,
            fps=fps_list[0],
            probs_start=probs_start,
            probs_end=probs_end,
            K_label=k_label,
        )


def _build_segment_targets(
    assistant_text: str,
    total_timesteps: int,
    *,
    n_fullframes: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate boundary supervision from assistant JSON annotation."""

    try:
        annotation = json.loads(assistant_text)
    except json.JSONDecodeError as exc:  # noqa: F841
        raise ValueError("assistant_text must be a JSON string with label_info action_config.")

    configs: Iterable[Dict[str, Any]] = annotation.get("label_info", {}).get("action_config", [])
    configs = list(configs)
    if not configs:
        raise ValueError("annotation.label_info.action_config is empty – cannot build targets.")

    probs_start = torch.zeros(total_timesteps, dtype=torch.float32)
    probs_end = torch.zeros_like(probs_start)

    for seg in configs:
        start_frame = seg.get("start_frame")
        end_frame = seg.get("end_frame")
        if start_frame is None or end_frame is None:
            continue
        start_idx = int(round(start_frame * total_timesteps / max(n_fullframes, 1)))
        end_idx = int(round(end_frame * total_timesteps / max(n_fullframes, 1)))
        start_idx = min(max(start_idx, 0), total_timesteps - 1)
        end_idx = min(max(end_idx, 0), total_timesteps - 1)
        probs_start[start_idx] = 1.0
        probs_end[end_idx] = 1.0

    eps = 1e-8
    ps = gaussian_filter1d(probs_start.numpy(), sigma=1)
    pe = gaussian_filter1d(probs_end.numpy(), sigma=1)
    ps = ps / (ps.max() + eps)
    pe = pe / (pe.max() + eps)
    probs_start = torch.from_numpy(ps).float()
    probs_end = torch.from_numpy(pe).float()

    num_segments = max(len(configs), 1)
    k_label = torch.tensor([num_segments - 1], dtype=torch.long)

    return probs_start, probs_end, k_label


def build_dataloader(
    processor: AutoProcessor,
    *,
    dataset_root: Optional[str] = None,
    batch_size: int = 1,
    shuffle: bool = False,
    num_workers: int = 0,
) -> DataLoader:
    dataset = VideoActionDataset(processor=processor, dataset_root=dataset_root)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=build_collator(processor),
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )


def build_dataloader_for_eval(
    processor: AutoProcessor,
    *,
    dataset_root: Optional[str] = None,
    batch_size: int = 1,
    shuffle: bool = False,
    num_workers: int = 0,
) -> DataLoader:
    dataset = VideoActionDatasetForEval(processor=processor, dataset_root=dataset_root, max_samples=30)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=build_collator(processor),
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )


if __name__ == "__main__":
    # 简单测试数据集与数据加载器
    _load_samples_from_root("test_data")
