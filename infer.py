from models.qwen3_action_model import ActionSegmentationModel
from transformers import Qwen3VLProcessor
import torch


def build_model(args):
    processor = Qwen3VLProcessor.from_pretrained(args.hf_model, trust_remote_code=True, use_fast=True)
    model = ActionSegmentationModel.from_pretrained("/mnt/e/qwen3_action_segmentation_2B/checkpoint-90")
    return model, processor


def infer(model, processor, video_path: str):
    model.eval()

    inputs = processor.apply_cha
    with torch.inference_mode():
        outputs = model.generate(
            processor=processor,
            inputs_lm={k: v.cuda() for k, v in inputs_lm.items()},
            video_frames=inputs["video_frames"].cuda(),
            video_mask=inputs["video_mask"].cuda(),
            num_frames=inputs["num_frames"].cuda(),
            max_length=512,
            num_beams=5,
            temperature=1.0,
            top_p=0.9,
            repetition_penalty=1.2,
        )

    generated_text = processor.decode(outputs[0], skip_special_tokens=True)
    return generated_text
