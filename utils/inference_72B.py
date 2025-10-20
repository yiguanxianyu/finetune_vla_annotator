import time
import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from transformers import AutoProcessor, Qwen2_5_VLProcessor, Qwen2_5_VLForConditionalGeneration
from vision_process import process_vision_info
import torch

if __name__ == "__main__":
    # default processer
    # from given config file
    # processor = Qwen2_5_VLProcessor.from_config(r"config\qwen25vl_7b.json")

    # default: Load the model on the available device(s)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-72B-Instruct-AWQ", dtype=torch.float16, device_map="auto"
    )

    # The default range for the number of visual tokens per image in the model is 4-16384.
    # You can set min_pixels and max_pixels according to your needs, such as a token range of 256-1280, to balance performance and cost.
    min_pixels = 256 * 28 * 28
    max_pixels = 1280 * 28 * 28
    processor = AutoProcessor.from_pretrained(
        "Qwen/Qwen2.5-VL-3B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels, use_fast=True
    )

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": "test_data/observations/616/843991/videos/head_color.mp4",
                },
                {"type": "text", "text": "Describe this video."},
            ],
        }
    ]

    # Preparation for inference
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
        **video_kwargs,
    )
    inputs = inputs.to("cuda").to(torch.float16)
    time1 = time.time()
    with torch.inference_mode():
        # Inference
        generated_ids = model.generate(**inputs, max_new_tokens=2048)
        generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
    print(output_text)
    print(f"Total time: {time.time() - time1}s")
