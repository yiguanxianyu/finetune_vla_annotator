from peft import get_peft_model, LoraConfig


def load_lora_model(base_model: str, args):
    lora_config = LoraConfig(
        r=16,  # 秩
        lora_alpha=16,  # alpha值
        lora_dropout=0.1,
        inference_mode=False,
        task_type="CAUSAL_LM",  # 任务类型
    )

    peft_model = get_peft_model(base_model, lora_config)
    peft_model.print_trainable_parameters()

    return peft_model
