from peft import LoraConfig, get_peft_model
import torch.nn as nn

def freeze_all_params(model: nn.Module):
    for p in model.parameters():
        p.requires_grad = False

def add_lora_to_cdit(model: nn.Module, r=16, alpha=32, dropout=0.05):
    lora_config = LoraConfig(
    r=r,
    lora_alpha=alpha,
    lora_dropout=dropout,
    bias="none",
    target_modules=[
        "qkv",
        "proj",
        "out_proj",
        # "fc1",
        # "fc2",
    ],
    init_lora_weights=True,
    task_type="FEATURE_EXTRACTION", 
)

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model