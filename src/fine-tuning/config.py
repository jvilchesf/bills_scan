from typing import List
from pydantic_settings import BaseSettings


class Config(BaseSettings):
    # ---- W&B ----
    wb_experiment_name: str = ""
    wb_project_name: str = ""
    wb_enable: bool = True

    # ---- App / model ----
    # Your current field:
    modal_app_name: str = "lfm2-vl-crafted"
    model_app_name: str = "lfm2-vl-crafted"
    model_name: str = "LiquidAI/LFM2-VL-450M"

    # ---- Volumes / paths ----
    # Your current fields:
    model_volume: str = "model"
    files_volume: str = "files"
    cache_volume: str = "cache"

    # A single, canonical output path used by training
    output_path: str = "model/lfm2-vl-crafted-model"

    # ---- App config ----
    gpu: str = "T4"
    max_hr_training: int = 2

    # ---- LoRA ----
    lora_target_modules: List[str] = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
    ]  # lora_target_modules: List[str] = ["q_proj", "k_proj"]
    lora_alpha: int = 32  # 2:1 ratio with r
    lora_dropout: float = 0.05  # low dropout for small dataset
    lora_rank: int = 16  # balanced rank

    # Collate function
    trainig_mode: bool = True  # define if it is necessary to enchance image

    # Import model
    max_image_tokens: int = 1024

    # ---- Dataset ----
    hf_dataset: str = "jvilchesf/receipts-2025"

    # ---- Fine-tuning hyperparams ----
    num_epochs: int = 30
    batch_size: int = 1
    grad_accum: int = 16  # effective batch = 4
    learning_rate: float = 2e-5
    warmup_steps: int = 10
    save_steps: int = 20
