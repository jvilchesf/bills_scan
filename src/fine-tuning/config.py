# config.py - Updated with better hyperparameters to prevent overfitting

from typing import List
from pydantic_settings import BaseSettings


class Config(BaseSettings):
    # ---- W&B ----
    wb_experiment_name: str = ""
    wb_project_name: str = ""
    wb_enable: bool = True

    # ---- App / model ----
    modal_app_name: str = "lfm2-vl-crafted"
    model_app_name: str = "lfm2-vl-crafted"
    model_name: str = "LiquidAI/LFM2-VL-1.6B"

    # ---- Volumes / paths ----
    model_volume: str = "model"
    files_volume: str = "files"
    cache_volume: str = "cache"
    output_path: str = "model/lfm2-vl-crafted-model"

    # ---- App config ----
    gpu: str = "T4"
    max_hr_training: int = 2

    # ---- Focused LoRA Configuration for Image Processing ----
    lora_target_modules: List[str] = [
        "q_proj",
        "k_proj", 
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj", 
        "down_proj"
    ]
    lora_alpha: int = 32                    # Reduced back for stability
    lora_dropout: float = 0.15              # Increased dropout to prevent overfitting
    lora_rank: int = 32                     # Higher capacity for complex vision-text mapping

    # ---- Training mode ----
    training_mode: bool = True
    max_image_tokens: int = 1024

    # ---- Dataset ----
    hf_dataset: str = "jvilchesf/receipts-2025"

    # ---- OPTIMIZED HYPERPARAMETERS FOR BETTER ACCURACY ----
    num_epochs: int = 4                     # Reduced to prevent overfitting on small dataset
    batch_size: int = 1                     # Keep current (GPU memory constraint)
    grad_accum: int = 4                     # Keep current  
    learning_rate: float = 1.5e-5           # Balanced for learning without overfitting
    warmup_steps: int = 50                  # Increased for more stable warmup
    save_steps: int = 20                    # Must be multiple of eval_steps
    eval_steps: int = 20                    # Frequent evaluation to catch overfitting early

    # ---- Reduced Regularization ----
    weight_decay: float = 0.1               # Strong regularization to prevent overfitting
    max_grad_norm: float = 0.5              # Reduced for stability

    # ---- Early Stopping (Anti-Overfitting) ----
    early_stopping_patience: int = 2        # Stop quickly if validation stops improving
    early_stopping_threshold: float = 0.02  # More sensitive threshold

    # ---- Optimized Augmentation ----
    augmentation_enabled: bool = True
    augmentation_prob: float = 0.6          # Reduced from 0.8
    augmentations_per_sample: int = 3       # Increased augmentation for more variety
