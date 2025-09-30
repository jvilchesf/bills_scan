from transformers import AutoModelForImageTextToText, AutoProcessor
import torch
from peft import LoraConfig, get_peft_model
from .config import Config
from pathlib import Path


def import_model(config: Config):
    # Import Processor that helps to create batch of datasets
    processor = AutoProcessor.from_pretrained(
        config.model_name,
        trust_remote_code=True,
        max_image_tokens=config.max_image_tokens,
    )

    # Import model
    model = AutoModelForImageTextToText.from_pretrained(
        config.model_name,  # The model identifier (e.g., "microsoft/Florence-2-large")
        torch_dtype=torch.bfloat16,  # Sets model weights to bfloat16 precision to reduce memory usage
        # while maintaining good numerical stability
        trust_remote_code=True,  # Allows execution of custom code from the model repository
        # Required for models with custom modeling code
        device_map="auto",  # Automatically distributes model layers across available devices
        # (CPU/GPU) for optimal memory usage and performance
        low_cpu_mem_usage=True,  # Minimizes CPU RAM usage during model loading by loading
        # weights directly to target device when possible
        max_memory={0: "40GiB"},  # Sets maximum memory limit for device 0 (first GPU)
        # Prevents out-of-memory errors by capping GPU usage
    )

    # peft = parameters efficient fine tunning
    # BALANCED configuration: good capacity with reasonable speed
    peft_config = LoraConfig(
        lora_alpha=config.lora_alpha,  # Scaling factor for LoRA weights (controls learning rate scaling)
        # Higher values make LoRA updates more influential
        # 2:1 ratio with r maintains stable training dynamics
        lora_dropout=config.lora_dropout,  # Dropout probability applied to LoRA layers during training
        # Low value (5%) reduces overfitting risk while preserving learning
        # Especially important for small datasets
        r=config.lora_rank,  # Rank of LoRA adaptation matrices (controls model capacity)
        # Higher rank = more parameters = better adaptation but slower training
        # 16 is balanced: 4x more capacity than r=4, 2x less than r=32
        bias="none",  # Whether to adapt bias parameters ("none", "all", or "lora_only")
        # "none" keeps original biases frozen, focusing adaptation on weights
        target_modules=config.lora_target_modules,  # List of module names to apply LoRA adaptation to
        # Typically attention layers (q_proj, v_proj, etc.)
        task_type="CAUSAL_LM",  # Type of task for the model (affects how LoRA is applied)
        # "CAUSAL_LM" for autoregressive language generation tasks
    )

    # put together lora config + model
    model = get_peft_model(model, peft_config)

    # print trainable parameters
    model.print_trainable_parameters()
    return model, processor


def zip_model(config: Config):
    import zipfile
    import tempfile

    # The model is saved with absolute path in config
    model_dir = Path(f"/{config.output_path}")

    if not model_dir.exists():
        # Try without leading slash
        model_dir = Path(config.output_path)

    if not model_dir.exists():
        return {"error": f"No model found at {config.output_path}"}
        # Create a zip file of the model
    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp_file:
        with zipfile.ZipFile(tmp_file.name, "w", zipfile.ZIP_DEFLATED) as zipf:
            for file_path in model_dir.rglob("*"):
                if file_path.is_file():
                    zipf.write(file_path, file_path.relative_to(model_dir))

        with open(tmp_file.name, "rb") as f:
            zip_content = f.read()

    return {"model_zip": zip_content}
