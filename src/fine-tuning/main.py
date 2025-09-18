import modal
import wandb
import os
from datetime import datetime
from .model import import_model, zip_model
from .config import Config
from .dataset import get_dataset
from .train import train_model

# Config
config = Config()

# Create a modal app
app = modal.App(config.modal_app_name)

# Create image
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.0.0",
        "torchvision",
        "transformers>=4.55.0",
        "trl>=0.12.0",
        "datasets>=2.0.0",
        "accelerate>=0.33.0",
        "peft>=0.12.0",
        "huggingface-hub>=0.24.0",
        "pillow>=10.0.0",
        "pydantic-settings>=2.0.0",
        "bitsandbytes>=0.43.0",
        "hf-transfer>=0.1.9",
        "torchao>=0.13.0",
        "wandb>=0.21.4",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

# Create volumes
model_volume = modal.Volume.from_name(config.model_volume, create_if_missing=True)
files_volume = modal.Volume.from_name(config.files_volume, create_if_missing=True)
cache_volume = modal.Volume.from_name(config.cache_volume, create_if_missing=True)


# Use the image set on top of the main function
@app.function(
    image=image,
    volumes={
        "/model": model_volume,
        "/files": files_volume,
        "/cache": cache_volume,
    },
    gpu=config.gpu,
    secrets=[
        modal.Secret.from_name("huggingface-secret"),
        modal.Secret.from_name("wandb-secret"),
    ],
    timeout=60 * 60 * config.max_hr_training,
    memory=32768,
)
def fine_tune_lfm2_vl(config: Config):
    """
    Fine tuning image to text model
    """
    try:
        # Setup wandb if enabled
        # Set experiment name for wandb
        experiment_name = (
            f"lfm2_vl_crafted_exp_{datetime.now().strftime('%d_%m_%Y_%H_%M_%S')}"
        )

        project_name = config.model_app_name

        config.wb_experiment_name = experiment_name
        config.wb_project_name = project_name
        if config.wb_enable:
            # Login to wandb
            wandb_api_key = os.getenv("WANDB_API_KEY")
            if wandb_api_key:
                wandb.login(key=wandb_api_key)

                # Set environment variables for transformers integration
                os.environ["WANDB_PROJECT"] = config.wb_project_name
                os.environ["WANDB_LOG_MODEL"] = "checkpoint"

                # Initialize wandb
                wandb.init(
                    project=config.wb_project_name,
                    name=config.wb_experiment_name,
                    config=config.__dict__,
                )
                print(f"WandB initialized: {wandb.run.url}")
            else:
                print("Warning: WANDB_API_KEY not found, disabling wandb")
                config.wb_enable = False

        # Import model and processor from hf
        model, processor = import_model(config)
        # Import dataset
        train_dataset, test_dataset = get_dataset(config)
        # Train Model
        result = train_model(config, model, processor, train_dataset, test_dataset)
        if result:
            return {
                "status": "success",
                "model_path": config.output_path,
                "train_samples": len(train_dataset),
                "val_samples": len(test_dataset) if test_dataset else 0,
            }

        # Clean up experiment tracking
        if config.wb_enable:
            wandb.finish()
    except Exception as e:
        print(f"Error trying to run fine tune: {e}")
        # Raise an expception
        raise e


@app.local_entrypoint()
def main():
    """
    Entry point function for modal to call the process that really fine tune the model
    """

    config = Config()

    # Fine tune model
    print(
        f"Start fine tuning process at date: {datetime.now().strftime('%d-%m-%Y %H:%M:%S')}"
    )
    result = fine_tune_lfm2_vl.remote(config)
    if result:
        print(
            f"Result of training process: {result}, at: {datetime.now().strftime('%d-%m-%Y %H:%M:%S')}"
        )

    # Download model to local machine
    print("Download model to local device")
    model_data = download_personal_model.remote(config)
    if "model_zip" in model_data:
        with open("model/balanced_personal_receipt_model.zip", "wb") as f:
            f.write(model_data["model_zip"])
        print("Model downloaded as balanced_personal_receipt_model.zip")
    else:
        print("Error downloading model:", model_data.get("error", "Unknown error"))


if __name__ == "__main__":
    main()


@app.function(
    image=image,
    volumes={"/model": model_volume},
    timeout=300,
)
def download_personal_model(config: Config):
    """Download the trained personal receipt model"""

    # Go to the folder model in the modal image docker and get the data
    return zip_model(config)
