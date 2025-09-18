from .config import Config
from datasets import load_dataset, Dataset
from PIL import Image, ImageEnhance
import io
import json
import os
import math

# ANTI-HALLUCINATION system message
system_message = (
    "Extract receipt information as JSON. Output only valid JSON with these fields:\n"
    '{"store_name": "", "date": "YYYY-MM-DD", "items": [], "total": "", "currency": "SEK"}\n'
    "Extract exactly what you see. Use empty string for unclear text."
)


def get_dataset(config: Config):
    """
    Download dataset from hugging face with optimized caching
    """
    try:
        dataset_key = config.hf_dataset.replace("/", "-")
        # Use the mounted volume instead of /tmp
        train_cache = f"/{config.cache_volume}/formatted_train_{dataset_key}"
        test_cache = f"/{config.cache_volume}/formatted_test_{dataset_key}"

        # Check if cache directories exist
        if os.path.exists(train_cache) and os.path.exists(test_cache):
            print("Loading cached datasets...")
            train_dataset = Dataset.load_from_disk(train_cache)
            test_dataset = Dataset.load_from_disk(test_cache)
            print(f"train dataset length: {len(train_dataset)}")
            print(f"test dataset length: {len(test_dataset)}")
            return train_dataset, test_dataset

        print("📥 Loading raw datasets...")
        # Load raw datasets
        train_dataset = load_dataset(
            config.hf_dataset,
            split="train",
            streaming=False,
        )
        test_dataset = load_dataset(
            config.hf_dataset,
            split="test",
            streaming=False,
        )

        print(f"train dataset length: {len(train_dataset)}")
        print(f"test dataset length: {len(test_dataset)}")

        # Process datasets with optimizations
        print("🔄 Processing train dataset...")
        train_dataset = train_dataset.map(
            format_dataset,
            num_proc=4,  # Parallel processing
            load_from_cache_file=True,
            desc="Formatting train data",
        )

        print("🔄 Processing test dataset...")
        test_dataset = test_dataset.map(
            format_dataset,
            num_proc=4,  # Parallel processing
            load_from_cache_file=True,
            desc="Formatting test data",
        )

        # Save processed datasets for future use
        print("💾 Saving processed datasets to cache...")
        train_dataset.save_to_disk(train_cache)
        test_dataset.save_to_disk(test_cache)

        print("✅ Processing complete - future runs will be instant!")

        return train_dataset, test_dataset

    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None, None


def format_dataset(sample):
    """
    Format sample for training - optimized for speed
    """
    enhanced_image = enhance_receipt_image(sample["image"])
    # Create target JSON efficiently
    target_json = {
        "store_name": sample.get("store_name", ""),
        "date": sample.get("date", ""),
        "items": sample.get("items", []),
        "total": sample.get("total", ""),
        "currency": sample.get("currency", "SEK"),
    }

    # Return formatted messages
    return {
        "messages": [
            {"role": "system", "content": [{"type": "text", "text": system_message}]},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": enhanced_image},  # Keep image as-is
                    {
                        "type": "text",
                        "text": "Extract the receipt information from this image. Output ONLY a JSON object with store_name, date, items, total, and currency.",
                    },
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps(target_json, ensure_ascii=False),
                    }
                ],
            },
        ]
    }


def enhance_receipt_image(image):
    """
    Enhance image for better OCR performance.
    Applied once during dataset preparation.
    """
    # Convert bytes to PIL Image if needed
    if isinstance(image, dict) and "bytes" in image:
        image = Image.open(io.BytesIO(image["bytes"]))
    elif isinstance(image, bytes):
        image = Image.open(io.BytesIO(image))
    elif not hasattr(image, "convert"):  # Not a PIL Image
        try:
            image = Image.open(io.BytesIO(image))
        except:
            print("Warning: Could not process image")
            return image

    # Ensure RGB
    if image.mode != "RGB":
        image = image.convert("RGB")

    image = image.resize(
        (512, 512), Image.LANCZOS
    )  # Enhance contrast for better text visibility
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(1.3)

    # Enhance sharpness
    enhancer = ImageEnhance.Sharpness(image)
    image = enhancer.enhance(1.2)

    # Optional: Enhance brightness slightly for dark receipts
    enhancer = ImageEnhance.Brightness(image)
    image = enhancer.enhance(1.1)

    return image
