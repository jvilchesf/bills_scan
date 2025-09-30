# dataset.py - Updated with CORRECT receipt schema
from datasets import load_dataset
from pathlib import Path
from .config import Config
from .augmentation import ReceiptAugmenter
import json


def get_dataset(config: Config):
    """
    Load and prepare the dataset with proper receipt schema
    """
    cache_dir = Path("/cache/datasets")
    cache_dir.mkdir(exist_ok=True, parents=True)

    # Cache paths
    train_cache_path = cache_dir / "train_augmented_v2.arrow"
    test_cache_path = cache_dir / "test_processed_v2.arrow"

    # Force refresh to use new schema and prompts
    force_refresh = True  # SET TO TRUE to regenerate with vision-focused prompts

    if not force_refresh and train_cache_path.exists() and test_cache_path.exists():
        print("Loading cached augmented datasets...")
        from datasets import Dataset

        train_dataset = Dataset.from_file(str(train_cache_path))
        test_dataset = Dataset.from_file(str(test_cache_path))
        print(
            f"Loaded cached datasets - train: {len(train_dataset)}, test: {len(test_dataset)}"
        )
        return train_dataset, test_dataset

    # Load raw dataset
    print("Loading raw datasets...")
    dataset = load_dataset(config.hf_dataset)

    # Split dataset
    if "train" in dataset and "test" in dataset:
        train_dataset = dataset["train"]
        test_dataset = dataset["test"]
    else:
        # Create train/test split if not present
        dataset = dataset["train"].train_test_split(test_size=0.15, seed=42)
        train_dataset = dataset["train"]
        test_dataset = dataset["test"]

    print(
        f"Original dataset sizes - train: {len(train_dataset)}, test: {len(test_dataset)}"
    )

    # Initialize augmenters
    train_augmenter = ReceiptAugmenter(
        training_mode=True, augmentation_prob=config.augmentation_prob
    )
    test_augmenter = ReceiptAugmenter(training_mode=False, augmentation_prob=0.0)

    # Create augmented versions of training data
    print("Creating augmented training samples...")
    augmented_samples = []

    # Reduced augmentation to prevent extreme overfitting
    num_augmentations_per_sample = config.augmentations_per_sample  # Reduced from 3

    for idx, sample in enumerate(train_dataset):
        # Add original sample (with basic preprocessing)
        original_sample = sample.copy()
        original_sample["image"] = test_augmenter(sample["image"])
        augmented_samples.append(original_sample)

        # Add augmented versions
        for aug_idx in range(num_augmentations_per_sample):
            aug_sample = sample.copy()
            aug_sample["image"] = train_augmenter(sample["image"])
            aug_sample["is_augmented"] = True
            aug_sample["augmentation_idx"] = aug_idx
            augmented_samples.append(aug_sample)

        if (idx + 1) % 10 == 0:
            print(f"Processed {idx + 1}/{len(train_dataset)} samples...")

    # Create new dataset from augmented samples
    from datasets import Dataset

    train_dataset = Dataset.from_list(augmented_samples)
    print(f"Augmented training dataset size: {len(train_dataset)}")

    # Process test dataset (no augmentation, just preprocessing)
    print("Processing test dataset...")

    def preprocess_test_sample(sample):
        sample["image"] = test_augmenter(sample["image"])
        return sample

    test_dataset = test_dataset.map(
        preprocess_test_sample, desc="Preprocessing test images"
    )

    # Format datasets with CORRECT schema
    print("Formatting datasets with proper receipt schema...")

    def format_for_training(sample):
        """Format sample with COMPLETE receipt structure"""

        # SPECIFIC system message forcing image attention
        system_message = "You must carefully examine the receipt image and extract the EXACT text visible. Return JSON with: store_name (exact text from image), date (YYYY-MM-DD format from image), items array (actual items from receipt), total (exact amount from image), currency (SEK/EUR/USD from image). DO NOT make up generic values - read the actual receipt."

        user_prompt = "Parse this receipt."  # Build the expected output - MATCHING THE ACTUAL DATA STRUCTURE
        items = sample.get("items", [])

        # Ensure items is a list
        if not isinstance(items, list):
            items = []

        # Format items properly
        formatted_items = []
        for item in items:
            if isinstance(item, dict):
                formatted_items.append(
                    {
                        "name": item.get("name", ""),
                        "quantity": str(item.get("quantity", "1")),
                        "unit_price": str(item.get("unit_price", "0")),
                        "total_price": str(item.get("total_price", "0")),
                    }
                )

        # Build complete response
        assistant_response = {
            "store_name": sample.get("store_name", ""),
            "date": sample.get("date", ""),
            "items": formatted_items,
            "total": str(sample.get("total", "")),
            "currency": sample.get("currency", "SEK"),
        }

        # Convert to JSON string
        assistant_text = json.dumps(assistant_response, ensure_ascii=False, indent=2)

        # Build messages
        messages = [
            {"role": "system", "content": [{"type": "text", "text": system_message}]},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": sample["image"]},
                    {"type": "text", "text": user_prompt},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": assistant_text}],
            },
        ]

        sample["messages"] = messages
        return sample

    # Apply formatting
    train_dataset = train_dataset.map(
        format_for_training, desc="Formatting training data with full schema"
    )

    test_dataset = test_dataset.map(
        format_for_training, desc="Formatting test data with full schema"
    )

    # Verify formatting by checking a sample
    print("\nSample formatted message structure:")
    sample = train_dataset[0]
    if "messages" in sample:
        for msg in sample["messages"]:
            if msg["role"] == "assistant":
                content = msg["content"][0]["text"] if msg["content"] else ""
                print(f"Assistant response preview (first 500 chars):")
                print(content[:500])
                break

    # Save to cache
    print("\nSaving formatted datasets to cache...")
    train_dataset.save_to_disk(str(train_cache_path))
    test_dataset.save_to_disk(str(test_cache_path))

    print(
        f"Final dataset sizes - train: {len(train_dataset)}, test: {len(test_dataset)}"
    )

    return train_dataset, test_dataset
