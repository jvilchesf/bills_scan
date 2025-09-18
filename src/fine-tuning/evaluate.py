import modal
from pathlib import Path
import json

# Create Modal app
app = modal.App("lfm2-vl-receipts-evaluate")

# Define the container with necessary dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.0.0",
        "hf-transfer>=0.1.9",
        "torchvision",
        "transformers>=4.55.0",
        "datasets>=2.0.0",
        "accelerate>=0.33.0",
        "peft>=0.12.0",
        "huggingface-hub>=0.24.0",
        "pillow>=10.0.0",
        "bitsandbytes>=0.43.0",
        "scikit-learn>=1.3.0",
        "pandas>=2.0.0",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

# Use the same volumes from training
model_volume = modal.Volume.from_name("model", create_if_missing=False)
files_volume = modal.Volume.from_name("files", create_if_missing=False)
cache_volume = modal.Volume.from_name("cache", create_if_missing=False)


@app.function(
    image=image,
    gpu="T4",  # Same GPU as training
    volumes={
        "/model": model_volume,
        "/files": files_volume,
        "/cache": cache_volume,
    },
    timeout=3600,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    memory=32768,  # Same as training
)
def evaluate_model(num_samples: int = 0, save_predictions: bool = True):
    """Evaluate the fine-tuned LFM2-VL-450M model on receipt test set"""
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from datasets import load_dataset
    from PIL import Image, ImageEnhance
    import io
    from peft import PeftModel
    import time
    import pandas as pd
    import re

    def enhance_image(image):
        """EXACT same enhancement as training"""
        if image.mode != "RGB":
            image = image.convert("RGB")
        # Add contrast enhancement for receipts
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.2)
        return image

    def clean_json_output(text):
        """Clean the generated text to extract valid JSON"""
        # Remove any text before the first {
        if "{" in text:
            start_idx = text.find("{")
            text = text[start_idx:]

        # Remove any text after the last }
        if "}" in text:
            end_idx = text.rfind("}") + 1
            text = text[:end_idx]

        # Remove common artifacts
        text = text.replace("```json", "").replace("```", "")
        text = text.replace("<|im_end|>", "").replace("<|endoftext|>", "")
        text = text.replace("</s>", "").replace("<s>", "")

        # Fix trailing commas
        text = re.sub(r",(\s*[}\]])", r"\1", text)

        return text.strip()

    def calculate_metrics(predictions, ground_truths):
        """Calculate evaluation metrics"""
        metrics = {
            "valid_json_rate": 0,
            "exact_match": 0,
            "store_name_accuracy": 0,
            "date_accuracy": 0,
            "total_accuracy": 0,
            "currency_accuracy": 0,
            "items_count_accuracy": 0,
            "partial_match": 0,
        }

        for pred, gt in zip(predictions, ground_truths):
            # Check if prediction is valid JSON
            try:
                pred_json = json.loads(clean_json_output(pred))
                metrics["valid_json_rate"] += 1

                # Parse ground truth
                gt_json = json.loads(gt)

                # Exact match
                if pred_json == gt_json:
                    metrics["exact_match"] += 1

                # Field-by-field comparison
                if (
                    pred_json.get("store_name", "").lower()
                    == gt_json.get("store_name", "").lower()
                ):
                    metrics["store_name_accuracy"] += 1

                if pred_json.get("date") == gt_json.get("date"):
                    metrics["date_accuracy"] += 1

                if pred_json.get("total") == gt_json.get("total"):
                    metrics["total_accuracy"] += 1

                if pred_json.get("currency") == gt_json.get("currency"):
                    metrics["currency_accuracy"] += 1

                # Items count (with tolerance)
                pred_items = len(pred_json.get("items", []))
                gt_items = len(gt_json.get("items", []))
                if abs(pred_items - gt_items) <= 1:
                    metrics["items_count_accuracy"] += 1

                # Partial match (at least 3 fields correct)
                correct_fields = sum(
                    [
                        pred_json.get("store_name", "").lower()
                        == gt_json.get("store_name", "").lower(),
                        pred_json.get("date") == gt_json.get("date"),
                        pred_json.get("total") == gt_json.get("total"),
                        pred_json.get("currency") == gt_json.get("currency"),
                    ]
                )
                if correct_fields >= 3:
                    metrics["partial_match"] += 1

            except json.JSONDecodeError:
                continue
            except Exception as e:
                print(f"Error processing sample: {e}")
                continue

        # Convert to percentages
        n = len(predictions)
        for key in metrics:
            metrics[key] = (metrics[key] / n) * 100 if n > 0 else 0

        return metrics

    def load_model_and_processor():
        """Load the fine-tuned model with LoRA adapter"""
        model_id = "LiquidAI/LFM2-VL-450M"
        lora_path = "/model/lfm2-vl-crafted-model"

        # Check if LoRA adapter exists
        if not Path(lora_path).exists():
            raise FileNotFoundError(
                f"Model not found at {lora_path}. Please train the model first."
            )

        print("Loading processor...")
        processor = AutoProcessor.from_pretrained(
            lora_path,
            trust_remote_code=True,
            # Don't specify max_image_tokens to use default
        )

        print("Loading base model...")
        base_model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto",
            low_cpu_mem_usage=True,
            max_memory={0: "36GiB"},
        )

        print("Loading LoRA adapter...")
        model = PeftModel.from_pretrained(
            base_model,
            lora_path,
            torch_dtype=torch.bfloat16,
        )

        # Set to evaluation mode
        model.eval()

        print(f"Model loaded successfully!")
        if torch.cuda.is_available():
            print(
                f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB"
            )

        return model, processor

    def run_inference(model, processor, image):
        """Run inference on a single image - EXACT same as training"""
        # EXACT same system message as training
        system_message = (
            "Extract receipt information as JSON. Output only valid JSON with these fields:\n"
            '{"store_name": "", "date": "YYYY-MM-DD", "items": [], "total": "", "currency": "SEK"}\n'
            "Extract exactly what you see. Use empty string for unclear text."
        )

        user_prompt = "Extract the receipt information from this image. Output ONLY a JSON object with store_name, date, items, total, and currency."

        # Format messages EXACTLY like training
        messages = [
            {"role": "system", "content": [{"type": "text", "text": system_message}]},
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": user_prompt},
                ],
            },
        ]

        # Apply chat template
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Process inputs
        inputs = processor(
            text=text, images=[image], return_tensors="pt", padding=True
        ).to(model.device)

        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,  # Match training max_length
                temperature=0.1,  # Low temperature for consistency
                do_sample=False,
                num_beams=2,  # Add beam search
                top_p=0.9,
                repetition_penalty=1.0,
                pad_token_id=processor.tokenizer.pad_token_id,
                eos_token_id=processor.tokenizer.eos_token_id,
            )

        # Decode output
        generated_text = processor.decode(outputs[0], skip_special_tokens=True)

        # Extract assistant response
        if "assistant" in generated_text.lower():
            parts = generated_text.split("assistant")
            if len(parts) > 1:
                generated_text = parts[-1].strip()

        return clean_json_output(generated_text)

    print("=" * 60)
    print("Starting Receipt Model Evaluation")
    print("=" * 60)

    # Load model and processor
    model, processor = load_model_and_processor()

    # Load test dataset
    print("\nLoading test dataset...")
    dataset = load_dataset("jvilchesf/receipts-2025")
    test_dataset = dataset.get("test", dataset["train"])  # Fallback to train if no test

    # Limit samples if specified
    if num_samples > 0 and num_samples < len(test_dataset):
        test_dataset = test_dataset.select(range(num_samples))

    print(f"Evaluating on {len(test_dataset)} samples...")

    # Run evaluation
    predictions = []
    ground_truths = []
    inference_times = []

    for idx, sample in enumerate(test_dataset):
        print(f"\rProcessing sample {idx + 1}/{len(test_dataset)}", end="")

        # Get image
        image = sample["image"]
        if isinstance(image, dict) and "bytes" in image:
            image = Image.open(io.BytesIO(image["bytes"]))

        # Apply same enhancement as training
        image = enhance_image(image)

        # Measure inference time
        start_time = time.time()
        prediction = run_inference(model, processor, image)
        inference_time = time.time() - start_time

        predictions.append(prediction)

        # Create ground truth
        gt = {
            "store_name": sample.get("store_name", ""),
            "date": sample.get("date", ""),
            "items": sample.get("items", []),
            "total": sample.get("total", ""),
            "currency": sample.get("currency", "SEK"),
        }
        ground_truths.append(json.dumps(gt, ensure_ascii=False))
        inference_times.append(inference_time)

        # Show sample predictions periodically
        if (idx + 1) % 5 == 0:
            print(f"\n\nSample {idx + 1}:")
            print(f"Prediction: {prediction[:100]}...")
            print(f"Ground Truth: {ground_truths[-1][:100]}...")
            print(f"Inference time: {inference_time:.2f}s")

    print("\n\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)

    # Calculate metrics
    metrics = calculate_metrics(predictions, ground_truths)
    metrics["avg_inference_time"] = sum(inference_times) / len(inference_times)
    metrics["total_samples"] = len(test_dataset)

    # Display results
    print("\nPerformance Metrics:")
    print("-" * 40)
    print(f"Total Samples:          {metrics['total_samples']}")
    print(f"Valid JSON Rate:        {metrics['valid_json_rate']:.1f}%")
    print(f"Exact Match:            {metrics['exact_match']:.1f}%")
    print(f"Partial Match (3/4):    {metrics['partial_match']:.1f}%")
    print()
    print("Field Accuracy:")
    print(f"  Store Name:           {metrics['store_name_accuracy']:.1f}%")
    print(f"  Date:                 {metrics['date_accuracy']:.1f}%")
    print(f"  Total:                {metrics['total_accuracy']:.1f}%")
    print(f"  Currency:             {metrics['currency_accuracy']:.1f}%")
    print(f"  Items Count (±1):     {metrics['items_count_accuracy']:.1f}%")
    print()
    print(f"Avg Inference Time:     {metrics['avg_inference_time']:.2f}s")

    # Save results if requested
    if save_predictions:
        results = {
            "metrics": metrics,
            "predictions": predictions[:10],  # Save first 10 for inspection
            "ground_truths": ground_truths[:10],
            "num_samples": len(test_dataset),
            "model_path": "/model/lfm2-vl-crafted-model",
        }

        output_path = Path("/files/evaluation_results.json")
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_path}")

        # Save detailed predictions
        df = pd.DataFrame(
            {
                "prediction": predictions,
                "ground_truth": ground_truths,
                "inference_time": inference_times,
            }
        )
        csv_path = Path("/files/detailed_predictions.csv")
        df.to_csv(csv_path, index=False)
        print(f"Detailed predictions saved to: {csv_path}")

    return metrics


@app.function(
    image=image,
    volumes={"/files": files_volume},
    timeout=300,
)
def download_results():
    """Download evaluation results"""
    import zipfile
    import tempfile

    results_files = [
        Path("/files/evaluation_results.json"),
        Path("/files/detailed_predictions.csv"),
    ]

    existing_files = [f for f in results_files if f.exists()]

    if not existing_files:
        return {"error": "No evaluation results found"}

    # Create a zip file of results
    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp_file:
        with zipfile.ZipFile(tmp_file.name, "w", zipfile.ZIP_DEFLATED) as zipf:
            for file_path in existing_files:
                zipf.write(file_path, file_path.name)

        with open(tmp_file.name, "rb") as f:
            zip_content = f.read()

    return {"results_zip": zip_content}


@app.local_entrypoint()
def main(num_samples: int = 0, save_predictions: bool = True, download: bool = True):
    """
    Evaluate the trained receipt model

    Args:
        num_samples: Number of test samples to evaluate (0 for all)
        save_predictions: Whether to save predictions to file
        download: Whether to download results locally
    """
    print("Starting Receipt Model Evaluation on Modal...")

    # Run evaluation
    metrics = evaluate_model.remote(
        num_samples=num_samples, save_predictions=save_predictions
    )

    print("\nFinal Metrics Summary:")
    print("-" * 40)
    print(f"Valid JSON Rate:        {metrics['valid_json_rate']:.1f}%")
    print(f"Exact Match:            {metrics['exact_match']:.1f}%")
    print(f"Partial Match:          {metrics['partial_match']:.1f}%")
    print(f"Store Name Accuracy:    {metrics['store_name_accuracy']:.1f}%")
    print(f"Total Accuracy:         {metrics['total_accuracy']:.1f}%")
    print(f"Avg Inference Time:     {metrics['avg_inference_time']:.2f}s")

    # Download results if requested
    if download and save_predictions:
        print("\nDownloading evaluation results...")
        results_data = download_results.remote()

        if "results_zip" in results_data:
            with open("receipt_evaluation_results.zip", "wb") as f:
                f.write(results_data["results_zip"])
            print("Results downloaded as receipt_evaluation_results.zip")
        else:
            print("Error:", results_data.get("error", "Unknown error"))


if __name__ == "__main__":
    import sys

    num_samples = 0  # Default to all samples
    save = True
    download = True

    if len(sys.argv) > 1:
        num_samples = int(sys.argv[1])
    if len(sys.argv) > 2:
        save = sys.argv[2].lower() == "true"
    if len(sys.argv) > 3:
        download = sys.argv[3].lower() == "true"

    main(num_samples=num_samples, save_predictions=save, download=download)
