# 🧾 Receipt OCR Fine-Tuning with LFM2-VL-1.6B  

A production-ready fine-tuning pipeline for training the **LiquidAI LFM2-VL-1.6B** vision-language model on **Swedish receipt OCR tasks** using **Modal cloud infrastructure**.  

---

## 📋 Overview  
This project implements a complete end-to-end pipeline for fine-tuning a vision-language model to extract structured data from receipt images. It uses **Parameter-Efficient Fine-Tuning (PEFT)** with **LoRA adapters** to optimize the model for receipt OCR while maintaining efficiency and preventing overfitting.  

### Key Features
- 🚀 **Cloud-Native**: Runs entirely on Modal cloud infrastructure  
- 🎯 **PEFT/LoRA**: Memory-efficient fine-tuning using LoRA adapters  
- 📊 **W&B Integration**: Real-time training metrics and experiment tracking  
- 🖼️ **Image Enhancement**: Advanced preprocessing with data augmentation  
- 💾 **Smart Caching**: Preprocessed datasets cached for faster iterations  
- 🔍 **Comprehensive Evaluation**: Detailed accuracy metrics and predictions
- 🛡️ **Anti-Overfitting**: Strong regularization and early stopping for small datasets  

---

## 🏗️ Architecture  
The pipeline consists of three main components:  
1. **Data Processing**: Loads receipt images and annotations, applies enhancement, formats for training  
2. **Model Training**: Fine-tunes LFM2-VL-1.6B using LoRA adapters with optimized hyperparameters  
3. **Evaluation**: Tests model performance on held-out data with detailed metrics  

---

## 📁 Project Structure  

```
src/
└── fine-tuning/
    ├── __init__.py
    ├── config.py           # Centralized configuration management
    ├── dataset.py          # Dataset loading and preprocessing
    ├── model.py            # Model initialization and LoRA setup
    ├── collate_fn.py       # Batch collation and label masking
    ├── train.py            # Training loop implementation
    ├── main.py             # Training entry point
    └── evaluate.py         # Model evaluation pipeline
```

---

## 📄 File Descriptions  

### `config.py`  
Manages all hyperparameters and configuration settings using Pydantic:  
- Model selection and LoRA parameters  
- Training hyperparameters (learning rate, batch size, epochs)  
- Modal infrastructure settings (GPU type, volumes)  
- W&B experiment tracking configuration  

### `dataset.py`  
Handles data preparation and caching:  
- Downloads receipt dataset from HuggingFace  
- Applies image enhancement (resize, contrast, sharpness)  
- Formats data into conversation format for training  
- Implements smart caching to avoid reprocessing  

### `model.py`  
Sets up the model architecture:  
- Loads base LFM2-VL-1.6B model  
- Configures LoRA adapters for efficient fine-tuning  
- Manages model serialization and downloading  

### `collate_fn.py`  
Prepares batches for training:  
- Processes images and text through model processor  
- Creates masked labels (only trains on assistant responses)  
- Handles tokenization and padding  

### `train.py`  
Implements the training loop:  
- Configures SFTTrainer with optimized settings  
- Manages checkpointing and model saving  
- Integrates with W&B for metrics tracking  

### `main.py`  
Main entry point for training:  
- Orchestrates the complete training pipeline  
- Manages Modal app lifecycle  
- Downloads trained model to local storage  

### `evaluate.py`  
Comprehensive evaluation system:  
- Loads fine-tuned model with LoRA weights  
- Runs inference on test dataset  
- Calculates detailed accuracy metrics  
- Exports results as JSON and CSV  

---

## 🚀 Usage

### Prerequisites

1. Install dependencies:
```bash
pip install modal

2. Set up Modal account and authenticate

```bash
modal setup
```

---

3. Configure secrets in Modal dashboard

- `huggingface-secret`: Your HuggingFace API token  
- `wandb-secret`: Your Weights & Biases API key

---

## 🏋️ Training

Run the training pipeline:

```bash
uv run modal run -m src.fine-tuning.main
```

This will:

- Download and preprocess the receipt dataset  
- Fine-tune the model for configured epochs  
- Save checkpoints to Modal volumes  
- Download final model as `balanced_personal_receipt_model.zip`

---

## ✅ Evaluation

Evaluate the trained model:

```bash
uv run modal run -m src.fine-tuning.evaluate
```

**Options:**

- `num_samples`: Number of test samples (`0` for all)  
- `save_predictions`: Save detailed results (default: `True`)  
- `download`: Download results locally (default: `True`)

---

## ⚙️ Optimized Configuration

Key hyperparameters in `config.py` (optimized for small datasets):

```python
# Model Architecture
model_name: str = "LiquidAI/LFM2-VL-1.6B"
max_image_tokens: int = 1024

# Enhanced LoRA Configuration
lora_rank: int = 32                     # Higher capacity for vision-text mapping
lora_alpha: int = 32                    # Balanced scaling factor
lora_dropout: float = 0.15              # Strong dropout to prevent overfitting
lora_target_modules: List[str] = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj"  # Include MLP layers
]

# Anti-Overfitting Training Parameters
num_epochs: int = 4                     # Reduced to prevent overfitting
batch_size: int = 1                     # GPU memory constraint
grad_accum: int = 4                     # Effective batch size = 4
learning_rate: float = 1.5e-5           # Balanced for vision learning
warmup_steps: int = 50                  # Extended warmup for stability
weight_decay: float = 0.1               # Strong L2 regularization

# Data Augmentation (Anti-Overfitting)
augmentation_prob: float = 0.6          # Probability of augmentation
augmentations_per_sample: int = 3       # Multiple variations per image

# Early Stopping
early_stopping_patience: int = 2        # Stop if no improvement
eval_steps: int = 20                    # Frequent validation checks

# Infrastructure
gpu: str = "T4"
max_hr_training: int = 2
```

---

## 📊 Performance Metrics

The evaluation pipeline tracks:

- **Valid JSON Rate**: Percentage of valid JSON outputs  
- **Exact Match**: Complete accuracy across all fields  
- **Field Accuracy**: Individual accuracy for store name, date, total, currency  
- **Items Count Accuracy**: Accuracy of extracted items (with ±1 tolerance)  
- **Partial Match**: At least 3/4 fields correct  
- **Inference Time**: Average processing time per receipt

---

## 🔧 Optimization Tips

### **For Small Datasets (< 500 samples)**
- **Use strong regularization**: High dropout (0.15+), weight decay (0.1+)
- **Limit epochs**: 3-5 epochs max to prevent overfitting
- **Increase augmentation**: 3+ variations per sample for diversity
- **Early stopping**: Monitor validation loss closely

### **Image Resolution**: Balance between detail and memory usage  
- Calculate optimal size: `(image_size / 16)² ≤ max_image_tokens`
- Current config: `(512/16)² = 1024 ≤ 1024` ✅

### **LoRA Configuration**: 
- **Rank 32**: Good balance for vision-language tasks
- **Dropout 0.15**: Strong regularization for small datasets
- **Target modules**: Include both attention and MLP layers

### **Learning Rate Tuning**:
- **1e-5 to 2e-5**: Sweet spot for vision adaptation
- **Too low (< 5e-6)**: Model won't learn to read images
- **Too high (> 3e-5)**: Unstable gradients, hallucination

### **Data Augmentation Strategy**:
- **3 augmentations per sample**: Rotation, brightness, contrast, noise
- **Probability 0.6**: Not too aggressive to maintain readability

### **Monitoring Training**:
- **Watch eval vs train loss**: Gap indicates overfitting
- **Gradient norms**: Should stay < 2.0 for stability
- **Token accuracy**: Should improve steadily

### **Cache Management**:
```bash
# Clear cache when changing configuration
rm -rf /cache/datasets/train_augmented_v2.arrow
rm -rf /cache/datasets/test_processed_v2.arrow
```

---

## 🐛 Common Issues & Solutions

### **Training Issues**

**ValueError: saving steps not multiple of eval steps**
```bash
--load_best_model_at_end requires saving steps to be a round multiple of eval steps
```
**Solution**: Ensure `save_steps` is multiple of `eval_steps` (both set to 20 in current config)

**Model Hallucinating Instead of Reading Images**
- **Symptom**: Valid JSON but wrong content (generic store names, wrong dates)
- **Solution**: Increase learning rate to 1.5e-5+, target vision-language layers
- **Prevention**: Use vision-focused LoRA modules, strong system prompts

**Loss Oscillating Between 5-8**
- **Cause**: Learning rate too high or insufficient regularization
- **Solution**: Reduce LR to 1e-5, increase dropout to 0.15+, add weight decay

### **Performance Issues**

**Dimension Mismatch Error**  
```
RuntimeError: The size of tensor a (3600) must match the size of tensor b (1024)
```
**Solution**: Ensure image resolution matches `max_image_tokens` setting (512px → 1024 tokens)

**Low Accuracy on Evaluation**
- Check if model is reading images vs memorizing patterns
- Run sample evaluation to inspect actual predictions
- Increase vision-language learning with higher LR
- Ensure training data has correct image-text alignment

**Overfitting on Small Datasets**
- **Symptoms**: Training loss << validation loss, exact match 0%
- **Solution**: Reduce epochs (4 max), increase dropout (0.15), more augmentation
- **Prevention**: Use early stopping with patience=2

### **Infrastructure Issues**

**Out of Memory**
- Reduce batch size or gradient accumulation  
- Lower `max_image_tokens` from 1024 to 512
- Use smaller LoRA rank (16 instead of 32)

**Training Too Slow**
- Current config: ~10-15s per step (normal for vision-language)
- Reduce augmentations if needed
- Consider smaller model if speed critical

---

## 📈 Expected Results

### **Training Metrics**
- **Training Loss**: Should decrease from ~7.0 to 4.0-5.0 range
- **Validation Loss**: Should follow training loss closely (no overfitting)
- **Gradient Norm**: Should stay below 2.0 for stability
- **Token Accuracy**: Progressive improvement 40% → 60%+

### **Evaluation Targets**
- **Valid JSON Rate**: 95-100% (structure learning)
- **Store Name Accuracy**: 70-85% (vision-text alignment)
- **Date Accuracy**: 75-90% (format standardization)  
- **Total Amount Accuracy**: 80-95% (numerical extraction)
- **Currency Accuracy**: 85-95% (context understanding)

### **Performance Characteristics**
- **Inference Time**: 3-5 seconds per receipt
- **Memory Usage**: ~3GB GPU memory
- **Training Time**: ~45-60 minutes (4 epochs, 308 steps)

---

## 🔬 Configuration Evolution

### **Problem**: Low Accuracy with Hallucination
**Initial Issue**: Model generated valid JSON but with hallucinated content (wrong store names, dates, amounts)

**Root Cause Analysis**:
- Model learned JSON structure without vision-text connection
- Learning rate too low for vision adaptation
- Insufficient regularization for small dataset (154 samples)

### **Solution**: Vision-Focused Anti-Overfitting Strategy

**Key Changes Applied**:
1. **Enhanced LoRA Targeting**: Added MLP layers for better vision-language mapping
2. **Balanced Learning Rate**: Increased to 1.5e-5 for vision learning without instability  
3. **Strong Regularization**: 15% dropout + 0.1 weight decay to prevent memorization
4. **Data Augmentation**: 3x augmentations per sample for 616 total training examples
5. **Early Stopping**: Automatic halt if validation stops improving
6. **Reduced Epochs**: 4 epochs max to prevent overfitting on small dataset

**Results**: 
- Stable gradient norms (< 2.0)
- Progressive loss reduction without oscillation
- Vision-language connection establishment
- Prevention of hallucination patterns

---

## 🤝 Contributing

1. Fork the repository  
2. Create a feature branch  
3. Make your changes  
4. Test thoroughly using the evaluation pipeline  
5. Submit a pull request


---

## 🙏 Acknowledgments

- LiquidAI for the LFM2-VL model  
- Modal for cloud infrastructure  
- Hugging Face for model hosting and datasets  
- Weights & Biases for experiment tracking

---

## 📧 Contact

For questions or issues, please open a GitHub issue or contact the maintainers.
