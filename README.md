# 🧾 Receipt OCR Fine-Tuning with LFM2-VL-450M  

A production-ready fine-tuning pipeline for training the **LiquidAI LFM2-VL-450M** vision-language model on **Swedish receipt OCR tasks** using **Modal cloud infrastructure**.  

---

## 📋 Overview  
This project implements a complete end-to-end pipeline for fine-tuning a vision-language model to extract structured data from receipt images. It uses **Parameter-Efficient Fine-Tuning (PEFT)** with **LoRA adapters** to optimize the model for receipt OCR while maintaining efficiency.  

### Key Features
- 🚀 **Cloud-Native**: Runs entirely on Modal cloud infrastructure  
- 🎯 **PEFT/LoRA**: Memory-efficient fine-tuning using LoRA adapters  
- 📊 **W&B Integration**: Real-time training metrics and experiment tracking  
- 🖼️ **Image Enhancement**: Automatic preprocessing for receipt images  
- 💾 **Smart Caching**: Preprocessed datasets cached for faster iterations  
- 🔍 **Comprehensive Evaluation**: Detailed accuracy metrics and predictions  

---

## 🏗️ Architecture  
The pipeline consists of three main components:  
1. **Data Processing**: Loads receipt images and annotations, applies enhancement, formats for training  
2. **Model Training**: Fine-tunes LFM2-VL-450M using LoRA adapters with optimized hyperparameters  
3. **Evaluation**: Tests model performance on held-out data with detailed metrics  

---

## 📁 Project Structure  
src/
└── fine-tuning/
├── init.py
├── config.py           # Centralized configuration management
├── dataset.py          # Dataset loading and preprocessing
├── model.py            # Model initialization and LoRA setup
├── collate_fn.py       # Batch collation and label masking
├── train.py            # Training loop implementation
├── main.py             # Training entry point
└── evaluate.py         # Model evaluation pipeline

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
- Loads base LFM2-VL-450M model  
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

## ⚙️ Configuration

Key hyperparameters in `config.py`:

```python
# Model Architecture
model_name: str = "LiquidAI/LFM2-VL-450M"
max_image_tokens: int = 1600

# LoRA Configuration
lora_rank: int = 16
lora_alpha: int = 32
lora_dropout: float = 0.05
lora_target_modules: List[str] = ["q_proj", "k_proj", "v_proj", "o_proj"]

# Training Parameters
num_epochs: int = 15
batch_size: int = 1
grad_accum: int = 8
learning_rate: float = 3e-6
warmup_steps: int = 10

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

**Image Resolution**: Balance between detail and memory usage  
- Calculate optimal size: `(image_size / 16)² ≤ max_image_tokens`

**LoRA Rank**: Higher rank = more capacity but slower training  
- Start with `16`, increase to `32` or `64` for complex tasks

**Learning Rate**: Critical for convergence  
- Too low (`< 1e-6`): Slow or no learning  
- Too high (`> 1e-4`): Unstable training

**Data Caching**: Preprocessed images cached automatically  
- Clear cache when changing enhancement:  
  ```bash
  rm -rf /cache/formatted_*
  ```

---

## 🐛 Common Issues

**Dimension Mismatch Error**  
```
RuntimeError: The size of tensor a (3600) must match the size of tensor b (1024)
```
**Solution**: Ensure image resolution matches `max_image_tokens` setting.

**Low Accuracy Results**
- Verify label masking is working correctly  
- Increase training epochs or LoRA rank  
- Check image enhancement settings  
- Ensure consistent date/text formatting

**Out of Memory**
- Reduce batch size or gradient accumulation  
- Lower `max_image_tokens`  
- Use smaller LoRA rank

---

## 📈 Results

Example metrics from evaluation:

- **Valid JSON Rate**: 100%  
- **Store Name Accuracy**: Target 80%+  
- **Date Accuracy**: Target 80%+  
- **Total Amount Accuracy**: Target 85%+

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
