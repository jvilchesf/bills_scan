from trl import SFTConfig, SFTTrainer
from .config import Config
from .collate_fn import create_collate_fn


def train_model(config: Config, model, processor, train_dataset, test_dataset):
    """
    Use trainer to fine tune model
    """
    output_dir = f"/{config.output_path}"  # Training configuration

    sft_config = SFTConfig(
        # ================================
        # OUTPUT & CHECKPOINTING
        # ================================
        output_dir=output_dir,  # Where to save model checkpoints and final model
        logging_dir=f"{output_dir}/logs",  # Where to save TensorBoard/W&B logs
        save_strategy="steps",  # Save checkpoints every `save_steps`
        save_steps=config.save_steps,  # Interval (in steps) between checkpoints
        save_total_limit=2,  # Keep only the last 2 checkpoints (old ones deleted)
        # ================================
        # TRAINING HYPERPARAMETERS
        # ================================
        num_train_epochs=config.num_epochs,  # Total number of epochs
        per_device_train_batch_size=config.batch_size,  # Batch size per GPU/CPU device
        gradient_accumulation_steps=config.grad_accum,  # Virtual batch size = batch_size * grad_accum
        learning_rate=config.learning_rate,  # Initial learning rate
        warmup_steps=config.warmup_steps,  # Steps to warm up the LR scheduler
        warmup_ratio=0.10,  # Alternatively, % of total steps for warmup
        lr_scheduler_type="cosine",  # Scheduler: cosine decay after warmup
        weight_decay=0.05,  # L2 regularization
        max_grad_norm=1.0,  # Gradient clipping (prevents exploding grads)
        optim="adamw_torch",  # Optimizer: AdamW in PyTorch
        adam_beta1=0.9,  # Adam β1 parameter (momentum)
        adam_beta2=0.999,  # Adam β2 parameter (smoothing)
        # ================================
        # LOGGING & EVALUATION
        # ================================
        logging_steps=5,  # Log training metrics every N steps
        eval_strategy="steps"
        if test_dataset
        else "no",  # Evaluate periodically if test set available
        eval_steps=20 if test_dataset else None,  # Frequency of evaluation
        load_best_model_at_end=True
        if test_dataset
        else False,  # Keep best model after training
        metric_for_best_model="eval_loss" if test_dataset else None,  # Selection metric
        greater_is_better=False,  # Lower loss is better (for eval_loss)
        # ================================
        # MEMORY & PERFORMANCE OPTIMIZATION
        # ================================
        gradient_checkpointing=True,  # Save memory by re-computing forward activations
        gradient_checkpointing_kwargs={
            "use_reentrant": False
        },  # More stable checkpointing
        bf16=True,  # Use bfloat16 (better for Ampere+ GPUs)
        fp16=False,  # Disable float16 (conflicts with bf16 sometimes)
        torch_compile=False,  # Torch compile (can optimize training speed, still experimental)
        dataloader_pin_memory=False,  # Avoid pinning to reduce memory overhead
        dataloader_num_workers=0,  # Workers for dataloader (0 = single-threaded)
        dataloader_drop_last=False,  # Keep last incomplete batch (don’t drop it)
        # ================================
        # MODEL / DATASET BEHAVIOR
        # ================================
        max_length=256,  # Max length for training sequences
        dataset_kwargs={
            "skip_prepare_dataset": True,  # Dataset already preprocessed
            "max_seq_length": 512,  # Max length used for tokenization
        },
        remove_unused_columns=False,  # Keep dataset columns not used by the model
        group_by_length=False,  # Don’t bucket sequences by length
        prediction_loss_only=True,  # Report only loss (not predictions)
        ddp_find_unused_parameters=False,  # Skip unused parameter checks (needed for DDP)
        # ================================
        # EXPERIMENT TRACKING (W&B)
        # ================================
        report_to="wandb"
        if config.wb_enable
        else "none",  # Enable Weights & Biases logging
        run_name=config.wb_experiment_name,  # W&B experiment name (appears in dashboard)
        # ================================
        # REPRODUCIBILITY
        # ================================
        seed=42,  # Fixed random seed for reproducibility
    )

    collate_fn = create_collate_fn(processor)
    print("Creating SFT trainer...")
    sft_trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=collate_fn,
        processing_class=processor,
    )

    sft_trainer.train()
    # Save with absolute path
    print(f"Saving model to {output_dir}...")
    sft_trainer.save_model(output_dir)
    processor.save_pretrained(output_dir)

    print("Training completed successfully!")

    return True  # Return success indicator
