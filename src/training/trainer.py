from transformers import TrainingArguments, Trainer
from src.utils.metrics import compute_metrics
import logging
import os

logger = logging.getLogger(__name__)


def get_training_args(cfg):
    """
    Create training arguments for the Trainer from config.

    Args:
        cfg: Hydra configuration object with training parameters.

    Returns:
        TrainingArguments: The training arguments object.
    """
    # Extract training configuration
    train_cfg = cfg.training
    output_dir = cfg.output_dir

    # Create training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy=train_cfg.evaluation_strategy,
        save_strategy=train_cfg.save_strategy,
        num_train_epochs=train_cfg.num_train_epochs,
        per_device_train_batch_size=train_cfg.per_device_train_batch_size,
        per_device_eval_batch_size=train_cfg.per_device_eval_batch_size,
        learning_rate=train_cfg.learning_rate,
        weight_decay=train_cfg.weight_decay,
        logging_dir=os.path.join(output_dir, "logs"),
        logging_steps=train_cfg.logging_steps,
        save_total_limit=train_cfg.save_total_limit,
        load_best_model_at_end=train_cfg.load_best_model_at_end,
        metric_for_best_model=train_cfg.metric_for_best_model,
        seed=cfg.seed,
        # Optional gradient accumulation for larger effective batch sizes
        gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 1),
        # Optional FP16 training
        fp16=train_cfg.get("fp16", False),
    )

    logger.info(f"Created training arguments with {train_cfg.num_train_epochs} epochs")
    return training_args


def setup_trainer(
        model,
        cfg,
        train_dataset,
        eval_dataset,
        tokenizer,
        data_collator,
):
    """
    Set up the Hugging Face Trainer for model fine-tuning.

    Args:
        model: The model to train.
        cfg: Hydra configuration.
        train_dataset: Training dataset.
        eval_dataset: Evaluation dataset.
        tokenizer: The tokenizer.
        data_collator: Data collator for batching.

    Returns:
        Trainer: The configured Trainer object.
    """
    # Get training arguments
    training_args = get_training_args(cfg)

    # Define metric computation with task name
    def compute_metrics_for_task(eval_pred):
        return compute_metrics(eval_pred, task_name=cfg.data.name)

    # Create the trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics_for_task,
    )

    return trainer