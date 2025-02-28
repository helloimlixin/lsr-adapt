#!/usr/bin/env python3
"""
Main script for LoRA fine-tuning of language models using Hydra for configuration.

This script orchestrates the end-to-end process of:
1. Loading and configuring a model with LoRA
2. Preparing datasets
3. Training the model
4. Evaluating results
5. Saving the fine-tuned model
"""

import os
import logging
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import random
import numpy as np
from pathlib import Path

from src.models.lora import load_lora_model
from src.data.dataset import prepare_dataset
from src.training.trainer import setup_trainer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    """
    Main function to run the training process.

    Args:
        cfg: Hydra configuration.
    """
    # Print configuration for debugging
    logger.info(f"Running with configuration:\n{OmegaConf.to_yaml(cfg)}")

    # Set random seeds for reproducibility
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        logger.warning("No GPU detected! Training will be slow.")

    # Create output directory if it doesn't exist
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    # Save the configuration for reproducibility
    with open(output_dir / "config.yaml", "w") as f:
        f.write(OmegaConf.to_yaml(cfg))

    # Load model and tokenizer with LoRA applied
    logger.info(f"Loading model {cfg.model.base_model} with LoRA (r={cfg.model.lora.r})")
    model, tokenizer = load_lora_model(
        model_name=cfg.model.base_model,
        num_labels=cfg.model.num_labels,
        target_names=cfg.model.lora.target_modules,
        r=cfg.model.lora.r,
        lora_alpha=cfg.model.lora.alpha
    )

    # Prepare dataset
    logger.info(f"Preparing {cfg.data.name} dataset")
    tokenized_datasets, data_collator = prepare_dataset(
        dataset_name=cfg.data.name,
        tokenizer=tokenizer,
        max_length=cfg.data.max_length,
        batch_size=cfg.data.preprocessing.get("batch_size", None)
    )

    # Set up trainer
    logger.info("Setting up trainer")
    trainer = setup_trainer(
        model=model,
        cfg=cfg,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator
    )

    # Train the model
    logger.info(f"Starting training for {cfg.training.num_train_epochs} epochs")
    train_result = trainer.train()

    # Log and save training results
    train_metrics = train_result.metrics
    trainer.save_model()
    trainer.log_metrics("train", train_metrics)
    trainer.save_metrics("train", train_metrics)
    trainer.save_state()

    # Evaluate the model
    logger.info("Evaluating final model")
    eval_results = trainer.evaluate()
    trainer.log_metrics("eval", eval_results)
    trainer.save_metrics("eval", eval_results)

    # Print final results
    logger.info(f"Final evaluation results: {eval_results}")

    # Save the final model and tokenizer
    final_model_dir = output_dir / "final_model"
    logger.info(f"Saving final model to {final_model_dir}")
    model.save_pretrained(final_model_dir)
    tokenizer.save_pretrained(final_model_dir)

    # Return evaluation results
    return eval_results


if __name__ == "__main__":
    main()