from datasets import load_dataset
from transformers import DataCollatorWithPadding
import logging

logger = logging.getLogger(__name__)


def load_glue_dataset(dataset_name):
    """
    Load a dataset from the GLUE benchmark.

    Args:
        dataset_name (str): Name of the GLUE dataset (e.g., 'mrpc', 'sst2').

    Returns:
        dataset: The loaded dataset with train, validation, and test splits.
    """
    logger.info(f"Loading GLUE dataset: {dataset_name}")
    return load_dataset("glue", dataset_name)


def tokenize_function(examples, tokenizer, max_length=128):
    """
    Tokenize text examples for model input.

    Args:
        examples: The examples to tokenize.
        tokenizer: The tokenizer to use.
        max_length (int): Maximum sequence length.

    Returns:
        dict: The tokenized examples.
    """
    # Handle sentence pairs (e.g., for MRPC, MNLI, QQP)
    if "sentence2" in examples:
        return tokenizer(
            examples["sentence1"],
            examples["sentence2"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )
    # Handle single sentences (e.g., for SST-2)
    else:
        return tokenizer(
            examples["sentence"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )


def prepare_dataset(dataset_name, tokenizer, max_length=128, batch_size=None):
    """
    Load and preprocess a GLUE dataset for training.

    This function handles:
    1. Loading the dataset
    2. Tokenizing all examples
    3. Creating a data collator for dynamic padding

    Args:
        dataset_name (str): Name of the GLUE dataset.
        tokenizer: The tokenizer to use.
        max_length (int): Maximum sequence length.
        batch_size (int, optional): Batch size for preprocessing.

    Returns:
        tuple: (tokenized_datasets, data_collator) - Preprocessed dataset and collator.
    """
    # Load dataset
    dataset = load_glue_dataset(dataset_name)

    # Define tokenization function
    def _tokenize(examples):
        return tokenize_function(examples, tokenizer, max_length)

    # Apply tokenization with batching for efficiency
    tokenized_datasets = dataset.map(
        _tokenize,
        batched=True,
        batch_size=batch_size,
        remove_columns=dataset["train"].column_names,  # Remove raw text columns
        desc=f"Tokenizing {dataset_name} dataset",
    )

    # Create data collator for dynamic padding during training
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Log dataset statistics
    logger.info(f"Dataset {dataset_name} loaded with splits: {list(tokenized_datasets.keys())}")
    for split in tokenized_datasets:
        logger.info(f"  {split}: {len(tokenized_datasets[split])} examples")

    return tokenized_datasets, data_collator