# Parameter-Efficient Fine-Tuning with Matrix Low-Separation Rank Structure

This project demonstrates parameter-efficient fine-tuning of transformer models using Low-Rank Adaptation (LoRA) with Hydra configuration management for better experiment tracking.

## What is Matrix Low Separation Rank?

TBD

## Project Structure

```
project_root/
│
├── src/                  # Source code
│   ├── models/           # Model implementations
│   │   └── lora.py       # LoRA architecture
│   ├── data/             # Data processing
│   │   └── dataset.py    # Dataset loading and preprocessing
│   ├── training/         # Training utilities
│   │   └── trainer.py    # Training configuration
│   └── utils/            # Utility functions
│       └── metrics.py    # Evaluation metrics
│
├── conf/                 # Hydra configuration
│   ├── config.yaml       # Main config
│   ├── model/            # Model configurations
│   │   └── roberta_lora.yaml
│   ├── data/             # Dataset configurations
│   │   └── mrpc.yaml
│   └── training/         # Training configurations
│       └── lora_training.yaml
│
├── main.py               # Main entry point
├── requirements.txt      # Dependencies
└── README.md             # Documentation
```

## Installation

```bash
# Clone the repository
git clone https://github.com/helloimlixin/lsr-adapt.git
cd lsr-adapt

# Create and activate a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Basic Usage

To run with the default configuration:

```bash
python main.py
```

This will:
1. Load RoBERTa-base and apply PEFT to the query and value projections
2. Fine-tune on the MRPC dataset for 20 epochs
3. Save the best model based on validation accuracy
4. Log metrics and results to the output directory

### Overriding Configuration

With Hydra, you can easily override any configuration parameter:

```bash
# Change LoRA rank to 16
python main.py model.lora.r=16

# Use a different dataset
python main.py data=sst2

# Change multiple parameters
python main.py model.lora.r=16 training.num_train_epochs=10 training.learning_rate=1e-4
```

### Running Multiple Experiments

You can leverage Hydra's multirun capability:

```bash
# Sweep over different LoRA ranks
python main.py --multirun model.lora.r=4,8,16,32

# Grid search over ranks and learning rates
python main.py --multirun model.lora.r=4,8,16 training.learning_rate=1e-4,5e-4,1e-3
```

## Extending the Project

### Adding New Models

1. Create a new configuration file in `conf/model/`
2. Implement any necessary model-specific code in `src/models/`

### Adding New Datasets

1. Create a new configuration file in `conf/data/`
2. Ensure the dataset preprocessing in `src/data/dataset.py` handles the new dataset

## Results

When fine-tuning RoBERTa-base on MRPC with LoRA (r=8), you should expect:
- Accuracy: ~85-88%
- F1 Score: ~89-92%
- Training Parameters: ~0.2% of full model parameters
- Training Time: Significantly faster than full fine-tuning

## References

- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- [Hydra Framework](https://hydra.cc/)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/index)