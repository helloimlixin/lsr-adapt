# LoRA and LSR Fine-Tuning with Hydra

This project demonstrates parameter-efficient fine-tuning of transformer models using both Low-Rank Adaptation (LoRA) and Low-Rank Structured Reparameterization (LSR) with Hydra configuration management for better experiment tracking.

## Parameter-Efficient Fine-Tuning Methods

### Low-Rank Adaptation (LoRA)

LoRA is a parameter-efficient fine-tuning technique that:

1. **Freezes the pre-trained model weights** entirely
2. **Injects trainable rank decomposition matrices** into each layer of the Transformer architecture
3. Significantly **reduces the number of parameters** needed for fine-tuning (often by >99%)

For example, fine-tuning RoBERTa-base (125M parameters) with LoRA might only require training ~0.1-0.5M parameters while achieving comparable performance to full fine-tuning.

### Low-Rank Structured Reparameterization (LSR)

LSR extends LoRA by using Kronecker products to achieve even greater parameter efficiency:

1. **Assumes weight matrices can be factored** (e.g., 768 = 32 × 24)
2. **Uses multiple Kronecker product terms** to represent the adaptation
3. **Reduces parameter count substantially** compared to standard LoRA

For example, where LoRA might use 24K parameters, LSR might use just 3.5K parameters (85% reduction) while maintaining comparable performance.

## Project Structure

```
project_root/
│
├── src/                  # Source code
│   ├── models/           # Model implementations
│   │   ├── __init__.py
│   │   ├── lora.py       # LoRA architecture implementation
│   │   └── lsr.py        # LSR architecture implementation
│   ├── data/             # Data processing
│   │   ├── __init__.py
│   │   └── dataset.py    # Dataset loading and preprocessing
│   ├── training/         # Training utilities
│   │   ├── __init__.py
│   │   └── trainer.py    # Training configuration
│   └── utils/            # Utility functions
│       ├── __init__.py
│       └── metrics.py    # Evaluation metrics
│
├── conf/                 # Hydra configuration
│   ├── config.yaml       # Main config
│   ├── model/            # Model configurations
│   │   ├── roberta_lora.yaml  # LoRA configuration
│   │   └── roberta_lsr.yaml   # LSR configuration
│   ├── data/             # Dataset configurations
│   │   └── mrpc.yaml     # MRPC dataset config
│   └── training/         # Training configurations
│       └── lora_training.yaml  # Training parameters
│
├── main.py               # Main entry point
├── requirements.txt      # Dependencies
└── README.md             # Documentation
```

### Code Organization

The project follows a modular design with clear separation of concerns:

- **Models**: Implementations of parameter-efficient fine-tuning methods (LoRA and LSR)
- **Data**: Dataset loading, preprocessing, and tokenization
- **Training**: Training loop and optimization configuration
- **Utils**: Evaluation metrics and helper functions
- **Conf**: Hydra configuration files for all components

This modular structure makes it easy to:
1. Add new parameter-efficient methods (like we did with LSR)
2. Support different datasets beyond MRPC
3. Experiment with various training configurations
4. Track and compare results across different approaches

## Parameter-Efficient Fine-Tuning Architecture

The project implements two different approaches to parameter-efficient fine-tuning, housed in a modular architecture that makes it easy to compare methods and extend to new techniques.

### Key Design Elements

#### 1. Adapter Pattern
Both LoRA and LSR use a common adapter pattern where:
- Original pre-trained weights are frozen
- Small trainable modules are injected at strategic points
- A scaling factor balances the influence of the adapters

#### 2. Shared Interface
The implementations share a consistent interface:
- `load_lora_model()` and `load_lsr_model()` functions with parallel parameters
- `replace_linear_with_lora()` and `replace_linear_with_lsr()` for transformer modification
- Consistent parameter naming and scaling approaches

#### 3. Dynamic Selection
The system dynamically selects the fine-tuning method based on configuration:
```python
if cfg.model.adaptation_type == "lsr":
    model, tokenizer = load_lsr_model(...)
else:  # Default to LoRA
    model, tokenizer = load_lora_model(...)
```

#### 4. Configuration-Driven Experimentation
Hydra configurations control all aspects of the methods:
- Which layers to adapt (query, key, value projections)
- Hyperparameters like rank, scaling factor, and number of terms
- Model selection and dataset choices

## Usage

### Basic Usage

To run with the default configuration (LoRA):

```bash
python main.py
```

To run with LSR instead:

```bash
python main.py model=roberta_lsr
```

This will:
1. Load RoBERTa-base and apply the selected adaptation method
2. Fine-tune on the MRPC dataset for 20 epochs
3. Save the best model based on validation accuracy
4. Log metrics and results to the output directory

### Overriding Configuration

With Hydra, you can easily override any configuration parameter:

```bash
# Change LoRA rank to 16
python main.py model.lora.r=16

# Try LSR with different parameters
python main.py model=roberta_lsr model.lsr.num_terms=8 model.lsr.r=4

# Use a different dataset
python main.py data=sst2

# Change multiple parameters
python main.py model.lora.r=16 training.num_train_epochs=10 training.learning_rate=1e-4
```

## Comparing Adaptation Methods

One of the key strengths of this project structure is the ability to systematically compare different parameter-efficient fine-tuning methods. Below are some example experiments that leverage Hydra's multirun capability.

### Direct Comparison with Default Settings

Compare LoRA and LSR approaches with their default settings:

```bash
python main.py --multirun model=roberta_lora,roberta_lsr
```

This will run both methods sequentially with their default hyperparameters.

### Comparing Parameter Efficiency

Compare different rank settings for both approaches:

```bash
# LoRA with different ranks
python main.py model=roberta_lora --multirun model.lora.r=4,8,16,32

# LSR with different ranks
python main.py model=roberta_lsr --multirun model.lsr.r=1,2,4,8

# LSR with different numbers of terms
python main.py model=roberta_lsr --multirun model.lsr.num_terms=4,8,16,32
```

### Grid Search Across Methods

You can even perform grid searches across different methods and parameters:

```bash
python main.py --multirun model=roberta_lora,roberta_lsr \
  training.learning_rate=1e-4,5e-4,1e-3
```

### Analysis Workflow

For a thorough comparison, we recommend:

1. Start with equal parameter budgets (e.g., LSR with r=2 vs LoRA with r=8)
2. Analyze not just final performance but also training dynamics
3. Compare inference speed (which should be similar for both methods)
4. Examine performance across different datasets and tasks

## Technical Deep Dive: LoRA vs LSR

Understanding the mathematical foundations of these approaches helps explain their efficiency advantages.

### LoRA: Low-Rank Adaptation

In LoRA, for a pre-trained weight matrix $W \in \mathbb{R}^{d \times k}$, the adaptation is:

$$W_{adapted} = W + \Delta W = W + BA$$

Where:
- $B \in \mathbb{R}^{d \times r}$ and $A \in \mathbb{R}^{r \times k}$ 
- $r \ll \min(d, k)$ is the low-rank dimension (typically 4-32)

The total parameter count for the adaptation is $r(d + k)$, which is much smaller than the original $d \times k$ parameters when $r$ is small.

### LSR: Low-Rank Structured Reparameterization

LSR extends LoRA by using Kronecker products. For a weight matrix that can be factored into dimensions $W \in \mathbb{R}^{d_1 d_2 \times k_1 k_2}$, the adaptation becomes:

$$W_{adapted} = W + \sum_{i=1}^{t} (B_{1i} \otimes B_{2i})(A_{1i} \otimes A_{2i})$$

Where:
- $A_{1i} \in \mathbb{R}^{r \times d_1}$, $A_{2i} \in \mathbb{R}^{r \times d_2}$ 
- $B_{1i} \in \mathbb{R}^{k_1 \times r}$, $B_{2i} \in \mathbb{R}^{k_2 \times r}$ 
- $\otimes$ denotes the Kronecker product
- $t$ is the number of terms in the sum (typically 4-32)

The parameter count becomes $t \cdot r(d_1 + d_2 + k_1 + k_2)$, which can be substantially smaller than LoRA's $r(d_1d_2 + k_1k_2)$ when the dimensions are large.

### Practical Comparison

For a 768×768 linear layer:

| Method           | Configuration            | Parameter Count | % of Original |
|------------------|--------------------------|-----------------|---------------|
| Full Fine-tuning | -                        | 589,824         | 100%          |
| LoRA             | r=8                      | 12,288          | 2.08%         |
| LoRA             | r=16                     | 24,576          | 4.17%         |
| LSR              | r=2, t=16, factors=32×24 | 3,584           | 0.61%         |
| LSR              | r=4, t=8, factors=32×24  | 3,584           | 0.61%         |

The vectorized implementation in this codebase ensures that, despite the mathematical complexity, LSR remains computationally efficient during both training and inference.

## Performance Results

Empirical results show the effectiveness of both parameter-efficient fine-tuning methods:

### LoRA Results
When fine-tuning RoBERTa-base on MRPC with LoRA (r=8), you should expect:
- Accuracy: ~85-88%
- F1 Score: ~89-92%
- Training Parameters: ~0.2% of full model parameters
- Training Time: Significantly faster than full fine-tuning

### LSR Results
When fine-tuning RoBERTa-base on MRPC with LSR (r=2, num_terms=16), you should expect:
- Accuracy: ~84-87%
- F1 Score: ~88-91%
- Training Parameters: ~0.03% of full model parameters (approximately 7x fewer than LoRA)
- Training Time: Similar to LoRA, both much faster than full fine-tuning

### Performance vs. Parameter Count

A key insight is that LSR achieves nearly the same performance as LoRA while using a fraction of the parameters. This is particularly valuable for:

1. **Memory-constrained environments**: When fine-tuning very large models on limited hardware
2. **Deployment scenarios**: Where model size directly impacts inference costs
3. **Multitask adaptation**: When maintaining multiple task-specific adaptations simultaneously

The optimal choice between LoRA and LSR depends on your specific constraints and requirements.

## Extending the Project

The modular design makes it easy to extend this framework in several directions:

### Adding New Adaptation Methods

To implement a new parameter-efficient method:

1. Create a new implementation file in `src/models/` (e.g., `src/models/new_method.py`)
2. Implement the core adapter class (following the pattern of `LoRALinear` or `LSRLinear`)
3. Add functions to apply the method to a model (like `replace_linear_with_lora`)
4. Create a model loader function (similar to `load_lora_model`)
5. Add a configuration file in `conf/model/` (e.g., `roberta_new_method.yaml`)
6. Update `main.py` to recognize and initialize the new method

### Adding New Models

To support a new base model architecture:

1. Create a new configuration file in `conf/model/`
2. Implement any necessary model-specific code in `src/models/`
3. Ensure the adaptation methods can properly target the right layers in the new architecture

### Adding New Datasets

To add support for new datasets:

1. Create a new configuration file in `conf/data/`
2. Ensure the dataset preprocessing in `src/data/dataset.py` handles the new dataset
3. Consider adding dataset-specific metrics if needed in `src/utils/metrics.py`

### Customizing Training

To modify the training process:

1. Update or create a new configuration in `conf/training/`
2. Extend the trainer implementation in `src/training/trainer.py` if needed

## References

- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- [Hydra Framework](https://hydra.cc/)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/index)
- [The Power of Kronecker Products in Matrix Factorization](https://proceedings.neurips.cc/paper/2013/file/d3d9446802a44259755d38e6d163e820-Paper.pdf)
- [Parameter-Efficient Transfer Learning with Diff Pruning](https://arxiv.org/abs/2012.07463)