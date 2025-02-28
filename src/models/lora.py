import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import RobertaForSequenceClassification, AutoTokenizer


class LoRALinear(nn.Module):
    """
    A LoRA-augmented linear layer that adds low-rank adaptation to a frozen pre-trained layer.

    This implementation freezes the original weights and adds trainable low-rank matrices
    that modify the behavior of the layer without changing the original parameters.
    """

    def __init__(self, original_linear: nn.Linear, r: int = 8, lora_alpha: float = 32):
        """
        Initialize a LoRA-augmented linear layer.

        Args:
            original_linear (nn.Linear): The pre-trained linear layer to adapt.
            r (int): The low-rank dimension - smaller values mean fewer parameters.
            lora_alpha (float): Scaling factor for the LoRA update.
        """
        super().__init__()
        self.in_features = original_linear.in_features
        self.out_features = original_linear.out_features
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r if r > 0 else 1.0

        # Use the original (pre-trained) weight and bias.
        self.weight = original_linear.weight
        self.bias = original_linear.bias

        # Freeze the base model's parameters.
        self.weight.requires_grad = False
        if self.bias is not None:
            self.bias.requires_grad = False

        # Initialize LoRA parameters if r > 0.
        if r > 0:
            # lora_A: projects input into a lower-dimensional space (r × in_features)
            self.lora_A = nn.Parameter(torch.randn(r, self.in_features) * 0.01)
            # lora_B: projects back to the output space (out_features × r)
            self.lora_B = nn.Parameter(torch.randn(self.out_features, r) * 0.01)
        else:
            self.lora_A = None
            self.lora_B = None

    def forward(self, x):
        """
        Forward pass that combines original layer output with LoRA adaptation.

        Args:
            x: Input tensor

        Returns:
            Output tensor with LoRA adaptation applied
        """
        # Compute the frozen linear transformation.
        base = F.linear(x, self.weight, self.bias)

        if self.r > 0:
            # Compute the LoRA update: x → lora_A → lora_B
            lora_intermediate = F.linear(x, self.lora_A)  # Shape: [batch, seq_len, r]
            lora_update = F.linear(lora_intermediate, self.lora_B)  # Shape: [batch, seq_len, out_features]

            # Scale and add the update.
            return base + self.scaling * lora_update

        return base


def replace_linear_with_lora(module: nn.Module, target_names=("query", "value"), r=8, lora_alpha=32):
    """
    Recursively replace nn.Linear layers with LoRA-augmented versions.

    This function traverses the model and replaces specific linear layers based on their names.

    Args:
        module (nn.Module): The module to modify.
        target_names (tuple): Names of linear layers to target for replacement.
        r (int): The low-rank dimension for LoRA.
        lora_alpha (float): Scaling factor for LoRA updates.
    """
    for name, child in module.named_children():
        if isinstance(child, nn.Linear) and any(t in name for t in target_names):
            setattr(module, name, LoRALinear(child, r=r, lora_alpha=lora_alpha))
        else:
            replace_linear_with_lora(child, target_names=target_names, r=r, lora_alpha=lora_alpha)


def load_lora_model(model_name, num_labels, target_names, r, lora_alpha):
    """
    Load a pretrained model and apply LoRA to specific layers.

    Args:
        model_name (str): The name of the pretrained model.
        num_labels (int): Number of output labels for classification.
        target_names (tuple): tuple of layer names to apply LoRA to.
        r (int): The low-rank dimension.
        lora_alpha (float): Scaling factor for the LoRA update.

    Returns:
        tuple: (model, tokenizer) - The model with LoRA applied and its tokenizer.
    """
    # Load the pretrained model and tokenizer
    model = RobertaForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Apply LoRA to specified layers
    replace_linear_with_lora(model, target_names=target_names, r=r, lora_alpha=lora_alpha)

    # Freeze all parameters except those of the LoRA modules
    trainable_param_count = 0
    total_param_count = 0

    for name, param in model.named_parameters():
        total_param_count += param.numel()
        if "lora_" not in name:
            param.requires_grad = False
        else:
            trainable_param_count += param.numel()

    # Log parameter efficiency stats
    print(f"LoRA parameters: {trainable_param_count:,} ({trainable_param_count / total_param_count:.2%} of total)")

    return model, tokenizer