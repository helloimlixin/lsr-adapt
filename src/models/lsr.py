import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import RobertaForSequenceClassification, AutoTokenizer
import logging

logger = logging.getLogger(__name__)


def batched_kron(A, B):
    """
    Compute the Kronecker product for a batch of matrix pairs in a vectorized way.

    Args:
        A: Tensor of shape (N, a, b)
        B: Tensor of shape (N, c, d)

    Returns:
        Tensor of shape (N, a*c, b*d) where each slice is the Kronecker product of
        the corresponding slices in A and B.
    """
    N, a, b = A.shape
    N2, c, d = B.shape
    if N != N2:
        raise ValueError("The batch dimensions of A and B must match.")

    # Compute an outer product for each batch element using einsum
    # This yields a tensor of shape (N, a, c, b, d)
    kron_out = torch.einsum('nab,ncd->nacbd', A, B)

    # Reshape to (N, a*c, b*d)
    return kron_out.reshape(N, a * c, b * d)


class LSRLinear(nn.Module):
    """
    LSR-Augmented Linear Layer using multiple Kronecker product terms.

    This implementation extends LoRA by using structured factorization with
    Kronecker products to achieve greater parameter efficiency.
    """

    def __init__(self, original_linear: nn.Linear, r: int = 4, lora_alpha: float = 16,
                 num_terms: int = 16, factor_a: int = 32, factor_b: int = 24):
        """
        Initialize an LSR-augmented linear layer.

        Args:
            original_linear: A pre-trained nn.Linear layer whose weights are frozen.
            r: The bottleneck dimension.
            lora_alpha: Scaling factor.
            num_terms: Number of Kronecker product terms to sum.
            factor_a: First factor of the input/output dimensions.
            factor_b: Second factor of the input/output dimensions.

        Note:
            This implementation assumes in_features and out_features can be factored
            as factor_a * factor_b (e.g., 768 = 32 * 24).
        """
        super().__init__()
        self.in_features = original_linear.in_features
        self.out_features = original_linear.out_features
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r if r > 0 else 1.0
        self.num_terms = num_terms

        # Verify dimensions can be factored as expected
        if self.in_features != factor_a * factor_b:
            raise ValueError(f"Input features ({self.in_features}) must equal {factor_a}*{factor_b}")
        if self.out_features != factor_a * factor_b:
            raise ValueError(f"Output features ({self.out_features}) must equal {factor_a}*{factor_b}")

        # Use the original (pre-trained) weight and bias (and freeze them)
        self.weight = original_linear.weight
        self.bias = original_linear.bias
        self.weight.requires_grad = False
        if self.bias is not None:
            self.bias.requires_grad = False

        if r > 0:
            # For the A-side (down-projection): factors that multiply to in_features
            # A1: (num_terms, r, factor_a)
            # A2: (num_terms, r, factor_b)
            self.lora_A1 = nn.Parameter(torch.randn(num_terms, r, factor_a) * 0.01)
            self.lora_A2 = nn.Parameter(torch.randn(num_terms, r, factor_b) * 0.01)

            # For the B-side (up-projection): factors that multiply to out_features
            # B1: (num_terms, factor_a, r)
            # B2: (num_terms, factor_b, r)
            self.lora_B1 = nn.Parameter(torch.randn(num_terms, factor_a, r) * 0.01)
            self.lora_B2 = nn.Parameter(torch.randn(num_terms, factor_b, r) * 0.01)
        else:
            self.lora_A1 = self.lora_A2 = None
            self.lora_B1 = self.lora_B2 = None

    def forward(self, x):
        """
        Forward pass that combines original layer output with LSR adaptation.

        Args:
            x: Input tensor

        Returns:
            Output tensor with LSR adaptation applied
        """
        # Compute the frozen base linear transformation
        base = F.linear(x, self.weight, self.bias)

        if self.r > 0:
            # --- Compute the A-side update in a fully vectorized way ---
            # Each term i produces kron(A1[i], A2[i]) of shape (r*r, factor_a*factor_b)
            A_kron = batched_kron(self.lora_A1, self.lora_A2)  # shape: (num_terms, r*r, in_features)
            # Sum over the num_terms dimension
            A_update = A_kron.sum(dim=0)  # shape: (r*r, in_features)
            # Apply the down-projection
            lora_intermediate = F.linear(x, A_update)  # shape: (batch_size, seq_len, r*r)

            # --- Compute the B-side update in a fully vectorized way ---
            B_kron = batched_kron(self.lora_B1, self.lora_B2)  # shape: (num_terms, out_features, r*r)
            B_update = B_kron.sum(dim=0)  # shape: (out_features, r*r)
            # Apply the up-projection
            lora_update = F.linear(lora_intermediate, B_update)  # shape: (batch_size, seq_len, out_features)

            # Add the scaled update to the base output
            return base + self.scaling * lora_update

        return base


def replace_linear_with_lsr(module: nn.Module, target_names=("query", "value"),
                            r=4, lora_alpha=16, num_terms=16, factor_a=32, factor_b=24):
    """
    Recursively replace nn.Linear layers with LSR-augmented versions.

    Args:
        module: The module to modify.
        target_names: Names of linear layers to target for replacement.
        r: The low-rank dimension for LSR.
        lora_alpha: Scaling factor for updates.
        num_terms: Number of Kronecker product terms to use.
        factor_a: First factor of input/output dimensions.
        factor_b: Second factor of input/output dimensions.
    """
    for name, child in module.named_children():
        if isinstance(child, nn.Linear) and any(t in name for t in target_names):
            try:
                setattr(module, name, LSRLinear(
                    child,
                    r=r,
                    lora_alpha=lora_alpha,
                    num_terms=num_terms,
                    factor_a=factor_a,
                    factor_b=factor_b
                ))
                logger.info(f"Replaced {name} with LSR (r={r}, terms={num_terms})")
            except ValueError as e:
                logger.warning(f"Could not replace {name}: {str(e)}")
        else:
            # Recursively apply to child modules
            replace_linear_with_lsr(
                child,
                target_names=target_names,
                r=r,
                lora_alpha=lora_alpha,
                num_terms=num_terms,
                factor_a=factor_a,
                factor_b=factor_b
            )


def load_lsr_model(model_name, num_labels, target_names, r, lora_alpha, num_terms, factor_a, factor_b):
    """
    Load a pretrained model and apply LSR to specific layers.

    Args:
        model_name: The name of the pretrained model.
        num_labels: Number of output labels for classification.
        target_names: List of layer names to apply LSR to.
        r: The low-rank dimension.
        lora_alpha: Scaling factor for the LSR update.
        num_terms: Number of Kronecker product terms to use.
        factor_a: First factor for input/output dimensions.
        factor_b: Second factor for input/output dimensions.

    Returns:
        tuple: (model, tokenizer) - The model with LSR applied and its tokenizer.
    """
    # Load the pretrained model and tokenizer
    model = RobertaForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Apply LSR to specified layers
    replace_linear_with_lsr(
        model,
        target_names=target_names,
        r=r,
        lora_alpha=lora_alpha,
        num_terms=num_terms,
        factor_a=factor_a,
        factor_b=factor_b
    )

    # Freeze all parameters except those of the LSR modules
    trainable_param_count = 0
    total_param_count = 0

    for name, param in model.named_parameters():
        total_param_count += param.numel()
        if "lora_" not in name:
            param.requires_grad = False
        else:
            trainable_param_count += param.numel()

    # Log parameter efficiency stats
    logger.info(f"LSR parameters: {trainable_param_count:,} ({trainable_param_count / total_param_count:.2%} of total)")

    return model, tokenizer