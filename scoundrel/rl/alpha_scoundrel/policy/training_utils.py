"""
Shared training utilities for alpha_scoundrel policies.
Contains loss computation, metrics, and gradient tracking functions.
"""
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from typing import Optional, Dict
from scoundrel.rl.utils import mask_logits


def compute_loss(
    logits: torch.Tensor,
    target_probs: torch.Tensor,
    action_mask: torch.Tensor,
    hard_loss_weight: float = 0.0,
    focal_gamma: float = 2.0
) -> torch.Tensor:
    """
    Compute focused MSE loss adapting Focal Loss concept to regression.

    Uses Focal Loss adapted for regression: L_focal = |y - ŷ|^γ * (y - ŷ)²
    This down-weights "easy" examples (small errors) and focuses on "hard" examples (large errors).

    Args:
        logits: Model output [batch_size, 5]
        target_probs: Target distribution [batch_size, 5]
        action_mask: Boolean mask [batch_size, 5]
        hard_loss_weight: Weight for hard classification loss (0.0 to 1.0).
                          0.0 = pure focused MSE (new behavior)
                          1.0 = pure hard classification on best action
                          Recommended: 0.0 for pure focused MSE
        focal_gamma: Gamma parameter for focal MSE modulation (> 0 focuses on hard examples)

    Returns:
        Scalar loss tensor
    """
    masked_logits = mask_logits(logits, action_mask)

    # Check for all-masked cases (all logits are -inf)
    # This can happen if all actions are invalid, which shouldn't occur but handle gracefully
    max_logits = masked_logits.max(dim=-1, keepdim=True)[0]
    masked_logits = torch.where(
        max_logits == float('-inf'),
        torch.zeros_like(masked_logits),
        masked_logits
    )

    # Get predicted probabilities
    pred_probs = F.softmax(masked_logits, dim=-1)

    # Add small epsilon to avoid numerical issues with zero probabilities
    target_probs_safe = target_probs + 1e-8
    target_probs_safe = target_probs_safe / target_probs_safe.sum(dim=-1, keepdim=True)

    # Compute per-element squared errors: (y - ŷ)²
    squared_errors = (target_probs_safe - pred_probs) ** 2

    # Apply focal modulation: |y - ŷ|^γ * (y - ŷ)²
    error_magnitudes = torch.abs(target_probs_safe - pred_probs)
    focal_weights = error_magnitudes ** focal_gamma
    focused_mse = focal_weights * squared_errors

    # Sum over action dimensions to get per-sample loss
    soft_loss = focused_mse.sum(dim=-1)

    if hard_loss_weight > 0.0:
        # Hard loss: classification loss on best action
        # This forces the model to get the top action right, not just approximate the distribution
        best_action = torch.argmax(target_probs, dim=-1)  # [batch_size]
        hard_loss = F.cross_entropy(masked_logits, best_action, reduction='none')

        # Combine soft and hard losses
        loss = (1.0 - hard_loss_weight) * soft_loss + hard_loss_weight * hard_loss
    else:
        loss = soft_loss

    # Filter out any NaN or Inf values
    loss = torch.where(torch.isfinite(loss), loss, torch.zeros_like(loss))

    return loss.mean()


def compute_metrics(
    logits: torch.Tensor,
    target_probs: torch.Tensor,
    action_mask: torch.Tensor,
    writer: Optional[SummaryWriter] = None,
    global_step: Optional[int] = None,
    prefix: str = "",
    focal_gamma: float = 2.0
) -> Dict[str, float]:
    """
    Compute comprehensive metrics for evaluation including focused MSE.

    Args:
        logits: Model output [batch_size, 5]
        target_probs: Target distribution [batch_size, 5]
        action_mask: Boolean mask [batch_size, 5]
        writer: Optional TensorBoard writer
        global_step: Optional global step for logging
        prefix: Optional prefix for metric names
        focal_gamma: Gamma parameter for focal MSE computation

    Returns:
        Dictionary with metrics: kl_div, accuracy, mse, focused_mse, per_action_probs, per_action_targets
    """
    masked_logits = mask_logits(logits, action_mask)

    # Handle all-masked cases
    max_logits = masked_logits.max(dim=-1, keepdim=True)[0]
    masked_logits = torch.where(
        max_logits == float('-inf'),
        torch.zeros_like(masked_logits),
        masked_logits
    )

    log_probs = F.log_softmax(masked_logits, dim=-1)
    pred_probs = F.softmax(masked_logits, dim=-1)

    # Add epsilon to target_probs for numerical stability
    target_probs_safe = target_probs + 1e-8
    target_probs_safe = target_probs_safe / target_probs_safe.sum(dim=-1, keepdim=True)

    # Compute KL divergence: sum(target * log(target / pred))
    # Using manual computation for better numerical stability
    kl_div = (target_probs_safe * (torch.log(target_probs_safe + 1e-8) - log_probs)).sum(dim=-1)
    kl_div = torch.where(torch.isfinite(kl_div), kl_div, torch.zeros_like(kl_div))
    kl_div = kl_div.mean()

    # Compute accuracy with tie-aware handling
    # If multiple actions have equal max probability in target, any of them is correct
    # Use tolerance for floating point comparisons to handle numerical precision issues
    pred_actions = torch.argmax(pred_probs, dim=-1)  # [batch_size]
    target_max_probs = target_probs.max(dim=-1)[0]  # [batch_size]
    # Find all actions that have the max probability (handles ties)
    # Use small tolerance to account for floating point precision errors
    tolerance = 1e-6
    target_max_mask = (target_probs >= target_max_probs.unsqueeze(-1) - tolerance)  # [batch_size, 5]
    # Check if predicted action is among the tied max actions
    # Use gather to check if pred_actions index has max probability
    accuracy = target_max_mask.gather(1, pred_actions.unsqueeze(-1)).squeeze(-1).float().mean()

    # Standard MSE
    mse = F.mse_loss(pred_probs, target_probs)

    # Focused MSE: |y - ŷ|^γ * (y - ŷ)²
    squared_errors = (target_probs_safe - pred_probs) ** 2
    error_magnitudes = torch.abs(target_probs_safe - pred_probs)
    focal_weights = error_magnitudes ** focal_gamma
    focused_mse = (focal_weights * squared_errors).mean()

    metrics = {
        'kl_div': kl_div.item(),
        'accuracy': accuracy.item(),
        'mse': mse.item(),
        'focused_mse': focused_mse.item(),
    }
    
    # Always compute per-action metrics (needed for validation accumulation)
    for action_idx in range(5):
        pred_mean = pred_probs[:, action_idx].mean().item()
        target_mean = target_probs[:, action_idx].mean().item()
        # Use keys without prefix for accumulation (prefix is only for TensorBoard logging)
        metrics[f'prob_action_{action_idx}_pred'] = pred_mean
        metrics[f'prob_action_{action_idx}_target'] = target_mean
    
    # Log to TensorBoard if writer is provided
    if writer is not None and global_step is not None:
        for action_idx in range(5):
            action_name = f"action_{action_idx}" if action_idx < 4 else "action_avoid"
            pred_mean = metrics[f'prob_action_{action_idx}_pred']
            target_mean = metrics[f'prob_action_{action_idx}_target']
            
            writer.add_scalar(f'{prefix}Probabilities/{action_name}_pred', pred_mean, global_step)
            writer.add_scalar(f'{prefix}Probabilities/{action_name}_target', target_mean, global_step)
            writer.add_scalar(f'{prefix}Probabilities/{action_name}_error', abs(pred_mean - target_mean), global_step)
    
    return metrics


def compute_gradient_norm(model: torch.nn.Module) -> float:
    """
    Compute the L2 norm of all gradients.
    
    Args:
        model: PyTorch model
        
    Returns:
        Gradient norm
    """
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm ** 0.5

