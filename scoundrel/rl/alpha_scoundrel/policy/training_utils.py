"""
Shared training utilities for alpha_scoundrel policies.
Contains loss computation, metrics, and gradient tracking functions.
"""
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from typing import Optional, Dict
from scoundrel.rl.utils import mask_logits


def compute_loss(logits: torch.Tensor, target_probs: torch.Tensor, action_mask: torch.Tensor) -> torch.Tensor:
    """
    Compute cross-entropy loss with action masking.
    
    Args:
        logits: Model output [batch_size, 5]
        target_probs: Target distribution [batch_size, 5]
        action_mask: Boolean mask [batch_size, 5]
        
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
    
    log_probs = F.log_softmax(masked_logits, dim=-1)
    
    # Add small epsilon to avoid numerical issues with zero probabilities
    target_probs_safe = target_probs + 1e-8
    target_probs_safe = target_probs_safe / target_probs_safe.sum(dim=-1, keepdim=True)
    
    # Compute cross-entropy: -sum(target * log(pred))
    loss = -(target_probs_safe * log_probs).sum(dim=-1)
    
    # Filter out any NaN or Inf values
    loss = torch.where(torch.isfinite(loss), loss, torch.zeros_like(loss))
    
    return loss.mean()


def compute_metrics(
    logits: torch.Tensor,
    target_probs: torch.Tensor,
    action_mask: torch.Tensor,
    writer: Optional[SummaryWriter] = None,
    global_step: Optional[int] = None,
    prefix: str = ""
) -> Dict[str, float]:
    """
    Compute comprehensive metrics for evaluation.
    
    Args:
        logits: Model output [batch_size, 5]
        target_probs: Target distribution [batch_size, 5]
        action_mask: Boolean mask [batch_size, 5]
        writer: Optional TensorBoard writer
        global_step: Optional global step for logging
        prefix: Optional prefix for metric names
        
    Returns:
        Dictionary with metrics: kl_div, accuracy, mse, per_action_probs, per_action_targets
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
    
    pred_actions = torch.argmax(pred_probs, dim=-1)
    target_actions = torch.argmax(target_probs, dim=-1)
    accuracy = (pred_actions == target_actions).float().mean()
    mse = F.mse_loss(pred_probs, target_probs)
    
    metrics = {
        'kl_div': kl_div.item(),
        'accuracy': accuracy.item(),
        'mse': mse.item(),
    }
    
    if writer is not None and global_step is not None:
        for action_idx in range(5):
            action_name = f"action_{action_idx}" if action_idx < 4 else "action_avoid"
            pred_mean = pred_probs[:, action_idx].mean().item()
            target_mean = target_probs[:, action_idx].mean().item()
            metrics[f'{prefix}prob_{action_name}_pred'] = pred_mean
            metrics[f'{prefix}prob_{action_name}_target'] = target_mean
            
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

