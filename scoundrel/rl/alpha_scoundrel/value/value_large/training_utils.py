"""
Shared training utilities for alpha_scoundrel value networks.
Contains loss computation, metrics, and gradient tracking functions.
Adapted from policy training utilities for regression tasks.
"""
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from typing import Optional, Dict


def compute_loss(
    pred_values: torch.Tensor,
    target_values: torch.Tensor,
    focal_gamma: float = 2.0
) -> torch.Tensor:
    """
    Compute focused MSE loss adapting Focal Loss concept to regression.

    Uses Focal Loss adapted for regression: L_focal = |y - ŷ|^γ * (y - ŷ)²
    This down-weights "easy" examples (small errors) and focuses on "hard" examples (large errors).

    Args:
        pred_values: Model predictions [batch_size, 1]
        target_values: Target values [batch_size]
        focal_gamma: Gamma parameter for focal MSE modulation (> 0 focuses on hard examples)

    Returns:
        Scalar loss tensor
    """
    pred_values = pred_values.squeeze(-1)  # [batch_size]

    # Compute per-element squared errors: (y - ŷ)²
    squared_errors = (target_values - pred_values) ** 2

    if focal_gamma > 0.0:
        # Apply focal modulation: |y - ŷ|^γ * (y - ŷ)²
        error_magnitudes = torch.abs(target_values - pred_values)
        focal_weights = error_magnitudes ** focal_gamma
        focused_mse = focal_weights * squared_errors
    else:
        # Pure MSE when focal_gamma = 0
        focused_mse = squared_errors

    # Filter out any NaN or Inf values
    loss = torch.where(torch.isfinite(focused_mse), focused_mse, torch.zeros_like(focused_mse))

    return loss.mean()


def compute_metrics(
    pred_values: torch.Tensor,
    target_values: torch.Tensor,
    writer: Optional[SummaryWriter] = None,
    global_step: Optional[int] = None,
    prefix: str = "",
    focal_gamma: float = 2.0
) -> Dict[str, float]:
    """
    Compute comprehensive metrics for value evaluation.

    Args:
        pred_values: Model predictions [batch_size, 1]
        target_values: Target values [batch_size]
        writer: Optional TensorBoard writer
        global_step: Optional global step for logging
        prefix: Optional prefix for metric names
        focal_gamma: Gamma parameter for focal MSE computation

    Returns:
        Dictionary with metrics: mse, mae, rmse, mean_error, focused_mse
    """
    pred_values = pred_values.squeeze(-1)  # [batch_size]

    mse = F.mse_loss(pred_values, target_values)
    mae = F.l1_loss(pred_values, target_values)
    rmse = torch.sqrt(mse)
    mean_error = (pred_values - target_values).mean()

    # Focused MSE: |y - ŷ|^γ * (y - ŷ)²
    squared_errors = (target_values - pred_values) ** 2
    if focal_gamma > 0.0:
        error_magnitudes = torch.abs(target_values - pred_values)
        focal_weights = error_magnitudes ** focal_gamma
        focused_mse = (focal_weights * squared_errors).mean()
    else:
        focused_mse = mse

    # Additional statistics
    pred_mean = pred_values.mean()
    target_mean = target_values.mean()
    pred_std = pred_values.std()
    target_std = target_values.std()

    metrics = {
        'mse': mse.item(),
        'mae': mae.item(),
        'rmse': rmse.item(),
        'mean_error': mean_error.item(),
        'focused_mse': focused_mse.item(),
        'pred_mean': pred_mean.item(),
        'target_mean': target_mean.item(),
        'pred_std': pred_std.item(),
        'target_std': target_std.item(),
    }

    if writer is not None and global_step is not None:
        writer.add_scalar(f'{prefix}MSE', mse.item(), global_step)
        writer.add_scalar(f'{prefix}MAE', mae.item(), global_step)
        writer.add_scalar(f'{prefix}RMSE', rmse.item(), global_step)
        writer.add_scalar(f'{prefix}MeanError', mean_error.item(), global_step)
        writer.add_scalar(f'{prefix}FocusedMSE', focused_mse.item(), global_step)
        writer.add_scalar(f'{prefix}PredMean', pred_mean.item(), global_step)
        writer.add_scalar(f'{prefix}TargetMean', target_mean.item(), global_step)
        writer.add_scalar(f'{prefix}PredStd', pred_std.item(), global_step)
        writer.add_scalar(f'{prefix}TargetStd', target_std.item(), global_step)

    return metrics


def compute_gradient_norm(model: torch.nn.Module) -> float:
    """
    Compute the L2 norm of all gradients.

    Args:
        model: PyTorch model

    Returns:
        Gradient norm (returns 0.0 if NaN/inf detected)
    """
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            if torch.isnan(param_norm) or torch.isinf(param_norm):
                return float('nan')
            total_norm += param_norm.item() ** 2
    result = total_norm ** 0.5
    if torch.isnan(torch.tensor(result)) or torch.isinf(torch.tensor(result)):
        return float('nan')
    return result
