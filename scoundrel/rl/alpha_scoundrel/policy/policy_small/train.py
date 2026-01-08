"""
Training script for Policy Small with Advanced Features.

Features:
- Hybrid loss with focal MSE and hard classification
- Gradient clipping and dropout regularization
- Temperature sharpening for target distributions
- Learning rate scheduling (warmup + cosine decay)
"""

import argparse
import torch
import math
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from typing import Optional, Dict
import time
from datetime import datetime

from scoundrel.rl.alpha_scoundrel.policy.policy_small.constants import (
    BATCH_SIZE,
    CHECKPOINT_DIR,
    CHECKPOINT_INTERVAL,
    CHECKPOINT_PREFIX,
    DEFAULT_MCTS_LOGS_DIR,
    DROPOUT_RATE,
    EPOCHS,
    FOCAL_GAMMA,
    HARD_LOSS_WEIGHT,
    LR,
    MAX_GAMES,
    MAX_GRAD_NORM,
    MIN_LR_RATIO,
    STACK_SEQ_LEN,
    TEMPERATURE,
    TRAIN_VAL_SPLIT,
    USE_Q_WEIGHTS,
    WARMUP_EPOCHS,
)
from scoundrel.rl.alpha_scoundrel.policy.policy_small.network import PolicySmallNet
from scoundrel.rl.alpha_scoundrel.policy.policy_small.data_loader import create_dataloaders
from scoundrel.rl.translator import ScoundrelTranslator
from scoundrel.rl.utils import get_device, get_pin_memory
from scoundrel.rl.alpha_scoundrel.policy.training_utils import (
    compute_loss,
    compute_metrics,
    compute_gradient_norm,
)


def get_lr_scheduler(optimizer, warmup_epochs: int, total_epochs: int, min_lr_ratio: float = 1.0):
    """
    Create learning rate scheduler.

    If warmup_epochs=0 and min_lr_ratio=1.0, returns constant LR.
    Otherwise uses warmup + cosine decay.

    Args:
        optimizer: PyTorch optimizer
        warmup_epochs: Number of warmup epochs (0 for no warmup)
        total_epochs: Total training epochs
        min_lr_ratio: Minimum LR as ratio of initial LR (1.0 for constant)

    Returns:
        LambdaLR scheduler
    """
    def lr_lambda(epoch):
        # Constant LR if no warmup and no decay
        if warmup_epochs == 0 and min_lr_ratio >= 1.0:
            return 1.0

        if epoch < warmup_epochs:
            # Linear warmup
            return (epoch + 1) / max(1, warmup_epochs)
        else:
            # Cosine decay (or constant if min_lr_ratio == 1.0)
            if min_lr_ratio >= 1.0:
                return 1.0
            progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
            return min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def _save_training_summary(
    checkpoint_dir: Path,
    final_train_metrics: Optional[Dict[str, float]],
    final_val_metrics: Optional[Dict[str, float]],
    epochs: int,
    batch_size: int,
    lr: float,
):
    """
    Save training summary files: results.txt, constants.txt, and network.txt
    
    Args:
        checkpoint_dir: Directory to save summary files
        final_train_metrics: Final training metrics from last epoch
        final_val_metrics: Final validation metrics from last epoch
        epochs: Total number of epochs
        batch_size: Batch size used
        lr: Learning rate used
    """
    base_dir = Path(__file__).parent
    
    # Save results.txt
    results_path = checkpoint_dir / "results.txt"
    with open(results_path, 'w') as f:
        f.write("Training Results Summary\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Total Epochs: {epochs}\n")
        f.write(f"Batch Size: {batch_size}\n")
        f.write(f"Learning Rate: {lr}\n\n")
        
        if final_train_metrics:
            f.write("Final Training Metrics:\n")
            f.write("-" * 30 + "\n")
            for key, value in final_train_metrics.items():
                if isinstance(value, float):
                    f.write(f"  {key}: {value:.6f}\n")
                else:
                    f.write(f"  {key}: {value}\n")
            f.write("\n")
        
        if final_val_metrics:
            f.write("Final Validation Metrics:\n")
            f.write("-" * 30 + "\n")
            
            # Write main metrics first (loss, kl_div, accuracy, mse)
            main_metrics = ['loss', 'kl_div', 'accuracy', 'mse']
            for key in main_metrics:
                if key in final_val_metrics:
                    value = final_val_metrics[key]
                    if isinstance(value, float):
                        f.write(f"  {key}: {value:.6f}\n")
                    else:
                        f.write(f"  {key}: {value}\n")
            
            f.write("\n")
            f.write("Per-Action Probabilities:\n")
            f.write("-" * 30 + "\n")
            
            # Write per-action metrics grouped by action
            action_names = ['Pick Card 1', 'Pick Card 2', 'Pick Card 3', 'Pick Card 4', 'Avoid']
            for action_idx in range(5):
                action_name = action_names[action_idx]
                pred_key = f'prob_action_{action_idx}_pred'
                target_key = f'prob_action_{action_idx}_target'
                error_key = f'prob_action_{action_idx}_error'
                
                f.write(f"  {action_name} (Action {action_idx}):\n")
                if pred_key in final_val_metrics:
                    f.write(f"    Predicted: {final_val_metrics[pred_key]:.6f}\n")
                if target_key in final_val_metrics:
                    f.write(f"    Target:    {final_val_metrics[target_key]:.6f}\n")
                if error_key in final_val_metrics:
                    f.write(f"    Error:     {final_val_metrics[error_key]:.6f}\n")
                f.write("\n")
    
    # Save constants.txt
    constants_path = base_dir / "constants.py"
    if constants_path.exists():
        output_constants_path = checkpoint_dir / "constants.txt"
        with open(constants_path, 'r') as src, open(output_constants_path, 'w') as dst:
            dst.write(src.read())
    
    # Save network.txt
    network_path = base_dir / "network.py"
    if network_path.exists():
        output_network_path = checkpoint_dir / "network.txt"
        with open(network_path, 'r') as src, open(output_network_path, 'w') as dst:
            dst.write(src.read())


def train_epoch(
    model: PolicySmallNet,
    train_loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    writer: Optional[SummaryWriter] = None,
    epoch: int = 0,
    hard_loss_weight: float = 1.0,
    focal_gamma: float = 2.0,
    max_grad_norm: float = 1.0
) -> Dict[str, float]:
    """
    Train for one epoch with comprehensive logging.
    
    Args:
        model: Model to train
        train_loader: Training data loader
        optimizer: Optimizer
        device: Device to use
        writer: Optional TensorBoard writer
        epoch: Current epoch number
        
    Returns:
        Dictionary with training metrics
    """
    model.train()
    total_loss = 0.0
    total_samples = 0
    batch_count = 0
    
    epoch_start_time = time.time()
    
    for batch_idx, (scalar_features, stack_sums, total_stats, target_probs, action_mask) in enumerate(train_loader):
        batch_start_time = time.time()
        
        scalar_features = scalar_features.to(device)
        stack_sums = stack_sums.to(device)
        target_probs = target_probs.to(device)
        action_mask = action_mask.to(device)
        
        logits = model(scalar_features, stack_sums)
        loss = compute_loss(
            logits, target_probs, action_mask,
            hard_loss_weight=hard_loss_weight,
            focal_gamma=focal_gamma
        )
        
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        grad_norm = compute_gradient_norm(model)

        optimizer.step()
        
        batch_size = scalar_features.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size
        batch_count += 1
        
        if writer is not None:
            global_step = epoch * len(train_loader) + batch_idx
            writer.add_scalar('Train/BatchLoss', loss.item(), global_step)
            writer.add_scalar('Train/GradientNorm', grad_norm, global_step)
            writer.add_scalar('Train/LearningRate', optimizer.param_groups[0]['lr'], global_step)
            
            if batch_idx % 10 == 0:
                metrics = compute_metrics(logits, target_probs, action_mask, writer, global_step, prefix='Train/')
                writer.add_scalar('Train/Accuracy', metrics['accuracy'], global_step)
        
        batch_time = time.time() - batch_start_time
        if writer is not None:
            writer.add_scalar('Train/BatchTime', batch_time, epoch * len(train_loader) + batch_idx)
    
    epoch_time = time.time() - epoch_start_time
    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
    samples_per_sec = total_samples / epoch_time if epoch_time > 0 else 0.0
    
    # Compute epoch-level accuracy
    model.eval()
    epoch_accuracy = 0.0
    with torch.no_grad():
        for scalar_features, stack_sums, total_stats, target_probs, action_mask in train_loader:
            scalar_features = scalar_features.to(device)
            stack_sums = stack_sums.to(device)
            total_stats = total_stats.to(device)
            target_probs = target_probs.to(device)
            action_mask = action_mask.to(device)
            
            logits = model(scalar_features, stack_sums)
            metrics = compute_metrics(logits, target_probs, action_mask)
            epoch_accuracy += metrics['accuracy'] * scalar_features.size(0)
    epoch_accuracy = epoch_accuracy / total_samples if total_samples > 0 else 0.0
    model.train()
    
    metrics = {
        'loss': avg_loss,
        'accuracy': epoch_accuracy,
        'samples_per_sec': samples_per_sec,
        'epoch_time': epoch_time,
    }
    
    if writer is not None:
        writer.add_scalar('Train/EpochLoss', avg_loss, epoch)
        writer.add_scalar('Train/EpochAccuracy', epoch_accuracy, epoch)
        writer.add_scalar('Train/SamplesPerSec', samples_per_sec, epoch)
        writer.add_scalar('Train/EpochTime', epoch_time, epoch)
    
    return metrics


def validate(
    model: PolicySmallNet,
    val_loader,
    device: torch.device,
    writer: Optional[SummaryWriter] = None,
    epoch: int = 0
) -> Dict[str, float]:
    """
    Validate the model with comprehensive logging.
    
    Args:
        model: Model to validate
        val_loader: Validation data loader
        device: Device to use
        writer: Optional TensorBoard writer
        epoch: Current epoch number
        
    Returns:
        Dictionary with validation metrics
    """
    model.eval()
    total_loss = 0.0
    total_samples = 0
    
    all_metrics = {'kl_div': 0.0, 'accuracy': 0.0, 'mse': 0.0}
    per_action_metrics = {f'prob_action_{i}_pred': 0.0 for i in range(5)}
    per_action_metrics.update({f'prob_action_{i}_target': 0.0 for i in range(5)})
    
    with torch.no_grad():
        for batch_idx, (scalar_features, stack_sums, total_stats, target_probs, action_mask) in enumerate(val_loader):
            scalar_features = scalar_features.to(device)
            stack_sums = stack_sums.to(device)
            total_stats = total_stats.to(device)
            target_probs = target_probs.to(device)
            action_mask = action_mask.to(device)
            
            logits = model(scalar_features, stack_sums)
            loss = compute_loss(logits, target_probs, action_mask)
            
            metrics = compute_metrics(
                logits,
                target_probs,
                action_mask,
                writer=None,  # Don't log during loop, log epoch-level metrics at end
                global_step=None,
                prefix='Val/'
            )
            
            batch_size = scalar_features.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            
            for key in all_metrics:
                if key in metrics:
                    all_metrics[key] += metrics[key] * batch_size
            
            for key in per_action_metrics:
                if key in metrics:
                    per_action_metrics[key] += metrics[key] * batch_size
    
    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
    avg_metrics = {k: v / total_samples if total_samples > 0 else 0.0 for k, v in all_metrics.items()}
    avg_per_action = {k: v / total_samples if total_samples > 0 else 0.0 for k, v in per_action_metrics.items()}
    
    # Compute action errors
    action_errors = {}
    for action_idx in range(5):
        pred_key = f'prob_action_{action_idx}_pred'
        target_key = f'prob_action_{action_idx}_target'
        if pred_key in avg_per_action and target_key in avg_per_action:
            error = abs(avg_per_action[pred_key] - avg_per_action[target_key])
            action_errors[f'prob_action_{action_idx}_error'] = error
    
    result = {
        'loss': avg_loss,
        **avg_metrics,
        **avg_per_action,
        **action_errors,
    }
    
    if writer is not None:
        writer.add_scalar('Val/EpochLoss', avg_loss, epoch)
        writer.add_scalar('Val/KL_Div', avg_metrics['kl_div'], epoch)
        writer.add_scalar('Val/Accuracy', avg_metrics['accuracy'], epoch)
        writer.add_scalar('Val/MSE', avg_metrics['mse'], epoch)
        
        for action_idx in range(5):
            action_name = f"action_{action_idx}" if action_idx < 4 else "action_avoid"
            pred_key = f'prob_action_{action_idx}_pred'
            target_key = f'prob_action_{action_idx}_target'
            if pred_key in avg_per_action:
                writer.add_scalar(f'Val/Probabilities/{action_name}_pred', avg_per_action[pred_key], epoch)
                writer.add_scalar(f'Val/Probabilities/{action_name}_target', avg_per_action[target_key], epoch)
                writer.add_scalar(
                    f'Val/Probabilities/{action_name}_error',
                    abs(avg_per_action[pred_key] - avg_per_action[target_key]),
                    epoch
                )
    
    return result


def train(
    mcts_logs_dir: Path,
    batch_size: int = BATCH_SIZE,
    epochs: int = EPOCHS,
    lr: float = LR,
    checkpoint_dir: Path = None,
    resume_from: Optional[Path] = None,
    max_games: Optional[int] = MAX_GAMES,
    tensorboard: bool = True,
    train_val_split: float = TRAIN_VAL_SPLIT,
    temperature: float = TEMPERATURE,
    use_q_weights: bool = USE_Q_WEIGHTS,
    hard_loss_weight: float = HARD_LOSS_WEIGHT,
    focal_gamma: float = FOCAL_GAMMA,
    dropout_rate: float = DROPOUT_RATE,
    max_grad_norm: float = MAX_GRAD_NORM,
    warmup_epochs: int = WARMUP_EPOCHS,
    min_lr_ratio: float = MIN_LR_RATIO,
):
    """
    Main training function with advanced features.

    Args:
        mcts_logs_dir: Directory containing MCTS log JSON files
        batch_size: Batch size for training
        epochs: Number of training epochs
        lr: Learning rate
        checkpoint_dir: Directory to save checkpoints
        resume_from: Path to checkpoint to resume from
        max_games: Maximum number of games to load (None = all)
        tensorboard: Whether to enable TensorBoard logging
        train_val_split: Fraction of data for training
        temperature: Temperature for sharpening target distributions (< 1.0 sharpens)
        use_q_weights: Whether to weight visits by Q-values from MCTS
        hard_loss_weight: Weight for hard classification loss (0.0 = pure soft loss)
        focal_gamma: Gamma parameter for focal MSE loss
        dropout_rate: Dropout rate for regularization
        max_grad_norm: Maximum gradient norm for clipping
        warmup_epochs: Number of warmup epochs for learning rate
        min_lr_ratio: Minimum learning rate ratio for decay
    """
    device = get_device()
    print(f"Using device: {device}")
    
    base_dir = Path(__file__).parent
    
    # Create unique timestamp for this training run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"run_{timestamp}"
    
    # Create checkpoint directory with timestamp subfolder
    if checkpoint_dir is None:
        checkpoint_dir = base_dir / CHECKPOINT_DIR / run_id
    else:
        checkpoint_dir = Path(checkpoint_dir) / run_id
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Create unique run directory with timestamp for TensorBoard
    if tensorboard:
        runs_dir = base_dir / "runs"
        runs_dir.mkdir(parents=True, exist_ok=True)
        log_dir = runs_dir / run_id
    else:
        log_dir = None
    
    writer = SummaryWriter(log_dir=str(log_dir)) if tensorboard else None
    
    if writer:
        writer.add_hparams(
            {
                'batch_size': batch_size,
                'learning_rate': lr,
                'epochs': epochs,
                'train_val_split': train_val_split,
                'max_games': max_games if max_games else -1,
            },
            {}
        )
    
    translator = ScoundrelTranslator(stack_seq_len=STACK_SEQ_LEN)
    from scoundrel.game.game_manager import GameManager
    engine = GameManager()
    dummy_state = engine.restart()
    dummy_scalar, _ = translator.encode_state(dummy_state)
    scalar_input_dim = dummy_scalar.shape[1]
    
    print(f"Loading data from {mcts_logs_dir}...")
    train_loader, val_loader = create_dataloaders(
        log_dir=mcts_logs_dir,
        translator=translator,
        batch_size=batch_size,
        train_val_split=train_val_split,
        max_games=max_games,
        num_workers=0,
        temperature=temperature,
        use_q_weights=use_q_weights,
    )
    
    model = PolicySmallNet(
        scalar_input_dim=scalar_input_dim,
        dropout_rate=dropout_rate
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Create learning rate scheduler
    scheduler = get_lr_scheduler(optimizer, warmup_epochs, epochs, min_lr_ratio)
    
    if writer:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        writer.add_text('Model/TotalParams', str(total_params), 0)
        writer.add_text('Model/TrainableParams', str(trainable_params), 0)
        
        dummy_scalar_tensor = dummy_scalar.to(device)
        dummy_stack_sums = torch.zeros((1, 3)).to(device)
        try:
            writer.add_graph(model, (dummy_scalar_tensor, dummy_stack_sums))
        except Exception as e:
            print(f"Warning: Could not log model graph: {e}")
    
    start_epoch = 0
    
    # Resume from checkpoint if specified
    if resume_from and Path(resume_from).exists():
        print(f"Resuming from checkpoint: {resume_from}")
        checkpoint = torch.load(resume_from, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint.get('epoch', 0) + 1
        print(f"Resumed from epoch {start_epoch - 1}")
        
        if writer:
            writer.add_text('Training/ResumedFrom', str(resume_from), 0)
    
    print(f"\nStarting training for {epochs} epochs...")
    print(f"Train samples: {len(train_loader.dataset)}, Val samples: {len(val_loader.dataset)}")
    
    if writer:
        writer.add_scalar('Data/TrainSamples', len(train_loader.dataset), 0)
        writer.add_scalar('Data/ValSamples', len(val_loader.dataset), 0)
    
    final_train_metrics = None
    final_val_metrics = None
    
    try:
        for epoch in range(start_epoch, epochs):
            train_metrics = train_epoch(
                model, train_loader, optimizer, device, writer, epoch,
                hard_loss_weight=hard_loss_weight,
                focal_gamma=focal_gamma,
                max_grad_norm=max_grad_norm
            )
            val_metrics = validate(model, val_loader, device, writer, epoch)
            
            # Track final metrics
            final_train_metrics = train_metrics
            final_val_metrics = val_metrics
            
            print(f"Epoch {epoch+1}/{epochs}: "
                  f"Train Loss: {train_metrics['loss']:.6f}, "
                  f"Train Acc: {train_metrics['accuracy']:.4f}, "
                  f"Val Loss: {val_metrics['loss']:.6f}, "
                  f"Val KL: {val_metrics['kl_div']:.6f}, "
                  f"Val Acc: {val_metrics['accuracy']:.4f}, "
                  f"Speed: {train_metrics['samples_per_sec']:.1f} samples/sec")
            
            # Step the learning rate scheduler
            scheduler.step()

            if (epoch + 1) % CHECKPOINT_INTERVAL == 0 or (epoch + 1) == epochs:
                checkpoint_path = checkpoint_dir / f"{CHECKPOINT_PREFIX}{epoch+1}.pt"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_metrics['loss'],
                    'train_accuracy': train_metrics['accuracy'],
                    'val_loss': val_metrics['loss'],
                    'val_accuracy': val_metrics['accuracy'],
                    'val_kl_div': val_metrics['kl_div'],
                    'scalar_input_dim': scalar_input_dim,
                }, checkpoint_path)
                print(f"Saved checkpoint: {checkpoint_path}")
                
                if writer:
                    writer.add_text('Checkpoints/Latest', str(checkpoint_path), epoch)
        
        print("Training complete!")
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user (Ctrl+C)")
        print("Saving summary files...")
    finally:
        # Save summary files (always, even on interruption)
        if final_train_metrics is not None or final_val_metrics is not None:
            _save_training_summary(checkpoint_dir, final_train_metrics, final_val_metrics, epochs, batch_size, lr)
            print(f"Summary files saved to {checkpoint_dir}")
        
        if writer:
            writer.flush()
            writer.close()


def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(description="Train Policy Small network on MCTS visit distributions")
    parser.add_argument(
        "--mcts-logs-dir",
        type=str,
        default=DEFAULT_MCTS_LOGS_DIR,
        help=f"Directory containing MCTS log JSON files (default: {DEFAULT_MCTS_LOGS_DIR})"
    )
    parser.add_argument(
        "--max-games",
        type=int,
        default=None,
        help="Maximum number of games to load (default: all)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help=f"Batch size (default: {BATCH_SIZE})"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=EPOCHS,
        help=f"Number of epochs (default: {EPOCHS})"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=LR,
        help=f"Learning rate (default: {LR})"
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="Directory to save checkpoints (default: checkpoints/)"
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Path to checkpoint to resume from"
    )
    parser.add_argument(
        "--no-tensorboard",
        action="store_true",
        help="Disable TensorBoard logging"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=TEMPERATURE,
        help=f"Temperature for sharpening target distributions (default: {TEMPERATURE})"
    )
    parser.add_argument(
        "--use-q-weights",
        action="store_true",
        help="Weight visits by their Q-values from MCTS"
    )
    parser.add_argument(
        "--hard-loss-weight",
        type=float,
        default=HARD_LOSS_WEIGHT,
        help=f"Weight for hard classification loss (default: {HARD_LOSS_WEIGHT})"
    )
    parser.add_argument(
        "--focal-gamma",
        type=float,
        default=FOCAL_GAMMA,
        help=f"Gamma parameter for focal MSE loss (default: {FOCAL_GAMMA})"
    )
    parser.add_argument(
        "--dropout-rate",
        type=float,
        default=DROPOUT_RATE,
        help=f"Dropout rate for regularization (default: {DROPOUT_RATE})"
    )
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=MAX_GRAD_NORM,
        help=f"Maximum gradient norm for clipping (default: {MAX_GRAD_NORM})"
    )
    parser.add_argument(
        "--warmup-epochs",
        type=int,
        default=WARMUP_EPOCHS,
        help=f"Number of warmup epochs for learning rate (default: {WARMUP_EPOCHS})"
    )
    parser.add_argument(
        "--min-lr-ratio",
        type=float,
        default=MIN_LR_RATIO,
        help=f"Minimum learning rate ratio for decay (default: {MIN_LR_RATIO})"
    )

    args = parser.parse_args()
    
    train(
        mcts_logs_dir=Path(args.mcts_logs_dir),
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        checkpoint_dir=Path(args.checkpoint_dir) if args.checkpoint_dir else None,
        resume_from=Path(args.resume_from) if args.resume_from else None,
        max_games=args.max_games,
        tensorboard=not args.no_tensorboard,
        temperature=args.temperature,
        use_q_weights=args.use_q_weights,
        hard_loss_weight=args.hard_loss_weight,
        focal_gamma=args.focal_gamma,
        dropout_rate=args.dropout_rate,
        max_grad_norm=args.max_grad_norm,
        warmup_epochs=args.warmup_epochs,
        min_lr_ratio=args.min_lr_ratio,
    )


if __name__ == "__main__":
    main()

