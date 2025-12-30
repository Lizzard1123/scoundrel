import argparse
import torch
import torch.nn.functional as F
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from typing import Optional, Dict
import time
from datetime import datetime

from scoundrel.rl.alpha_scoundrel.value.value_large.constants import (
    BATCH_SIZE,
    CHECKPOINT_DIR,
    CHECKPOINT_INTERVAL,
    CHECKPOINT_PREFIX,
    DEFAULT_MCTS_LOGS_DIR,
    EPOCHS,
    LR,
    MAX_GRAD_NORM,
    MAX_GAMES,
    STACK_SEQ_LEN,
    TRAIN_VAL_SPLIT,
)
from scoundrel.rl.alpha_scoundrel.value.value_large.network import ValueLargeNet
from scoundrel.rl.alpha_scoundrel.value.value_large.data_loader import create_dataloaders
from scoundrel.rl.translator import ScoundrelTranslator
from scoundrel.rl.utils import get_device, get_pin_memory


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
            for key, value in final_val_metrics.items():
                if isinstance(value, float):
                    f.write(f"  {key}: {value:.6f}\n")
                else:
                    f.write(f"  {key}: {value}\n")
    
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


def compute_value_metrics(
    pred_values: torch.Tensor,
    target_values: torch.Tensor,
    writer: Optional[SummaryWriter] = None,
    global_step: Optional[int] = None,
    prefix: str = ""
) -> Dict[str, float]:
    """
    Compute comprehensive metrics for value evaluation.
    
    Args:
        pred_values: Model output [batch_size, 1]
        target_values: Target values [batch_size]
        writer: Optional TensorBoard writer
        global_step: Optional global step for logging
        prefix: Optional prefix for metric names
        
    Returns:
        Dictionary with metrics: mse, mae, rmse, mean_error
    """
    pred_values = pred_values.squeeze(-1)  # [batch_size]
    
    mse = F.mse_loss(pred_values, target_values)
    mae = F.l1_loss(pred_values, target_values)
    rmse = torch.sqrt(mse)
    mean_error = (pred_values - target_values).mean()
    
    metrics = {
        'mse': mse.item(),
        'mae': mae.item(),
        'rmse': rmse.item(),
        'mean_error': mean_error.item(),
        'pred_mean': pred_values.mean().item(),
        'target_mean': target_values.mean().item(),
    }
    
    if writer is not None and global_step is not None:
        writer.add_scalar(f'{prefix}MSE', mse.item(), global_step)
        writer.add_scalar(f'{prefix}MAE', mae.item(), global_step)
        writer.add_scalar(f'{prefix}RMSE', rmse.item(), global_step)
        writer.add_scalar(f'{prefix}MeanError', mean_error.item(), global_step)
        writer.add_scalar(f'{prefix}PredMean', metrics['pred_mean'], global_step)
        writer.add_scalar(f'{prefix}TargetMean', metrics['target_mean'], global_step)
        writer.add_scalar(f'{prefix}ValueRange', pred_values.max().item() - pred_values.min().item(), global_step)
    
    return metrics


def train_epoch(
    model: ValueLargeNet,
    train_loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    writer: Optional[SummaryWriter] = None,
    epoch: int = 0
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
    
    for batch_idx, (scalar_features, sequence_features, unknown_stats, target_values) in enumerate(train_loader):
        batch_start_time = time.time()
        
        scalar_features = scalar_features.to(device)
        sequence_features = sequence_features.to(device)
        unknown_stats = unknown_stats.to(device)
        target_values = target_values.to(device)
        
        # Check for NaN/inf in inputs
        if torch.isnan(scalar_features).any() or torch.isinf(scalar_features).any():
            print(f"Warning: NaN/inf detected in scalar_features at batch {batch_idx}")
            continue
        if torch.isnan(sequence_features).any() or torch.isinf(sequence_features).any():
            print(f"Warning: NaN/inf detected in sequence_features at batch {batch_idx}")
            continue
        if torch.isnan(unknown_stats).any() or torch.isinf(unknown_stats).any():
            print(f"Warning: NaN/inf detected in unknown_stats at batch {batch_idx}")
            continue
        if torch.isnan(target_values).any() or torch.isinf(target_values).any():
            print(f"Warning: NaN/inf detected in target_values at batch {batch_idx}")
            continue
        
        pred_values = model(scalar_features, sequence_features, unknown_stats)
        
        # Check for NaN/inf in predictions
        if torch.isnan(pred_values).any() or torch.isinf(pred_values).any():
            print(f"Warning: NaN/inf detected in pred_values at batch {batch_idx}")
            print(f"  pred_values stats: min={pred_values.min().item():.6f}, max={pred_values.max().item():.6f}, mean={pred_values.mean().item():.6f}")
            continue
        
        loss = F.mse_loss(pred_values.squeeze(-1), target_values)
        
        # Check for NaN/inf in loss
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"Warning: NaN/inf loss detected at batch {batch_idx}")
            print(f"  pred_values stats: min={pred_values.min().item():.6f}, max={pred_values.max().item():.6f}, mean={pred_values.mean().item():.6f}")
            print(f"  target_values stats: min={target_values.min().item():.6f}, max={target_values.max().item():.6f}, mean={target_values.mean().item():.6f}")
            continue
        
        optimizer.zero_grad()
        loss.backward()
        
        # Clip gradients to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
        
        grad_norm = compute_gradient_norm(model)
        
        # Skip optimizer step if gradient norm is NaN/inf
        if torch.isnan(torch.tensor(grad_norm)) or torch.isinf(torch.tensor(grad_norm)):
            print(f"Warning: NaN/inf gradient norm detected at batch {batch_idx}, skipping optimizer step")
            print(f"  Loss: {loss.item():.6f}")
            continue
        
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
                metrics = compute_value_metrics(pred_values, target_values, writer, global_step, prefix='Train/')
        
        batch_time = time.time() - batch_start_time
        if writer is not None:
            writer.add_scalar('Train/BatchTime', batch_time, epoch * len(train_loader) + batch_idx)
    
    epoch_time = time.time() - epoch_start_time
    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
    samples_per_sec = total_samples / epoch_time if epoch_time > 0 else 0.0
    
    # Compute epoch-level metrics
    model.eval()
    epoch_metrics = {'mse': 0.0, 'mae': 0.0, 'rmse': 0.0, 'mean_error': 0.0}
    with torch.no_grad():
        for scalar_features, sequence_features, unknown_stats, target_values in train_loader:
            scalar_features = scalar_features.to(device)
            sequence_features = sequence_features.to(device)
            unknown_stats = unknown_stats.to(device)
            target_values = target_values.to(device)
            
            pred_values = model(scalar_features, sequence_features, unknown_stats)
            metrics = compute_value_metrics(pred_values, target_values)
            
            batch_size = scalar_features.size(0)
            for key in epoch_metrics:
                if key in metrics:
                    epoch_metrics[key] += metrics[key] * batch_size
    epoch_metrics = {k: v / total_samples if total_samples > 0 else 0.0 for k, v in epoch_metrics.items()}
    model.train()
    
    metrics = {
        'loss': avg_loss,
        **epoch_metrics,
        'samples_per_sec': samples_per_sec,
        'epoch_time': epoch_time,
    }
    
    if writer is not None:
        writer.add_scalar('Train/EpochLoss', avg_loss, epoch)
        writer.add_scalar('Train/EpochMSE', epoch_metrics['mse'], epoch)
        writer.add_scalar('Train/EpochMAE', epoch_metrics['mae'], epoch)
        writer.add_scalar('Train/EpochRMSE', epoch_metrics['rmse'], epoch)
        writer.add_scalar('Train/SamplesPerSec', samples_per_sec, epoch)
        writer.add_scalar('Train/EpochTime', epoch_time, epoch)
    
    return metrics


def validate(
    model: ValueLargeNet,
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
    
    all_metrics = {'mse': 0.0, 'mae': 0.0, 'rmse': 0.0, 'mean_error': 0.0}
    
    with torch.no_grad():
        for batch_idx, (scalar_features, sequence_features, unknown_stats, target_values) in enumerate(val_loader):
            scalar_features = scalar_features.to(device)
            sequence_features = sequence_features.to(device)
            unknown_stats = unknown_stats.to(device)
            target_values = target_values.to(device)
            
            pred_values = model(scalar_features, sequence_features, unknown_stats)
            loss = F.mse_loss(pred_values.squeeze(-1), target_values)
            
            metrics = compute_value_metrics(
                pred_values,
                target_values,
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
    
    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
    avg_metrics = {k: v / total_samples if total_samples > 0 else 0.0 for k, v in all_metrics.items()}
    
    result = {
        'loss': avg_loss,
        **avg_metrics,
    }
    
    if writer is not None:
        writer.add_scalar('Val/EpochLoss', avg_loss, epoch)
        writer.add_scalar('Val/MSE', avg_metrics['mse'], epoch)
        writer.add_scalar('Val/MAE', avg_metrics['mae'], epoch)
        writer.add_scalar('Val/RMSE', avg_metrics['rmse'], epoch)
        writer.add_scalar('Val/MeanError', avg_metrics['mean_error'], epoch)
    
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
):
    """
    Main training function with comprehensive logging.
    
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
    )
    
    model = ValueLargeNet(scalar_input_dim=scalar_input_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    if writer:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        writer.add_text('Model/TotalParams', str(total_params), 0)
        writer.add_text('Model/TrainableParams', str(trainable_params), 0)
        
        dummy_scalar_tensor = dummy_scalar.to(device)
        dummy_seq_tensor = torch.zeros((1, STACK_SEQ_LEN), dtype=torch.long).to(device)
        dummy_unknown_stats = torch.zeros((1, 3), dtype=torch.float32).to(device)
        try:
            writer.add_graph(model, (dummy_scalar_tensor, dummy_seq_tensor, dummy_unknown_stats))
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
            train_metrics = train_epoch(model, train_loader, optimizer, device, writer, epoch)
            val_metrics = validate(model, val_loader, device, writer, epoch)
            
            # Track final metrics
            final_train_metrics = train_metrics
            final_val_metrics = val_metrics
            
            print(f"Epoch {epoch+1}/{epochs}: "
                  f"Train Loss: {train_metrics['loss']:.6f}, "
                  f"Train MSE: {train_metrics['mse']:.6f}, "
                  f"Val Loss: {val_metrics['loss']:.6f}, "
                  f"Val MSE: {val_metrics['mse']:.6f}, "
                  f"Val MAE: {val_metrics['mae']:.6f}, "
                  f"Speed: {train_metrics['samples_per_sec']:.1f} samples/sec")
            
            if (epoch + 1) % CHECKPOINT_INTERVAL == 0 or (epoch + 1) == epochs:
                checkpoint_path = checkpoint_dir / f"{CHECKPOINT_PREFIX}{epoch+1}.pt"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_metrics['loss'],
                    'train_mse': train_metrics['mse'],
                    'val_loss': val_metrics['loss'],
                    'val_mse': val_metrics['mse'],
                    'val_mae': val_metrics['mae'],
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
    parser = argparse.ArgumentParser(description="Train Value Large network on MCTS final scores")
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
    )


if __name__ == "__main__":
    main()
