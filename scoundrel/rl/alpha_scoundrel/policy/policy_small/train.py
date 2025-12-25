import argparse
import torch
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from typing import Optional, Dict
import time

from scoundrel.rl.alpha_scoundrel.policy.policy_small.constants import (
    BATCH_SIZE,
    CHECKPOINT_DIR,
    CHECKPOINT_INTERVAL,
    CHECKPOINT_PREFIX,
    DEFAULT_MCTS_LOGS_DIR,
    EPOCHS,
    LR,
    MAX_GAMES,
    STACK_SEQ_LEN,
    TRAIN_VAL_SPLIT,
)
from scoundrel.rl.alpha_scoundrel.policy.policy_small.network import PolicySmallNet
from scoundrel.rl.alpha_scoundrel.policy.policy_small.data_loader import create_dataloaders
from scoundrel.rl.translator import ScoundrelTranslator
from scoundrel.rl.utils import get_device, get_pin_memory, default_paths
from scoundrel.rl.alpha_scoundrel.policy.training_utils import (
    compute_loss,
    compute_metrics,
    compute_gradient_norm,
)


def train_epoch(
    model: PolicySmallNet,
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
    
    for batch_idx, (scalar_features, stack_sums, target_probs, action_mask) in enumerate(train_loader):
        batch_start_time = time.time()
        
        scalar_features = scalar_features.to(device)
        stack_sums = stack_sums.to(device)
        target_probs = target_probs.to(device)
        action_mask = action_mask.to(device)
        
        logits = model(scalar_features, stack_sums)
        loss = compute_loss(logits, target_probs, action_mask)
        
        optimizer.zero_grad()
        loss.backward()
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
        for scalar_features, stack_sums, target_probs, action_mask in train_loader:
            scalar_features = scalar_features.to(device)
            stack_sums = stack_sums.to(device)
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
        for batch_idx, (scalar_features, stack_sums, target_probs, action_mask) in enumerate(val_loader):
            scalar_features = scalar_features.to(device)
            stack_sums = stack_sums.to(device)
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
    
    result = {
        'loss': avg_loss,
        **avg_metrics,
        **avg_per_action,
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
    if checkpoint_dir is None:
        checkpoint_dir = base_dir / CHECKPOINT_DIR
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    log_dir, _ = default_paths(base_dir, "dummy.pt") if tensorboard else (None, None)
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
    
    model = PolicySmallNet(scalar_input_dim=scalar_input_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
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
    
    for epoch in range(start_epoch, epochs):
        train_metrics = train_epoch(model, train_loader, optimizer, device, writer, epoch)
        val_metrics = validate(model, val_loader, device, writer, epoch)
        
        print(f"Epoch {epoch+1}/{epochs}: "
              f"Train Loss: {train_metrics['loss']:.6f}, "
              f"Train Acc: {train_metrics['accuracy']:.4f}, "
              f"Val Loss: {val_metrics['loss']:.6f}, "
              f"Val KL: {val_metrics['kl_div']:.6f}, "
              f"Val Acc: {val_metrics['accuracy']:.4f}, "
              f"Speed: {train_metrics['samples_per_sec']:.1f} samples/sec")
        
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
    
    if writer:
        writer.flush()
        writer.close()
    
    print("Training complete!")


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

