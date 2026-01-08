"""
REINFORCE-style Policy Gradient Training for Self-Play.

Implements policy gradient updates:
    loss = -log(π(a|s)) * R

Where:
- π(a|s) = probability of action a given state s
- R = reward (+1 for win, -1 for loss)

This reinforces actions from winning games and discourages actions from losing games.
"""

import time
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Optional, Dict
from torch.utils.tensorboard import SummaryWriter

from scoundrel.rl.alpha_scoundrel.policy.policy_large.network import PolicyLargeNet
from scoundrel.rl.alpha_scoundrel.self_play.rl_data_loader import create_rl_dataloaders
from scoundrel.rl.alpha_scoundrel.self_play.constants import (
    TENSORBOARD_LOG_INTERVAL,
    TENSORBOARD_HISTOGRAM_INTERVAL,
    LOG_WEIGHT_HISTOGRAMS,
    LOG_GRADIENT_HISTOGRAMS,
)
from scoundrel.rl.translator import ScoundrelTranslator
from scoundrel.game.game_manager import GameManager


def compute_policy_gradient_loss(
    logits: torch.Tensor,
    actions: torch.Tensor,
    rewards: torch.Tensor,
    action_mask: torch.Tensor,
    entropy_coef: float = 0.01,
) -> Dict[str, torch.Tensor]:
    """
    Compute REINFORCE policy gradient loss.

    Loss = -log(π(a|s)) * R + entropy_bonus

    Args:
        logits: Raw policy logits [batch_size, 5]
        actions: Actions taken [batch_size]
        rewards: Rewards for each sample [batch_size] (+1 win, -1 loss)
        action_mask: Valid actions mask [batch_size, 5]
        entropy_coef: Coefficient for entropy bonus (encourages exploration)

    Returns:
        Dictionary with 'loss', 'policy_loss', 'entropy', 'mean_reward'
    """
    # Mask invalid actions with large negative value
    masked_logits = logits.clone()
    masked_logits[~action_mask] = float('-inf')

    # Compute log probabilities
    log_probs = F.log_softmax(masked_logits, dim=-1)

    # Get log probability of the action taken
    action_log_probs = log_probs.gather(1, actions.unsqueeze(-1)).squeeze(-1)

    # Variance Reduction: Subtract baseline (batch mean)
    # This centers the returns, reducing gradient variance especially with small batch sizes
    advantages = rewards - rewards.mean()

    # Policy gradient loss: -log(π(a|s)) * (R - b)
    # Negative because we want to maximize expected reward
    policy_loss = -(action_log_probs * advantages).mean()

    # Entropy bonus to encourage exploration
    probs = F.softmax(masked_logits, dim=-1)
    # Only compute entropy over valid actions
    probs_masked = probs * action_mask.float()
    probs_masked = probs_masked / probs_masked.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    entropy = -(probs_masked * (probs_masked + 1e-8).log()).sum(dim=-1).mean()

    # Total loss = policy loss - entropy bonus
    total_loss = policy_loss - entropy_coef * entropy

    return {
        'loss': total_loss,
        'policy_loss': policy_loss,
        'entropy': entropy,
        'mean_reward': rewards.mean(),
    }


def compute_gradient_norm(model: torch.nn.Module) -> float:
    """Compute the L2 norm of all gradients."""
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            if torch.isnan(param_norm) or torch.isinf(param_norm):
                return float('nan')
            total_norm += param_norm.item() ** 2
    return total_norm ** 0.5


def train_rl_epoch(
    model: PolicyLargeNet,
    train_loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    writer: Optional[SummaryWriter] = None,
    epoch: int = 0,
    max_grad_norm: float = 1.0,
    entropy_coef: float = 0.01,
) -> Dict[str, float]:
    """
    Train for one epoch using REINFORCE policy gradients.

    Args:
        model: Policy network
        train_loader: Training data loader
        optimizer: Optimizer
        device: Device to use
        writer: Optional TensorBoard writer
        epoch: Current epoch number
        max_grad_norm: Maximum gradient norm for clipping
        entropy_coef: Entropy bonus coefficient

    Returns:
        Dictionary with training metrics
    """
    model.train()
    total_loss = 0.0
    total_policy_loss = 0.0
    total_entropy = 0.0
    total_reward = 0.0
    total_correct = 0
    total_win_correct = 0
    total_win_samples = 0
    total_loss_correct = 0
    total_loss_samples = 0
    total_samples = 0
    
    epoch_start_time = time.time()

    for batch_idx, batch in enumerate(train_loader):
        batch_start_time = time.time()
        
        # Unpack batch (10 elements for RL training)
        (scalar_features, sequence_features, unknown_stats, total_stats,
         room_features, room_mask, dungeon_len, actions, rewards, action_mask) = batch

        scalar_features = scalar_features.to(device)
        sequence_features = sequence_features.to(device)
        unknown_stats = unknown_stats.to(device)
        total_stats = total_stats.to(device)
        room_features = room_features.to(device)
        room_mask = room_mask.to(device)
        dungeon_len = dungeon_len.to(device)
        actions = actions.to(device)
        rewards = rewards.to(device)
        action_mask = action_mask.to(device)

        # Forward pass
        logits = model(
            scalar_features, sequence_features, unknown_stats, total_stats,
            room_features=room_features, room_mask=room_mask, dungeon_len=dungeon_len
        )

        # Compute policy gradient loss
        loss_dict = compute_policy_gradient_loss(
            logits, actions, rewards, action_mask, entropy_coef
        )

        optimizer.zero_grad()
        loss_dict['loss'].backward()

        # Compute gradient norm before clipping
        grad_norm_before = compute_gradient_norm(model)

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        
        # Compute gradient norm after clipping
        grad_norm_after = compute_gradient_norm(model)

        optimizer.step()

        batch_size = scalar_features.size(0)
        total_loss += loss_dict['loss'].item() * batch_size
        total_policy_loss += loss_dict['policy_loss'].item() * batch_size
        total_entropy += loss_dict['entropy'].item() * batch_size
        total_reward += loss_dict['mean_reward'].item() * batch_size
        total_samples += batch_size
        
        # Compute accuracy metrics
        masked_logits = logits.clone()
        masked_logits[~action_mask] = float('-inf')
        predicted_actions = masked_logits.argmax(dim=-1)
        correct = (predicted_actions == actions).float()
        total_correct += correct.sum().item()
        
        # Track win/loss accuracy
        win_mask = rewards > 0
        loss_mask = rewards < 0
        if win_mask.any():
            total_win_correct += correct[win_mask].sum().item()
            total_win_samples += win_mask.sum().item()
        if loss_mask.any():
            total_loss_correct += correct[loss_mask].sum().item()
            total_loss_samples += loss_mask.sum().item()

        batch_time = time.time() - batch_start_time
        
        # TensorBoard logging per batch
        if writer is not None and batch_idx % TENSORBOARD_LOG_INTERVAL == 0:
            global_step = epoch * len(train_loader) + batch_idx
            
            # Loss metrics
            writer.add_scalar('RL_Train/BatchLoss', loss_dict['loss'].item(), global_step)
            writer.add_scalar('RL_Train/PolicyLoss', loss_dict['policy_loss'].item(), global_step)
            writer.add_scalar('RL_Train/Entropy', loss_dict['entropy'].item(), global_step)
            writer.add_scalar('RL_Train/MeanReward', loss_dict['mean_reward'].item(), global_step)
            
            # Gradient metrics
            writer.add_scalar('RL_Train/GradientNormBefore', grad_norm_before, global_step)
            writer.add_scalar('RL_Train/GradientNormAfter', grad_norm_after, global_step)
            writer.add_scalar('RL_Train/GradientClipped', float(grad_norm_before > max_grad_norm), global_step)
            
            # Learning rate
            writer.add_scalar('RL_Train/LearningRate', optimizer.param_groups[0]['lr'], global_step)
            
            # Batch accuracy
            batch_accuracy = correct.mean().item()
            writer.add_scalar('RL_Train/BatchAccuracy', batch_accuracy, global_step)
            
            # Timing
            writer.add_scalar('RL_Train/BatchTime', batch_time, global_step)
            writer.add_scalar('RL_Train/SamplesPerSec', batch_size / batch_time if batch_time > 0 else 0, global_step)
            
            # Probability distribution stats
            probs = F.softmax(masked_logits, dim=-1)
            writer.add_scalar('RL_Train/MaxProb', probs.max(dim=-1).values.mean().item(), global_step)
            writer.add_scalar('RL_Train/MinProb', probs[action_mask].min().item(), global_step)
        
        # Weight and gradient histograms (less frequent)
        if writer is not None and batch_idx % TENSORBOARD_HISTOGRAM_INTERVAL == 0:
            global_step = epoch * len(train_loader) + batch_idx
            
            if LOG_WEIGHT_HISTOGRAMS:
                for name, param in model.named_parameters():
                    if param.requires_grad:
                        writer.add_histogram(f'Weights/{name}', param.data, global_step)
            
            if LOG_GRADIENT_HISTOGRAMS:
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        writer.add_histogram(f'Gradients/{name}', param.grad.data, global_step)

    epoch_time = time.time() - epoch_start_time
    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
    avg_policy_loss = total_policy_loss / total_samples if total_samples > 0 else 0.0
    avg_entropy = total_entropy / total_samples if total_samples > 0 else 0.0
    avg_reward = total_reward / total_samples if total_samples > 0 else 0.0
    accuracy = total_correct / total_samples if total_samples > 0 else 0.0
    win_accuracy = total_win_correct / total_win_samples if total_win_samples > 0 else 0.0
    loss_accuracy = total_loss_correct / total_loss_samples if total_loss_samples > 0 else 0.0
    samples_per_sec = total_samples / epoch_time if epoch_time > 0 else 0.0

    metrics = {
        'loss': avg_loss,
        'policy_loss': avg_policy_loss,
        'entropy': avg_entropy,
        'mean_reward': avg_reward,
        'accuracy': accuracy,
        'win_accuracy': win_accuracy,
        'loss_accuracy': loss_accuracy,
        'samples_per_sec': samples_per_sec,
        'epoch_time': epoch_time,
    }

    if writer is not None:
        # Epoch-level metrics
        writer.add_scalar('RL_Train/EpochLoss', avg_loss, epoch)
        writer.add_scalar('RL_Train/EpochPolicyLoss', avg_policy_loss, epoch)
        writer.add_scalar('RL_Train/EpochEntropy', avg_entropy, epoch)
        writer.add_scalar('RL_Train/EpochMeanReward', avg_reward, epoch)
        writer.add_scalar('RL_Train/EpochAccuracy', accuracy, epoch)
        writer.add_scalar('RL_Train/WinAccuracy', win_accuracy, epoch)
        writer.add_scalar('RL_Train/LossAccuracy', loss_accuracy, epoch)
        writer.add_scalar('RL_Train/SamplesPerSecEpoch', samples_per_sec, epoch)
        writer.add_scalar('RL_Train/EpochTime', epoch_time, epoch)

    return metrics


def validate_rl(
    model: PolicyLargeNet,
    val_loader,
    device: torch.device,
    writer: Optional[SummaryWriter] = None,
    epoch: int = 0,
    entropy_coef: float = 0.01,
) -> Dict[str, float]:
    """
    Validate the model on held-out data.

    Args:
        model: Policy network
        val_loader: Validation data loader
        device: Device to use
        writer: Optional TensorBoard writer
        epoch: Current epoch number
        entropy_coef: Entropy bonus coefficient

    Returns:
        Dictionary with validation metrics
    """
    model.eval()
    total_loss = 0.0
    total_policy_loss = 0.0
    total_entropy = 0.0
    total_correct = 0
    total_win_correct = 0
    total_win_samples = 0
    total_loss_correct = 0
    total_loss_samples = 0
    total_samples = 0

    with torch.no_grad():
        for batch in val_loader:
            (scalar_features, sequence_features, unknown_stats, total_stats,
             room_features, room_mask, dungeon_len, actions, rewards, action_mask) = batch

            scalar_features = scalar_features.to(device)
            sequence_features = sequence_features.to(device)
            unknown_stats = unknown_stats.to(device)
            total_stats = total_stats.to(device)
            room_features = room_features.to(device)
            room_mask = room_mask.to(device)
            dungeon_len = dungeon_len.to(device)
            actions = actions.to(device)
            rewards = rewards.to(device)
            action_mask = action_mask.to(device)

            logits = model(
                scalar_features, sequence_features, unknown_stats, total_stats,
                room_features=room_features, room_mask=room_mask, dungeon_len=dungeon_len
            )

            loss_dict = compute_policy_gradient_loss(
                logits, actions, rewards, action_mask, entropy_coef
            )

            # Compute accuracy (how often model agrees with action taken)
            masked_logits = logits.clone()
            masked_logits[~action_mask] = float('-inf')
            predicted_actions = masked_logits.argmax(dim=-1)
            correct = (predicted_actions == actions).float()

            # Track accuracy separately for wins and losses
            win_mask = rewards > 0
            loss_mask = rewards < 0

            batch_size = scalar_features.size(0)
            total_loss += loss_dict['loss'].item() * batch_size
            total_policy_loss += loss_dict['policy_loss'].item() * batch_size
            total_entropy += loss_dict['entropy'].item() * batch_size
            total_correct += correct.sum().item()

            if win_mask.any():
                total_win_correct += correct[win_mask].sum().item()
                total_win_samples += win_mask.sum().item()
            if loss_mask.any():
                total_loss_correct += correct[loss_mask].sum().item()
                total_loss_samples += loss_mask.sum().item()

            total_samples += batch_size

    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
    avg_policy_loss = total_policy_loss / total_samples if total_samples > 0 else 0.0
    avg_entropy = total_entropy / total_samples if total_samples > 0 else 0.0
    accuracy = total_correct / total_samples if total_samples > 0 else 0.0
    win_accuracy = total_win_correct / total_win_samples if total_win_samples > 0 else 0.0
    loss_accuracy = total_loss_correct / total_loss_samples if total_loss_samples > 0 else 0.0

    metrics = {
        'loss': avg_loss,
        'policy_loss': avg_policy_loss,
        'entropy': avg_entropy,
        'accuracy': accuracy,
        'win_accuracy': win_accuracy,
        'loss_accuracy': loss_accuracy,
    }

    if writer is not None:
        writer.add_scalar('RL_Val/Loss', avg_loss, epoch)
        writer.add_scalar('RL_Val/PolicyLoss', avg_policy_loss, epoch)
        writer.add_scalar('RL_Val/Entropy', avg_entropy, epoch)
        writer.add_scalar('RL_Val/Accuracy', accuracy, epoch)
        writer.add_scalar('RL_Val/WinAccuracy', win_accuracy, epoch)
        writer.add_scalar('RL_Val/LossAccuracy', loss_accuracy, epoch)

    return metrics


def train_policy_rl(
    games_dir: Path,
    checkpoint_path: Path,
    pretrained_checkpoint: Optional[Path],
    writer: Optional[SummaryWriter],
    iteration: int,
    device: torch.device,
    epochs: int = 20,
    batch_size: int = 256,
    lr: float = 1e-4,
    max_grad_norm: float = 1.0,
    entropy_coef: float = 0.01,
    reward_type: str = "binary",
    stack_seq_len: int = 40,
    verbose: bool = False,
) -> Dict[str, float]:
    """
    Train policy network using REINFORCE policy gradients.

    Args:
        games_dir: Directory containing game data
        checkpoint_path: Path to save trained checkpoint
        pretrained_checkpoint: Path to pretrained checkpoint to fine-tune from
        writer: TensorBoard writer
        iteration: Current iteration number
        device: Device to use
        epochs: Number of training epochs
        batch_size: Batch size
        lr: Learning rate
        max_grad_norm: Gradient clipping threshold
        entropy_coef: Entropy bonus coefficient
        reward_type: How to compute rewards ("binary", "normalized", "scaled")
        stack_seq_len: Sequence length for translator
        verbose: Print progress information

    Returns:
        Final validation metrics
    """
    training_start_time = time.time()
    
    # Set up translator and data
    translator = ScoundrelTranslator(stack_seq_len=stack_seq_len)
    dummy_state, _ = translator.encode_state(GameManager().restart())
    scalar_input_dim = dummy_state.shape[1]

    # Create RL data loaders
    train_loader, val_loader = create_rl_dataloaders(
        log_dir=games_dir,
        translator=translator,
        batch_size=batch_size,
        train_val_split=0.9,
        max_games=None,
        reward_type=reward_type,
    )

    # Initialize model
    model = PolicyLargeNet(scalar_input_dim=scalar_input_dim).to(device)
    
    # Log model info to TensorBoard
    if writer is not None:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        writer.add_scalar('Policy/TotalParams', total_params, iteration)
        writer.add_scalar('Policy/TrainableParams', trainable_params, iteration)
        writer.add_scalar('Policy/TrainSamples', len(train_loader.dataset), iteration)
        writer.add_scalar('Policy/ValSamples', len(val_loader.dataset), iteration)
        
        # Log hyperparameters
        writer.add_text(f'Policy/Hyperparams_Iter{iteration}', 
            f"LR: {lr}, Entropy: {entropy_coef}, Reward: {reward_type}, Epochs: {epochs}", iteration)

    # Load pretrained weights if available
    pretrained_epoch = None
    pretrained_iteration = None
    if pretrained_checkpoint and pretrained_checkpoint.exists():
        if verbose:
            print(f"  Loading pretrained policy weights from: {pretrained_checkpoint}")
        checkpoint = torch.load(pretrained_checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        pretrained_epoch = checkpoint.get('epoch', 0)
        pretrained_iteration = checkpoint.get('iteration', 0)
        if verbose:
            print(f"    Loaded from epoch {pretrained_epoch}, iteration {pretrained_iteration}")
        
        if writer is not None:
            writer.add_text(f'Policy/PretrainedFrom_Iter{iteration}', str(pretrained_checkpoint), iteration)
    else:
        if verbose:
            print("  No pretrained policy checkpoint found, training from scratch")

    # Initialize optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.1)

    best_loss = float('inf')
    best_accuracy = 0.0
    final_metrics = {}
    best_epoch = 0

    if verbose:
        print(f"  Train samples: {len(train_loader.dataset)}, Val samples: {len(val_loader.dataset)}")
        print(f"  Reward type: {reward_type}, Entropy coef: {entropy_coef}")

    # Training loop
    for epoch in range(epochs):
        global_epoch = epoch + iteration * epochs
        
        train_metrics = train_rl_epoch(
            model, train_loader, optimizer, device, writer,
            epoch=epoch,
            max_grad_norm=max_grad_norm,
            entropy_coef=entropy_coef,
        )

        val_metrics = validate_rl(
            model, val_loader, device, writer,
            epoch=epoch,
            entropy_coef=entropy_coef,
        )

        final_metrics = val_metrics
        
        # Step the scheduler
        scheduler.step()

        if verbose:
            print(f"  Epoch {epoch+1}/{epochs}: "
                  f"Loss: {train_metrics['loss']:.4f}, "
                  f"Entropy: {train_metrics['entropy']:.4f}, "
                  f"Val Acc: {val_metrics['accuracy']:.4f}, "
                  f"Win Acc: {val_metrics['win_accuracy']:.4f}, "
                  f"LR: {scheduler.get_last_lr()[0]:.2e}")

        # Track best model (lowest loss)
        if val_metrics['loss'] < best_loss:
            best_loss = val_metrics['loss']
            best_accuracy = val_metrics['accuracy']
            best_epoch = epoch
            
            torch.save({
                'epoch': epoch,
                'global_epoch': global_epoch,
                'iteration': iteration,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': val_metrics['loss'],
                'val_accuracy': val_metrics['accuracy'],
                'val_win_accuracy': val_metrics['win_accuracy'],
                'val_loss_accuracy': val_metrics['loss_accuracy'],
                'train_loss': train_metrics['loss'],
                'train_accuracy': train_metrics['accuracy'],
                'training_type': 'reinforce',
                'reward_type': reward_type,
                'entropy_coef': entropy_coef,
                'lr': lr,
                'scalar_input_dim': scalar_input_dim,
                'pretrained_checkpoint': str(pretrained_checkpoint) if pretrained_checkpoint else None,
                'pretrained_epoch': pretrained_epoch,
                'pretrained_iteration': pretrained_iteration,
            }, checkpoint_path)
            
            if writer is not None:
                writer.add_scalar('Policy/BestValLoss', best_loss, epoch)
                writer.add_scalar('Policy/BestValAccuracy', best_accuracy, epoch)

    training_time = time.time() - training_start_time
    
    if verbose:
        print(f"  Best RL policy - Epoch: {best_epoch+1}, Loss: {best_loss:.4f}, Accuracy: {best_accuracy:.4f}")
        print(f"  Policy training completed in {training_time:.1f}s")
    
    # Include additional info in returned metrics
    final_metrics['best_loss'] = best_loss
    final_metrics['best_accuracy'] = best_accuracy
    final_metrics['best_epoch'] = best_epoch
    final_metrics['training_time'] = training_time

    return final_metrics

