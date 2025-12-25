import argparse
from pathlib import Path
import torch

from scoundrel.models.game_state import GameState
from scoundrel.rl.viewer import run_state_viewer
from scoundrel.rl.alpha_scoundrel.value.value_large.network import ValueLargeNet
from scoundrel.rl.alpha_scoundrel.value.value_large.constants import STACK_SEQ_LEN
from scoundrel.rl.alpha_scoundrel.value.value_large.data_loader import compute_unknown_stats
from scoundrel.rl.translator import ScoundrelTranslator
from scoundrel.game.game_manager import GameManager


def _load_model(checkpoint_path: Path, scalar_input_dim: int) -> ValueLargeNet:
    """
    Load trained model from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        scalar_input_dim: Scalar input dimension
        
    Returns:
        Loaded model in eval mode
    """
    model = ValueLargeNet(scalar_input_dim=scalar_input_dim)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    return model


def _get_colored_score_text(state: GameState) -> str:
    """
    Get colored score text based on game state.
    
    Args:
        state: Current game state
        
    Returns:
        Colored score string
    """
    score = state.score
    
    # Determine color based on state
    if state.exit:
        color = "yellow"
    elif state.lost:
        color = "red"
    elif score > 0:
        color = "green"
    elif score == 0:
        color = "yellow"
    else:
        color = "red"
    
    return f"[{color}]{score:+d}[/{color}]"


def _get_colored_predicted_text(predicted_value: float) -> str:
    """
    Get colored text for predicted value.
    
    Args:
        predicted_value: Predicted final score
        
    Returns:
        Colored predicted value string
    """
    # Determine color based on predicted value
    if predicted_value > 0:
        color = "green"
    elif predicted_value == 0:
        color = "yellow"
    else:
        color = "red"
    
    return f"[{color}]{predicted_value:+.1f}[/{color}]"


def run_viewer(checkpoint: Path, label: str, seed: int = None):
    """
    Run interactive viewer for Value Large model.
    Shows game state with current score and predicted final score in title.
    
    Args:
        checkpoint: Path to checkpoint file
        label: Label to display in UI
        seed: Optional seed for deterministic deck shuffling (same seed = same game sequence)
    """
    translator = ScoundrelTranslator(stack_seq_len=STACK_SEQ_LEN)
    
    # Initialize model
    temp_engine = GameManager(seed=seed)
    init_state = temp_engine.restart()
    s_scal, _ = translator.encode_state(init_state)
    scalar_input_dim = s_scal.shape[1]
    
    checkpoint_data = torch.load(checkpoint, map_location="cpu")
    if isinstance(checkpoint_data, dict) and 'scalar_input_dim' in checkpoint_data:
        scalar_input_dim = checkpoint_data['scalar_input_dim']
    
    model = _load_model(checkpoint, scalar_input_dim=scalar_input_dim)
    device = torch.device("cpu")  # Use CPU for inference
    
    def get_title_fn(state: GameState) -> str:
        """Get dynamic title with current score and predicted value."""
        base_title = f"{label}"
        if checkpoint.name:
            base_title += f" — {checkpoint.name}"
        
        # Get current score
        score_text = _get_colored_score_text(state)
        
        # Get predicted value
        with torch.no_grad():
            scalar_features, sequence_features = translator.encode_state(state)
            unknown_stats = compute_unknown_stats(state)
            
            # Add batch dimension
            scalar_features = scalar_features.to(device)
            sequence_features = sequence_features.to(device)
            unknown_stats = unknown_stats.unsqueeze(0).to(device)
            
            predicted_value = model(scalar_features, sequence_features, unknown_stats)
            predicted_value = predicted_value.item()
        
        predicted_text = _get_colored_predicted_text(predicted_value)
        
        return f"{base_title} | Score: {score_text} | Predicted: {predicted_text}"
    
    run_state_viewer(
        label=label,
        checkpoint_name=checkpoint.name,
        seed=seed,
        get_title_fn=get_title_fn,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="View a Value Large checkpoint playing Scoundrel.")
    default_ckpt = Path(__file__).parent / "checkpoints" / "value_large_epoch_100.pt"
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(default_ckpt) if default_ckpt.exists() else None,
        help="Path to Value Large checkpoint (.pt).",
    )
    parser.add_argument(
        "--label",
        type=str,
        default="Value Large",
        help="Label shown in the actions panel title.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed for deterministic deck shuffling (same seed = same game sequence)"
    )
    return parser.parse_args()


def main():
    """Entry point for console script."""
    args = parse_args()
    if args.checkpoint is None:
        raise ValueError("No checkpoint specified and no default checkpoint found. Use --checkpoint to specify a checkpoint.")
    
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    run_viewer(ckpt_path, args.label, seed=args.seed)


if __name__ == "__main__":
    main()
