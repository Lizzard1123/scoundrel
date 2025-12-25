import argparse
from pathlib import Path
from typing import Any

import torch

from scoundrel.game.game_manager import GameManager
from scoundrel.models.game_state import Action, GameState
from scoundrel.rl.alpha_scoundrel.value.value_large.network import ValueLargeNet
from scoundrel.rl.alpha_scoundrel.value.value_large.constants import STACK_SEQ_LEN
from scoundrel.rl.translator import ScoundrelTranslator
from scoundrel.rl.viewer import run_interactive_viewer


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


def _get_value_prediction(model: ValueLargeNet, translator: ScoundrelTranslator, state: GameState):
    """
    Get value prediction from model.
    
    Args:
        model: Trained ValueLargeNet model
        translator: ScoundrelTranslator for encoding states
        state: Current game state
        
    Returns:
        Tuple of (action_enum, value_prediction)
    """
    s_scal, s_seq = translator.encode_state(state)
    mask = translator.get_action_mask(state)

    with torch.no_grad():
        value_pred = model(s_scal, s_seq)
        value = float(value_pred.squeeze().item())

    # For value network viewer, we'll use a simple greedy policy based on value estimates
    # This is a placeholder - in practice, you'd want to evaluate each action's value
    # For now, we'll just return a random valid action
    valid_actions = [i for i in range(5) if mask[i]]
    if valid_actions:
        action_idx = valid_actions[0]  # Simple: take first valid action
    else:
        action_idx = 0
    
    action_enum = translator.decode_action(action_idx)
    return action_enum, value


def run_viewer(checkpoint: Path, label: str, seed: int = None):
    """
    Run interactive viewer for trained Value Large model.
    
    Args:
        checkpoint: Path to checkpoint file
        label: Label to display in UI
        seed: Optional seed for deterministic deck shuffling (same seed = same game sequence)
    """
    translator = ScoundrelTranslator(stack_seq_len=STACK_SEQ_LEN)
    
    temp_engine = GameManager(seed=seed)
    init_state = temp_engine.restart()
    s_scal, _ = translator.encode_state(init_state)
    scalar_input_dim = s_scal.shape[1]
    
    checkpoint_data = torch.load(checkpoint, map_location="cpu")
    if isinstance(checkpoint_data, dict) and 'scalar_input_dim' in checkpoint_data:
        scalar_input_dim = checkpoint_data['scalar_input_dim']
    
    model = _load_model(checkpoint, scalar_input_dim=scalar_input_dim)
    
    def get_value_action(state: GameState) -> tuple[Action, Any]:
        """Get action and value prediction."""
        return _get_value_prediction(model, translator, state)
    
    run_interactive_viewer(
        get_action_fn=get_value_action,
        label=label,
        checkpoint_name=checkpoint.name,
        seed=seed,
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

