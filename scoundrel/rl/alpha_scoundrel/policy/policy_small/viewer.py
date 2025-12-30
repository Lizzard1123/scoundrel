import argparse
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from scoundrel.game.game_manager import GameManager
from scoundrel.models.game_state import Action, GameState
from scoundrel.rl.alpha_scoundrel.policy.policy_small.network import PolicySmallNet
from scoundrel.rl.alpha_scoundrel.policy.policy_small.constants import STACK_SEQ_LEN
from scoundrel.rl.alpha_scoundrel.policy.policy_small.data_loader import (
    compute_stack_sums,
    compute_total_stats,
)
from scoundrel.rl.translator import ScoundrelTranslator
from scoundrel.rl.utils import mask_logits
from scoundrel.rl.viewer import run_interactive_viewer


def _load_model(checkpoint_path: Path, scalar_input_dim: int) -> PolicySmallNet:
    """
    Load trained model from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        scalar_input_dim: Scalar input dimension
        
    Returns:
        Loaded model in eval mode
    """
    model = PolicySmallNet(scalar_input_dim=scalar_input_dim)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    return model


def _greedy_action(model: PolicySmallNet, translator: ScoundrelTranslator, state: GameState):
    """
    Get greedy action from model.
    
    Args:
        model: Trained PolicySmallNet model
        translator: ScoundrelTranslator for encoding states
        state: Current game state
        
    Returns:
        Tuple of (action_enum, action_probs)
    """
    s_scal, _ = translator.encode_state(state)
    stack_sums = compute_stack_sums(state)
    total_stats = compute_total_stats(state)
    mask = translator.get_action_mask(state)

    with torch.no_grad():
        logits = model(s_scal, stack_sums.unsqueeze(0), total_stats.unsqueeze(0))
        masked_logits = mask_logits(logits, mask)
        probs = F.softmax(masked_logits, dim=-1)
        action_idx = int(torch.argmax(probs).item())

    action_enum = translator.decode_action(action_idx)
    return action_enum, probs.squeeze(0)


def run_viewer(checkpoint: Path, label: str, seed: int = None):
    """
    Run interactive viewer for trained Policy Small model.
    
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
    
    def get_policy_action(state: GameState) -> tuple[Action, Any]:
        """Get policy action."""
        return _greedy_action(model, translator, state)
    
    run_interactive_viewer(
        get_action_fn=get_policy_action,
        label=label,
        checkpoint_name=checkpoint.name,
        seed=seed,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="View a Policy Small checkpoint playing Scoundrel.")
    default_ckpt = Path(__file__).parent / "checkpoints" / "policy_small_epoch_100.pt"
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(default_ckpt) if default_ckpt.exists() else None,
        help="Path to Policy Small checkpoint (.pt).",
    )
    parser.add_argument(
        "--label",
        type=str,
        default="Policy Small",
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

