import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from scoundrel.game.game_manager import GameManager
from scoundrel.models.game_state import Action
from scoundrel.rl.transformer_mcts.ppo_agent import PPOAgent
from scoundrel.rl.transformer_mcts.transformer_mcts import _default_paths
from scoundrel.rl.transformer_mcts.translator import ScoundrelTranslator


def _load_agent(checkpoint_path: Path, input_dim: int) -> PPOAgent:
    agent = PPOAgent(input_dim=input_dim)
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    agent.policy.load_state_dict(state_dict)
    agent.policy_old.load_state_dict(state_dict)
    agent.policy.eval()
    agent.policy_old.eval()
    return agent


def _format_action(action: Action) -> str:
    if action == Action.AVOID:
        return "avoid"
    if action in {Action.USE_1, Action.USE_2, Action.USE_3, Action.USE_4}:
        return f"use {action.value + 1}"
    return action.name.lower()


def _greedy_action(agent: PPOAgent, translator: ScoundrelTranslator, state):
    s_scal, s_seq = translator.encode_state(state)
    mask = translator.get_action_mask(state)

    with torch.no_grad():
        logits, _ = agent.policy(s_scal, s_seq)
        masked_logits = logits.clone()
        masked_logits[~mask.unsqueeze(0)] = float("-inf")
        probs = F.softmax(masked_logits, dim=-1)
        action_idx = int(torch.argmax(probs).item())

    action_enum = translator.decode_action(action_idx)
    return action_enum, probs.squeeze(0)


def run_viewer(checkpoint: Path, label: str):
    engine = GameManager()
    translator = ScoundrelTranslator()

    # Determine input dim
    init_state = engine.restart()
    s_scal, _ = translator.encode_state(init_state)
    agent = _load_agent(checkpoint, input_dim=s_scal.shape[1])

    actions_title = f"{label} — {checkpoint.name}"

    state = engine.get_state()
    while not state.exit:
        # Sample next action greedily
        action_enum, probs = _greedy_action(agent, translator, state)
        action_text = _format_action(action_enum)
        ui_text = f"Next action: [bold]{action_text}[/bold]"

        if state.game_over:
            ui_text += " | press 'r' to restart or 'q' to quit"

        engine.ui.display_game_state(
            state,
            actions_override=ui_text,
            actions_title=actions_title,
        )

        user = input("Space=step | r=restart | q=quit: ").strip().lower()

        if user in ("q", "quit"):
            engine.execute_turn(Action.EXIT)
            state = engine.get_state()
            break
        if user in ("r", "restart"):
            state = engine.restart()
            continue
        if state.game_over:
            # Ignore other inputs while game over
            continue
        if user in ("", " ", "s", "step"):
            engine.execute_turn(action_enum)
            state = engine.get_state()
            continue
        # Unrecognized input: loop and resample
        state = engine.get_state()


def parse_args():
    parser = argparse.ArgumentParser(description="View a checkpoint playing Scoundrel.")
    default_logdir, default_ckpt = _default_paths(Path(__file__).parent)
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(default_ckpt),
        help="Path to PPO checkpoint (.pt).",
    )
    parser.add_argument(
        "--label",
        type=str,
        default="PPO (latest)",
        help="Label shown in the actions panel title.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    run_viewer(ckpt_path, args.label)
