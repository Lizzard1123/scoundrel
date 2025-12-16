import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from collections import deque

from scoundrel.game.game_manager import GameManager
from scoundrel.rl.transformer_mlp.constants import (
    TRAIN_MAX_EPISODES,
    TRAIN_MAX_STEPS_PER_EPISODE,
    TRAIN_SAVE_INTERVAL,
    TRAIN_UPDATE_TIMESTEP,
    TRAIN_RESUME_FROM,
    STACK_SEQ_LEN,
)
from scoundrel.rl.transformer_mlp.ppo_agent import PPOAgent
from scoundrel.rl.translator import ScoundrelTranslator

# Simple Memory Buffer
class Memory:
    def __init__(self):
        self.actions = []
        self.states_scal = []
        self.states_seq = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
        self.masks = []

    def clear(self):
        del self.actions[:]
        del self.states_scal[:]
        del self.states_seq[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]
        del self.masks[:]

def _default_paths(base_dir: Path):
    runs = base_dir / "runs"
    checkpoints = base_dir / "checkpoints" / "ppo_latest.pt"
    runs.mkdir(parents=True, exist_ok=True)
    checkpoints.parent.mkdir(parents=True, exist_ok=True)
    return runs, checkpoints


def train_scoundrel(
    engine_instance=None,
    max_episodes=TRAIN_MAX_EPISODES,
    log_dir=None,
    checkpoint_path=None,
    tensorboard=True,
    save_interval=TRAIN_SAVE_INTERVAL,
    resume_path=TRAIN_RESUME_FROM,
):
    """
    Main entrypoint to train the model.
    """
    base_dir = Path(__file__).parent
    default_logdir, default_checkpoint = _default_paths(base_dir)
    log_dir = Path(log_dir) if log_dir else default_logdir
    checkpoint_path = Path(checkpoint_path) if checkpoint_path else default_checkpoint
    log_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    if engine_instance is None:
        engine = GameManager()
    else:
        engine = engine_instance

    print(f"--- Starting Scoundrel Training (PPO) ---")

    # Initialize objects
    translator = ScoundrelTranslator(stack_seq_len=STACK_SEQ_LEN)
    game_state = engine.restart()
    s_scal, s_seq = translator.encode_state(game_state)
    scalar_dim = s_scal.shape[1]

    agent = PPOAgent(input_dim=scalar_dim)
    memory = Memory()
    writer = SummaryWriter(log_dir=str(log_dir)) if tensorboard else None

    # Optional resume
    if resume_path:
        resume_path = Path(resume_path)
        if resume_path.exists():
            state_dict = torch.load(resume_path, map_location="cpu")
            agent.policy.load_state_dict(state_dict)
            agent.policy_old.load_state_dict(state_dict)
            print(f"Resumed training from checkpoint: {resume_path}")
        else:
            print(f"Resume path not found, starting fresh: {resume_path}")

    reward_window = deque(maxlen=TRAIN_UPDATE_TIMESTEP)
    best_raw_score = float("-inf")
    update_timestep = TRAIN_UPDATE_TIMESTEP
    time_step = 0
    update_step = 0

    for i_episode in range(1, max_episodes+1):
        state = engine.restart()
        ep_reward = 0

        for t in range(TRAIN_MAX_STEPS_PER_EPISODE): # Max steps per game
            time_step += 1

            # 1. Translate State
            s_scal, s_seq = translator.encode_state(state)
            mask = translator.get_action_mask(state)

            # 2. Select Action
            action, log_prob, _ = agent.select_action(s_scal, s_seq, mask)

            # 3. Execute in Engine
            # (Bridge: Decode index 0-4 to engine command)
            engine_action = translator.decode_action(action)
            engine.execute_turn(engine_action)
            next_state = engine.get_state()
            done = next_state.game_over
            # Normalize reward to [0, 1]: min=-188, max=30, range=218
            reward = ((next_state.score + 188) / 218) if done else 0

            # 4. Save to Memory
            memory.states_scal.append(s_scal)
            memory.states_seq.append(s_seq)
            memory.actions.append(action)
            memory.logprobs.append(log_prob)
            memory.rewards.append(reward)
            memory.is_terminals.append(done)
            memory.masks.append(mask)

            state = next_state
            ep_reward += reward

            # 5. Update PPO
            if time_step % update_timestep == 0:
                metrics = agent.update(memory)
                if writer and metrics:
                    writer.add_scalar("loss/total", metrics["loss"], update_step)
                    writer.add_scalar("loss/policy", metrics["policy_loss"], update_step)
                    writer.add_scalar("loss/value", metrics["value_loss"], update_step)
                    writer.add_scalar("policy/entropy", metrics["entropy"], update_step)
                    writer.add_scalar("policy/approx_kl", metrics["approx_kl"], update_step)
                update_step += 1
                memory.clear()
                time_step = 0

            if done:
                break

        reward_window.append(ep_reward)
        # Convert normalized reward back to raw score for best tracking
        current_raw_score = ep_reward * 218 - 188
        best_raw_score = max(best_raw_score, current_raw_score)

        # TensorBoard logging
        if writer:
            writer.add_scalar("episode/reward", ep_reward, i_episode)
            writer.add_scalar("episode/steps", t + 1, i_episode)
            writer.add_scalar("episode/best_raw_score_so_far", best_raw_score, i_episode)
            writer.add_scalar("episode/avg_reward_window", sum(reward_window) / len(reward_window), i_episode)


        # Periodic checkpointing
        if save_interval and (i_episode % save_interval == 0):
            periodic_checkpoint = checkpoint_path.parent / f"ppo_episode_{i_episode}.pt"
            torch.save(agent.policy.state_dict(), periodic_checkpoint)

    print("--- Training Complete ---")
    # Final checkpoint
    torch.save(agent.policy.state_dict(), checkpoint_path)
    if writer:
        writer.flush()
        writer.close()
    return agent

def sample_from_model(agent, engine_state):
    """
    Example of how to use the trained model for a single inference step.
    Useful for running the GUI or debugging.
    """
    translator = ScoundrelTranslator()

    # Encode
    s_scal, s_seq = translator.encode_state(engine_state)
    mask = translator.get_action_mask(engine_state)

    # Infer
    agent.policy.eval() # Set to eval mode
    with torch.no_grad():
        logits, _ = agent.policy(s_scal, s_seq)

        # Apply mask
        logits[~mask.unsqueeze(0)] = float('-inf')
        probs = F.softmax(logits, dim=-1)

        # Greedy selection for inference (or sample if you prefer variety)
        action_idx = int(torch.argmax(probs).item())

    action_str = translator.decode_action(action_idx)

    print(f"Model sees probabilities: {probs.numpy()}")
    print(f"Model chooses: {action_str}")
    return action_idx

def parse_args():
    parser = argparse.ArgumentParser(description="Train Scoundrel PPO with optional TensorBoard logging.")
    parser.add_argument("--max-episodes", type=int, default=TRAIN_MAX_EPISODES, help="Number of episodes to train.")
    parser.add_argument("--logdir", type=str, default=None, help="Directory for TensorBoard runs.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to save the final checkpoint.")
    parser.add_argument("--save-interval", type=int, default=TRAIN_SAVE_INTERVAL, help="Episodes between checkpoints (0 to save only at end).")
    parser.add_argument("--resume-from", type=str, default=TRAIN_RESUME_FROM, help="Optional checkpoint path to resume training from.")
    parser.add_argument("--no-tensorboard", action="store_true", help="Disable TensorBoard logging.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    trained_agent = train_scoundrel(
        max_episodes=args.max_episodes,
        log_dir=args.logdir,
        checkpoint_path=args.checkpoint,
        tensorboard=not args.no_tensorboard,
        save_interval=args.save_interval,
        resume_path=args.resume_from,
    )

    # Sample run
    # mock_env = GameManager()
    # state = mock_env.restart()
    # sample_from_model(trained_agent, state)