import torch
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from collections import deque

from scoundrel.game.game_manager import GameManager
from scoundrel.rl.transformer_mcts.ppo_agent import PPOAgent
from scoundrel.rl.transformer_mcts.translator import ScoundrelTranslator

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

def train_scoundrel(engine_instance=None, max_episodes=1000):
    """
    Main entrypoint to train the model.
    """
    if engine_instance is None:
        engine = GameManager()
    else:
        engine = engine_instance

    print(f"--- Starting Scoundrel Training (PPO) ---")

    # Initialize objects
    translator = ScoundrelTranslator()
    game_state = engine.restart()
    s_scal, s_seq = translator.encode_state(game_state)
    scalar_dim = s_scal.shape[1]

    agent = PPOAgent(input_dim=scalar_dim)
    memory = Memory()

    running_reward = 0
    update_timestep = 2000
    time_step = 0

    for i_episode in range(1, max_episodes+1):
        state = engine.restart()
        ep_reward = 0

        for t in range(100): # Max steps per game
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
            done = not next_state.game_over
            reward = next_state.score

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
                agent.update(memory)
                memory.clear()
                time_step = 0

            if done:
                break

        running_reward += ep_reward

        # Console Logging
        if i_episode % 50 == 0:
            avg_reward = running_reward / 50
            print(f"Episode {i_episode} \t Avg Reward: {avg_reward:.2f}")
            running_reward = 0

    print("--- Training Complete ---")
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

if __name__ == "__main__":
    # Test run with Mock Engine
    trained_agent = train_scoundrel(max_episodes=200)

    # Sample run
    mock_env = GameManager()
    state = mock_env.restart()
    sample_from_model(trained_agent, state)