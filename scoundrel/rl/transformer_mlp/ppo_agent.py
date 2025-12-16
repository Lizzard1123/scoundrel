import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from scoundrel.rl.transformer_mlp.constants import EPS_CLIP, GAMMA, K_EPOCHS, LR
from scoundrel.rl.transformer_mlp.network import ScoundrelNet


class PPOAgent:
    def __init__(self, input_dim):
        self.policy = ScoundrelNet(input_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=LR)
        self.policy_old = ScoundrelNet(input_dim)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.MseLoss = nn.MSELoss()

    def select_action(self, scalar_in, seq_in, mask):
        """
        Selects an action given state, handling masking for invalid moves.
        """
        with torch.no_grad():
            logits, val = self.policy_old(scalar_in, seq_in)

            # Apply Masking: Set invalid actions to -inf
            masked_logits = logits.clone()
            masked_logits[~mask.unsqueeze(0)] = float('-inf')

            probs = F.softmax(masked_logits, dim=-1)
            dist = Categorical(probs)
            action = dist.sample()

            return int(action.item()), dist.log_prob(action), val

    def update(self, memory):
        # Convert memory lists to tensors
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (GAMMA * discounted_reward)
            rewards.insert(0, discounted_reward)

        rewards = torch.tensor(rewards, dtype=torch.float32)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7) # Normalize

        # Stack tensors
        old_states_scal = torch.cat(memory.states_scal)
        old_states_seq = torch.cat(memory.states_seq)
        old_actions = torch.tensor(memory.actions)
        old_logprobs = torch.tensor(memory.logprobs)
        old_masks = torch.stack(memory.masks)

        # PPO Update Loop
        last_metrics = {}
        for _ in range(K_EPOCHS):
            logits, state_values = self.policy(old_states_scal, old_states_seq)

            # Re-apply masking for update
            masked_logits = logits.clone()
            # We must iterate or broadcast the mask carefully
            # Ideally, masking happens inside the model or loss, but here we just mask logits before softmax
            masked_logits[~old_masks] = float('-inf')

            dist = Categorical(F.softmax(masked_logits, dim=-1))
            logprobs = dist.log_prob(old_actions)
            dist_entropy = dist.entropy()
            state_values = torch.squeeze(state_values)

            # Ratios
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Surrogate Loss
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-EPS_CLIP, 1+EPS_CLIP) * advantages

            policy_loss = (-torch.min(surr1, surr2)).mean()
            value_loss = self.MseLoss(state_values, rewards)
            entropy = dist_entropy.mean()

            loss = policy_loss + 0.5 * value_loss - 0.01 * entropy

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Keep the latest metrics for optional logging
            approx_kl = (old_logprobs.detach() - logprobs).mean()
            last_metrics = {
                "loss": loss.detach().item(),
                "policy_loss": policy_loss.detach().item(),
                "value_loss": value_loss.detach().item(),
                "entropy": entropy.detach().item(),
                "approx_kl": approx_kl.detach().item(),
            }

        self.policy_old.load_state_dict(self.policy.state_dict())
        return last_metrics
