"""
MIT License

Copyright (c) 2025 Roby Joehanes

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

@author: Roby Joehanes

"""
# PPO (Proximal Policy Optimization) on continuous action, with GAE

import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
gamma = 0.99
gae_lambda = 0.95
clip_epsilon = 0.2
entropy_beta = 0.002 # Encouraging exploration works, but more wobbly
actor_lr = 3e-4
critic_lr = 1e-3
update_epochs = 10
batch_size = 1024
max_steps = 32768     # Bigger steps to encourage data diversity works
max_episodes = 30000   # Needs much longer episodes to converge
last_episode = 0

# Environment
env = gym.make("Pendulum-v1")
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
action_bound = float(env.action_space.high[0])  # 2.0

# === Actor ===
class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh()
        )
        self.mean_layer = nn.Linear(64, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))  # learnable std

    def forward(self, state):
        x = self.net(state)
        mu = self.mean_layer(x)
        std = torch.exp(self.log_std) # Standard std
        #std = torch.clamp(torch.exp(self.log_std), min=1e-3) # Standard std, with min 1e-3
        #std = 0.1 + 0.9 * (torch.exp(self.log_std))
        return mu, std

# === Critic ===
class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
    def forward(self, state):
        return self.net(state).squeeze(-1)

actor = Actor().to(device)
critic = Critic().to(device)
actor_optim = optim.Adam(actor.parameters(), lr=actor_lr)
critic_optim = optim.Adam(critic.parameters(), lr=critic_lr)

# === GAE ===
def compute_gae(rewards, values, dones, next_value, gamma, lam):
    values = values + [next_value]
    gae = 0
    advantages = []
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * values[t+1] * (1 - dones[t]) - values[t]
        gae = delta + gamma * lam * (1 - dones[t]) * gae
        advantages.insert(0, gae)
    return advantages

episode = 0
episode_rewards = []

while episode < max_episodes:
    states, actions, rewards, dones = [], [], [], []
    log_probs, values, entropies = [], [], []
    steps_collected = 0
    ep_reward = 0

    while steps_collected < max_steps:
        state, _ = env.reset()
        done = False

        while not done and steps_collected < max_steps:
            state_tensor = torch.tensor(state, dtype=torch.float32).to(device)
            mu, std = actor(state_tensor)
            #print(f"Mean Std (policy output): {std.mean().item():.4f}")
            dist = Normal(mu, std)
            action = dist.sample()
            #action = torch.tanh(action) * action_bound  # squash to [-2, 2], tanh normalization, problem specific
            log_prob = dist.log_prob(action).sum(dim=-1)
            entropy = dist.entropy().sum(dim=-1)

            # Clamp action to env limits
            clipped_action = action.clamp(-action_bound, action_bound)
            next_state, reward, terminated, truncated, _ = env.step(clipped_action.cpu().numpy())
            done = terminated or truncated

            with torch.no_grad():
                value = critic(state_tensor)

            states.append(state_tensor)
            actions.append(action)
            rewards.append(reward)
            dones.append(done)
            log_probs.append(log_prob)
            values.append(value)
            entropies.append(entropy)

            state = next_state
            ep_reward += reward
            steps_collected += 1

            if done:
                episode_rewards.append(ep_reward)
                ep_reward = 0
                episode += 1

    # Bootstrap value
    with torch.no_grad():
        next_state_tensor = torch.tensor(next_state, dtype=torch.float32).to(device)
        next_value = critic(next_state_tensor).item()

    values_list = [v.item() for v in values]
    advantages = compute_gae(rewards, values_list, dones, next_value, gamma, gae_lambda)
    returns = [a + v for a, v in zip(advantages, values_list)]

    # Convert to tensors
    states_tensor = torch.stack(states)
    actions_tensor = torch.stack(actions)
    old_log_probs_tensor = torch.stack(log_probs).detach()
    advantages_tensor = torch.tensor(advantages, dtype=torch.float32).to(device)
    returns_tensor = torch.tensor(returns, dtype=torch.float32).to(device)

    # Normalize advantages
    advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-8)

    # === PPO Update ===
    for _ in range(update_epochs):
        for start in range(0, len(states_tensor), batch_size):
            end = start + batch_size
            b_states = states_tensor[start:end]
            b_actions = actions_tensor[start:end]
            b_old_log_probs = old_log_probs_tensor[start:end]
            b_advantages = advantages_tensor[start:end]
            b_returns = returns_tensor[start:end]

            mu, std = actor(b_states)
            dist = Normal(mu, std)
            new_log_probs = dist.log_prob(b_actions).sum(dim=-1)
            entropy = dist.entropy().sum(dim=-1)

            ratio = torch.exp(new_log_probs - b_old_log_probs)
            clipped_ratio = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon)
            policy_loss = -torch.min(ratio * b_advantages, clipped_ratio * b_advantages).mean()
            entropy_loss = -entropy.mean()

            values_pred = critic(b_states)
            value_loss = nn.functional.mse_loss(values_pred, b_returns)

            actor_optim.zero_grad()
            (policy_loss + entropy_beta * entropy_loss).backward()
            actor_optim.step()

            critic_optim.zero_grad()
            value_loss.backward()
            critic_optim.step()

    if episode - last_episode >= 100:
        avg = sum(episode_rewards[-10:]) / 10
        last_episode = episode
        print(f"Episode {episode} | Avg Reward: {avg:.1f} | Max: {max(episode_rewards):.1f}")

env.close()
