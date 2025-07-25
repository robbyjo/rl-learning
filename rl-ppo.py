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
# PPO (Proximal Policy Optimization), with GAE

import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Hyperparameters ===
gamma = 0.99
gae_lambda = 0.95
clip_epsilon = 0.2
entropy_beta = 0.01
actor_lr = 3e-4
critic_lr = 1e-3
update_epochs = 4
batch_size = 64
max_steps = 5*1024  # collect this many steps before updating
# If max_steps is too small, this will limit rewards. 2048 --> max rewards = 200, 4096 --> max rewards = 400

max_episodes = 4000

# === Environment ===
env = gym.make("CartPole-v1")
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

# === Actor & Critic ===
class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1)
        )
    def forward(self, state):
        return self.net(state)

class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
    def forward(self, state):
        return self.net(state).squeeze(-1)

actor = Actor().to(device)
critic = Critic().to(device)
actor_optim = optim.Adam(actor.parameters(), lr=actor_lr)
critic_optim = optim.Adam(critic.parameters(), lr=critic_lr)

# === GAE computation ===
def compute_gae(rewards, values, dones, next_value, gamma, lam):
    values = values + [next_value]
    gae = 0
    advantages = []
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * values[t+1] * (1 - dones[t]) - values[t]
        gae = delta + gamma * lam * (1 - dones[t]) * gae
        advantages.insert(0, gae)
    return advantages

# === Main PPO Training Loop ===
episode = 0
while episode < max_episodes:
    states, actions, rewards, dones = [], [], [], []
    log_probs, values, entropies, episode_rewards = [], [], [], []
    ep_reward = 0

    steps_collected = 0
    while steps_collected < max_steps:
        state, _ = env.reset()
        done = False
        while not done and steps_collected < max_steps:
            state_tensor = torch.tensor(state, dtype=torch.float32).to(device)
            probs = actor(state_tensor)
            dist = Categorical(probs)
            action = dist.sample()

            next_state, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated
            ep_reward += reward

            with torch.no_grad():
                value = critic(state_tensor)

            log_prob = dist.log_prob(action)
            entropy = dist.entropy()

            states.append(state_tensor)
            actions.append(action)
            rewards.append(reward)
            dones.append(done)
            log_probs.append(log_prob)
            values.append(value)
            entropies.append(entropy)

            state = next_state
            steps_collected += 1
            if done:
                episode_rewards.append(ep_reward)
                ep_reward = 0
                episode += 1

    # === Bootstrap last value ===
    with torch.no_grad():
        next_state_tensor = torch.tensor(next_state, dtype=torch.float32).to(device)
        next_value = critic(next_state_tensor).item()

    # === Compute returns and advantages ===
    values_list = [v.item() for v in values]
    advantages = compute_gae(rewards, values_list, dones, next_value, gamma, gae_lambda)
    returns = [adv + val for adv, val in zip(advantages, values_list)]

    # === Convert to tensors ===
    states_tensor = torch.stack(states)
    actions_tensor = torch.stack(actions)
    old_log_probs_tensor = torch.stack(log_probs).detach()
    advantages_tensor = torch.tensor(advantages, dtype=torch.float32).to(device)
    returns_tensor = torch.tensor(returns, dtype=torch.float32).to(device)

    # === Normalize advantages ===
    advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-8)

    # === PPO Updates ===
    for _ in range(update_epochs):
        for start in range(0, len(states_tensor), batch_size):
            end = start + batch_size
            batch_states = states_tensor[start:end]
            batch_actions = actions_tensor[start:end]
            batch_old_log_probs = old_log_probs_tensor[start:end]
            batch_advantages = advantages_tensor[start:end]
            batch_returns = returns_tensor[start:end]

            # Recompute distribution with current actor
            probs = actor(batch_states)
            dist = Categorical(probs)
            new_log_probs = dist.log_prob(batch_actions)
            entropy = dist.entropy()

            ratio = torch.exp(new_log_probs - batch_old_log_probs)
            clipped = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon)
            policy_loss = -torch.min(ratio * batch_advantages, clipped * batch_advantages).mean()
            entropy_loss = -entropy.mean()

            # Critic loss
            values_pred = critic(batch_states)
            value_loss = nn.functional.mse_loss(values_pred, batch_returns)

            actor_optim.zero_grad()
            (policy_loss + entropy_beta * entropy_loss).backward()
            actor_optim.step()

            critic_optim.zero_grad()
            value_loss.backward()
            critic_optim.step()

    # === Report every N episodes ===
    if episode % 10 == 0:
        print(f"Episode {episode} | Avg reward (last 10): {sum(episode_rewards[-10:]) / 10:.2f}")
        #print(f"Episode {episode} | Avg reward: {sum(rewards) / len(dones):.2f}")

env.close()
