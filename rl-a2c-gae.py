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
# Advantage Actor Critic (A2C) algorithm with Generalized Advantage Estimation (GAE)
# Bias-Variance tradeoff with lambda = 1 being more REINFORCE like and lambda = 0 being more A2C like

import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

#env = gym.make("CartPole-v1")
env = gym.make("LunarLander-v3")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === NETWORK DEFINITIONS ===

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, state):
        return self.net(state)

class Critic(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, state):
        return self.net(state)  # shape: (batch, 1)

# === HYPERPARAMETERS ===

gamma = 0.99
gae_lambda = 0.75
entropy_beta = 0.03
actor_lr = 1e-3
critic_lr = 1e-3
max_episodes = 1000

# === INIT MODELS ===

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

actor = Actor(state_dim, action_dim).to(device)
critic = Critic(state_dim).to(device)

actor_optimizer = optim.Adam(actor.parameters(), lr=actor_lr)
critic_optimizer = optim.Adam(critic.parameters(), lr=critic_lr)

# === GAE FUNCTION ===

def compute_gae(rewards, values, dones, gamma, lam):
    advantages = []
    gae = 0
    values = values + [0.0]  # append dummy value for s_{t+1}
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * values[t + 1] * (1 - dones[t]) - values[t]
        gae = delta + gamma * lam * (1 - dones[t]) * gae
        advantages.insert(0, gae)
    return advantages

# === TRAINING LOOP ===

for episode in range(max_episodes):
    state, _ = env.reset()
    done = False
    ep_reward = 0

    log_probs = []
    values = []
    rewards = []
    dones = []
    entropies = []
    states = []

    while not done:
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)  # shape: (1, state_dim)

        probs = actor(state_tensor)
        dist = Categorical(probs)
        action = dist.sample()

        log_prob = dist.log_prob(action)
        entropy = dist.entropy()

        value = critic(state_tensor)  # shape: (1, 1)

        next_state, reward, terminated, truncated, _ = env.step(action.item())
        done = terminated or truncated

        log_probs.append(log_prob)
        values.append(value.squeeze(0))  # shape: (1,) to ()
        rewards.append(reward)
        dones.append(done)
        entropies.append(entropy)
        states.append(state_tensor.squeeze(0))  # shape: (state_dim,)
        ep_reward += reward

        state = next_state

    # === Compute GAE advantages ===
    with torch.no_grad():
        values_for_gae = [v.item() for v in values]
        advantages = compute_gae(rewards, values_for_gae, dones, gamma, gae_lambda)
        returns = [a + v for a, v in zip(advantages, values_for_gae)]

    # === Convert to tensors ===
    log_probs = torch.stack(log_probs).to(device)
    entropies = torch.stack(entropies).to(device)
    values_tensor = torch.stack(values).to(device)  # requires grad!
    advantages_tensor = torch.tensor(advantages, dtype=torch.float32).to(device)
    returns_tensor = torch.tensor(returns, dtype=torch.float32).to(device)  # doesn't need grad

    # === Normalize advantage ===
    advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-8)

    # === Losses ===
    actor_loss = -(log_probs * advantages_tensor.detach()).mean() - entropy_beta * entropies.mean()
    critic_loss = nn.functional.mse_loss(values_tensor.squeeze(), returns_tensor)

    # === Backprop ===
    actor_optimizer.zero_grad()
    actor_loss.backward()
    actor_optimizer.step()

    critic_optimizer.zero_grad()
    critic_loss.backward()
    critic_optimizer.step()

    if episode % 10 == 0:
        print(f"Episode {episode} - Reward: {ep_reward:.1f}")

env.close()
