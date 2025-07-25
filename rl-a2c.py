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
# Advantage Actor Critic (A2C) algorithm
# A simple Policy Gradient algorithm
# It's a synchronous actor-critic algorithm that combines:
# * Policy gradient (actor) - learning to choose actions
# * Value estimation (critic) - learning expected return from states
# Slow and steady convergence

import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

env = gym.make("CartPole-v1")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === MODEL DEFINITIONS ===

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
        return self.net(state).squeeze(-1)  # shape: (batch,)

# === HYPERPARAMETERS ===

gamma = 0.99
actor_lr = 1e-3
critic_lr = 1e-3
entropy_beta = 0.03
max_episodes = 1000

# === INIT MODELS + OPTIMIZERS ===

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

actor = Actor(state_dim, action_dim).to(device)
critic = Critic(state_dim).to(device)

actor_optimizer = optim.Adam(actor.parameters(), lr=actor_lr)
critic_optimizer = optim.Adam(critic.parameters(), lr=critic_lr)

# === TRAINING LOOP ===

for episode in range(max_episodes):
    state, _ = env.reset()
    done = False
    episode_reward = 0

    while not done:
        state_tensor = torch.tensor(state, dtype=torch.float32).to(device)

        # === ACTOR: Sample action from current policy ===
        probs = actor(state_tensor)
        dist = Categorical(probs)
        action = dist.sample()

        log_prob = dist.log_prob(action)
        entropy = dist.entropy()

        next_state, reward, terminated, truncated, _ = env.step(action.item())
        done = terminated or truncated

        next_state_tensor = torch.tensor(next_state, dtype=torch.float32).to(device)

        # === CRITIC: Get value estimates ===
        value = critic(state_tensor)
        next_value = critic(next_state_tensor).detach()

        # === TD Target and Advantage ===
        td_target = reward + gamma * next_value * (1 - int(done))
        advantage = td_target - value

        # === ACTOR LOSS (Policy Gradient + Entropy Bonus) ===
        actor_loss = -log_prob * advantage.detach() - entropy_beta * entropy

        # === CRITIC LOSS (MSE) ===
        critic_loss = nn.functional.mse_loss(value, td_target)

        # === BACKPROP ===
        actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_optimizer.step()

        critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_optimizer.step()

        state = next_state
        episode_reward += reward

    if episode % 10 == 0:
        print(f"Episode {episode} - Total reward: {episode_reward:.1f}")

env.close()
