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
# REINFORCE algorithm
# A simple Policy Gradient algorithm

import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

env = gym.make("CartPole-v1")
#env = gym.make("CartPole-v1", render_mode="human")
#env.metadata['render_fps'] = 180
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Policy Network
class PolicyNet(nn.Module):
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

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
policy_net = PolicyNet(state_dim, action_dim).to(device)
optimizer = optim.Adam(policy_net.parameters(), lr=1e-2)
gamma = 0.99
max_episodes = 1000

# Training Loop
for episode in range(max_episodes):
    state, _ = env.reset()
    log_probs = []
    rewards = []

    done = False
    while not done:
        state_tensor = torch.tensor(state, dtype=torch.float32).to(device)
        probs = policy_net(state_tensor)
        dist = Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        next_state, reward, terminated, truncated, _ = env.step(action.item())
        done = terminated or truncated

        log_probs.append(log_prob)
        rewards.append(reward)

        state = next_state

    # Compute return G_t for each time step
    returns = []
    G = 0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    returns = torch.tensor(returns, dtype=torch.float32).to(device)

    # Normalize returns for stability
    returns = (returns - returns.mean()) / (returns.std() + 1e-8)

    # Compute loss
    loss = 0
    for log_prob, G in zip(log_probs, returns):
        loss += -log_prob * G

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Logging
    if episode % 10 == 0:
        total_reward = sum(rewards)
        print(f"Episode {episode}: Total Reward = {total_reward}")

env.close()
