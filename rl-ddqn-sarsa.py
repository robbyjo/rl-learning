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
# Double DQN (DDQN)

import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import random
import numpy as np
from collections import deque

# Hyperparameters
gamma = 0.99
epsilon = 0.2
batch_size = 64
lr = 0.001
buffer_size = 10000
target_update_freq = 10
num_episodes = 500

env = gym.make("MountainCar-v0", render_mode="human")
env.metadata['render_fps'] = 180

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

# Replay Buffer
replay_buffer = deque(maxlen=buffer_size)

# Simple MLP
class QNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.net(x)

policy_net = QNet(state_dim, action_dim)
target_net = QNet(state_dim, action_dim)
target_net.load_state_dict(policy_net.state_dict())
optimizer = optim.Adam(policy_net.parameters(), lr=lr)

def get_action(state):
    if random.random() < epsilon:
        return env.action_space.sample()
    else:
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            return policy_net(state).argmax().item()

def train():
    if len(replay_buffer) < batch_size:
        return

    batch = random.sample(replay_buffer, batch_size)
    states, actions, rewards, next_states, dones = zip(*batch)

    states = torch.tensor(states, dtype=torch.float32)
    actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(1)
    rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
    next_states = torch.tensor(next_states, dtype=torch.float32)
    dones = torch.tensor(dones, dtype=torch.bool).unsqueeze(1)

    q_values = policy_net(states).gather(1, actions)
    # get next action under current policy (SARSA-style)
    next_actions = torch.tensor([
        get_action(ns) for ns in next_states
    ]).unsqueeze(1)

    with torch.no_grad():
        # DOUBLE DQN SARSA CHANGE:
        target_q_values = target_net(next_states).gather(1, next_actions)
        q_targets = rewards + gamma * target_q_values * (~dones)

    loss = nn.MSELoss()(q_values, q_targets)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Training Loop
for episode in range(num_episodes):
    state, _ = env.reset()
    total_reward = 0
    done = False

    while not done:
        action = get_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        replay_buffer.append((state, action, reward, next_state, done))
        state = next_state
        total_reward += reward

        train()

    if episode % target_update_freq == 0:
        target_net.load_state_dict(policy_net.state_dict())
        print(f"Episode {episode}: target net updated")

env.close()
