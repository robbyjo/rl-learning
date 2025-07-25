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
# Dueling DQN
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gymnasium as gym
import random
from collections import deque

class DuelingDQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DuelingDQN, self).__init__()
        self.feature = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU()
        )
        self.value_stream = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        x = self.feature(x)
        value = self.value_stream(x)
        advantage = self.advantage_stream(x)
        return value + advantage - advantage.mean(dim=1, keepdim=True)

# Setup
#env = gym.make("MountainCar-v0", render_mode="human") # Dueling DQN doesn't shine here
env = gym.make("LunarLander-v3", render_mode="human")
env.metadata['render_fps'] = 180
input_dim = env.observation_space.shape[0]
output_dim = env.action_space.n
policy_net = DuelingDQN(input_dim, output_dim)
target_net = DuelingDQN(input_dim, output_dim)
target_net.load_state_dict(policy_net.state_dict())
optimizer = optim.Adam(policy_net.parameters(), lr=0.001)
buffer = deque(maxlen=10000)

# Training loop config
batch_size = 64
gamma = 0.99
epsilon = 0.2
update_target_every = 10
num_episodes = 1000

for episode in range(num_episodes):
    state, _ = env.reset()
    total_reward = 0
    done = False

    while not done:
        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                q_values = policy_net(state_tensor)
                action = q_values.argmax().item()

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        buffer.append((state, action, reward, next_state, done))
        state = next_state
        total_reward += reward

        if len(buffer) >= batch_size:
            transitions = random.sample(buffer, batch_size)
            states, actions, rewards, next_states, dones = zip(*transitions)

            states = torch.FloatTensor(states)
            actions = torch.LongTensor(actions).unsqueeze(1)
            rewards = torch.FloatTensor(rewards).unsqueeze(1)
            next_states = torch.FloatTensor(next_states)
            dones = torch.BoolTensor(dones).unsqueeze(1)

            q_vals = policy_net(states).gather(1, actions)

            with torch.no_grad():
                next_q_vals = policy_net(next_states)
                next_action = next_q_vals.argmax(1, keepdim=True)
                q_next = target_net(next_states).gather(1, next_action)
    
                # SARSA chooses the actual next action taken (even if suboptimal)
                # Replace with epsilon-greedy policy:
                sarsa_next_action = []
                for ns in next_states:
                    if np.random.rand() < epsilon:
                        sarsa_next_action.append(random.randint(0, output_dim - 1))
                    else:
                        sarsa_next_action.append(next_q_vals[0].argmax().item())
                sarsa_next_action = torch.LongTensor(sarsa_next_action).unsqueeze(1)
    
                q_next = target_net(next_states).gather(1, sarsa_next_action)
                q_target = rewards + gamma * q_next * (~dones)

            loss = nn.MSELoss()(q_vals, q_target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    if episode % update_target_every == 0:
        target_net.load_state_dict(policy_net.state_dict())

    if episode % 10 == 0:
        print(f"Episode {episode} - Total reward: {total_reward}")

env.close()
