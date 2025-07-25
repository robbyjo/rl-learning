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
# TD Deep Deterministic Policy Gradient (TD3)

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
import random
from collections import deque
from torch.utils.tensorboard import SummaryWriter

# ========== ENVIRONMENT SETUP ==========
env = gym.make("Pendulum-v1")
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])

# ========== HYPERPARAMETERS ==========
lr_actor = 1e-3
lr_critic = 1e-3
gamma = 0.99
tau = 0.005
policy_noise = 0.2       # TD3 ONLY
noise_clip = 0.5         # TD3 ONLY
policy_delay = 2         # TD3 ONLY
max_episodes = 2000
max_steps = 500
batch_size = 64
buffer_capacity = 200000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========== REPLAY BUFFER ==========
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, float(done)))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return (torch.FloatTensor(state).to(device),
                torch.FloatTensor(action).to(device),
                torch.FloatTensor(reward).unsqueeze(1).to(device),
                torch.FloatTensor(next_state).to(device),
                torch.FloatTensor(done).unsqueeze(1).to(device))

    def __len__(self):
        return len(self.buffer)

# ========== ACTOR NETWORK ==========
class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.out = nn.Linear(256, action_dim)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return torch.tanh(self.out(x)) * max_action

# ========== TD3: DUAL CRITIC NETWORK ==========
class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        # Q1
        self.q1_fc1 = nn.Linear(state_dim + action_dim, 256)
        self.q1_fc2 = nn.Linear(256, 256)
        self.q1_out = nn.Linear(256, 1)
        # Q2
        self.q2_fc1 = nn.Linear(state_dim + action_dim, 256)
        self.q2_fc2 = nn.Linear(256, 256)
        self.q2_out = nn.Linear(256, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], dim=1)
        # Q1
        q1 = F.relu(self.q1_fc1(sa))
        q1 = F.relu(self.q1_fc2(q1))
        q1 = self.q1_out(q1)
        # Q2
        q2 = F.relu(self.q2_fc1(sa))
        q2 = F.relu(self.q2_fc2(q2))
        q2 = self.q2_out(q2)
        return q1, q2

    def Q1(self, state, action):
        sa = torch.cat([state, action], dim=1)
        q1 = F.relu(self.q1_fc1(sa))
        q1 = F.relu(self.q1_fc2(q1))
        return self.q1_out(q1)

# ========== INITIALIZE ==========
actor = Actor().to(device)
actor_target = Actor().to(device)
actor_target.load_state_dict(actor.state_dict())

critic = Critic().to(device)
critic_target = Critic().to(device)
critic_target.load_state_dict(critic.state_dict())

actor_optimizer = torch.optim.Adam(actor.parameters(), lr=lr_actor)
critic_optimizer = torch.optim.Adam(critic.parameters(), lr=lr_critic)

replay_buffer = ReplayBuffer(buffer_capacity)
writer = SummaryWriter(comment="-TD3-Pendulum")

# ========== TRAIN LOOP ==========
total_steps = 0

for episode in range(max_episodes):
    state, _ = env.reset()
    episode_reward = 0

    for step in range(max_steps):
        total_steps += 1
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        action = actor(state_tensor).cpu().data.numpy().flatten()
        noise = np.random.normal(0, 0.1, size=action_dim)
        action = np.clip(action + noise, -max_action, max_action)

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        replay_buffer.push(state, action, reward, next_state, done)

        state = next_state
        episode_reward += reward

        # TRAINING
        if len(replay_buffer) >= batch_size:
            states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

            # TD3 TARGET ACTION NOISE
            with torch.no_grad():
                noise = (torch.randn_like(actions) * policy_noise).clamp(-noise_clip, noise_clip)
                next_actions = (actor_target(next_states) + noise).clamp(-max_action, max_action)
                target_q1, target_q2 = critic_target(next_states, next_actions)
                target_q = rewards + gamma * (1 - dones) * torch.min(target_q1, target_q2)

            # CRITIC UPDATE
            current_q1, current_q2 = critic(states, actions)
            critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

            critic_optimizer.zero_grad()
            critic_loss.backward()
            critic_optimizer.step()

            # TD3: DELAYED ACTOR UPDATE
            if total_steps % policy_delay == 0:
                actor_loss = -critic.Q1(states, actor(states)).mean()
                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()

                # TARGET NETWORK SOFT UPDATE
                for param, target_param in zip(actor.parameters(), actor_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

                for param, target_param in zip(critic.parameters(), critic_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    writer.add_scalar("Reward/Episode", episode_reward, episode)
    if episode % 10 == 0:
        print(f"Episode {episode}, Reward: {episode_reward:.2f}")

env.close()
writer.close()
