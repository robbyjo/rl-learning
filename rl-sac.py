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
# Soft Actor-Critic (SAC) for Pendulum-v1
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
import random
from collections import deque
from torch.distributions import Normal
from torch.utils.tensorboard import SummaryWriter

# Environment setup
env = gym.make("Pendulum-v1")
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])

# Hyperparameters
lr_actor = 3e-4
lr_critic = 3e-4
lr_alpha = 3e-4
gamma = 0.99
tau = 0.005
alpha = 0.2  # Entropy coefficient
automatic_entropy_tuning = True
target_entropy = -action_dim
batch_size = 64
buffer_capacity = 100000
max_episodes = 500
max_steps = 200

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Replay Buffer
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

# Actor Network
class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.mean = nn.Linear(256, action_dim)
        self.log_std = nn.Linear(256, action_dim)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mean = self.mean(x)
        log_std = self.log_std(x).clamp(-20, 2)
        std = log_std.exp()
        return mean, std

    def sample(self, state):
        mean, std = self.forward(state)
        normal = Normal(mean, std)
        x_t = normal.rsample()
        action = torch.tanh(x_t) * max_action
        log_prob = normal.log_prob(x_t).sum(dim=-1, keepdim=True)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6).sum(dim=-1, keepdim=True)
        return action, log_prob

# Critic Network
class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        self.q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, state, action):
        sa = torch.cat([state, action], dim=-1)
        return self.q1(sa), self.q2(sa)

# Initialization
actor = Actor().to(device)
critic = Critic().to(device)
critic_target = Critic().to(device)
critic_target.load_state_dict(critic.state_dict())

actor_optimizer = torch.optim.Adam(actor.parameters(), lr=lr_actor)
critic_optimizer = torch.optim.Adam(critic.parameters(), lr=lr_critic)

if automatic_entropy_tuning:
    log_alpha = torch.zeros(1, requires_grad=True, device=device)
    alpha_optimizer = torch.optim.Adam([log_alpha], lr=lr_alpha)

replay_buffer = ReplayBuffer(buffer_capacity)
writer = SummaryWriter(comment="-SAC-Pendulum")

for episode in range(max_episodes):
    state, _ = env.reset()
    episode_reward = 0

    for step in range(max_steps):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        action, _ = actor.sample(state_tensor)
        action_np = action.cpu().detach().numpy()[0]
        next_state, reward, terminated, truncated, _ = env.step(action_np)
        done = terminated or truncated
        replay_buffer.push(state, action_np, reward, next_state, done)

        state = next_state
        episode_reward += reward

        if len(replay_buffer) > batch_size:
            states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
            with torch.no_grad():
                next_actions, next_log_probs = actor.sample(next_states)
                q1_next, q2_next = critic_target(next_states, next_actions)
                q_target = torch.min(q1_next, q2_next) - alpha * next_log_probs
                target_q = rewards + gamma * (1 - dones) * q_target

            q1, q2 = critic(states, actions)
            critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)
            critic_optimizer.zero_grad()
            critic_loss.backward()
            critic_optimizer.step()

            actions_pi, log_pi = actor.sample(states)
            q1_pi, q2_pi = critic(states, actions_pi)
            actor_loss = (alpha * log_pi - torch.min(q1_pi, q2_pi)).mean()

            actor_optimizer.zero_grad()
            actor_loss.backward()
            actor_optimizer.step()

            if automatic_entropy_tuning:
                alpha_loss = -(log_alpha * (log_pi + target_entropy).detach()).mean()
                alpha_optimizer.zero_grad()
                alpha_loss.backward()
                alpha_optimizer.step()
                alpha = log_alpha.exp().item()

            for param, target_param in zip(critic.parameters(), critic_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    writer.add_scalar("Reward/Episode", episode_reward, episode)
    if episode % 10 == 0:
        print(f"Episode {episode}, Reward: {episode_reward:.2f}, Alpha: {alpha:.4f}")

env.close()
writer.close()
