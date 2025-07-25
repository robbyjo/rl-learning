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
# Deep Deterministic Policy Gradient (DDPG)

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
from torch.utils.tensorboard import SummaryWriter

# === Hyperparameters ===
ENV_NAME = "Pendulum-v1"
MAX_EPISODES = 500
MAX_STEPS = 200
GAMMA = 0.99
TAU = 0.005
ACTOR_LR = 1e-4
CRITIC_LR = 1e-3
BUFFER_SIZE = int(1e6)
BATCH_SIZE = 256
ACTION_NOISE_STD = 0.1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Actor Network ===
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, action_bound):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, action_dim), nn.Tanh()
        )
        self.action_bound = action_bound

    def forward(self, state):
        return self.net(state) * self.action_bound

# === Critic Network ===
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        return self.net(x)

# === Replay Buffer ===
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, s, a, r, s_, d):
        self.buffer.append((s, a, r, s_, d))

    def sample(self, batch_size):
        samples = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*samples)
        return (torch.FloatTensor(states).to(device),
                torch.FloatTensor(actions).to(device),
                torch.FloatTensor(rewards).unsqueeze(1).to(device),
                torch.FloatTensor(next_states).to(device),
                torch.FloatTensor(dones).unsqueeze(1).to(device))

    def __len__(self):
        return len(self.buffer)

# === Soft update ===
def soft_update(target, source, tau):
    for t_param, s_param in zip(target.parameters(), source.parameters()):
        t_param.data.copy_(tau * s_param.data + (1 - tau) * t_param.data)

# === Main DDPG Training Loop ===
env = gym.make(ENV_NAME)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
action_bound = float(env.action_space.high[0])

actor = Actor(state_dim, action_dim, action_bound).to(device)
target_actor = Actor(state_dim, action_dim, action_bound).to(device)
target_actor.load_state_dict(actor.state_dict())

critic = Critic(state_dim, action_dim).to(device)
target_critic = Critic(state_dim, action_dim).to(device)
target_critic.load_state_dict(critic.state_dict())

actor_optimizer = optim.Adam(actor.parameters(), lr=ACTOR_LR)
critic_optimizer = optim.Adam(critic.parameters(), lr=CRITIC_LR)

replay_buffer = ReplayBuffer(BUFFER_SIZE)
writer = SummaryWriter(comment="-DDPG-Pendulum")

for episode in range(MAX_EPISODES):
    state, _ = env.reset()
    episode_reward = 0

    for step in range(MAX_STEPS):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        action = actor(state_tensor).cpu().detach().numpy()[0]
        noise = np.random.normal(0, ACTION_NOISE_STD, size=action_dim)
        noisy_action = np.clip(action + noise, -action_bound, action_bound)

        next_state, reward, terminated, truncated, _ = env.step(noisy_action)
        done = terminated or truncated
        replay_buffer.add(state, noisy_action, reward, next_state, done)

        state = next_state
        episode_reward += reward

        writer.add_scalar("Reward/Episode", episode_reward, episode)
        if len(replay_buffer) > BATCH_SIZE:
            states, actions, rewards, next_states, dones = replay_buffer.sample(BATCH_SIZE)

            # === Critic Update ===
            with torch.no_grad():
                next_actions = target_actor(next_states)
                q_next = target_critic(next_states, next_actions)
                q_target = rewards + GAMMA * q_next * (1 - dones)

            q_values = critic(states, actions)
            critic_loss = nn.MSELoss()(q_values, q_target)

            critic_optimizer.zero_grad()
            critic_loss.backward()
            critic_optimizer.step()

            # === Actor Update ===
            actor_loss = -critic(states, actor(states)).mean()

            actor_optimizer.zero_grad()
            actor_loss.backward()
            actor_optimizer.step()

            # === Target Network Update ===
            soft_update(target_actor, actor, TAU)
            soft_update(target_critic, critic, TAU)

            # Tensor board logging
            writer.add_scalar("Loss/Actor", actor_loss.item(), episode)
            writer.add_scalar("Loss/Critic", critic_loss.item(), episode)
            writer.add_scalar("Std/Noise", np.std(noise), episode)

        if done:
            break

    # Tensor board logging
    if episode % 10 == 0:
        with torch.no_grad():
            sampled_actions = []
            for _ in range(100):
                s, _ = env.reset()
                s_tensor = torch.FloatTensor(s).unsqueeze(0).to(device)
                a = actor(s_tensor).cpu().numpy()[0]
                sampled_actions.append(a)

            sampled_actions = np.array(sampled_actions)
            for i in range(action_dim):
                writer.add_histogram(f"Action/Dim_{i}", sampled_actions[:, i], episode)
    print(f"Episode {episode} | Reward: {episode_reward:.2f}")

writer.close()
env.close()
