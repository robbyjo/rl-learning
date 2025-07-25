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
# Asynchronous Advantage Actor Critic (A3C) algorithm

import gymnasium as gym
import torch
import torch.nn as nn
import torch.multiprocessing as mp
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np

device = torch.device("cpu")  # A3C is CPU-friendly

# === Shared Actor-Critic ===
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU()
        )
        self.policy = nn.Linear(128, action_dim)
        self.value = nn.Linear(128, 1)

    def forward(self, x):
        x = self.fc(x)
        return F.softmax(self.policy(x), dim=-1), self.value(x)

# === Worker process ===
def worker(worker_id, global_model, optimizer, counter, max_eps, gamma):
    env = gym.make("CartPole-v1")
    local_model = ActorCritic(env.observation_space.shape[0], env.action_space.n).to(device)
    local_model.load_state_dict(global_model.state_dict())

    while True:
        state, _ = env.reset()
        done = False
        ep_reward = 0
        values, log_probs, rewards = [], [], []

        while not done:
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
            probs, value = local_model(state_tensor)
            dist = Categorical(probs)
            action = dist.sample()

            next_state, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated
            ep_reward += reward

            log_probs.append(dist.log_prob(action))
            values.append(value)
            rewards.append(reward)

            state = next_state

            if done:
                # Compute returns
                R = 0
                returns = []
                for r in reversed(rewards):
                    R = r + gamma * R
                    returns.insert(0, R)
                returns = torch.tensor(returns, dtype=torch.float32).to(device)
                values = torch.cat(values)
                log_probs = torch.stack(log_probs)
                advantage = returns - values.squeeze()

                policy_loss = -(log_probs * advantage.detach()).mean()
                value_loss = advantage.pow(2).mean()
                loss = policy_loss + 0.5 * value_loss

                optimizer.zero_grad()
                loss.backward()

                # Push gradients to shared model
                for local_param, global_param in zip(local_model.parameters(), global_model.parameters()):
                    global_param._grad = local_param.grad
                optimizer.step()

                # Sync with global model
                local_model.load_state_dict(global_model.state_dict())

                with counter.get_lock():
                    counter.value += 1
                    if counter.value >= max_eps:
                        print(f"Worker {worker_id} finished at {counter.value} episodes")
                        return
                print(f"Worker {worker_id} | Episode {counter.value} | Reward: {ep_reward}")
                break

# === Master launcher ===
if __name__ == "__main__":
    mp.set_start_method('spawn')  # For macOS/Windows

    env = gym.make("CartPole-v1")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    global_model = ActorCritic(state_dim, action_dim).to(device)
    global_model.share_memory()

    optimizer = optim.Adam(global_model.parameters(), lr=1e-3)
    counter = mp.Value('i', 0)
    max_eps = 300

    num_workers = mp.cpu_count() // 2
    processes = []
    for worker_id in range(num_workers):
        p = mp.Process(target=worker, args=(worker_id, global_model, optimizer, counter, max_eps, 0.99))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    print("Training complete.")

env.close()
