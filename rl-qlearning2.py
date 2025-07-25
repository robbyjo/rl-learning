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
# Q-learning Continuous states version
import gymnasium as gym
import numpy as np

env = gym.make("MountainCar-v0", render_mode="human")
env.metadata['render_fps'] = 180

num_actions = env.action_space.n
state_size = env.observation_space.shape[0]

# One linear model per action
weights = np.random.randn(num_actions, state_size)
biases = np.zeros(num_actions)

alpha = 0.1   # learning rate
gamma = 0.99   # discount
epsilon = 0.1
num_episodes = 500

for episode in range(num_episodes):
    state, _ = env.reset()
    done = False

    while not done:
        # Epsilon-greedy policy
        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            q_values = np.dot(weights, state) + biases
            action = np.argmax(q_values)

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        # Q-learning TD target
        q_current = np.dot(weights[action], state) + biases[action]
        q_next = np.dot(weights, next_state) + biases
        target = reward + gamma * np.max(q_next)

        td_error = target - q_current

        # Gradient step: update weights and bias for chosen action
        weights[action] += alpha * td_error * state
        biases[action] += alpha * td_error

        state = next_state

    if episode % 50 == 0:
        print(f"Episode {episode}")

env.close()
