# -*- coding: utf-8 -*-
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

Reinforcement Learning Value Iteration
"""

import numpy as np
import random
from dataclasses import dataclass
# From Reinfrocement Learning Bootcamp
# https://sites.google.com/view/deep-rl-bootcamp/lectures
# Lecture 1, slide 12 with added cliffs at the bottom
# https://drive.google.com/file/d/0BxXI_RttTZAhVXBlMUVkQ1BVVDQ/view?resourcekey=0-p0o3Gw0KfHMcLbWrj-UNUg

grid = np.array([
    [  0,   0,   0,  10],
    [  0,   0,   0, -10],
    [  0,   0,   0,   0],
    [-10, -10, -10, -10]], np.int32)

# Moves
U = 8; R = 4; L = 2; D = 1

# Encoding legal moves
moves = np.array([
    [   R+D,   R+L,    R+L+D,     0],
    [   U+D,     0,    U+R+D,     0],
    [ U+R+D, R+L+D,  U+R+L+D, U+L+D],
    [     0,     0,        0,     0]], np.int32)

values = np.full_like(grid, float('-Inf'))
policies = np.full_like(grid, 0)

@dataclass
class Q:
    action: int
    state: tuple

gamma = 0.9
noise = 0.2 # Chance not being able to move to desired direction
start = (2,0)

#res = tuple(map(sum, zip(test_tup1, test_tup2)))

def get_possible_moves(cur_state: tuple) -> list:
    possible_moves = []
    val = moves[cur_state]
    if val == 0:
        return possible_moves
    if (val & U == U):
        possible_moves.append(U)
    if (val & D == D):
        possible_moves.append(D)
    if (val & L == L):
        possible_moves.append(L)
    if (val & R == R):
        possible_moves.append(R)
    return possible_moves

def get_next_state(cur_state: tuple) -> Q:
    possible_moves = get_possible_moves(cur_state)
    if len(possible_moves) == 0:
        return None
    pick = random.choice(possible_moves)
    possible_moves.remove(pick)
    x = random.random()
    if (x < noise):
        pick = random.choice(possible_moves)
    if pick == U:
        next_state = (cur_state[0] - 1, cur_state[1])
    elif pick == L:
        next_state = (cur_state[0], cur_state[1] - 1)
    elif pick == R:
        next_state = (cur_state[0], cur_state[1] + 1)
    elif pick == D:
        next_state = (cur_state[0] + 1, cur_state[1])
    return Q(pick, next_state)

def make_episode(start_state: tuple) -> list:
    episode = [Q(0, start_state)]
    cur_state = start_state
    while True:
        q = get_next_state(cur_state)
        if (q is None):
            break
        episode.append(q)
        cur_state = q.state
    return episode

def calc_reward(episode: list) -> list:
    val = grid[episode[-1].state]
    rewards = [val] * len(episode)
    for i in range(len(episode) - 1, -1, -1):
        val = gamma * val + grid[episode[i].state]
        rewards[i] = val
    return rewards

def update_values(episode: list) -> float:
    rewards = calc_reward(episode)
    sum_abs_diff = 0
    for i in range(len(rewards)):
        cur_state = episode[i].state
        if (rewards[i] > values[cur_state]):
            sum_abs_diff = sum_abs_diff + abs(rewards[i] - values[cur_state])
            values[cur_state] = rewards[i]
    return sum_abs_diff

def update_policies():
    for i in range(len(values)):
        for j in range(len(values[i])):
            cur_max = float('-Inf')
            cur_pick = 0
            for pick in get_possible_moves((i, j)):
                if pick == U:
                    next_state = (i - 1, j)
                elif pick == L:
                    next_state = (i, j - 1)
                elif pick == R:
                    next_state = (i, j + 1)
                elif pick == D:
                    next_state = (i + 1, j)
                val = values[next_state]
                if val > cur_max:
                    cur_max = val
                    cur_pick = pick
            policies[(i, j)] = cur_pick
        pass
    pass

delta = 1
while delta > 1e-4:
    eps = make_episode(start)
    delta = update_values(eps)
    update_policies()

