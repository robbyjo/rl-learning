# rl-learning
This is a collection of Python programs that I created while learning Reinforcement Learning. Everything is licensed in MIT license so that you can customize them however you want.

## Requirements
* Everything requires numpy. I used v1.26.4.
* Except for the first one (rl-value.py), all the others require gymnasium package since I used gymnasium's toy problems to test algorithms. I used v1.1.1.
* DQN, DDQN, Dueling DQN, A2C, PPO, DDPG, TD3, SAC require pytorch. I used v2.6.0+cu126.
* DDPG, TD3, and SAC require Tensorboard. I used v2.20.0.

I figured that these version numbers need not be strictly identical, but caveat emptor.

## How to learn
Apologies, the instruction is a bit sparse. I assume that everyone's path is a little different. Mine was as follows:
* rl-value.py for Value iteration
* rl-qlearning.py for Q-learning
* rl-sarsa.py for SARSA variation
* rl-qlearning2.py for Q-learning with continuous states (hopeless, but useful conceptually)
* rl-sarsa2.py for SARSA variation with continuous states (again, hopeless, but useful conceptually)
* rl-dqn.py for Deep Q-Network (DQN)
* rl-prioritized-replay-buffer.py Prioritized Replay Buffer add-on that you can add to many RL algorithm from DQN onward (optional)
* rl-dqn2.py for DQN with Reward-shaping
* rl-dqn-anneal.py for DQN with epsilon-annealing
* rl-ddqn.py Double DQN
* rl-ddqn-sarsa.py Double DQN with SARSA variation
* rl-duel-dqn.py Dueling DQN
* rl-duel-dqn-sarsa.py Dueling DQN with SARSA variation
* rl-reinforce.py REINFORCE algorithm (a policy-gradient algorithm)
* rl-reinforce2.py REINFORCE algorithm with Entropy Regularization and Baseline Value Estimator
* rl-a2c.py Advantage Actor Critic (A2C) algorithm
* rl-a2c-gae.py A2C with Generalized Advantage Estimation (GAE)
* rl-a3c.py Asynchronous A2C algorithm
* rl-ppo.py Proximal Policy Optimization (PPO) algorithm.
* rl-ppo-continuous.py PPO algorithm for continuous policy
* rl-ddpg.py Deep Deterministic Policy Gradient (DDPG)
* rl-td3.py Twin-Delayed Deep Deterministic Policy Gradient (TD3)
* rl-sac.py Soft Actor Critic (SAC)

You may want to view Berkeley's Deep Reinforcement Learning Bootcamp:
* https://sites.google.com/view/deep-rl-bootcamp/lectures
* https://rail.eecs.berkeley.edu/deeprlcourse/

Hope this helps.

Roby Joehanes
