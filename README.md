# Empirical Analysis of Maximization Bias in Deep RL

This repository contains a modular Deep Reinforcement Learning framework developed to investigate and visualize the phenomenon of value overestimation—commonly known as maximization bias—in standard Q-learning.

Using the `LunarLander-v3` environment from Gymnasium, this project implements standard Deep Q-Networks (DQN) and Double DQN (DDQN) from scratch in PyTorch, establishing an empirical comparison between the two algorithms.

## 🚀 Key Features

* **Modular Framework:** Clean, object-oriented implementation of Q-Networks, Replay Buffers, and RL Agents in PyTorch.
* **Algorithmic Toggle:** A unified agent capable of seamlessly switching between DQN and DDQN update rules.
* **Empirical Validation:** Custom evaluation scripts to isolate and measure the divergence between neural network Q-value estimates and true Monte Carlo returns.

## 🧮 Theoretical Background

### The Problem: Maximization Bias in DQN

In standard Q-learning and DQN, the target relies on a $\max$ operator over estimated action values. Because these estimates are inherently noisy (especially early in training), taking the maximum of these noisy estimates introduces a systemic positive bias. Over time, the network overestimates the true value of states.

The standard DQN TD-target is calculated as:


$$Y^{DQN}_t = R_{t+1} + \gamma \max_a Q(S_{t+1}, a; \theta^-)$$

### The Solution: Double Q-Learning

Double DQN (DDQN) addresses this by decoupling action *selection* from action *evaluation*. The online network ($\theta$) is used to find the greedy action, while the target network ($\theta^-$) is used to evaluate the value of that specific action.

The DDQN TD-target is calculated as:


$$Y^{DDQN}_t = R_{t+1} + \gamma Q(S_{t+1}, \arg\max_a Q(S_{t+1}, a; \theta); \theta^-)$$

## 📊 Results and Visualizations

By tracking a fixed set of states periodically during training, we successfully visualize the maximization bias.

<img width="1577" height="577" alt="image" src="https://github.com/user-attachments/assets/2b0bfc04-a0e8-418f-8d7a-3660128212d8" />


**Key Findings:**

1. **Overestimation Gap:** In the standard DQN formulation, the estimated Q-values severely detach from the true Monte Carlo returns, creating a massive artificial inflation of state values (the red shaded area).
2. **DDQN Stability:** The Double DQN implementation successfully anchors the estimated Q-values to the true returns (the blue lines track tightly together), empirically demonstrating the mathematical reduction of maximization bias.
3. **Environment Solved:** Despite the varying value estimations, both agents successfully map optimal policies, reliably achieving scores $> 200$ to solve the `LunarLander-v3` environment.
