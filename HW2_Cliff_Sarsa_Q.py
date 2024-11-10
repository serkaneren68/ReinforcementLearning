import numpy as np
import gym
import matplotlib.pyplot as plt

# Initialize the Cliff Walking environment
env = gym.make("CliffWalking-v0")
grid_shape = (env.unwrapped.shape[0], env.unwrapped.shape[1])  # Grid dimensions

# Helper function for epsilon-greedy policy
def epsilon_greedy(Q, state, epsilon):
    if np.random.random() < epsilon:
        return np.random.choice(env.action_space.n)
    return np.argmax(Q[state])

# Sarsa Algorithm
def sarsa(env, episodes, alpha, gamma, epsilon):
    Q = np.zeros((env.observation_space.n, env.action_space.n))
    rewards = []  # Track rewards for each episode
    for episode in range(episodes):
        state, _ = env.reset()
        action = epsilon_greedy(Q, state, epsilon)
        done = False
        total_reward = 0
        while not done:
            next_state, reward, done, _, _ = env.step(action)
            total_reward += reward
            next_action = epsilon_greedy(Q, next_state, epsilon)
            Q[state][action] += alpha * (reward + gamma * Q[next_state][next_action] - Q[state][action])
            state, action = next_state, next_action
        rewards.append(total_reward)
    return Q, rewards

# Q-learning Algorithm
def q_learning(env, episodes, alpha, gamma, epsilon):
    Q = np.zeros((env.observation_space.n, env.action_space.n))
    rewards = []  # Track rewards for each episode
    for episode in range(episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0
        while not done:
            action = epsilon_greedy(Q, state, epsilon)
            next_state, reward, done, _, _ = env.step(action)
            total_reward += reward
            Q[state][action] += alpha * (reward + gamma * np.max(Q[next_state]) - Q[state][action])
            state = next_state
        rewards.append(total_reward)
    return Q, rewards

# Print the policy in grid format
def print_policy(policy, grid_shape):
    action_map = {0: "↑", 1: "→", 2: "↓", 3: "←"}  # Map actions to arrows
    grid_policy = np.array([action_map[action] for action in policy]).reshape(grid_shape)

    # Replace Start, Goal, and Cliff cells
    grid_policy[-1, 0] = "S"  # Start
    grid_policy[-1, -1] = "G"  # Goal
    for i in range(1, grid_shape[1] - 1):
        grid_policy[-1, i] = "C"  # Cliff

    # Print the policy grid
    for row in grid_policy:
        print(" ".join(row))

# Print Q-values
def print_q_values(Q, grid_shape):
    print("\nQ-values (state-action values):")
    for state in range(env.observation_space.n):
        row, col = state // grid_shape[1], state % grid_shape[1]
        print(f"State ({row}, {col}): {Q[state]}")

# Plot training rewards comparison
def plot_rewards(rewards_sarsa, rewards_qlearning):
    plt.figure(figsize=(10, 6))
    plt.plot(
        np.convolve(rewards_sarsa, np.ones(10) / 10, mode="valid"),
        label="SARSA (Moving Avg)", color="red", alpha=0.8, linewidth=2
    )
    plt.plot(
        np.convolve(rewards_qlearning, np.ones(10) / 10, mode="valid"),
        label="Q-Learning (Moving Avg)", color="blue", alpha=0.8, linewidth=2
    )

    plt.title("Episode vs Sum of Rewards", fontsize=14)
    plt.xlabel("Episode", fontsize=12)
    plt.ylabel("Sum of Rewards", fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(color="gray", linestyle="--", linewidth=0.5)
    plt.show()

# Set hyperparameters
episodes = 500
alpha = 0.1
gamma = 0.9
epsilon = 0.1

# Train Sarsa and Q-learning
Q_sarsa, rewards_sarsa = sarsa(env, episodes, alpha, gamma, epsilon)
Q_qlearning, rewards_qlearning = q_learning(env, episodes, alpha, gamma, epsilon)

# Extract policies
policy_sarsa = np.argmax(Q_sarsa, axis=1)
policy_qlearning = np.argmax(Q_qlearning, axis=1)

# Plot rewards
plot_rewards(rewards_sarsa, rewards_qlearning)

# Print policies
print("Optimal Policy (SARSA):")
print_policy(policy_sarsa, grid_shape)

print("\nOptimal Policy (Q-Learning):")
print_policy(policy_qlearning, grid_shape)

# Print Q-values
print("\nQ-values for SARSA:")
print_q_values(Q_sarsa, grid_shape)

print("\nQ-values for Q-Learning:")
print_q_values(Q_qlearning, grid_shape)
