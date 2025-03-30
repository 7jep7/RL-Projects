"""
Train PPO on HalfCheetah-v5 using stable-baselines3.

This script sets up the environment, initializes the agent, trains it, 
and logs metrics for TensorBoard. The trained model is saved to disk.

Structure:
- env: HalfCheetah-v5 (17D obs, 6D continuous action)
- algo: PPO (with MLP policy)
- logging: logs/halfcheetah/
- model output: models/ppo_halfcheetah.zip
"""

import os
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure

# Create output folders
os.makedirs("logs/halfcheetah", exist_ok=True)
os.makedirs("models", exist_ok=True)

# Create and wrap environment
env = gym.make("HalfCheetah-v5")
env = Monitor(env)

print("Observation space:", env.observation_space)
print("Action space:", env.action_space)

# Configure logger for TensorBoard
new_logger = configure("logs/halfcheetah", ["stdout", "tensorboard"])

# Initialize PPO agent
model = PPO(
    policy="MlpPolicy",
    env=env,
    verbose=1,
    tensorboard_log="./logs/halfcheetah",
)

model.set_logger(new_logger)

# Train the agent
model.learn(total_timesteps=500_000)

# Save model
model.save("models/ppo_halfcheetah")

env.close()
