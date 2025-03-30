"""
test_lunarlander.py
---------------------

Tests the PPO agent for the LunarLander-v3 environment.
- Model is saved to (if not, run train_lunarlander.py): models/ppo_lunarlander.zip

To run:
    python scripts/test_lunarlander.py

"""

from stable_baselines3 import PPO
import gymnasium as gym
import time

# Load model
model = PPO.load("models/ppo_lunarlander")

# Create environment with rendering
env = gym.make("LunarLander-v3", render_mode="human")

# Run a few episodes
for episode in range(3):
    obs, _ = env.reset()
    done = False
    total_reward = 0

    while not done:
        action, _ = model.predict(obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        done = terminated or truncated
        time.sleep(0.02)

    print(f"Episode {episode+1} reward: {total_reward:.2f}")

env.close()
