from stable_baselines3 import PPO
import gymnasium as gym
import time

# Load your saved model
model = PPO.load("ppo_lunarlander")

# Create a renderable environment
env = gym.make("LunarLander-v3", render_mode="human")

# Run 3 episodes
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

    print(f"Episode {episode+1} Total Reward: {total_reward:.2f}")

env.close()
