import gymnasium as gym
import time

env = gym.make("LunarLander-v3", render_mode="human")

for episode in range(3):
    obs, info = env.reset()
    done = False
    total_reward = 0

    while not done:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward

        time.sleep(0.02)  # slow down for smoother viewing

    print(f"Episode {episode + 1} reward: {total_reward:.2f}")

env.close()
