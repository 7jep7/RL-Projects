import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3 import SAC
import time

env = gym.make("HalfCheetah-v5", render_mode="human")
obs, _ = env.reset()

# Run one random step to trigger render
obs, _, _, _, _ = env.step(env.action_space.sample())

# #Use ppo_halfcheetah_gpu (faster training using parallel environments) or ppo_halfcheetah (slower training)
# model = PPO.load("models/ppo_halfcheetah_gpu", env=env)

#Or, for SAC-trained (soft actor-critic) models: models/sac_halfcheetah/sac_checkpoint_250000_steps.zip
model = SAC.load("models/sac_halfcheetah/sac_checkpoint_250000_steps", env=env)


total_reward = 0

while True:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, _ = env.step(action)
    total_reward += reward

    time.sleep(0.02)  # Slow things down (feel free to tweak: 0.03, 0.05)

    if terminated or truncated:
        break

print(f"Total reward: {total_reward:.2f}")
env.close()
