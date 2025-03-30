from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
import gymnasium as gym

# Setup environment with monitoring and vectorization
env = DummyVecEnv([lambda: Monitor(gym.make("LunarLander-v3"))])

# Create PPO model with TensorBoard logging
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./logs/lunarlander/", device="cpu")

try:
    model.learn(total_timesteps=200_000)
except KeyboardInterrupt:
    print("Training interrupted. Saving model...")
    model.save("models/ppo_lunarlander_partial")

# Save final model
model.save("models/ppo_lunarlander")
env.close()
