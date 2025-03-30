"""
Train PPO on HalfCheetah-v5 using vectorized environments and GPU acceleration.
Compared to train_halfcheetah.py, this version uses:
- 4 parallel environments (SubprocVecEnv)
- CUDA backend for faster model training

The "normal" version took around 1 second for 1k timesteps on Jonas' Robohub. (482s for 501760 timesteps)
This one takes: 1s for 3k timesteps, roughly (Using NUM_ENVS=4). In total: 194s for 507904 timesteps. 2-3x as fast.


"""

import os
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure

# Create output folders
os.makedirs("logs/halfcheetah_gpu", exist_ok=True)
os.makedirs("models", exist_ok=True)

# Use 4 parallel environments
NUM_ENVS = 4

def make_env():
    def _init():
        env = gym.make("HalfCheetah-v5")
        return Monitor(env)
    return _init

if __name__ == "__main__":

    env = SubprocVecEnv([make_env() for _ in range(NUM_ENVS)])

    # Configure TensorBoard logging
    new_logger = configure("logs/halfcheetah_gpu", ["stdout", "tensorboard"])

    # Initialize PPO agent (using GPU + compatible hyperparams)
    model = PPO(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        device="cuda",  # <--- GPU!
        n_steps=2048,
        batch_size=256,
        n_epochs=10,
        tensorboard_log="./logs/halfcheetah_gpu"
    )

    model.set_logger(new_logger)

    # Train the agent
    model.learn(total_timesteps=500_000)

    # Save model
    model.save("models/ppo_halfcheetah_gpu")

    env.close()
