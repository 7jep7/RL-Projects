"""
Train SAC on HalfCheetah-v5 with GPU acceleration and faster config:
- 500k timesteps
- Larger batch size
- Smaller buffer
- Checkpointing
"""

import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback
import torch

# Optimized config
TOTAL_TIMESTEPS = 500_000
BATCH_SIZE = 512            # Larger updates
BUFFER_SIZE = 300_000       # Smaller buffer to save memory
SAVE_FREQ = 250_000         # Save halfway and at the end

env = gym.make("HalfCheetah-v5")
env = Monitor(env)

model = SAC(
    policy="MlpPolicy",
    env=env,
    verbose=1,
    batch_size=BATCH_SIZE,
    buffer_size=BUFFER_SIZE,
    tensorboard_log="logs/halfcheetah_sac",
    device="cuda" if torch.cuda.is_available() else "cpu"
)

checkpoint_callback = CheckpointCallback(
    save_freq=SAVE_FREQ,
    save_path="models/sac_halfcheetah/",
    name_prefix="sac_checkpoint"
)

model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=checkpoint_callback)

model.save("models/sac_halfcheetah/final_model")

print("SAC training complete. Saved to models/sac_halfcheetah/")
