import gymnasium as gym
from stable_baselines3 import A2C,PPO
import os

model_dir="./models/PPO"
env=gym.make("LunarLander-v2",render_mode="human")
env.reset()
model_path=f"{model_dir}/"