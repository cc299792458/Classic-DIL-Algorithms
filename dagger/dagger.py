import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split

from utils.torch_utils import StudentModel

"""
    写DAgger.
"""


# Create a function to wrap the environment with Monitor and any other wrappers you need
def wrap_env(env_id):
    env = gym.make(env_id, render_mode='human')
    env = Monitor(env, "./logs/")  # Specify the directory to save the logs
    env = DummyVecEnv([lambda: env])  # Wrap in a DummyVecEnv for compatibility
    return env
    
if __name__ == '__main__':
    env_id = 'CartPole-v1'

    # ##### 1. Train an expert #####
    # ./train_expert.py

    # ##### 2. Load the expert and rollout to collect data #####
    # ./train_expert.py