import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
import numpy as np
from tqdm import tqdm
from utils.gym_utils import wrap_env
import os

# # Create a function to wrap the environment with Monitor and any other wrappers you need
# def wrap_env(env_id):
#     env = gym.make(env_id, render_mode=None)
#     env = Monitor(env, "./data_collecting_logs/")  # Specify the directory to save the logs
#     env = DummyVecEnv([lambda: env])  # Wrap in a DummyVecEnv for compatibility
#     return env

# if __name__ == '__main__':
#     # env_id = 'CartPole-v1'
#     env_id = 'LunarLander-v2'
#     env = wrap_env(env_id)  # Wrap the environment
#     ppo_expert = PPO('MlpPolicy', env, verbose=1).load('./training_logs/'+env_id+'/ppo_expert')
    
#     expert_data = dict(obs=[], action=[])
#     total_timesteps = int(5e4)
#     obs = env.reset()
#     for _ in tqdm(range(total_timesteps)):
#         action, _ = ppo_expert.predict(observation=obs, deterministic=True)
#         new_obs, reward, done, info = env.step(actions=action)
#         expert_data['obs'].append(obs)
#         expert_data['action'].append(action)
#         if done:
#             obs = env.reset()
#         else:
#             obs = new_obs
#     for key in expert_data.keys():
#         expert_data[key] = np.array(expert_data[key])
#     np.save('./data_collecting_logs/expert_data', expert_data)

if __name__ == '__main__':
    # env_id = 'CartPole-v1'
    env_id = 'LunarLander-v2'
    log_dir = './data_collecting_logs/' + env_id + '/'
    model_dir = './training_logs/' + env_id + '/'
    model_path = model_dir + 'ppo_expert.zip'

    # Ensure the log and model directories exist
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    env = wrap_env(env_id, log_dir)  # Wrap the environment with the specified log directory
    ppo_expert = PPO.load(model_path, env=env)
    
    expert_data = dict(obs=[], actions=[])
    total_timesteps = int(5e4)
    obs = env.reset()
    for _ in tqdm(range(total_timesteps)):
        action, _ = ppo_expert.predict(observation=obs, deterministic=True)
        new_obs, rewards, dones, info = env.step(action)
        expert_data['obs'].append(obs[0])  # Adjust for the observation shape
        expert_data['actions'].append(action[0])  # Adjust for the action shape
        if dones[0]:
            obs = env.reset()
        else:
            obs = new_obs

    for key in expert_data.keys():
        expert_data[key] = np.array(expert_data[key])
    np.save(log_dir + 'expert_data.npy', expert_data)  # Save the expert data in the specified log directory