import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
import numpy as np
from tqdm import tqdm


# Create a function to wrap the environment with Monitor and any other wrappers you need
def wrap_env(env_id):
    env = gym.make(env_id, render_mode=None)
    env = Monitor(env, "./data_collecting_logs/")  # Specify the directory to save the logs
    env = DummyVecEnv([lambda: env])  # Wrap in a DummyVecEnv for compatibility
    return env

if __name__ == '__main__':
    env_id = 'CartPole-v1'
    env = wrap_env(env_id)  # Wrap the environment
    ppo_expert = PPO('MlpPolicy', env, verbose=1).load('./training_logs/ppo_expert')
    
    expert_data = dict(obs=[], action=[])
    total_timesteps = int(5e4)
    obs = env.reset()
    for _ in tqdm(range(total_timesteps)):
        action, _ = ppo_expert.predict(observation=obs, deterministic=True)
        new_obs, reward, done, info = env.step(actions=action)
        expert_data['obs'].append(obs)
        expert_data['action'].append(action)
        if done:
            obs = env.reset()
        else:
            obs = new_obs
    for key in expert_data.keys():
        expert_data[key] = np.array(expert_data[key])
    np.save('./data_collecting_logs/expert_data', expert_data)