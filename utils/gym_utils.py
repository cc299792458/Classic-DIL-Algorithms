import gymnasium as gym
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv
import os


# # Create a function to wrap the environment with Monitor and any other wrappers you need
# def wrap_env(env_id, log_path=None, render_mode=None):
#     env = gym.make(env_id, render_mode=render_mode)
#     env = Monitor(env, log_path)  # Specify the directory to save the logs
#     env = DummyVecEnv([lambda: env])  # Wrap in a DummyVecEnv for compatibility
#     return env

def make_env(env_id, log_path=None, rank=0, render_mode=None):
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param log_path: (str) the path to save the log files
    :param rank: (int) index of the subprocess env
    :param render_mode: (str) the render mode of the gym environment
    """
    def _init():
        env = gym.make(env_id, render_mode=render_mode)
        if log_path is not None:
            # Ensure the directory for the rank exists
            os.makedirs(log_path, exist_ok=True)
            env = Monitor(env, os.path.join(log_path, str(rank)))
        return env
    return _init

def wrap_env(env_id, num_envs=4, log_path=None, render_mode=None):
    """
    Wrap the environment to support parallel execution.

    :param env_id: (str) the environment ID
    :param num_envs: (int) the number of parallel environments
    :param log_path: (str) the path to save the log files
    :param render_mode: (str) the render mode of the gym environment
    """
    # Create a list of environment instances
    envs = [make_env(env_id, log_path, rank, render_mode) for rank in range(num_envs)]
    # Create the vectorized environment
    env = SubprocVecEnv(envs)
    return env

def show_env(env_id, render_mode='human'):
    env = gym.make(env_id, render_mode=render_mode)
    env.reset()
    for _ in range(1000):
        env.render()
        action = env.action_space.sample()  # Choose an random action
        observation, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
    env.close()

if __name__ == '__main__':
    # show_env(env_id='CartPole-v1', render_mode='human')
    show_env(env_id='LunarLander-v2', render_mode='human')