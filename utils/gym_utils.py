import gymnasium as gym
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv


# Create a function to wrap the environment with Monitor and any other wrappers you need
def wrap_env(env_id, log_path=None, render_mode=None):
    env = gym.make(env_id, render_mode=render_mode)
    env = Monitor(env, log_path)  # Specify the directory to save the logs
    env = DummyVecEnv([lambda: env])  # Wrap in a DummyVecEnv for compatibility
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