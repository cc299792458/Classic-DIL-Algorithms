import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv


# Create a function to wrap the environment with Monitor and any other wrappers you need
def wrap_env(env_id):
    env = gym.make(env_id, render_mode=None)
    env = Monitor(env, "./training_logs/")  # Specify the directory to save the logs
    env = DummyVecEnv([lambda: env])  # Wrap in a DummyVecEnv for compatibility
    return env


if __name__ == '__main__':
    env_id = 'CartPole-v1'
    env = wrap_env(env_id)  # Wrap the environment

    # Define a callback for evaluation
    eval_callback = EvalCallback(env,
                                 best_model_save_path='./training_logs/best_model/',
                                 log_path='./training_logs/results/',
                                 eval_freq=1e4,  # Evaluate every 10000 steps
                                 deterministic=True,
                                 render=False)

    # Define a callback for stopping training once a reward threshold is reached
    # stop_callback = StopTrainingOnRewardThreshold(reward_threshold=195, verbose=1)
    # callback = [eval_callback, stop_callback]  # Combine callbacks

    ppo = PPO('MlpPolicy', env, verbose=1)
    ppo.learn(total_timesteps=3e4, callback=eval_callback)
    ppo.save("./training_logs/ppo_expert")

    # Evaluate the trained model
    mean_reward, std_reward = evaluate_policy(ppo, env, n_eval_episodes=10)
    print(f"Mean reward = {mean_reward} +/- {std_reward}")