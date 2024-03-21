import os
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from utils.gym_utils import wrap_env

if __name__ == '__main__':
    env_id = 'LunarLander-v2'
    cur_path = './training_logs/'+env_id+'/'
    # Ensure the log directory exists
    os.makedirs(cur_path, exist_ok=True)
    env = wrap_env(env_id, log_path=cur_path+'monitor/', render_mode=None)  # Assuming no need for rendering during training

    # Define a callback for evaluation
    eval_callback = EvalCallback(env,
                                 best_model_save_path=cur_path+'best_model/',
                                 log_path=cur_path+'results/',
                                 eval_freq=1e4,  # Evaluate every 10000 steps
                                 deterministic=True,
                                 render=False)

    # Check if a pre-trained model exists
    model_path = cur_path+'ppo_expert.zip'
    if os.path.exists(model_path):
        ppo = PPO.load(model_path, env=env)
        print("Loaded a previously trained model.")
    else:
        ppo = PPO('MlpPolicy', env, verbose=1)
        print("Created a new model.")

    # Evaluate the trained model before training
    mean_reward, std_reward = evaluate_policy(ppo, env, n_eval_episodes=10)
    print(f"Before training - Mean reward = {mean_reward} +/- {std_reward}")
    
    # Continue training
    ppo.learn(total_timesteps=2e5, callback=eval_callback)
    # Save the model
    ppo.save(model_path)

    # Evaluate the trained model again
    mean_reward, std_reward = evaluate_policy(ppo, env, n_eval_episodes=10)
    print(f"After training - Mean reward = {mean_reward} +/- {std_reward}")
