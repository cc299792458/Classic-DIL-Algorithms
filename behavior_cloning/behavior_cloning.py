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

"""
    第一次写了一个BehaviorCloning. 虽然不知道标不标准，但至少结果在CartPole-v1这个简单任务上表现的还是挺好的。
    还是挺好玩的。
    2024.3.12
"""


def plot_loss_curves(training_losses, validation_losses):
    epochs = range(1, len(training_losses) + 1)
    plt.plot(epochs, training_losses, label='Training Loss')
    plt.plot(epochs, validation_losses, label='Validation Loss')
    plt.title('Training and Validation Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

# Create a function to wrap the environment with Monitor and any other wrappers you need
def wrap_env(env_id):
    env = gym.make(env_id, render_mode='human')
    env = Monitor(env, "./logs/")  # Specify the directory to save the logs
    env = DummyVecEnv([lambda: env])  # Wrap in a DummyVecEnv for compatibility
    return env

def custom_evaluate_model(model, env, num_episodes=100, render=False):
    episode_rewards = []
    for _ in range(num_episodes):
        obs = env.reset()
        done = False
        total_reward = 0.0
        while not done:
            if render:
                env.render()  # Render the environment
            with torch.no_grad():
                obs_t = torch.tensor(obs).type(torch.float32)
                action_probabilities = torch.softmax(model.predict(obs_t), dim=1)
                action = torch.argmax(action_probabilities, dim=1).item()
            obs, reward, done, _ = env.step([action])
            total_reward += reward
        episode_rewards.append(total_reward)
    
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    return mean_reward, std_reward

class BahaviorCloningModel:
    def __init__(self, input_dim, output_dim, lr=3e-4):
        # This architechture is identical to that of PPO of SB3
        self.nn = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )
        self.optimizer = optim.Adam(self.nn.parameters(), lr=lr)
        if output_dim == 1:
            self.loss_function = nn.BCEWithLogitsLoss()
        else:
            self.loss_function = nn.CrossEntropyLoss()

    def predict(self, obs):
        return self.nn(obs)
    
    def state_dict(self):
        return self.nn.state_dict()

if __name__ == '__main__':
    env_id = 'CartPole-v1'
    # ##### 1. Train an expert #####
    # env = wrap_env(env_id)  # Wrap the environment

    # # Define a callback for evaluation
    # eval_callback = EvalCallback(env,
    #                              best_model_save_path='./logs/best_model/',
    #                              log_path='./logs/results/',
    #                              eval_freq=1e4,  # Evaluate every 10000 steps
    #                              deterministic=True,
    #                              render=False)

    # # Define a callback for stopping training once a reward threshold is reached
    # # stop_callback = StopTrainingOnRewardThreshold(reward_threshold=195, verbose=1)
    # # callback = [eval_callback, stop_callback]  # Combine callbacks

    # ppo = PPO('MlpPolicy', env, verbose=1)
    # ppo.learn(total_timesteps=3e4, callback=eval_callback)
    # ppo.save("./logs/ppo_expert")

    # # Evaluate the trained model
    # mean_reward, std_reward = evaluate_policy(ppo, env, n_eval_episodes=10)
    # print(f"Mean reward = {mean_reward} +/- {std_reward}")

    # ##### 2. Load the expert and rollout to collect data #####
    # env = wrap_env(env_id)  # Wrap the environment
    # ppo_expert = PPO('MlpPolicy', env, verbose=1).load('./logs/ppo_expert')
    
    # expert_data = dict(obs=[], action=[])
    # total_timesteps = int(5e4)
    # obs = env.reset()
    # for _ in tqdm(range(total_timesteps)):
    #     action, _ = ppo_expert.predict(observation=obs, deterministic=True)
    #     new_obs, reward, done, info = env.step(actions=action)
    #     expert_data['obs'].append(obs)
    #     expert_data['action'].append(action)
    #     if done:
    #         obs = env.reset()
    #     else:
    #         obs = new_obs
    # for key in expert_data.keys():
    #     expert_data[key] = np.array(expert_data[key])
    # np.save('./logs/expert_data', expert_data)

    # ##### 3. Create the BCModel and Train it
    # expert_data = np.load('./logs/expert_data.npy', allow_pickle=True).item()
    # expert_obs = expert_data['obs'].squeeze()
    # expert_action = expert_data['action']
    # expert_obs_t = torch.tensor(expert_obs).type(torch.float32)
    # expert_action_t = torch.tensor(expert_action).type(torch.long).squeeze()
    
    # dataset = TensorDataset(expert_obs_t, expert_action_t)
    # train_size = int(0.8 * len(dataset))
    # val_size = len(dataset) - train_size
    # train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    # val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    
    # bc_model = BahaviorCloningModel(input_dim=expert_obs.shape[-1], output_dim=np.max(expert_action)+1)

    # num_epochs = 200
    # training_losses = []
    # validation_losses = []
    # best_val_loss = float('inf')
    # # patience_counter = 0
    # # patience = 5
    # for epoch in tqdm(range(num_epochs)):
    #     total_loss = 0
    #     for inputs, labels in train_loader:
    #         bc_model.optimizer.zero_grad()
    #         outputs = bc_model.predict(inputs)
    #         loss = bc_model.loss_function(outputs, labels)
    #         loss.backward()
    #         bc_model.optimizer.step()
    #         total_loss += loss.item()
        
    #     avg_loss = total_loss / len(train_loader)
    #     training_losses.append(avg_loss)
    #     print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
        
    #     if epoch % 5 == 0:
    #         val_loss = 0
    #         with torch.no_grad():  # This tells PyTorch not to compute gradients during the validation pass, saving memory and computations
    #             for inputs, labels in val_loader:
    #                 outputs = bc_model.predict(inputs)
    #                 loss = bc_model.loss_function(outputs, labels)
    #                 val_loss += loss.item()

    #         avg_val_loss = val_loss / len(val_loader)
    #         if avg_val_loss < best_val_loss:
    #             best_val_loss = avg_val_loss
    #             patience_counter = 0
    #             torch.save(bc_model.state_dict(), './logs/best_behavior_cloning_model.pth')
    #             print(f"Best model saved at epoch {epoch+1} with validation loss: {avg_val_loss:.4f}")
    #         # else:
    #         #     patience_counter += 1
    #         validation_losses.append(avg_val_loss)
    #         print(f"Validation Loss: {avg_val_loss:.4f}")

    #         # if patience_counter >= patience:
    #         #     print(f"Stopping early at epoch {epoch+1}. No improvement in validation loss for {patience} epochs.")
    #         #     break

    # torch.save(bc_model.state_dict(), './logs/final_behavior_cloning_model.pth')
    # # training_losses.pop(-1)
    # validation_losses_full = [validation_losses[i//5] if i % 5 == 0 else validation_losses[i//5] for i in range(num_epochs)]
    # plot_loss_curves(training_losses, validation_losses_full)

    ##### 4. Load the bc model and evaluate it.
    env = wrap_env(env_id)
    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.n
    model = BahaviorCloningModel(input_dim, output_dim)
    model.nn.load_state_dict(torch.load('./logs/best_behavior_cloning_model.pth'))

    mean_reward, std_reward = custom_evaluate_model(model=model, env=env)
    print(f"Mean reward = {mean_reward} +/- {std_reward}")