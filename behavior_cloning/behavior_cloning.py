import os
import gymnasium as gym
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split

from utils.torch_utils import StudentModel, custom_evaluate_model
from utils.training_utils import plot_loss_curves

"""
    第一次写了一个BehaviorCloning. 虽然不知道标不标准，但至少结果在CartPole-v1这个简单任务上表现的还是挺好的。
    还是挺好玩的。
    2024.3.12
    Let's learn DAgger next time.
"""


# Create a function to wrap the environment with Monitor and any other wrappers you need
def wrap_env(env_id, render_mode=None):
    env = gym.make(env_id, render_mode=render_mode)
    env = Monitor(env, "./logs/")  # Specify the directory to save the logs
    env = DummyVecEnv([lambda: env])  # Wrap in a DummyVecEnv for compatibility
    return env

if __name__ == '__main__':
    env_id = 'CartPole-v1'
    # ##### 1. Train an expert #####
    # ./train_expert.py

    # ##### 2. Load the expert and rollout to collect data #####
    # ./train_expert.py

    # # ##### 3. Create the StudentModel Object and Train it
    # cur_dir = os.path.dirname(os.path.abspath(__file__))
    # expert_data = np.load('./data_collecting_logs/expert_data.npy', allow_pickle=True).item()
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
    
    # student_model = StudentModel(input_dim=expert_obs.shape[-1], output_dim=np.max(expert_action)+1)

    # num_epochs = 200
    # training_losses = []
    # validation_losses = []
    # best_val_loss = float('inf')
    # best_student_model_path = os.path.join(cur_dir, 'logs/best_student_model.pth')
    # final_student_model_path = os.path.join(cur_dir, 'logs/final_student_model.pth')
    # # patience_counter = 0
    # # patience = 5
    # for epoch in tqdm(range(num_epochs)):
    #     total_loss = 0
    #     for inputs, labels in train_loader:
    #         student_model.optimizer.zero_grad()
    #         outputs = student_model.predict(inputs)
    #         loss = student_model.loss_function(outputs, labels)
    #         loss.backward()
    #         student_model.optimizer.step()
    #         total_loss += loss.item()
        
    #     avg_loss = total_loss / len(train_loader)
    #     training_losses.append(avg_loss)
    #     print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
        
    #     if epoch % 5 == 0:
    #         val_loss = 0
    #         with torch.no_grad():  # This tells PyTorch not to compute gradients during the validation pass, saving memory and computations
    #             for inputs, labels in val_loader:
    #                 outputs = student_model.predict(inputs)
    #                 loss = student_model.loss_function(outputs, labels)
    #                 val_loss += loss.item()

    #         avg_val_loss = val_loss / len(val_loader)
    #         if avg_val_loss < best_val_loss:
    #             best_val_loss = avg_val_loss
    #             patience_counter = 0
    #             torch.save(student_model.state_dict(), best_student_model_path)
    #             print(f"Best model saved at epoch {epoch+1} with validation loss: {avg_val_loss:.4f}")
    #         # else:
    #         #     patience_counter += 1
    #         validation_losses.append(avg_val_loss)
    #         print(f"Validation Loss: {avg_val_loss:.4f}")

    #         # if patience_counter >= patience:
    #         #     print(f"Stopping early at epoch {epoch+1}. No improvement in validation loss for {patience} epochs.")
    #         #     break

    # torch.save(student_model.state_dict(), final_student_model_path)
    # # training_losses.pop(-1)
    # validation_losses_full = [validation_losses[i//5] if i % 5 == 0 else validation_losses[i//5] for i in range(num_epochs)]
    # plot_loss_curves(training_losses, validation_losses_full)

    ##### 4. Load the student model and evaluate it.
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    best_student_model_path = os.path.join(cur_dir, 'logs/best_student_model.pth')
    env = wrap_env(env_id, render_mode=None)    # render_mode='human'  
    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.n
    model = StudentModel(input_dim, output_dim)
    model.nn.load_state_dict(torch.load(best_student_model_path))

    mean_reward, std_reward = custom_evaluate_model(model=model, env=env)
    print(f"Mean reward = {mean_reward} +/- {std_reward}")