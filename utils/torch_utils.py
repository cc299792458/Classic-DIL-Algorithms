import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class StudentModel:
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