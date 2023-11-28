import os
import shutil
import traceback
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
import torch
import torch.nn as nn
import numpy as np
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from d2Env import DiabloIIGymEnv  # Import your custom environment


class CustomCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=64):
        super(CustomCNN, self).__init__(observation_space, features_dim)
        
        # CNN layers
        self.cnn = nn.Sequential(
            nn.Conv2d(12, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Sample a dummy input from the observation space
        dummy_input = observation_space['image'].sample()
        dummy_input_tensor = torch.as_tensor(dummy_input[None]).float()  # Add a batch dimension

        with torch.no_grad():
            n_flatten = self.cnn(dummy_input_tensor).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations):
        return self.linear(self.cnn(observations['image']))



# Function to create a new instance of the environment
def make_env(ip, game_port, flask_port, rank, log_dir):
    def _init():
        env = DiabloIIGymEnv(server_url=f'http://{ip}:{game_port}', flask_port=flask_port)
        env = Monitor(env, log_dir + str(rank))
        return env
    return _init

# Set up the directory where we'll save training logs
log_dir = "./logs/"
os.makedirs(log_dir, exist_ok=True)

# List of server IPs and ports
servers = [
    ('router.titogang.org', 5001, 8121), #Windows11 1
    ('router.titogang.org', 5002, 8122), #Windows11 2
    ('router.titogang.org', 5003, 8123), #Windows11 3
    ('router.titogang.org', 5004, 8124), #Windows11 4
    ('router.titogang.org', 5005, 8125), #Windows11 5
    ('router.titogang.org', 5006, 8126), #Windows11 6
    # ('192.168.150.107', 5000, 8130), #check
    ('router.titogang.org', 5009, 8129), #Asus
    #('router.titogang.org', 5010, 8130), #Surface
    # Add more server IPs, game ports, and flask ports as needed
]

class CustomCheckpointCallback(CheckpointCallback):
    def __init__(self, save_freq, save_path, name_prefix='rl_model', verbose=0):
        super(CustomCheckpointCallback, self).__init__(save_freq, save_path, name_prefix, verbose)

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            # Call the parent method to save a new checkpoint
            super(CustomCheckpointCallback, self)._on_step()

            # Get all checkpoint files
            checkpoints = [os.path.join(self.save_path, file) for file in os.listdir(self.save_path) if file.endswith('.zip')]

            # Sort the checkpoints by modification time
            checkpoints.sort(key=os.path.getmtime)

            # Remove all but the most recent checkpoint
            for checkpoint in checkpoints[:-1]:
                os.remove(checkpoint)
                print(f"Removed old checkpoint: {checkpoint}")

        return True

class SimpleCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(SimpleCallback, self).__init__(verbose)
        self.step_count = 0

    def _on_step(self) -> bool:
        self.step_count += 1
        if self.step_count % 100 == 0:
            print(f"Step number: {self.step_count}")
        return True

def train():
    # Create the vectorized environment
    env = SubprocVecEnv([make_env(ip, game_port, flask_port, i, log_dir) for i, (ip, game_port, flask_port) in enumerate(servers)])

    # Set random seed for reproducibility
    set_random_seed(0)

    ep_length = 2048 * 8

    callback = SimpleCallback()
    custom_checkpoint_callback = CustomCheckpointCallback(save_freq=1000, save_path='./checkpoints/', name_prefix='d2_model')


    # Define the policy_kwargs dict to pass to the PPO model
    policy_kwargs = dict(
        features_extractor_class=CustomCNN,
        features_extractor_kwargs=dict(features_dim=512),  # You can adjust this dimension
    )

    # Instantiate the agent
    model = PPO(
        "MultiInputPolicy",
        env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log="./d2_ppo_tensorboard/",
        device='mps',
        batch_size=512,
        n_steps=ep_length,
        n_epochs=1,
        gamma=0.999,
    )


    try:
        model.learn(total_timesteps=int(1e8), callback=[callback, custom_checkpoint_callback])
    except Exception as e:
        print(f"An error occurred during training: {e}")
        traceback.print_exc()  # This will print the full traceback

    # Save the agent
    model.save("d2_ppo_model")

    # Close the environment
    env.close()

    # Output to show training has finished
    print("Training finished!")

if __name__ == '__main__':
    train()
