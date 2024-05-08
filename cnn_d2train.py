import os
import traceback
import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, CallbackList
from cnn_d2Env import DiabloIIGymEnv  # Import your custom environment


# Function to create a new instance of the environment
def make_env(ip, game_port, flask_port, env_id, rank, log_dir):
    def _init():
        env = DiabloIIGymEnv(server_url=f'http://{ip}:{game_port}', flask_port=flask_port, env_id=env_id)
        env = Monitor(env, log_dir + str(rank))
        return env
    return _init

# Set up the directory where we'll save training logs
log_dir = "./logs/"
os.makedirs(log_dir, exist_ok=True)

def linear_schedule(initial_value):
    """
    Linear learning rate schedule.
    :param initial_value: (float or str) Initial learning rate which is linearly annealed to zero
    :return: (function) a function that takes the current progress remaining (from 1 to 0) and
      returns the learning rate as a float
    """
    if isinstance(initial_value, str):
        initial_value = float(initial_value)

    def func(progress_remaining):
        """
        Progress will decrease from 1 (beginning) to 0
        :param progress_remaining: (float)
        :return: (float) current learning rate
        """
        return progress_remaining * initial_value

    return func

# List of server IPs and ports
servers = [
    #('192.168.150.61', 5001, 8121), #Windows11 1
    #('192.168.150.62', 5002, 8122), #Windows11 2
    #('192.168.150.63', 5003, 8123, 'Win3'), #Windows11 3
    #('192.168.150.64', 5004, 8124, 'Win4'), #Windows11 4
    #('192.168.150.65', 5005, 8125), #Windows11 5
    #('192.168.150.66', 5006, 8126, 'Win6'), #Windows11 6
    #('192.168.150.156', 5000, 8127), #Windows10 7
    #('router.titogang.org', 5008, 8128), #Windows10 8
    ('router.titogang.org', 5009, 8129, 'Asus'), #Asus Bare Metal
    ('router.titogang.org', 5010, 8130, 'Surface'), #Surface
    #('router.titogang.org', 5011, 8131), #Windows10 11
    #('192.168.150.214', 5012, 8132), #Mac Intel Bare Metal
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


def train(checkpoint_path=None, device='cpu'):
    # Create the vectorized environment
    env = SubprocVecEnv([make_env(ip, game_port, flask_port, env_id, i, log_dir) for i, (ip, game_port, flask_port, env_id) in enumerate(servers)])
    set_random_seed(0)  # Set random seed for reproducibility

    # Learning rate schedule function
    learning_rate = linear_schedule(3e-4)

    callback = SimpleCallback()
    custom_checkpoint_callback = CustomCheckpointCallback(save_freq=500, save_path='./checkpoints/', name_prefix='d2_model')
    ep_lenght = 1024 * 8

    # Check if a checkpoint exists and load it; otherwise, start a new model
    if checkpoint_path and os.path.exists(checkpoint_path):
        model = PPO.load(checkpoint_path, env=env, device=device, tensorboard_log="./d2_ppo_tensorboard/", learning_rate=2.87e-4)
        model.n_steps = ep_lenght
        model.rollout_buffer.buffer_size = ep_lenght
        model.rollout_buffer.n_envs = len(servers)
        model.rollout_buffer.reset()
        model.set_env(env)  # Set the environment for the loaded model
        print(f"Continuing training from checkpoint: {checkpoint_path}")
    else:
        model = PPO("CnnPolicy", env, verbose=1, tensorboard_log="./d2_ppo_tensorboard/", learning_rate=2.87e-4, device=device, gamma=0.999, batch_size=512, n_steps=ep_lenght, n_epochs=1)
        print("Starting new training session")

    # Train the agent
    try:
        model.learn(total_timesteps=int(1e8), callback=[callback, custom_checkpoint_callback], reset_num_timesteps=not checkpoint_path)
    except Exception as e:
        print(f"An error occurred during training: {e}")
        traceback.print_exc()  # This will print the full traceback

    # Save the agent
    model.save("d2_ppo_model")
    env.close()  # Close the environment
    print("Training finished!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', help="Specify the device for training (e.g., 'cpu', 'cuda', 'mps')", default='cpu')
    args = parser.parse_args()

    checkpoint_dir = './checkpoints/'
    os.makedirs(checkpoint_dir, exist_ok=True)  # Ensure checkpoint directory exists

    latest_checkpoint = max([os.path.join(checkpoint_dir, f) for f in os.listdir(checkpoint_dir) if f.endswith('.zip')], default=None, key=os.path.getctime)
    train(latest_checkpoint, device=args.device)
