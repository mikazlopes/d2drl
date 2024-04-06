import warnings
import dreamerv3
from dreamerv3 import embodied
import numpy as np
import gym
from d2d3Env import DiabloIIGymEnv  # Import your custom environment
from dreamerv3.embodied.envs.from_gym import FromGym, CompatibleActionSpaceWrapper  # Import the necessary classes
from gym.wrappers import ResizeObservation
import cv2

    
warnings.filterwarnings('ignore', '.*truncated to dtype int32.*')

def main():

    # Configure DreamerV3
    config = embodied.Config(dreamerv3.configs['defaults'])
    config = config.update(dreamerv3.configs['xlarge'])
    config = config.update({
        'logdir': '/d2rl/logdir/run1',
        'run.train_ratio': 64,
        'run.log_every': 30,  # Seconds
        'batch_size': 16,
        'jax.prealloc': False,
        'encoder.mlp_keys': '$^',
        'decoder.mlp_keys': '$^',
        'encoder.cnn_keys': 'image',
        'decoder.cnn_keys': 'image',
    })
    config = embodied.Flags(config).parse()

    # Setup logging and agents
    logdir = embodied.Path(config.logdir)
    step = embodied.Counter()
    logger = embodied.Logger(step, [
        embodied.logger.TerminalOutput(),
        embodied.logger.JSONLOutput(logdir, 'metrics.jsonl'),
        embodied.logger.TensorBoardOutput(logdir),
    ])
 

    # # List of server IPs and ports
    # servers = [
    #     ('router.titogang.org', 5012, 8132),
    #     ('router.titogang.org', 5009, 8129),
    # ]

    # # Wrap each environment instance with FromGym
    # envs = [
    #     FromGym(
    #         DiabloIIGymEnv(server_url=f'http://{ip}:{game_port}', flask_port=flask_port),
    #         obs_key='image'
    #     ) for ip, game_port, flask_port in servers
    # ]
    

    # Wrap the batch of environments for DreamerV3
    # env = dreamerv3.wrap_env(embodied.BatchEnv(envs, parallel=True), config)

    env = FromGym(DiabloIIGymEnv(server_url='http://router.titogang.org:5009', flask_port=8129), obs_key='image')
    env = dreamerv3.wrap_env(env, config)
    env = embodied.BatchEnv(env, parallel=False)
    agent = dreamerv3.Agent(env.obs_space, env.act_space, step, config)
    replay = embodied.replay.Uniform(config.batch_length, config.replay_size, logdir / 'replay')
    args = embodied.Config(**config.run, logdir=config.logdir, batch_steps=config.batch_size * config.batch_length)
    embodied.run.train(agent, env, replay, logger, args)

if __name__ == '__main__':
    main()
