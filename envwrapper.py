from gym import Wrapper
from gym.spaces import Box
import numpy as np

class CompatibleActionSpaceWrapper(Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.action_space = self.convert_multidiscrete_to_box(env.action_space)
        self.observation_space = env.observation_space  # Pass through

    def convert_multidiscrete_to_box(self, multidiscrete_space):
        low = np.zeros_like(multidiscrete_space.nvec, dtype=np.float32)
        high = multidiscrete_space.nvec.astype(np.float32) - 1
        return Box(low=low, high=high, dtype=np.float32)

    def step(self, action):
        # Convert action back if necessary; here it's passed directly
        return self.env.step(action)

    def reset(self):
        return self.env.reset()
