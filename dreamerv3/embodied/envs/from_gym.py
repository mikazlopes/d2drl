import functools
from gym import Wrapper
from gym.spaces import Box
import embodied
import gym
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


class FromGym(embodied.Env):

  def __init__(self, env, obs_key='image', act_key='action', **kwargs):
    if isinstance(env, str):
      self._env = gym.make(env, **kwargs)
    else:
      assert not kwargs, kwargs
      self._env = env
    self._obs_dict = hasattr(self._env.observation_space, 'spaces')
    self._act_dict = hasattr(self._env.action_space, 'spaces')
    self._obs_key = obs_key
    self._act_key = act_key
    self._done = True
    self._info = None

  @property
  def info(self):
    return self._info

  @functools.cached_property
  def obs_space(self):
    if self._obs_dict:
      spaces = self._flatten(self._env.observation_space.spaces)
    else:
      spaces = {self._obs_key: self._env.observation_space}
    spaces = {k: self._convert(v) for k, v in spaces.items()}
    return {
        **spaces,
        'reward': embodied.Space(np.float32),
        'is_first': embodied.Space(bool),
        'is_last': embodied.Space(bool),
        'is_terminal': embodied.Space(bool),
    }

  @functools.cached_property
  def act_space(self):
    if self._act_dict:
      spaces = self._flatten(self._env.action_space.spaces)
    else:
      spaces = {self._act_key: self._env.action_space}
    spaces = {k: self._convert(v) for k, v in spaces.items()}
    spaces['reset'] = embodied.Space(bool)
    return spaces

  def step(self, action):
    if action['reset'] or self._done:
      self._done = False
      obs = self._env.reset()
      return self._obs(obs, 0.0, is_first=True)
    if self._act_dict:
      action = self._unflatten(action)
    else:
      action = action[self._act_key]
    obs, reward, self._done, self._info = self._env.step(action)
    return self._obs(
        obs, reward,
        is_last=bool(self._done),
        is_terminal=bool(self._info.get('is_terminal', self._done)))

  # def _obs(
  #     self, obs, reward, is_first=False, is_last=False, is_terminal=False):
  #   if not self._obs_dict:
  #     obs = {self._obs_key: obs}
  #   obs = self._flatten(obs)
  #   print(obs.items())
  #   print(reward)
  #   obs = {k: np.asarray(v) for k, v in obs.items()}
  #   obs.update(
  #       reward=np.float32(reward),
  #       is_first=is_first,
  #       is_last=is_last,
  #       is_terminal=is_terminal)
  #   return obs

  def _obs(self, obs, reward, is_first=False, is_last=False, is_terminal=False):
    # If observation is a tuple, assume the actual observation is the first element.
    if isinstance(obs, tuple):
        obs = obs[0]  # Assuming the actual observation is the first element of the tuple

    # Now, handle the case where the observation is directly a numpy array
    if isinstance(obs, np.ndarray):
        # Directly use the observation if it's already a numpy array
        updated_obs = {'image': np.asarray(obs, dtype=np.uint8)}
    elif isinstance(obs, dict):
        # If the observation is a dictionary, convert each value to a numpy array
        updated_obs = {k: np.asarray(v, dtype=np.uint8) for k, v in obs.items()}
    else:
        # Log an error or handle unexpected observation formats
        print("Unexpected observation format received:", type(obs))
        updated_obs = {}

    # Update the dictionary with additional metadata
    updated_obs.update({
        'reward': np.float32(reward),
        'is_first': is_first,
        'is_last': is_last,
        'is_terminal': is_terminal
    })

    return updated_obs



  
  def render(self):
    image = self._env.render('rgb_array')
    assert image is not None
    return image

  def close(self):
    try:
      self._env.close()
    except Exception:
      pass

  def _flatten(self, nest, prefix=None):
    result = {}
    for key, value in nest.items():
      key = prefix + '/' + key if prefix else key
      if isinstance(value, gym.spaces.Dict):
        value = value.spaces
      if isinstance(value, dict):
        result.update(self._flatten(value, key))
      else:
        result[key] = value
    return result

  def _unflatten(self, flat):
    result = {}
    for key, value in flat.items():
      parts = key.split('/')
      node = result
      for part in parts[:-1]:
        if part not in node:
          node[part] = {}
        node = node[part]
      node[parts[-1]] = value
    return result

  def _convert(self, space):
    if hasattr(space, 'n'):
      return embodied.Space(np.int32, (), 0, space.n)
    return embodied.Space(space.dtype, space.shape, space.low, space.high)
  

  

