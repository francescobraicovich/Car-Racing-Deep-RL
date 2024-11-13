import gymnasium as gym
import numpy as np

class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip):
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        for _ in range(self._skip):
            state, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            if terminated:
                break
        return state, total_reward, terminated, truncated, info

# Define the DiscretizeActionWrapper to convert the continuous action space into a discrete one
class DiscretizeActionWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super(DiscretizeActionWrapper, self).__init__(env)
        # Define a set of discrete actions
        self.actions = [
            np.array([0.0, 1.0, 0.0]),   # Full gas
            np.array([0.0, 0.0, 0.0]),   # No action
            np.array([0.0, 0.0, 1.0]),   # Full brake
            np.array([-1.0, 0.0, 0.0]),  # Turn left
            np.array([1.0, 0.0, 0.0]),   # Turn right
        ]
        self.action_space = gym.spaces.Discrete(len(self.actions))
        
    def action(self, action):
        return self.actions[action]
