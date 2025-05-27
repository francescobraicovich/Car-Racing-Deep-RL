"""
Custom Gymnasium wrappers for environment preprocessing.

This module provides wrappers for the CarRacing-v2 environment to:
- Skip frames to speed up training and reduce computational load.
- Discretize the continuous action space into a set of predefined discrete actions.
"""
import gymnasium as gym
import numpy as np


class SkipFrame(gym.Wrapper):
    """
    A Gymnasium wrapper that skips a specified number of frames for each action.
    
    This can speed up training by reducing the number of observations processed
    and can also help the agent learn more effectively by reducing
    the temporal correlation between consecutive states.
    """
    def __init__(self, env, skip):
        """
        Initialize the SkipFrame wrapper.

        Args:
            env: The Gymnasium environment to wrap.
            skip (int): The number of frames to skip for each action. 
                        The action is repeated for these skipped frames.
        """
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        """
        Take an action in the environment, repeating it for 'skip' number of frames.

        The rewards from all skipped frames are accumulated. The observation,
        termination status, truncation status, and info from the last frame in the
        sequence are returned.

        Args:
            action: The action to take.

        Returns:
            A tuple containing:
                - state: The observation from the last frame.
                - total_reward: The sum of rewards from all skipped frames.
                - terminated: Boolean indicating if the episode ended due to termination.
                - truncated: Boolean indicating if the episode ended due to truncation.
                - info: A dictionary containing additional information from the last frame.
        """
        total_reward = 0.0
        for _ in range(self._skip):
            state, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            if terminated or truncated:  # Check for truncation as well
                break
        return state, total_reward, terminated, truncated, info


class DiscretizeActionWrapper(gym.ActionWrapper):
    """
    A Gymnasium wrapper that converts a continuous action space into a discrete one.

    This is useful for environments like CarRacing-v2 where the original action
    space is continuous, but some reinforcement learning algorithms require a
    discrete action space.
    """
    def __init__(self, env):
        """
        Initialize the DiscretizeActionWrapper.

        Args:
            env: The Gymnasium environment to wrap.
        """
        super().__init__(env)
        # Define a set of discrete actions:
        # [Steering, Gas, Brake]
        # Steering: -1 (left) to +1 (right)
        # Gas: 0 to 1
        # Brake: 0 to 1 (represents brake, not reverse)
        self._actions = [
            np.array([0.0, 1.0, 0.0], dtype=np.float32),   # Full gas, no steering
            np.array([0.0, 0.0, 0.0], dtype=np.float32),   # No action (coast)
            np.array([0.0, 0.0, 0.8], dtype=np.float32),   # Full brake (0.8 is a common value)
            np.array([-1.0, 0.0, 0.0], dtype=np.float32),  # Sharp left, no gas/brake
            np.array([1.0, 0.0, 0.0], dtype=np.float32),   # Sharp right, no gas/brake
            # Potentially add more nuanced actions:
            # np.array([-0.5, 0.2, 0.0]), # Gentle left, light gas
            # np.array([0.5, 0.2, 0.0]),  # Gentle right, light gas
        ]
        self.action_space = gym.spaces.Discrete(len(self._actions))
        
    def action(self, action_idx):
        """
        Convert a discrete action index into its corresponding continuous action.

        Args:
            action_idx (int): The index of the discrete action.

        Returns:
            np.ndarray: The continuous action vector corresponding to the discrete action index.
        """
        return self._actions[action_idx]
