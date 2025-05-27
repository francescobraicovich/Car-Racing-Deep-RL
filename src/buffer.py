import random
from collections import deque

import numpy as np
import torch

# Set random seeds for reproducibility
SEED = 1  # PEP8: constants should be uppercase
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size):
        """
        Initialize a ReplayBuffer object.

        Args:
            action_size (int): Dimension of each action.
            buffer_size (int): Maximum size of buffer.
            batch_size (int): Size of each training batch.
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        
    def add(self, state, action, reward, next_state, done):
        """
        Add a new experience to memory.
        Experiences are stored on the CPU to save GPU memory.
        The batch will be moved to the appropriate device during the learning step.
        """
        # Ensure all tensor data is moved to CPU before storing.
        # This is important if data was previously on a GPU.
        if isinstance(state, torch.Tensor):
            state = state.cpu()
        if isinstance(action, torch.Tensor):
            action = action.cpu()
        # Assuming reward, next_state, done might also be tensors or need conversion
        if isinstance(reward, torch.Tensor):
            reward = reward.cpu()
        if isinstance(next_state, torch.Tensor):
            next_state = next_state.cpu()
        if isinstance(done, torch.Tensor):
            done = done.cpu()
        
        experience = (state, action, reward, next_state, done)
        self.memory.append(experience)
        
    def sample(self):
        """
        Randomly sample a batch of experiences from memory.
        All data is kept on CPU; the learning algorithm is responsible for moving it to the device.
        
        Returns:
            Tuple[torch.Tensor]: A tuple of tensors (states, actions, rewards, next_states, dones).
        """
        experiences = random.sample(self.memory, k=self.batch_size)
        
        # Convert list of tuples to separate tensors for states, actions, etc.
        # Ensure tensors are created on CPU.
        states = torch.stack([e[0] for e in experiences if e is not None]).float()
        actions = torch.stack([e[1] for e in experiences if e is not None]).long()
        rewards = torch.stack([e[2] for e in experiences if e is not None]).float()
        next_states = torch.stack([e[3] for e in experiences if e is not None]).float()
        dones = torch.stack([e[4] for e in experiences if e is not None]).float()
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        """
        Return the current size of internal memory.
        
        Returns:
            int: The number of experiences currently stored in the buffer.
        """
        return len(self.memory)