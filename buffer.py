import numpy as np
import random
import torch
from collections import deque

# Set random seeds for reproducibility
seed = 1
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)

# Define the Replay Buffer
# Define the Replay Buffer
class ReplayBuffer:
    def __init__(self, action_size, buffer_size, batch_size, seed=seed):
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.seed = random.seed(seed)
        
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        # Make sure each item is on the CPU
        state = state.cpu() if state.is_cuda else state
        action = action.cpu() if action.is_cuda else action
        reward = reward.cpu() if reward.is_cuda else reward
        next_state = next_state.cpu() if next_state.is_cuda else next_state
        done = done.cpu() if done.is_cuda else done
        
        self.memory.append((state, action, reward, next_state, done))
        
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)
        
        # Keep the sampled data on CPU
        states = torch.stack([e[0] for e in experiences if e is not None]).float()
        actions = torch.stack([e[1] for e in experiences if e is not None]).long()
        rewards = torch.stack([e[2] for e in experiences if e is not None]).float()
        next_states = torch.stack([e[3] for e in experiences if e is not None]).float()
        dones = torch.stack([e[4] for e in experiences if e is not None]).float()
        
        return (states, actions, rewards, next_states, dones)
    
    def __len__(self):
        return len(self.memory)


"""# Define the Replay Buffer
class ReplayBuffer:
    def __init__(self, action_size, buffer_size, batch_size, seed=seed):
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.seed = random.seed(seed)
        
    def add(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        
    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)
        
        states = torch.from_numpy(
            np.stack([e[0] for e in experiences if e is not None])
        ).float().to(device)
        actions = torch.from_numpy(
            np.vstack([e[1] for e in experiences if e is not None])
        ).long().to(device)
        rewards = torch.from_numpy(
            np.vstack([e[2] for e in experiences if e is not None])
        ).float().to(device)
        next_states = torch.from_numpy(
            np.stack([e[3] for e in experiences if e is not None])
        ).float().to(device)
        dones = torch.from_numpy(
            np.vstack([e[4] for e in experiences if e is not None]).astype(np.uint8)
        ).float().to(device)
        
        return (states, actions, rewards, next_states, dones)
    
    def __len__(self):
        return len(self.memory)"""