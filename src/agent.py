import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim

from buffer import ReplayBuffer
from models.q_network import QNetwork

# Set random seeds for reproducibility
SEED = 1 # PEP8: constants should be uppercase
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)

# Global SEED is set for reproducibility, but Agent might receive its own seed or device config.

class Agent:
    """Interacts with and learns from the environment using Double DQN."""

    def __init__(self, img_size, action_size, stack_size, gamma, learning_rate=0.001,
                 buffer_size=int(3e4), batch_size=256, target_update_freq=10000,
                 device='cpu', seed=None):
        """
        Initializes an Agent object.

        Args:
            img_size (tuple): Dimensions of each state (height, width).
            action_size (int): Number of possible actions.
            stack_size (int): Number of frames stacked to form a state for QNetwork.
            gamma (float): Discount factor for future rewards.
            learning_rate (float): Learning rate for the optimizer.
            buffer_size (int): Maximum size of the replay buffer.
            batch_size (int): Minibatch size for learning.
            target_update_freq (int): How often to update the target network (in agent steps).
            device (str or torch.device): Device to run the networks on ('cpu', 'cuda', 'mps').
            seed (int, optional): Random seed for QNetwork initialization. If None, global SEED is used.
        """
        self.action_size = action_size
        self.stack_size = stack_size # Store stack_size
        self.gamma = gamma
        self.batch_size = batch_size # Store batch_size for arange
        self.target_update_freq = target_update_freq
        self.device = torch.device(device) # Use the passed device string/object

        q_network_seed = seed if seed is not None else SEED # Use specific seed or global
        
        # Q-Networks: local for action selection, target for value estimation
        self.qnetwork_local = QNetwork(img_size, action_size, self.stack_size, seed=q_network_seed).to(self.device)
        self.qnetwork_target = QNetwork(img_size, action_size, self.stack_size, seed=q_network_seed).to(self.device) 
        self.qnetwork_target.load_state_dict(self.qnetwork_local.state_dict()) # Initialize target with local weights

        # Optimizer
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=learning_rate)
        
        # Replay memory
        # ReplayBuffer doesn't take a seed anymore in its constructor as per previous refactoring
        self.memory = ReplayBuffer(action_size, buffer_size=buffer_size, batch_size=batch_size)
        
        self.t_step = 0  # Counter for update steps
        self.loss_fn = nn.MSELoss()

        # Precompute arange array for efficient indexing in the learn method
        self.arange = torch.arange(self.batch_size, device=self.device)
    
    def step(self, state, action, reward, next_state, done):
        """
        Saves an experience in the replay memory and triggers learning if conditions are met.

        Args:
            state (array_like): The current state.
            action (int): The action taken.
            reward (float): The reward received.
            next_state (array_like): The next state.
            done (bool): Whether the episode has finished.
        """
        # Convert inputs to tensors and move to the designated device
        # Note: state normalization (e.g. /255.0) is expected to be done by the caller (e.g. in train.py)
        # Tensors are created on self.device directly.
        state_t = torch.tensor(state, dtype=torch.float32, device=self.device)
        action_t = torch.tensor(action, dtype=torch.long, device=self.device) # Action is index
        reward_t = torch.tensor(reward, dtype=torch.float32, device=self.device)
        next_state_t = torch.tensor(next_state, dtype=torch.float32, device=self.device)
        done_t = torch.tensor(done, dtype=torch.float32, device=self.device) # Boolean to float for multiplication

        # Save experience in replay memory
        # ReplayBuffer expects CPU tensors, agent.step inputs are numpy arrays or tensors.
        # If they are already tensors on self.device, they need to be moved to CPU for buffer.
        # The current ReplayBuffer.add() handles .cpu() if it's a tensor.
        # If state, action, etc., are numpy arrays, ReplayBuffer will handle them correctly.
        # If the inputs to this step() method are already tensors on `self.device`,
        # ReplayBuffer's .add() will move them to CPU.
        self.memory.add(state_t, action_t, reward_t, next_state_t, done_t)
        self.t_step += 1

        # Learn every 4 agent steps if enough samples are available in memory
        # This is a common hyperparameter, could also be configurable.
        learn_every_n_steps = 4 
        if len(self.memory) > self.batch_size and self.t_step % learn_every_n_steps == 0:
            experiences = self.memory.sample() # Samples are on CPU
            self.learn(experiences)
                
    def act(self, state, eps=0.0):
        """
        Returns actions for a given state as per the current policy (epsilon-greedy).
        Assumes `state` is a preprocessed (normalized) numpy array.
        Args:
            state (array_like): Current state.
            eps (float): Epsilon, for epsilon-greedy action selection.
        
        Returns:
            int: The action selected by the agent.
        """
        if random.random() > eps:
            # Exploitation: select the action with the highest Q-value
            state_t = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
            self.qnetwork_local.eval()  # Set network to evaluation mode
            with torch.no_grad():
                action_values = self.qnetwork_local(state_t)
            self.qnetwork_local.train()  # Set network back to training mode
            action = torch.argmax(action_values, dim=1).item()
        else:
            # Exploration: select a random action
            action = random.randint(0, self.action_size - 1)
        return action
    
    def learn(self, experiences):
        """
        Updates Q-network parameters using a batch of experiences.

        Args:
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples, on CPU.
        """
        states, actions, rewards, next_states, dones = experiences

        # Move experiences to the agent's designated device
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        # --- Compute Double DQN targets ---
        # 1. Get Q values from the local model for the current states and actions
        #    For chosen actions `actions`, get their Q values.
        q_expected = self.qnetwork_local(states).gather(1, actions.unsqueeze(1)).squeeze(1)


        # 2. Get Q values for the next states from the target model
        #    Select the best actions for next states using the local model (Double DQN)
        with torch.no_grad():
            # Use local network to choose the best action for the next state
            next_action_values_local = self.qnetwork_local(next_states)
            best_next_actions = torch.argmax(next_action_values_local, dim=1)
            
            # Get Q values for these best_next_actions from the target network
            q_targets_next = self.qnetwork_target(next_states).gather(1, best_next_actions.unsqueeze(1)).squeeze(1)
            
            # Compute Q targets for current states: R + gamma * Q_target_next * (1 - done)
            q_targets = rewards + (self.gamma * q_targets_next * (1 - dones))
    
        # Compute loss: MSE between Q_expected and Q_targets
        loss = self.loss_fn(q_expected, q_targets)

        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # --- Update target network ---
        if self.t_step % self.target_update_freq == 0:
            self.update_target()
    
    def update_target(self):
        """
        Updates the target network by copying the weights from the local network.
        This is a "hard" update. A "soft" update would be:
        τ * θ_local + (1 - τ) * θ_target, where τ is a small constant.
        """
        self.qnetwork_target.load_state_dict(self.qnetwork_local.state_dict())