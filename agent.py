import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from buffer import ReplayBuffer
from model import QNetwork
from torch.mps import is_available

# Set random seeds for reproducibility
seed = 1
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)

# Check if MPS is available and set the device accordingly
device = torch.device("mps" if torch.has_mps else "cpu")

# Define the Agent using Double DQN
class Agent:
    def __init__(self, img_size, action_size, gamma,learning_rate = 0.001, seed=seed):
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.gamma = gamma
        
        # Q-Network
        self.qnetwork_local = QNetwork(img_size, action_size).to(device)
        self.qnetwork_target = QNetwork(img_size, action_size).to(device)

        # Initialize target model weights with local model parameters
        self.qnetwork_target.load_state_dict(self.qnetwork_local.state_dict())

        # Optimizer and Replay Memory
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=learning_rate)
        self.memory = ReplayBuffer(action_size, buffer_size=int(3 * 1e4), batch_size=256)
        self.t_step = 0
        self.loss_fn = nn.MSELoss()

        # Precompute arange array for indexing in learn method
        self.arange = torch.arange(self.memory.batch_size).to(device)
    
    def step(self, state, action, reward, next_state, done):
        # Convert state, action, reward, next_state, done to appropriate device
        state = torch.tensor(state, dtype=torch.float32).to(device)
        action = torch.tensor(action).to(device)
        reward = torch.tensor(reward).to(device)
        next_state = torch.tensor(next_state, dtype=torch.float32).to(device)
        done = torch.tensor(done, dtype=torch.float32).to(device)

        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        self.t_step += 1

        if len(self.memory) > self.memory.batch_size and self.t_step % 4 == 0:
            experiences = self.memory.sample()
            self.learn(experiences)
                
    def act(self, state, eps=0.0):
        """Returns actions for given state as per current policy."""
        random_n = random.random()
        if random_n > eps:
            # Convert state to tensor and move to device
            state = torch.from_numpy(state).float().unsqueeze(0).to(device)
            with torch.no_grad():
                action_values = self.qnetwork_local(state)
            action = torch.argmax(action_values, dim=1).item()
        else:
            action = random.randint(0, self.action_size - 1)
        return action
    
    def learn(self, experiences):
        states, actions, rewards, next_states, dones = experiences

        # Move experiences to device
        states = states.to(device)
        actions = actions.to(device)
        rewards = rewards.to(device)
        next_states = next_states.to(device)
        dones = dones.to(device)

        # ------------------- Compute Double DQN targets ------------------- #
        # Get Q values from local model and select actions
        action_values = self.qnetwork_local(states)
        Q_expected = action_values[self.arange, actions]

        # Get Q values from target model using indices from local model
        with torch.no_grad():
            next_actions_values = self.qnetwork_target(next_states)
            #print('Next action values: ', next_actions_values)
            Q_targets_next, _ = torch.max(next_actions_values, dim=1)
            #print('Q_target next: ', Q_targets_next)
            #Q_targets_next = self.qnetwork_target(next_states)[self.arange, next_action_indices]
            Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))
            #print('Rewards: ', rewards)
            #print('Q_targets: ', Q_targets)
    
        # Compute loss
        loss = self.loss_fn(Q_expected, Q_targets)
        #print('Loss: ', loss)
        #print('')

        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # ------------------- Update target network ------------------- #
        if self.t_step % 10000 == 0:
            self.update_target()
    
    def update_target(self):
        """Update the target network with the local network weights"""
        self.qnetwork_target.load_state_dict(self.qnetwork_local.state_dict())

# Define the Agent using Double DQN
class Agent2:
    def __init__(self, action_size, gamma, seed=seed):
        self.action_size = action_size
        self.seed = random.seed(seed)
        
        # Q-Network
        self.qnetwork_local = QNetwork(action_size).to(device)
        self.qnetwork_target = QNetwork(action_size).to(device)

        # Initialise local model weights with xaiver uniform
        self.qnetwork_target.load_state_dict(self.qnetwork_local.state_dict())

        # Initialize target model parameters with local model parameters
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=0.001)
        self.memory = ReplayBuffer(action_size, buffer_size=int(1e6), batch_size=32)
        self.t_step = 0
        self.loss_fn = nn.SmoothL1Loss() 
    
    def step(self, state, action, reward, next_state, done):
        
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        self.t_step += 1

        if len(self.memory) > self.memory.batch_size and self.t_step % 4 == 0:
            experiences = self.memory.sample()
            self.learn(experiences, gamma=0.9)

                
    def act(self, state, eps=0.0):
        random_n = random.random()        
        
        # Epsilon-greedy action selection
        if random_n > eps:
            state = torch.from_numpy(state).float().unsqueeze(0)
            action_values = self.qnetwork_local(state)
            action = torch.argmax(action_values, axis=1).item()
            return action
        else:
            action = random.randint(0, self.action_size - 1)
            return action
    
    def learn(self, experiences, gamma):
        states, actions, rewards, next_states, dones = experiences

        # ------------------- Compute Double DQN targets ------------------- #
        # Get indices of best actions from local model
        action_values = self.qnetwork_local(states)
        Q_expected = action_values.gather(1, actions)

        # Get Q values from target model using indices from local model
        with torch.no_grad():
            next_actions_values = self.qnetwork_local(next_states)
            next_action_indices = torch.argmax(next_actions_values, dim=1)
            target_action_values = self.qnetwork_target(next_states)
            Q_targets_next = target_action_values.gather(1, next_action_indices.unsqueeze(1))
            Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Compute loss
        loss = self.loss_fn(Q_expected, Q_targets)

        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # ------------------- Update target network ------------------- #
        if self.t_step % 5000 == 0:
            self.update_target()
    
    def update_target(self):
        local_model = self.qnetwork_local
        target_model = self.qnetwork_target
        target_model.load_state_dict(local_model.state_dict())


# Define the Agent using Double DQN
"""class Agent:
    def __init__(self, action_size, gamma, seed=seed):
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.gamma = gamma
        
        # Q-Network
        self.qnetwork_local = QNetwork(action_size).to(device)
        self.qnetwork_target = QNetwork(action_size).to(device)

        # Initialise local model weights with xaiver uniform
        self.qnetwork_target.load_state_dict(self.qnetwork_local.state_dict())

        # Initialize target model parameters with local model parameters
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=0.001)
        self.memory = ReplayBuffer(action_size, buffer_size=int(1e6), batch_size=3)
        self.t_step = 0
        self.loss_fn = nn.MSELoss() 
    
    def step(self, state, action, reward, next_state, done):
        
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        self.t_step += 1

        if len(self.memory) > self.memory.batch_size and self.t_step % 2 == 0:
            experiences = self.memory.sample()
            self.learn(experiences)
                
    def act(self, state, eps=0.0):
        random_n = random.random()        
        
        # Epsilon-greedy action selection
        if random_n > eps:
            state = torch.from_numpy(state).float().unsqueeze(0)
            action_values = self.qnetwork_local(state)
            action = torch.argmax(action_values, axis=1).item()
            return action
        else:
            action = random.randint(0, self.action_size - 1)
            return action
    
    def learn(self, experiences):
        states, actions, rewards, next_states, dones = experiences

        # ------------------- Compute Double DQN targets ------------------- #
        # Get indices of best actions from local model
        action_values = self.qnetwork_local(states)
        Q_expected = action_values.gather(1, actions)

        # Get Q values from target model using indices from local model
        with torch.no_grad():
            next_actions_values = self.qnetwork_target(next_states)
            next_action_indices = torch.argmax(next_actions_values, dim=1)
            Q_targets = next_actions_values.gather(1, next_action_indices.unsqueeze(1))
            Q_targets = rewards + (self.gamma * Q_targets * (1 - dones))

        # Compute loss
        loss = self.loss_fn(Q_expected, Q_targets)

        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # ------------------- Update target network ------------------- #
        if self.t_step % 5000 == 0:
            self.update_target()
    
    def update_target(self):
        local_model = self.qnetwork_local
        target_model = self.qnetwork_target
        target_model.load_state_dict(local_model.state_dict())"""