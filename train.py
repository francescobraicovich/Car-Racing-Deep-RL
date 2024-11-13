import gymnasium as gym
import numpy as np
import random
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2
import time
import pygame
from agent import Agent
from wrapper import DiscretizeActionWrapper
from wrapper import SkipFrame
import os
from torch.mps import is_available
from concurrent.futures import ThreadPoolExecutor, as_completed
import gymnasium.wrappers as gym_wrap

# Set random seeds for reproducibility
seed = 1
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)

# Check if CUDA is available
device = "mps"

print(f"Using {device} device")

SHAPE = (84, 84) # Shape of the preprocessed frame
GAMMA = 0.9 # Discount factor
STACK_SIZE = 4 # Number of frames to stack
SKIP = 2 # Number of frames to skip
LEARNING_RATE = 0.0001 # Learning rate
EPS_START = 0.9 # Starting epsilon value
EPS_DECAY = 0.999 # Epsilon decay rate
MODEL_SAVED = False # Load the saved model
WHEN2SAVE = 250 # Save the model every n episodes
NUM2LOAD = 0 # Load the model from the nth episode
NUM_EPISODES = 5000 # Number of episodes to train
LENGTH_EPISODES = 350 # Maximum length of an episode

# Create the training environment
training_env = gym.make('CarRacing-v3', render_mode='rgb_array')
training_env = DiscretizeActionWrapper(training_env)
training_env = SkipFrame(training_env, skip=SKIP)
training_env = gym_wrap.GrayscaleObservation(training_env)
training_env = gym_wrap.ResizeObservation(training_env, shape=SHAPE)
training_env = gym_wrap.FrameStackObservation(training_env, stack_size=STACK_SIZE)

# Create the evaluation environment
evaluation_env = gym.make('CarRacing-v3', render_mode='human')
evaluation_env = DiscretizeActionWrapper(evaluation_env)
evaluation_env = gym_wrap.FrameStackObservation(evaluation_env, stack_size=STACK_SIZE)

def normalise_state(state):
    state = state / 255.0
    return state

# Preprocess the state (image)
def preprocess_state(state):
    """Normalize and transpose the state for the neural network."""
    new_state = np.zeros((state.shape[0], SHAPE[0], SHAPE[1]))
    for i in range(state.shape[0]):
        img = state[i]
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)  # Convert to grayscale
        img = cv2.resize(img, SHAPE)
        img = normalise_state(img)
        new_state[i] = img
    return new_state


def watch_agent(agent, n_episodes=1, max_t=1000, eps=0.01, time_limit=15, screen_number=0):
    """
    Watch the agent play the game on a specified screen.
    
    Args:
        agent: The trained agent
        n_episodes: Number of episodes to play
        max_t: Maximum number of timesteps per episode
        eps: Epsilon value for action selection
        time_limit: Time limit in seconds
        screen_number: Screen number to display the game (0 is primary display)
    """
    # Get information about connected displays
    pygame.init()
    displays = pygame.display.get_desktop_sizes()
    
    if screen_number >= len(displays):
        raise ValueError(f"Screen {screen_number} not found. Only {len(displays)} screens available.")
    
    # Calculate position for the selected screen
    x_offset = 0
    for i in range(screen_number):
        x_offset += displays[i][0]
    
    # Set the window position for the selected screen
    os.environ['SDL_VIDEO_WINDOW_POS'] = f"{x_offset},{0}"

    evaluation_env = gym.make('CarRacing-v3', render_mode='human')
    evaluation_env = DiscretizeActionWrapper(evaluation_env)
    evaluation_env = gym_wrap.FrameStackObservation(evaluation_env, stack_size=STACK_SIZE)
    
    # Get the Pygame window after it's created
    pygame.display.init()
    
    start_time = time.time()
    
    for i_episode in range(1, n_episodes + 1):
        state, _ = evaluation_env.reset()
        state = preprocess_state(state)
        score = 0
        
        for t in range(max_t):
            action = agent.act(state, eps)
            next_state, reward, terminated, truncated, _ = evaluation_env.step(action)
            next_state = preprocess_state(next_state)
            state = next_state
            score += reward
            
            if terminated or truncated or (time.time() - start_time) > time_limit:
                break
        
        print(f"Episode {i_episode} Score: {score}")
    
    evaluation_env.close()
    pygame.display.quit()
    pygame.quit()

# Training loop
def train_dqn(agent = None, env = training_env, n_episodes=1000, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=None):
    action_size = env.action_space.n
    if agent is None:
        agent = Agent(img_size=SHAPE, action_size=action_size, gamma=GAMMA, learning_rate=LEARNING_RATE)
    # ensure the learning rate is set
    agent.qnetwork_local.optimizer = torch.optim.Adam(agent.qnetwork_local.parameters(), lr=LEARNING_RATE)
    scores = []
    if eps_decay == None:
        eps_array = np.linspace(eps_start, eps_end, n_episodes)
        eps_array[np.arange(n_episodes) % 10 == 0] = 0.2
        eps_array[0] = eps_start
    else:
        eps = eps_start
    for i_episode in range(1, n_episodes+1):
        training_env = gym.make('CarRacing-v3', render_mode='rgb_array')
        training_env = DiscretizeActionWrapper(training_env)
        training_env = SkipFrame(training_env, skip=SKIP)
        training_env = gym_wrap.GrayscaleObservation(training_env)
        training_env = gym_wrap.ResizeObservation(training_env, shape=SHAPE)
        training_env = gym_wrap.FrameStackObservation(training_env, stack_size=STACK_SIZE)
        if eps_decay == None:
            eps = eps_array[i_episode-1]
        state, _ = training_env.reset()
        state = normalise_state(state)
        score = 0
        for t in tqdm(range(max_t), desc=f'Episode {i_episode + NUM2LOAD}'):
            action = agent.act(state, eps)
            next_state, reward, terminated, truncated, _ = training_env.step(action)
            next_state = normalise_state(next_state)
            done = terminated or truncated
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break
        scores.append(score)
        print(f'Episode {i_episode + NUM2LOAD}, Score: {score:.2f}, Eps: {eps:.2f}')
        
        if eps_decay != None:
            eps = eps * eps_decay
        
        # Every 2 episodes, watch the agent play for 15 seconds
        if i_episode % WHEN2SAVE == 0:
            print(f'Average score in the last {WHEN2SAVE} episodes: {np.mean(scores[-WHEN2SAVE:]):.2f}')
            torch.save(agent.qnetwork_local.state_dict(), f'dqn/dqn_carracing_{i_episode + NUM2LOAD}.pth')
            watch_agent(agent, n_episodes=1, max_t=1000, eps=0, time_limit=15)
        training_env.close()

    # Plot the scores
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.show()
    return agent

# Modified training loop with threading
def train_dqn_multithreaded(n_episodes=1000, max_t=1000, eps_start=1, eps_end=0.01, eps_decay=None, n_threads=16):
    scores = []
    eps_array = np.linspace(eps_start, eps_end, n_episodes)
    print('Eps: array:', eps_array)
    
    # Function to train a single episode
    def train_single_episode(i_episode):
        env = gym.make('CarRacing-v3', render_mode='rgb_array')
        env = DiscretizeActionWrapper(env)
        action_size = env.action_space.n
        agent = Agent(action_size=action_size, gamma=GAMMA)
        eps = eps_array[i_episode-1]
        state, _ = env.reset()
        state = preprocess_state(state)
        score = 0
        for t in range(max_t):
            action = agent.act(state, eps)
            next_state, reward, terminated, truncated, _ = env.step(action)
            next_state = preprocess_state(next_state)
            done = terminated or truncated
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break
        env.close()
        return score, agent
    
    # Run episodes in parallel
    with ThreadPoolExecutor(max_workers=n_threads) as executor:
        futures = {executor.submit(train_single_episode, i): i for i in range(1, n_episodes+1)}
        
        for future in tqdm(as_completed(futures), total=n_episodes, desc="Training Episodes"):
            score, agent = future.result()
            scores.append(score)
            
            i_episode = futures[future]
            if i_episode % 2 == 0:
                print(f'Episode {i_episode}, Average Score: {np.mean(scores[-1000:]):.2f}')

            if i_episode % 40 == 0:
                torch.save(agent.qnetwork_local.state_dict(), f'trained_models/dqn_carracing_{i_episode}.pth')

    # Plot the scores
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.show()
    
    return agent

# Evaluate the trained agent
def evaluate_agent(agent, env = evaluation_env, n_episodes=5):
    evaluation_env = gym.make('CarRacing-v3', render_mode='human')
    evaluation_env = DiscretizeActionWrapper(evaluation_env)
    evaluation_env = gym_wrap.FrameStackObservation(evaluation_env, stack_size=STACK_SIZE)
    for i_episode in range(1, n_episodes+1):
        state, _ = evaluation_env.reset()
        state = preprocess_state(state)
        score = 0
        while True:
            action = agent.act(state, eps=1)
            next_state, reward, terminated, truncated, _ = evaluation_env.step(action)
            next_state = preprocess_state(next_state)
            state = next_state
            score += reward
            if terminated or truncated:
                break
        print(f'Episode {i_episode}\tScore: {score}')
    evaluation_env.close()

# Main execution
if __name__ == "__main__":
    if MODEL_SAVED: 
        # load the trained model from the pth file
        agent = Agent(img_size=SHAPE, action_size=5, gamma=GAMMA)
        agent.qnetwork_local.load_state_dict(torch.load(f'trained_models/dqn_carracing_{NUM2LOAD}.pth'))
        print('Model loaded successfully')
        agent.update_target()
    else:
        agent = Agent(img_size=SHAPE, action_size=5, gamma=GAMMA)

    agent = train_dqn(agent=agent, n_episodes=NUM_EPISODES, max_t=LENGTH_EPISODES, eps_start=EPS_START, eps_decay=EPS_DECAY)
    
    # Save the trained model
    torch.save(agent.qnetwork_local.state_dict(), 'trained_models/dqn_carracing.pth')
    print('Model saved successfully')
    
    evaluate_agent(agent, 2)