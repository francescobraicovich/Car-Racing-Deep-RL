"""
Main training script for the DQN agent on the CarRacing-v2 environment.

This script handles:
- Environment setup and preprocessing using custom wrappers.
- Agent initialization.
- Training loop with epsilon-greedy exploration and experience replay.
- Periodic model saving.
- Evaluation of the trained agent.
- Watching the trained agent play.
"""
import gymnasium as gym
import numpy as np
import random
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import logging # Added
try:
    import tomllib # Python 3.11+
except ModuleNotFoundError:
    import toml # Fallback for older Python versions
from pathlib import Path

# Removed: cv2, pygame, torch.mps.is_available, concurrent.futures
# time and collections.deque are imported later as they are used in specific functions

from agent import Agent
from wrapper import DiscretizeActionWrapper, SkipFrame
import gymnasium.wrappers as gym_wrap


# --- Configuration Loading ---
def load_config(config_path_str="../configs/settings.toml"):
    # Construct path relative to this script's directory
    script_dir = Path(__file__).parent
    config_path = (script_dir / config_path_str).resolve()
    # Use print here as logger is not configured yet
    print(f"Attempting to load configuration from: {config_path}")
    try:
        with open(config_path, "rb") as f:
            try:
                return tomllib.load(f) # Python 3.11+
            except NameError: # tomllib does not exist (older Python)
                f.seek(0) # Rewind file for toml.load
                import toml # Ensure toml is imported here if tomllib failed
                return toml.load(f)
    except FileNotFoundError:
        print(f"ERROR: Configuration file not found at {config_path}. Exiting.")
        raise # Re-raise to stop execution if config is critical
    except Exception as e:
        print(f"ERROR: Error loading configuration from {config_path}: {e}. Exiting.")
        raise # Re-raise for other errors

CONFIG = load_config()

# --- Setup Output Directory (before logger to ensure log file path is valid) ---
OUTPUT_DIR = Path(CONFIG['training']['output_dir'])
MODEL_DIR_NAME = CONFIG['training']['model_dir']
MODEL_SAVE_DIR = OUTPUT_DIR / MODEL_DIR_NAME
OUTPUT_DIR.mkdir(parents=True, exist_ok=True) # Ensure base output directory exists for the log file
MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True) # Ensure model save directory exists

# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
    handlers=[
        logging.FileHandler(OUTPUT_DIR / "training.log"), # Log to a file in output_dir
        logging.StreamHandler()  # Log to console
    ]
)
logger = logging.getLogger(__name__)

# --- Global Constants and Configuration (from TOML) ---
logger.info("Configuration loaded successfully.")
SEED = CONFIG['training']['seed']
SHAPE = (CONFIG['environment']['shape_h'], CONFIG['environment']['shape_w'])
# GAMMA is used by Agent, passed during init
# LEARNING_RATE is used by Agent, passed during init
# EPS_START, EPS_END, EPS_DECAY_RATE are used by train_dqn, now from config
# BUFFER_SIZE, BATCH_SIZE, TARGET_UPDATE_FREQ_AGENT_STEPS are used by Agent, passed during init
NUM_TRAIN_EPISODES = CONFIG['training']['num_episodes']
MAX_EPISODE_LENGTH = CONFIG['training']['max_steps_per_episode']
MODEL_SAVE_FREQ_EPISODES = CONFIG['training']['model_save_freq_episodes']
LOAD_MODEL_ON_STARTUP = CONFIG['training']['load_model_on_startup']
MODEL_LOAD_EPISODE_NUM = CONFIG['training']['model_load_episode_num']

OUTPUT_DIR = Path(CONFIG['training']['output_dir'])
MODEL_DIR_NAME = CONFIG['training']['model_dir']
MODEL_SAVE_DIR = OUTPUT_DIR / MODEL_DIR_NAME
# MODEL_SAVE_DIR is created above.

SKIP_FRAMES_ENV = CONFIG['environment']['skip_frames'] 
STACK_SIZE_ENV = CONFIG['environment']['stack_size']   

# --- Seed Initialization ---
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)
logger.info(f"Global random seed set to: {SEED}")

# --- Device Handling (from TOML) ---
DEVICE_CONFIG = CONFIG['training']['device'].lower()
if DEVICE_CONFIG == "auto":
    if torch.backends.mps.is_available() and torch.backends.mps.is_built(): # Check if MPS is available and built
        DEVICE_STR = "mps"
    elif torch.cuda.is_available():
        DEVICE_STR = "cuda"
    else:
        DEVICE_STR = "cpu"
elif DEVICE_CONFIG in ["mps", "cuda", "cpu"]:
    DEVICE_STR = DEVICE_CONFIG
    if DEVICE_STR == "mps" and not (torch.backends.mps.is_available() and torch.backends.mps.is_built()):
        logger.warning("MPS configured but not available/built. Falling back to CPU.")
        DEVICE_STR = "cpu"
    elif DEVICE_STR == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA configured but not available. Falling back to CPU.")
        DEVICE_STR = "cpu"
else:
    logger.warning(f"Invalid device '{DEVICE_CONFIG}' in config. Falling back to CPU.")
    DEVICE_STR = "cpu"
DEVICE = torch.device(DEVICE_STR)
logger.info(f"Using device: {DEVICE}")


def normalize_state(state):
    """
    Normalize pixel values to the range [0, 1].
    Assumes state is a NumPy array (possibly a stack of frames).
    Gym wrappers should handle grayscaling and resizing.
    """
    return np.array(state, dtype=np.float32) / 255.0


import time # For frame_delay
def watch_agent(env, agent, 
                n_episodes=CONFIG['evaluation']['watch_episodes'], 
                max_t=CONFIG['evaluation']['watch_max_steps'], 
                frame_delay=0.03): 
    """
    Watch a trained agent play the game.

    The environment should be pre-configured with the desired render_mode ('human').
    The agent is expected to follow a greedy policy (epsilon=0).

    Args:
        env: The Gymnasium environment instance, already configured for human rendering.
        agent: The trained agent to watch.
        n_episodes (int): Number of episodes to watch.
        max_t (int): Maximum number of timesteps per episode.
        frame_delay (float): Delay between frames in seconds to make viewing easier (could be in config).
    """
    logger.info(f"Watching agent for {n_episodes} episode(s)...")
    # set the environment to 'human' render mode
    if env.render_mode != 'human':
        logger.warning(f"Environment render_mode is '{env.render_mode}', changing to 'human'.")
        env.render_mode = 'human'
    for i_episode in range(1, n_episodes + 1):
        state, _ = env.reset()
        state = normalize_state(state)
        episode_score = 0
        
        for t_step in range(max_t):
            env.render()
            action = agent.act(state, eps=CONFIG['evaluation']['watch_epsilon']) # Use epsilon from config
            next_state, reward, terminated, truncated, _ = env.step(action)
            next_state = normalize_state(next_state)
            state = next_state
            episode_score += reward
            
            if frame_delay > 0:
                time.sleep(frame_delay)

            if terminated or truncated:
                logger.info(f"Watch Episode {i_episode}: Finished after {t_step + 1} timesteps. Score: {episode_score:.2f}")
                break
        else: # Loop completed without break
            logger.info(f"Watch Episode {i_episode}: Max timesteps ({max_t}) reached. Score: {episode_score:.2f}")
    # Note: env.close() is handled by the caller.


from collections import deque # For scores_deque

def train_dqn(env, agent, 
              n_episodes=CONFIG['training']['num_episodes'], 
              max_t=CONFIG['training']['max_steps_per_episode'],
              eps_start=CONFIG['agent']['eps_start'], 
              eps_end=CONFIG['agent']['eps_end'], 
              eps_decay_rate=CONFIG['agent']['eps_decay_rate'],
              model_save_dir=MODEL_SAVE_DIR, # Use the global MODEL_SAVE_DIR
              initial_episode_num=0):
    """
    Train a Deep Q-Network agent.

    The environment `env` is reset at the beginning of each episode, not recreated.
    The agent's learning rate is assumed to be set during its initialization.

    Args:
        env: The Gymnasium environment for training (should not be re-created inside).
        agent: The agent to train.
        n_episodes (int): Number of episodes to train for.
        max_t (int): Maximum number of timesteps per episode.
        eps_start (float): Starting value of epsilon for epsilon-greedy policy.
        eps_end (float): Minimum value of epsilon.
        eps_decay_rate (float): Multiplicative factor (per episode) for decaying epsilon.
        model_save_dir (Path): Directory to save trained models.
        initial_episode_num (int): Starting episode number (e.g., if resuming training).
    """
    logger.info(f"Starting training: {n_episodes} episodes, max_t={max_t} steps/ep.")
    logger.info(f"Epsilon: start={eps_start:.3f}, end={eps_end:.3f}, decay={eps_decay_rate:.4f}")
    logger.info(f"Models will be saved to: {model_save_dir}")
    
    scores_deque = deque(maxlen=100) 
    all_scores = []
    current_eps = eps_start
    total_episodes_to_run = n_episodes + initial_episode_num

    for i_episode in range(1, n_episodes + 1):
        episode_num_overall = i_episode + initial_episode_num
        state, _ = env.reset()
        state = normalize_state(state)
        episode_score = 0
        num_steps_in_ep = 0
        
        desc = f'Ep {episode_num_overall}/{total_episodes_to_run} | Eps: {current_eps:.3f}'
        for t_step in tqdm(range(max_t), desc=desc, leave=False):
            action = agent.act(state, current_eps)
            next_state, reward, terminated, truncated, _ = env.step(action)
            next_state = normalize_state(next_state)
            done = terminated or truncated
            
            agent.step(state, action, reward, next_state, done)
            
            state = next_state
            episode_score += reward
            num_steps_in_ep = t_step + 1
            if done:
                break
        
        scores_deque.append(episode_score)
        all_scores.append(episode_score)
        avg_score_100 = np.mean(scores_deque)
        
        logger.info(
            f"Ep {episode_num_overall}/{total_episodes_to_run} | "
            f"Score: {episode_score:.2f} | Steps: {num_steps_in_ep} | "
            f"Eps: {current_eps:.4f} | Avg Score (100ep): {avg_score_100:.2f}"
        )
        
        current_eps = max(eps_end, eps_decay_rate * current_eps)
        
        if episode_num_overall % CONFIG['training']['model_save_freq_episodes'] == 0:
            logger.info(f"--- Saving model at episode {episode_num_overall} ---")
            logger.info(f"Average score over the last {len(scores_deque)} episodes: {avg_score_100:.2f}")
            save_path = model_save_dir / f'dqn_carracing_ep{episode_num_overall}.pth'
            try:
                torch.save(agent.qnetwork_local.state_dict(), save_path)
                logger.info(f"Model saved to {save_path}")
            except Exception as e:
                logger.error(f"Error saving model to {save_path}: {e}")
            logger.info("--- Model saving complete ---")

            # evaluate the agent
            eval_score = evaluate_agent(
                env=env, 
                agent=agent, 
                n_episodes=CONFIG['evaluation']['watch_episodes'], 
                max_t=CONFIG['evaluation']['watch_max_steps']
            )
            logger.info(f"Evaluation score after saving: {eval_score:.2f}")


    # Plot scores at the end
    plt.figure(figsize=(12, 6))
    plt.plot(np.arange(1, len(all_scores) + 1), all_scores, label='Episode Score')
    moving_avg = [np.mean(all_scores[max(0, i-100):i+1]) for i in range(len(all_scores))]
    plt.plot(np.arange(1, len(all_scores) + 1), moving_avg, label='Moving Average (100 episodes)', color='red', alpha=0.7)
    plt.ylabel('Score')
    plt.xlabel(f'Episode # (Starting from {initial_episode_num + 1})')
    plt.title('Training Scores with Moving Average')
    plt.legend()
    plt.grid(True)
    plot_filename = f'training_scores_ep{initial_episode_num + 1}_to_{episode_num_overall}.png'
    plot_save_path = model_save_dir.parent / plot_filename 
    plt.savefig(plot_save_path)
    logger.info(f"Training scores plot saved to {plot_save_path}")
    plt.show()
    
    return agent


def evaluate_agent(env, agent, 
                   n_episodes=CONFIG['evaluation']['watch_episodes'], 
                   max_t=CONFIG['evaluation']['watch_max_steps'], 
                   frame_delay=0.03): 
    """
    Evaluate a trained agent by running it in the environment.

    The environment should be pre-configured with the desired render_mode.
    The agent is expected to follow a greedy policy (epsilon=0).

    Args:
        env: The Gymnasium environment instance.
        agent: The trained agent to evaluate.
        n_episodes (int): Number of episodes for evaluation.
        max_t (int): Maximum number of timesteps per episode.
        frame_delay (float): Delay between frames in seconds for human viewing if render_mode is 'human'.
    """
    logger.info(f"Evaluating agent for {n_episodes} episode(s)...")
    total_scores = []
    for i_episode in range(1, n_episodes + 1):
        state, _ = env.reset()
        state = normalize_state(state)
        episode_score = 0
        for t_step in range(max_t):
            if env.render_mode == 'human':
                 env.render()
            action = agent.act(state, eps=CONFIG['evaluation']['watch_epsilon']) # Use epsilon from config
            next_state, reward, terminated, truncated, _ = env.step(action)
            next_state = normalize_state(next_state)
            state = next_state
            episode_score += reward
            
            if env.render_mode == 'human' and frame_delay > 0:
                time.sleep(frame_delay)

            if terminated or truncated:
                logger.info(f'Evaluation Episode {i_episode}: Score: {episode_score:.2f} (Timesteps: {t_step + 1})')
                break
        else: # Loop completed without break
            logger.info(f'Evaluation Episode {i_episode}: Score: {episode_score:.2f} (Max timesteps: {max_t} reached)')
        total_scores.append(episode_score)
    
    avg_eval_score = np.mean(total_scores)
    logger.info(f"Average evaluation score over {n_episodes} episodes: {avg_eval_score:.2f}")
    # Note: env.close() is handled by the caller.
    return avg_eval_score


if __name__ == "__main__":
    logger.info("--- Script Execution Started ---")
    
    # --- Helper to create environments using CONFIG ---
    def create_env(render_mode_key='render_mode_train', is_eval=False):
        """Creates and wraps the CarRacing environment using settings from CONFIG."""
        env_config = CONFIG['environment']
        render_mode = env_config[render_mode_key]
        
        logger.info(f"Creating environment: {env_config['env_name']} with render_mode='{render_mode}', is_eval={is_eval}")
        env = gym.make(env_config['env_name'], render_mode=render_mode)
        env = DiscretizeActionWrapper(env)
        
        if not is_eval: 
             env = SkipFrame(env, skip=env_config['skip_frames'])
        
        if env_config['grayscale']:
            env = gym_wrap.GrayscaleObservation(env, keep_dim=False)
            
        env = gym_wrap.ResizeObservation(env, shape=(env_config['shape_h'], env_config['shape_w']))
        env = gym_wrap.FrameStackObservation(env, stack_size=env_config['stack_size'])
        
        logger.info(f"Environment created. Observation space: {env.observation_space.shape}, Action space: {env.action_space.n}")
        return env

    # --- Initialize Agent using CONFIG ---
    logger.info("Initializing agent...")
    temp_train_env = create_env()
    action_size = temp_train_env.action_space.n
    temp_train_env.close() # Close temp env
    
    agent_params = CONFIG['agent']
    env_params = CONFIG['environment']
    agent_instance = Agent(
        img_size=SHAPE, 
        action_size=action_size,
        stack_size=env_params['stack_size'],
        gamma=agent_params['gamma'],
        learning_rate=agent_params['learning_rate'],
        buffer_size=agent_params['buffer_size'],
        batch_size=agent_params['batch_size'],
        target_update_freq=agent_params['target_update_freq_agent_steps'],
        device=DEVICE, 
        seed=SEED      
    )
    logger.info(f"Agent initialized: Device={DEVICE}, Action Size={action_size}, Stack Size={env_params['stack_size']}.")

    # --- Load Model (if specified in CONFIG) ---
    initial_episode_num = 0 # Default to 0 if not loading or if file not found
    if CONFIG['training']['load_model_on_startup']:
        load_ep_num = CONFIG['training']['model_load_episode_num']
        load_path = MODEL_SAVE_DIR / f'dqn_carracing_ep{load_ep_num}.pth'
        logger.info(f"Attempting to load model from: {load_path}")
        if load_path.exists():
            try:
                agent_instance.qnetwork_local.load_state_dict(torch.load(load_path, map_location=DEVICE))
                agent_instance.qnetwork_target.load_state_dict(torch.load(load_path, map_location=DEVICE))
                initial_episode_num = load_ep_num # Resume from this episode number
                logger.info(f"Model loaded successfully from {load_path}. Resuming training from episode {initial_episode_num}.")
            except Exception as e:
                logger.error(f"Error loading model from {load_path}: {e}. Training from scratch.")
                initial_episode_num = 0 # Ensure it's reset
        else:
            logger.warning(f"Model file not found at {load_path}. Training from scratch.")
            initial_episode_num = 0 # Ensure it's reset
    else:
        logger.info("Starting training from scratch (LOAD_MODEL_ON_STARTUP is false).")
    
    # --- Training Phase ---
    logger.info("--- Starting Training Phase ---")
    train_env_instance = create_env(render_mode_key='render_mode_train', is_eval=False)
    trained_agent = train_dqn(
        env=train_env_instance, 
        agent=agent_instance, 
        # Parameters below are now defaults in train_dqn, using values from CONFIG
        # n_episodes=CONFIG['training']['num_episodes'], 
        # max_t=CONFIG['training']['max_steps_per_episode'],
        # eps_start=CONFIG['agent']['eps_start'],
        # eps_end=CONFIG['agent']['eps_end'],
        # eps_decay_rate=CONFIG['agent']['eps_decay_rate'],
        # model_save_dir=MODEL_SAVE_DIR,
        initial_episode_num=initial_episode_num
    )
    train_env_instance.close() 
    logger.info("--- Training Phase Finished ---")
    
    # --- Save Final Model ---
    final_episode_count = initial_episode_num + CONFIG['training']['num_episodes']
    final_model_filename = f'dqn_carracing_final_ep{final_episode_count}.pth'
    final_model_path = MODEL_SAVE_DIR / final_model_filename
    try:
        torch.save(trained_agent.qnetwork_local.state_dict(), final_model_path)
        logger.info(f'Final model saved successfully to {final_model_path}')
    except Exception as e:
        logger.error(f"Error saving final model to {final_model_path}: {e}")
    
    # --- Evaluation Phase ---
    logger.info("--- Starting Evaluation Phase ---")
    eval_env_instance = create_env(render_mode_key='render_mode_watch', is_eval=True)
    # eval_config = CONFIG['evaluation'] # Already used in evaluate_agent defaults
    evaluate_agent(
        env=eval_env_instance, 
        agent=trained_agent
        # n_episodes and max_t use defaults from CONFIG['evaluation']
    )
    eval_env_instance.close()
    logger.info("--- Evaluation Phase Finished ---")
    
    # --- Watch Agent Phase ---
    logger.info("--- Starting Watch Agent Phase ---")
    watch_env_instance = create_env(render_mode_key='render_mode_watch', is_eval=True)
    # watch_config = CONFIG['evaluation'] # Already used in watch_agent defaults
    watch_agent(
        env=watch_env_instance, 
        agent=trained_agent
        # n_episodes and max_t use defaults from CONFIG['evaluation']
    )
    watch_env_instance.close()
    logger.info("--- Watch Agent Phase Finished ---")

    logger.info("--- Script Execution Finished ---")