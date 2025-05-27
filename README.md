# Deep Q-Learning Agent for CarRacing-v3

This project implements a Deep Q-Network (DQN) agent to learn and play the CarRacing-v3 environment from OpenAI Gymnasium, utilizing PyTorch. The agent learns to drive a car around a randomly generated racetrack by processing visual input.

## Key Features

-   **Modular Structure**: Code is organized into `src/` for core logic, `configs/` for settings, and `output/` for results.
-   **Configurable Parameters**: Most hyperparameters and settings are managed via `configs/settings.toml`, allowing easy experimentation.
-   **Deep Q-Network (DQN)**: Implements a DQN agent with:
    -   Convolutional Neural Network (CNN) to process game frames.
    -   Experience Replay Buffer for stable learning.
    -   Target Network for Q-value estimation stability.
    -   Epsilon-greedy exploration strategy with configurable decay.
-   **Environment Wrappers**:
    -   Frame skipping for efficiency.
    -   Action discretization to simplify the control space.
    -   Observation processing (grayscale, resize, frame stacking).
-   **Training & Evaluation**:
    -   Script for training the agent (`src/train.py`).
    -   Functionality to watch a trained agent play.
    -   Model saving and loading.
-   **Logging**: Structured logging of training progress to console and a log file (`output/training.log`).

## Project Structure

```
.
├── configs/
│   └── settings.toml       # Configuration file for all parameters
├── src/
│   ├── __init__.py
│   ├── agent.py            # DQN Agent implementation
│   ├── buffer.py           # Replay Buffer implementation
│   ├── models/
│   │   ├── __init__.py
│   │   └── q_network.py    # Q-Network model architecture
│   ├── train.py            # Main script for training and evaluation
│   └── wrapper.py          # Gymnasium environment wrappers
├── output/
│   ├── trained_models/     # Saved model checkpoints
│   └── training.log        # Log file for training runs
├── tests/                  # (Placeholder for future unit tests)
├── .gitignore
└── README.md
```

## Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd <repository_name>
    ```

2.  **Create a Python virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies:**
    The project requires Python 3.7+. You'll need PyTorch, Gymnasium, and other libraries.
    ```bash
    pip install torch torchvision torchaudio
    pip install gymnasium[box2d] # box2d is needed for CarRacing
    pip install numpy
    pip install matplotlib
    pip install tqdm
    pip install opencv-python # For image processing if not handled by wrappers
    pip install toml # Required if using Python < 3.11 for .toml config files
    ```
    *(Note: Specific PyTorch installation commands might vary based on your system and CUDA version. Refer to the [official PyTorch website](https://pytorch.org/get-started/locally/))*

## Usage

### 1. Configure Training
Edit `configs/settings.toml` to set your desired parameters for the environment, agent, training process, and paths. Key options include:
-   `environment.*`: Frame shape, skip, stack size.
-   `agent.*`: Gamma, learning rate, epsilon parameters, buffer size, batch size.
-   `training.*`: Number of episodes, seed, model saving frequency, device (auto, cpu, cuda, mps).

### 2. Run Training
Navigate to the `src/` directory and run the training script:
```bash
cd src
python train.py
```
-   Progress will be logged to the console and saved in `output/training.log`.
-   Models will be saved periodically to `output/trained_models/`.
-   To resume training from a saved model, set `load_model_on_startup = true` and `model_load_episode_num` in `configs/settings.toml`.

### 3. Watch a Trained Agent
The `train.py` script can also be used to watch a trained agent if `CONFIG['training']['num_episodes']` is set to 0 or a low number, and `CONFIG['training']['load_model_on_startup']` is true. The agent will play using the loaded model. (Alternatively, you can modify `train.py` to have a dedicated watch mode).
The `watch_agent` function in `train.py` is called periodically during training if enabled by `WHEN2SAVE` (now `model_save_freq_episodes`) > 0.

## Configuration System

The primary configuration file is `configs/settings.toml`. It is organized into sections:

-   `[environment]`: Settings for the CarRacing environment (e.g., frame dimensions, action space).
-   `[agent]`: Hyperparameters for the DQN agent (e.g., learning rate, discount factor, epsilon settings, buffer settings).
-   `[training]`: Parameters for the training loop (e.g., number of episodes, batch size, model save frequency, device selection).
-   `[evaluation]`: Settings for watching/evaluating the agent.

Refer to the comments within `settings.toml` for details on each parameter.

## Testing

(Placeholder)
To run tests in the future:
```bash
# pytest tests/
```

## License

(Placeholder - e.g., MIT License)

## Acknowledgements

(Placeholder - e.g., Based on or inspired by...)

```
