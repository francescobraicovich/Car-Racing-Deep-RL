# Car Racing with Deep Reinforcement Learning

This project implements a Deep Q-Learning (DQN) agent to play the CarRacing-v3 environment from OpenAI Gymnasium.

## Overview

The agent learns to drive a car around a randomly generated racetrack, trying to stay on the track while maintaining speed. The environment provides raw pixel data as input, which is processed and used by a deep neural network to determine optimal actions.

## Key Features

- Frame stacking (4 frames) for temporal information
- Frame skipping to reduce computation and improve learning
- Grayscale conversion and image resizing for efficient processing
- Discrete action space wrapper for simplified control
- Experience replay buffer for stable learning
- Epsilon-greedy exploration strategy

## Project Structure

- `train.py`: Main training loop and environment setup
- `agent.py`: DQN agent implementation 
- `buffer.py`: Experience replay buffer implementation
- `wrapper.py`: Custom environment wrappers for preprocessing
