# Pong AI

A project for training a reinforcement learning agent to play Atari Pong using Deep Q-Network (DQN) via Stable-Baselines3.

## Project Overview

- **Algorithm**: DQN with CNN policy (`CnnPolicy`) from Stable-Baselines3
- **Environment**: `ALE/Pong-v5` via Gymnasium + ALE-py
- **Platform**: Windows (key input uses Win32 API `GetAsyncKeyState`)

## Scripts

| File | Purpose |
|------|---------|
| `train_pong.py` | Train the DQN agent; saves checkpoints every 10k steps and final model to `model/pong_model` |
| `play_manual.py` | Play Pong manually with keyboard controls (no AI) |
| `terminator.py` | Play against the AI — you control the left paddle while the loaded model provides inference |

## Common Commands

```bash
# Install dependencies (uses uv)
uv sync

# Train the AI (headless)
python train_pong.py

# Train with live gameplay window
python train_pong.py --gameplay

# Play manually
python play_manual.py

# Play vs the trained AI
python terminator.py

# Play vs a specific model checkpoint
python terminator.py --model checkpoints/pong_dqn_10000_steps
```

## Controls (Manual / vs AI)

| Key | Action |
|-----|--------|
| `UP` arrow | Move paddle up |
| `DOWN` arrow | Move paddle down |
| `SPACE` | Serve / Fire |
| `ESC` | Quit |

## Project Structure

```
pong-ai/
├── train_pong.py        # Training script
├── play_manual.py       # Manual play (no AI)
├── terminator.py        # Human vs AI mode
├── model/               # Final saved model (pong_model.zip)
├── checkpoints/         # Intermediate checkpoints saved during training
├── pyproject.toml       # Project metadata and dependencies
└── requirements.txt     # Pip-compatible requirements
```

## Training Details

- **Total timesteps**: 100,000 (default)
- **Replay buffer size**: 100,000
- **Checkpoint frequency**: every 10,000 steps → `checkpoints/pong_dqn_<step>_steps.zip`
- **Final model**: `model/pong_model.zip`
- Ctrl+C during training saves an interrupt checkpoint to `model/pong_model_interrupt.zip`

## Dependencies

Key packages:
- `stable-baselines3` — DQN implementation
- `gymnasium` + `ale-py` — Atari environment
- `torch` — neural network backend
- `pygame` — rendering
- `tensorboard` — training metrics visualization
