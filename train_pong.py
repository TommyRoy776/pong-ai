import gymnasium as gym
import ale_py
from pathlib import Path
import torch
from stable_baselines3 import DQN
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.monitor import Monitor


env = gym.make("ALE/Pong-v5", render_mode="human")
env = AtariWrapper(env)
env = Monitor(env)  # tracks rewards, episode lengths

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"PyTorch {torch.__version__} | cuda available={torch.cuda.is_available()} | using device={device}")
if device != "cuda":
    raise RuntimeError(
        "GPU training requested but CUDA is not available. "
        "Install a CUDA-enabled PyTorch build (see https://pytorch.org/get-started/locally/) "
        "and ensure your NVIDIA driver/CUDA runtime is installed."
    )

model = DQN("CnnPolicy", env, verbose=1, device=device)

model.learn(total_timesteps=100_000)
model_dir = Path("model")
model_dir.mkdir(parents=True, exist_ok=True)
model_path = model_dir / "pong_model"
model.save(str(model_path))

obs, _ = env.reset()

while True:
    action, _ = model.predict(obs)
    obs, reward, terminated, truncated, _ = env.step(action)
    
    if terminated or truncated:
        obs, _ = env.reset()