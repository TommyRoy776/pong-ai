import gymnasium as gym
import ale_py
from pathlib import Path
import argparse
from stable_baselines3 import DQN
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor


def _str2bool(value: str) -> bool:
    value = value.strip().lower()
    if value in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value!r}")


parser = argparse.ArgumentParser()
parser.add_argument(
    "--gameplay",
    nargs="?",
    const="true",
    default=False,
    type=_str2bool,
    help="Show gameplay with render_mode='human' after training (e.g. --gameplay=true)",
)
args = parser.parse_args()


env = gym.make("ALE/Pong-v5", render_mode="human" if args.gameplay else None)
env = AtariWrapper(env)
env = Monitor(env)  # tracks rewards, episode lengths

model = DQN(
    "CnnPolicy",
    env,
    verbose=1,
    buffer_size=100_000,
    optimize_memory_usage=True,
)

total_timesteps = 100_000

checkpoint_dir = Path("checkpoints")
checkpoint_dir.mkdir(parents=True, exist_ok=True)
checkpoint_callback = CheckpointCallback(
    save_freq=10_000,
    save_path=str(checkpoint_dir),
    name_prefix="pong_dqn",
    save_replay_buffer=False,
    save_vecnormalize=False,
)

model_dir = Path("model")
model_dir.mkdir(parents=True, exist_ok=True)
model_path = model_dir / "pong_model"

interrupted = False

try:
    model.learn(total_timesteps=total_timesteps, callback=checkpoint_callback)
except KeyboardInterrupt:
    interrupted = True
    interrupt_path = model_dir / "pong_model_interrupt"
    model.save(str(interrupt_path))
    print(f"\nKeyboardInterrupt: saved model to {interrupt_path}")
finally:
    model.save(str(model_path))
    print(f"Saved model to {model_path}")

if interrupted:
    env.close()
    raise SystemExit(0)

if args.gameplay:
    obs, _ = env.reset()

    while True:
        action, _ = model.predict(obs)
        obs, reward, terminated, truncated, _ = env.step(action)

        if terminated or truncated:
            obs, _ = env.reset()