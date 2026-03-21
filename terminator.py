import gymnasium as gym
import ale_py
import argparse
from pathlib import Path
import os
import time
import ctypes
from stable_baselines3 import DQN
from stable_baselines3.common.atari_wrappers import AtariWrapper

class InputState:
    def __init__(self) -> None:
        self.up = False
        self.down = False
        self.fire = False


VK_UP = 0x26
VK_DOWN = 0x28
VK_SPACE = 0x20
VK_ESCAPE = 0x1B


def _is_pressed(vk_code: int) -> bool:
    if os.name != "nt":
        return False
    return (ctypes.windll.user32.GetAsyncKeyState(vk_code) & 0x8000) != 0


def _build_action_map(env):
    meanings = env.unwrapped.get_action_meanings()
    idx = {m: i for i, m in enumerate(meanings)}

    noop = idx.get("NOOP", 0)
    fire = idx.get("FIRE", noop)
    up_fire = idx.get("RIGHTFIRE")
    down_fire = idx.get("LEFTFIRE")

    # Pong often exposes movement as LEFT/RIGHT (even though the paddle moves vertically).
    up = idx.get("UP")
    down = idx.get("DOWN")
    if up is None and "RIGHT" in idx:
        up = idx["RIGHT"]
    if down is None and "LEFT" in idx:
        down = idx["LEFT"]

    # Fallbacks if UP/DOWN/LEFT/RIGHT aren't present.
    if up is None:
        up = noop
    if down is None:
        down = noop

    return {
        "meanings": meanings,
        "noop": noop,
        "fire": fire,
        "up": up,
        "down": down,
        "up_fire": up_fire,
        "down_fire": down_fire,
    }


def get_human_action(state: InputState, action_map):
    """Captures keyboard input.

    On Windows, uses global key polling so the Pong window can stay focused.
    """
    if os.name == "nt":
        if _is_pressed(VK_ESCAPE):
            return None
        state.up = _is_pressed(VK_UP)
        state.down = _is_pressed(VK_DOWN)
        state.fire = _is_pressed(VK_SPACE)

        if state.fire and state.up and not state.down and action_map.get("up_fire") is not None:
            return action_map["up_fire"]
        if state.fire and state.down and not state.up and action_map.get("down_fire") is not None:
            return action_map["down_fire"]
        if state.fire:
            return action_map["fire"]
        if state.up and not state.down:
            return action_map["up"]
        if state.down and not state.up:
            return action_map["down"]
        return action_map["noop"]

    # Non-Windows fallback: no global key polling implemented.
    return action_map["noop"]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="model/pong_model",
        help="Path prefix to a saved Stable-Baselines3 model ('.zip' is optional). Default: model/pong_model",
    )
    args = parser.parse_args()

    # 1. Setup Environment
    # Make controls responsive: avoid frame-skip and sticky actions.
    env = gym.make(
        "ALE/Pong-v5",
        render_mode="human",
        frameskip=1,
        repeat_action_probability=0.0,
    )
    env = AtariWrapper(
        env,
        noop_max=1,
        frame_skip=1,
        terminal_on_life_loss=False,
        clip_reward=False,
        action_repeat_probability=0.0,
    )

    action_map = _build_action_map(env)
    
    # 2. Load your AI
    model_path = Path(args.model)
    candidate_paths = [model_path]
    if model_path.suffix.lower() != ".zip":
        candidate_paths.insert(0, model_path.with_suffix(".zip"))

    if not any(p.exists() for p in candidate_paths):
        model_dir = Path("model")
        available = []
        if model_dir.exists():
            available = sorted(p.name for p in model_dir.glob("*.zip"))
        available_msg = "\n".join(f"- {name}" for name in available) if available else "(none)"
        raise FileNotFoundError(
            f"Could not find model at: {candidate_paths[0]}\n"
            f"Try: python terminator.py --model model/pong_model\n"
            f"Available models in 'model/':\n{available_msg}"
        )

    load_path = candidate_paths[0] if candidate_paths[0].exists() else candidate_paths[1]
    # For playing (inference), we don't need a large replay buffer.
    # Override it to avoid allocating many GB of RAM on load.
    model = DQN.load(
        str(load_path),
        env=env,
        custom_objects={"buffer_size": 1},
    )
    
    obs, info = env.reset()
    print("GAME START! Focus the Pong game window.")
    print("Controls: UP/DOWN arrows to move, SPACE to serve (FIRE), ESC to quit.")
    print(f"Detected action meanings: {action_map['meanings']}")
    print(f"Key mapping -> up:{action_map['up']} down:{action_map['down']} fire:{action_map['fire']} noop:{action_map['noop']}")

    input_state = InputState()
    
    try:
        while True:
            # AI decides its move
            ai_action, _ = model.predict(obs, deterministic=True)
            
            # You decide your move
            human_action = get_human_action(input_state, action_map)
            if human_action is None: break
            
            # NOTE: In standard Atari Pong, the 'env.step' only takes ONE action.
            # To truly play "Against" it, we usually let the AI control 
            # the right paddle and you control the left. 
            # However, the standard Pong-v5 is single-agent. 
            # So, we will 'intercept' the AI's move and use yours instead 
            # to see if you can beat the game's built-in opponent BETTER than the AI can!
            
            obs, reward, terminated, truncated, info = env.step(human_action)
            
            if terminated or truncated:
                obs, info = env.reset()

            # Slow it down to human speed (~60 FPS)
            time.sleep(1 / 60)
            
    except KeyboardInterrupt:
        print("Exiting...")
    finally:
        env.close()

if __name__ == "__main__":
    main()