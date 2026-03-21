import ale_py
import gymnasium as gym
import time
import os
import ctypes


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

    # Pong often exposes movement as LEFT/RIGHT (even though the paddle moves vertically).
    up = idx.get("UP")
    down = idx.get("DOWN")
    if up is None and "RIGHT" in idx:
        up = idx["RIGHT"]
    if down is None and "LEFT" in idx:
        down = idx["LEFT"]
    if up is None:
        up = noop
    if down is None:
        down = noop

    up_fire = idx.get("RIGHTFIRE")
    down_fire = idx.get("LEFTFIRE")

    return {
        "meanings": meanings,
        "noop": noop,
        "fire": fire,
        "up": up,
        "down": down,
        "up_fire": up_fire,
        "down_fire": down_fire,
    }

env = gym.make(
    "ALE/Pong-v5",
    render_mode="human",
    frameskip=1,
    repeat_action_probability=0.0,
)
obs, _ = env.reset()

action_map = _build_action_map(env)
print("Manual play: focus the Pong window.")
print("Controls: UP/DOWN arrows to move, SPACE to serve (FIRE), ESC to quit.")
print(f"Detected action meanings: {action_map['meanings']}")

while True:
    if _is_pressed(VK_ESCAPE):
        break

    up = _is_pressed(VK_UP)
    down = _is_pressed(VK_DOWN)
    fire = _is_pressed(VK_SPACE)

    if fire and up and not down and action_map.get("up_fire") is not None:
        action = action_map["up_fire"]
    elif fire and down and not up and action_map.get("down_fire") is not None:
        action = action_map["down_fire"]
    elif fire:
        action = action_map["fire"]
    elif up and not down:
        action = action_map["up"]
    elif down and not up:
        action = action_map["down"]
    else:
        action = action_map["noop"]

    obs, reward, terminated, truncated, _ = env.step(action)

    time.sleep(1 / 60)  # ~60 FPS

    if terminated or truncated:
        obs, _ = env.reset()

env.close()