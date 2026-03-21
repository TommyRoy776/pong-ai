import ale_py
import gymnasium as gym
import keyboard
import time

env = gym.make("ALE/Pong-v5", render_mode="human")
obs, _ = env.reset()

while True:
    if keyboard.is_pressed('up'):
        action = 2
    elif keyboard.is_pressed('down'):
        action = 3
    else:
        action = 0

    obs, reward, terminated, truncated, _ = env.step(action)

    time.sleep(0.02)  # 👈 add this (about 50 FPS)

    if terminated or truncated:
        obs, _ = env.reset()