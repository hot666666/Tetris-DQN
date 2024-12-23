import time
import gymnasium as gym
import rl_tetris.envs


env = gym.make("RL-Tetris-v0", render_mode="human")
obs, info = env.reset()

done = False
while True:
    env.render()

    action = env.action_space.sample()

    obs, _, done, _, info = env.step(action)

    time.sleep(1)

    if done:
        env.render()
        time.sleep(3)
        break
