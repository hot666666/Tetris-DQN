import time
import gymnasium as gym
from gymnasium.envs.registration import register

# from rl_tetris.envs.tetris import Tetris
# 로컬 환경 등록
register(
    id="RL-Tetris-v0",  # 환경의 고유 ID
    entry_point="rl_tetris.envs.tetris:Tetris",  # Tetris의 경로
)

env = gym.make("RL-Tetris-v0", render_mode="human")
env.reset()

done = False
while not done:
    env.render()
    action = env.action_space.sample()
    board_ob, _, done, _, info = env.step(action)

    for b in board_ob["board"]:
        print(b)
    print()
    print()

    time.sleep(1)
