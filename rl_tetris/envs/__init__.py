from gymnasium.envs.registration import register

register(
    id="RL-Tetris-v0",
    entry_point="rl_tetris.envs.tetris:Tetris",
)
