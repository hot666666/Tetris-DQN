import time
import gymnasium as gym

from rl_tetris.wrapper.Grouped import GroupedStepWrapper
from rl_tetris.wrapper.Observation import GroupedFeaturesObservation
# from rl_tetris.envs.tetris import Tetris

# env = gym.make("RL-Tetris-v0", render_mode="human")

env = gym.make("RL-Tetris-v0", render_mode="human")

env = GroupedStepWrapper(
    env, observation_wrapper=GroupedFeaturesObservation(env))
obs, info = env.reset()


done = False
while True:
    env.render()

    action = env.action_space.sample(obs["action_mask"])
    print(obs["action_mask"][-16:])

    obs, _, done, _, info = env.step(action)

    for b in info["board"]:
        print(b)
    print()
    print()

    time.sleep(1)

    if done:
        env.render()
        time.sleep(3)
        break
