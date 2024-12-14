import argparse

import torch
import gymnasium as gym

from rl_tetris.wrapper.Grouped import GroupedStepWrapper
from rl_tetris.wrapper.Observation import GroupedFeaturesObservation


def get_args():
    parser = argparse.ArgumentParser("""RL-Tetris 게임 환경을 이용한 강화학습""")
    parser.add_argument("--width", type=int, default=10,
                        help="The common width for all images")
    parser.add_argument("--height", type=int, default=20,
                        help="The common height for all images")
    parser.add_argument("--block_size", type=int,
                        default=30, help="Size of a block")
    parser.add_argument("--model_dir", type=str,
                        default=f"models")
    parser.add_argument("--model_name", type=str,
                        default=f"tetris_best_54088")

    args = parser.parse_args()
    return args


def test(opt):
    model_path = f"{opt.model_dir}/{opt.model_name}"

    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    else:
        torch.manual_seed(42)

    model = torch.load(model_path, map_location=torch.device("cpu")).eval()

    env = gym.make("RL-Tetris-v0", render_mode="human")
    env = GroupedStepWrapper(
        env, observation_wrapper=GroupedFeaturesObservation(env))

    obs, info = env.reset()

    done = False
    while not done:
        env.render()

        valid_features = obs["features"][obs["action_mask"] == 1]
        next_features = torch.from_numpy(valid_features).float()

        with torch.no_grad():
            q_values = model(next_features).squeeze(0)

        index = torch.argmax(q_values).item()
        action = info["action_mapping"][index]

        obs, _, done, _, info = env.step(action)


if __name__ == "__main__":
    opt = get_args()
    test(opt)
