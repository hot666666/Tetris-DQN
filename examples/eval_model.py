import argparse

import torch
import torch.nn as nn
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
                        default=f"tetris_2481_96439")

    args = parser.parse_args()
    return args


# DNN 모델
class DNN(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=64, output_dim=1):
        """게임 상태에 대한 휴리스틱한 정보(지워진 줄, 구멍, 인접열 차이 합, 높이 합)를 입력으로 받아서, q-value를 출력하는 DNN 모델"""
        super(DNN, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

        self.model.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.model(x)


def test(opt):
    model_path = f"{opt.model_dir}/{opt.model_name}"

    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    else:
        torch.manual_seed(42)

    # 모델 불러오기
    model = DNN()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    env = gym.make("RL-Tetris-v0", render_mode="animate")
    env = GroupedStepWrapper(
        env, observation_wrapper=GroupedFeaturesObservation(env))

    obs, info = env.reset()

    done = False
    while not done:
        if env.render_mode != "animate":
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
