import gymnasium as gym
import torch

from rl_tetris.envs.tetris import Tetris


class FeaturesObservation(gym.ObservationWrapper):
    def __init__(self, env: Tetris):
        super().__init__(env)

    def observation(self, observation):
        return {k: self.extract_board_features(observation[k]) for k in observation}

    def extract_board_features(self, board):
        """현재 보드 상태에 대한 특징(지워진 줄, 구멍, 인접열 차이 합, 높이 합)을 반환하는 메서드"""

        lines_cleared, board = self.env.unwrapped.clear_full_rows(board)
        holes = self.env.unwrapped.get_holes(board)
        bumpiness, height = self.env.unwrapped.get_bumpiness_and_height(board)

        return torch.FloatTensor([lines_cleared, holes, bumpiness, height])
