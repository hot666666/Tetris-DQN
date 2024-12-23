import gymnasium as gym
import numpy as np

from rl_tetris.envs.tetris import Tetris


class BoardObservation(gym.ObservationWrapper):
    def __init__(self, env: Tetris):
        super().__init__(env)

    def observation(self, observation):
        # TODO: piece가 회전에도 고정 observation에 맞도록, n*n 크기로 바꾸기 -> board에 piece를 합쳐서 반환
        return observation["boards"]


class GroupedFeaturesObservation(gym.ObservationWrapper):
    def __init__(self, env: Tetris):
        super().__init__(env)

    def observation(self, observation):
        # 마스크의 값이 1인 경우 extract_board_features 호출, 그렇지 않으면 0으로 채운 벡터 반환

        boards = observation["boards"]
        mask = observation["action_mask"]

        features = np.array([
            self.extract_board_features(board) if m == 1 else np.zeros(4)
            for board, m in zip(boards, mask)
        ])
        return features

    def extract_board_features(self, board):
        """현재 보드 상태에 대한 특징(지워진 줄, 구멍, 인접열 차이 합, 높이 합)을 반환하는 메서드"""

        lines_cleared, board = self.env.unwrapped.clear_full_rows_(board)
        holes = self.env.unwrapped.get_holes(board)
        bumpiness, height = self.env.unwrapped.get_bumpiness_and_height(board)

        return np.array([lines_cleared, holes, bumpiness, height], dtype=np.float32)
