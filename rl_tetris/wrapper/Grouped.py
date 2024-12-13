import gymnasium as gym
from gymnasium.spaces import Discrete
import numpy as np

from rl_tetris.envs.tetris import Tetris


class GroupedStepWrapper(gym.Wrapper):
    def __init__(self, env: Tetris, observation_wrapper=None):
        super().__init__(env)

        self.action_space = Discrete((env.unwrapped.width) * 4)
        self.valid_actions_mask = np.zeros(self.action_space.n, dtype=np.int8)
        self.observation_wrapper = observation_wrapper

    def encode_action(self, x, r):
        return x*4 + r

    def decode_action(self, action):
        return divmod(action, 4)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        obs = self.observation(obs)
        self.valid_actions_mask = self.get_valid_actions_mask(obs)
        info["action_mask"] = self.valid_actions_mask

        return obs, info

    def step(self, action):
        assert self.valid_actions_mask[action] == 1, f"Invalid action: {action}"

        x, r = self.decode_action(action)

        new_piece = [r[:] for r in self.env.unwrapped.piece]
        for _ in range(r):
            new_piece = self.env.unwrapped.get_rotated_piece(new_piece)

        self.env.unwrapped.x = x
        self.env.unwrapped.piece = new_piece

        obs, reward, done, truncted, info = self.env.step(
            self.env.unwrapped.actions.hard_drop
        )

        grouped_obs = self.observation(obs)
        self.valid_actions_mask = self.get_valid_actions_mask(
            grouped_obs)

        if wrapper := self.observation_wrapper:
            grouped_obs = wrapper.observation(grouped_obs)

        info["action_mask"] = self.valid_actions_mask

        return grouped_obs, reward, done, truncted, info

    def observation(self, observation):
        """현재 상태에서 가능한 모든 열(x)에서 가능한 모든 회전(r)에 대한 다음 상태를 반환하는 메서드"""

        grouped_observations = {}

        curr_piece = observation["piece"]
        piece_id = observation["p_id"]

        if piece_id == 0:
            num_rotations = 1
        elif piece_id < 4:
            num_rotations = 2
        else:
            num_rotations = 4

        for r in range(num_rotations):
            valid_xs = self.env.unwrapped.width - len(curr_piece[0])
            for x in range(valid_xs+1):
                piece = [r[:] for r in curr_piece]
                y = 0
                while not self.env.unwrapped.check_collision(piece, x, y+1):
                    y += 1
                self.env.unwrapped.truncate_overflow_piece(piece, x, y)

                board = self.env.unwrapped.get_board_with_piece(piece, x, y)

                grouped_observations[(x, r)] = board
            curr_piece = self.env.unwrapped.get_rotated_piece(curr_piece)
        return grouped_observations

    def get_valid_actions_mask(self, obs):
        """현재 관찰에서 유효한 액션에 대한 마스크 생성 메서드"""

        mask = np.zeros(self.action_space.n, dtype=np.int8)
        for (x, r) in obs.keys():
            action = self.encode_action(x, r)
            mask[action] = 1
        return mask
