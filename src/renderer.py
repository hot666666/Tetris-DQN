import cv2
import numpy as np

from src.game_state import GameState


class Renderer:
    """게임 이미지를 만들어주는 클래스"""

    PIECE_COLORS = [
        [0, 0, 0],  # Empty
        [127, 219, 255],  # O
        [255, 111, 97],  # I
        [192, 132, 151],  # S
        [168, 230, 207],  # Z
        [255, 210, 125],  # T
        [90, 125, 154],  # L
        [255, 160, 122],  # J
    ]

    HEADER_COLOR = [0, 0, 0]

    FONT = cv2.FONT_HERSHEY_SIMPLEX
    FONT_COLOR = [255, 255, 255]
    FONT_SCALE = 0.6

    def __init__(self, height=20, width=10,  block_size=30):
        self.height = height
        self.width = width
        self.block_size = block_size

        self.header_height = 2
        self.next_piece_size = (2*block_size) // 5

        self.header_right_padding = 10
        self.header_left_padding = block_size // 2

    def get_scaled_piece_pos(self, piece, x, y):
        """piece의 scale된 위치를 x, y에 맞게 반환하는 메서드"""

        top = y * self.block_size
        left = x * self.block_size
        bottom = top + len(piece)
        right = left + len(piece[0])

        return top, left, bottom, right

    def get_scaled_RGB_arr(self, arr):
        scale = self.next_piece_size if len(arr) == 5 else self.block_size

        ndarr = np.array(
            [[self.PIECE_COLORS[p] for p in row] for row in arr],
            dtype=np.uint8
        )
        scaled_ndarr = np.kron(ndarr, np.ones(
            (scale, scale, 1), dtype=np.uint8))
        return scaled_ndarr

    def render(self, game_state: GameState):
        # 보드 배열을 만들고 현재 블록을 추가
        board = self.get_board_ndarray(game_state.board)

        # 현재 블록 배열과 위치를 보드에 추가
        piece = self.get_piece_ndarray(game_state.piece)
        pos = self.get_scaled_piece_pos(piece, game_state.x, game_state.y)
        self.update_board_with(board, piece, pos)

        # 다음 블록 배열
        next_piece = self.get_next_piece_ndarray(game_state.next_piece)

        # 헤더 배열
        header = self.get_header_ndarray(next_piece)

        # 전체 게임 배열(헤더 + 보드)
        game = np.vstack((header, board))
        game = cv2.cvtColor(game, cv2.COLOR_RGB2BGR)  # RGB -> BGR

        # 점수 추가
        self.draw_header_score(game, game_state.score)

        cv2.imshow("Tetris with DQN", game)
        cv2.waitKey(1)

    def draw_header_score(self, game_ndarr, score):
        score_text = f"Score: {score}"

        cv2.putText(game_ndarr, score_text, (self.header_left_padding, 2*self.header_left_padding),
                    fontFace=self.FONT, fontScale=self.FONT_SCALE, color=self.FONT_COLOR)

    def update_board_with(self, board, piece, pos):
        # 보드에 현재 블록 추가
        top, left, bottom, right = pos
        board[top:bottom, left:right] = np.where(
            piece > 0, piece, board[top:bottom, left:right]
        )

        # 격자 추가
        board[[i * self.block_size for i in range(self.height)], :, :] = 0
        board[:, [i * self.block_size for i in range(self.width)], :] = 0

    def get_board_ndarray(self, board):
        board = self.get_scaled_RGB_arr(board)

        return board

    def get_piece_ndarray(self, piece):
        piece = self.get_scaled_RGB_arr(piece)

        return piece

    def get_header_ndarray(self, next_piece):
        # 헤더 영역 배열 생성
        header = np.full(
            (self.block_size * 2, self.width * self.block_size, 3),
            fill_value=self.HEADER_COLOR,
            dtype=np.uint8)

        # 다음 블록 이미지 추가
        next_piece_x = (self.width-self.header_height) * \
            self.block_size-self.header_right_padding
        header[0:next_piece.shape[0],
               next_piece_x:next_piece_x+next_piece.shape[1]] = next_piece

        # 밑줄 추가
        header[-1:] = 128

        return header

    def get_next_piece_ndarray(self, next_piece):
        # 5x5로 패딩
        padded_piece = np.zeros((5, 5), dtype=int)
        piece_h, piece_w = len(next_piece), len(next_piece[0])
        padded_piece[1:1+piece_h, 1:1+piece_w] = next_piece

        next_piece = self.get_scaled_RGB_arr(padded_piece)
        return next_piece
