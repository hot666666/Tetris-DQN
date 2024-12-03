from dataclasses import dataclass


@dataclass(frozen=True)
class GameState:
    """랜더링을 위한 게임 상태 데이터"""
    board: list
    piece: list
    x: int
    y: int
    score: int
    next_piece: list
