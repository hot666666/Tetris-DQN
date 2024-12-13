from dataclasses import dataclass


@dataclass(frozen=True)
class GameStates:
    """랜더링을 위한 게임 상태 데이터"""
    board: list
    score: int
    next_piece: list
