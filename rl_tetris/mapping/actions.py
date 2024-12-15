from dataclasses import dataclass


@dataclass
class GameActions:
    """에이전트가 취할 수 있는 행동들 정의"""

    move_left: int = 0
    move_right: int = 1
    move_down: int = 2
    rotate: int = 3
    hard_drop: int = 4
