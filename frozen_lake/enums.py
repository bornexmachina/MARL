from enum import IntEnum, auto
import random


class Actions(IntEnum):
    LEFT = auto()
    RIGHT = auto()
    DOWN = auto()
    UP = auto()

    @classmethod
    def sample(cls) -> "Actions":
        return random.choice(list(cls))
    
    @classmethod
    def get_actions(cls) -> list["Actions"]:
        return list(cls)
    

class LakeState(IntEnum):
    SAFE = auto()
    FROZEN = auto()
    HOLE = auto()
    GOAL = auto()
