from enum import IntEnum, auto
import random
import numpy as np


class Actions(IntEnum):
    LEFT = auto()
    RIGHT = auto()
    DOWN = auto()
    UP = auto()

    @classmethod
    def sample(cls):
        return random.choice([cls.LEFT, cls.RIGHT, cls.DOWN, cls.UP])
    

class LakeState(IntEnum):
    SAFE = auto()
    FROZEN = auto()
    HOLE = auto()
    GOAL = auto()


class Environment:
    def __init__(self, lake: np.ndarray, reward_map: dict[int, int], starting_position: tuple[int, int]) -> None:
        """
        https://towardsdatascience.com/q-learning-for-beginners-2837b777741/
        Initial configuration is from the blog
            --> lets see whether tehe results are the same
        """
        self.lake = lake
        self.reward_map = reward_map
        self.position = starting_position

    def turn_left(self) -> None:
        """
        if we are at the left most border, stay there. Else, decrease position's y index
        --> have to wrap my head around row/cols and actions
        """
        new_y = max(0, self.position[1] - 1)
        self.position = (self.position[0], new_y)


    def turn_right(self) -> None:
        """
        if we are at the right most border, stay there. Else, increase position's y index
        """
        new_y = min(self.lake.shape[1] - 1, self.position[1] + 1)
        self.position = (self.position[0], new_y)

    def turn_up(self) -> None:
        """
        if we are at the top most border, stay there. Else, decrease position's x index
        """
        new_x = max(0, self.position[0] - 1)
        self.position = (new_x, self.position[1])

    def turn_down(self) -> None:
        """
        if we are at the bottom most border, stay there. Else, increase position's x index
        """
        new_x = min(self.lake.shape[0] - 1, self.position[0] + 1)
        self.position = (new_x, self.position[1])

    def take_action(self, action: int) -> None:
        if action == Actions.LEFT:
            self.turn_left()
        if action == Actions.RIGHT:
            self.turn_right()
        if action == Actions.UP:
            self.turn_up()
        if action == Actions.DOWN:
            self.turn_down()
        raise ValueError("--- illegal action has been provided ---")
    
    def get_instantaneous_reward(self) -> int:
        current_state = self.lake[self.position]
        return self.reward_map[current_state]
    
    def is_terminal(self) -> bool:
        current_state = self.lake[self.position]
        return current_state in [LakeState.HOLE, LakeState.GOAL]
