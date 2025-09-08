import random
import numpy as np
from itertools import product
from enums import Actions, LakeState


class Environment:
    def __init__(self, lake: np.ndarray, reward_map: dict[int, int]) -> None:
        """
        https://towardsdatascience.com/q-learning-for-beginners-2837b777741/
        Initial configuration is from the blog
            --> lets see whether tehe results are the same
        """
        self.lake = lake
        self.reward_map = reward_map

    def turn_left(self, position: tuple[int, int]) -> tuple[int, int]:
        """
        if we are at the left most border, stay there. Else, decrease position's y index
        --> have to wrap my head around row/cols and actions
        """
        new_y = max(0, position[1] - 1)
        return (position[0], new_y)


    def turn_right(self, position: tuple[int, int]) -> tuple[int, int]:
        """
        if we are at the right most border, stay there. Else, increase position's y index
        """
        new_y = min(self.lake.shape[1] - 1, position[1] + 1)
        return (position[0], new_y)

    def turn_up(self, position: tuple[int, int]) -> tuple[int, int]:
        """
        if we are at the top most border, stay there. Else, decrease position's x index
        """
        new_x = max(0, position[0] - 1)
        return (new_x, position[1])

    def turn_down(self, position: tuple[int, int]) -> tuple[int, int]:
        """
        if we are at the bottom most border, stay there. Else, increase position's x index
        """
        new_x = min(self.lake.shape[0] - 1, position[0] + 1)
        return (new_x, position[1])

    def take_action(self, action: int, position: tuple[int, int]) -> tuple[int, int]:
        if action == Actions.LEFT:
            return self.turn_left(position)
        if action == Actions.RIGHT:
            return self.turn_right(position)
        if action == Actions.UP:
            return self.turn_up(position)
        if action == Actions.DOWN:
            return self.turn_down(position)
        raise ValueError(f"--- illegal action has been provided: {action} ---")
    
    def get_instantaneous_reward(self, position: tuple[int, int]) -> int:
        current_state = self.lake[position]
        return self.reward_map[current_state]
    
    def is_terminal(self, position: tuple[int, int]) -> bool:
        current_state = self.lake[position]
        return current_state in [LakeState.HOLE, LakeState.GOAL]
    
    def all_states(self) -> list[tuple[int, int]]:
        return list(product(range(self.lake.shape[0]), range(self.lake.shape[1])))
