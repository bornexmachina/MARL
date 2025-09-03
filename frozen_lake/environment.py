from enum import IntEnum, auto
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns


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
    def __init__(self):
        """
        https://towardsdatascience.com/q-learning-for-beginners-2837b777741/
        Initial configuration is from the blog
            --> lets see whether tehe results are the same
        """
        self.lake = np.array([[LakeState.SAFE, LakeState.FROZEN, LakeState.FROZEN, LakeState.FROZEN],
                              [LakeState.FROZEN, LakeState.HOLE, LakeState.FROZEN, LakeState.HOLE],
                              [LakeState.FROZEN, LakeState.FROZEN, LakeState.FROZEN, LakeState.HOLE],
                              [LakeState.HOLE, LakeState.FROZEN, LakeState.FROZEN, LakeState.GOAL]])
        self.position = (0, 0)
        
    def visualize_initial_env(self):
        custom_colors = ["white", "lightsteelblue", "darkslateblue", "gold"]
        cmap = ListedColormap(custom_colors)

        fig, ax = plt.subplots(figsize=(4, 4))
        
        ax.imshow(self.lake, cmap=cmap, origin="upper", interpolation="nearest", extent=(0, 4, 0, 4))
        ax.grid(color="lightgrey", linewidth=0.5)
        ax.set_xticks(np.arange(0, 5, 1))
        ax.set_yticks(np.arange(0, 5, 1))
        
        return ax

    def visualize_trajectory(self, trajectory):
        """
        trajectory is a list of tuples. each tuple is x,y coordinate of the lake
        --> we need a transformation from our indexing of the array to actual plot
        """
        ax = self.visualize_initial_env()
        offset = 0.5
        for i, point in enumerate(trajectory):
            color = "lightgrey"
            if i == len(trajectory) - 1:
                color = "orangered"
            x = point[1] + offset
            y = len(self.lake) - 1 - point[0] + offset
            
            ax.scatter(x, y, color=color, s=50, marker="D")

        plt.show()

    def turn_left(self):
        pass

    def turn_right(self):
        pass

    def turn_up(self):
        pass

    def turn_down(self):
        pass
