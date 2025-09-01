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
        self.lake = [[LakeState.SAFE, LakeState.FROZEN, LakeState.FROZEN, LakeState.FROZEN],
                    [LakeState.FROZEN, LakeState.HOLE, LakeState.FROZEN, LakeState.HOLE],
                    [LakeState.FROZEN, LakeState.FROZEN, LakeState.FROZEN, LakeState.HOLE],
                    [LakeState.HOLE, LakeState.FROZEN, LakeState.FROZEN, LakeState.GOAL]]
        
    def visualize(self):
        data = np.array(self.env)
        custom_colors = ["white", "lightsteelblue", "darkslateblue", "gold"]
        cmap = ListedColormap(custom_colors)
        plt.imshow(data, cmap=cmap, interpolation='nearest')
        plt.xticks([])
        plt.yticks([])
        plt.show()

    def visualize_trajectory(self, trajectory):
        pass