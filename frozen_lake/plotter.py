import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colors import ListedColormap
import seaborn as sns


def visualize_lake(lake) -> Axes:
    """
    plot the frozen lake
    """
    custom_colors = ["white", "lightsteelblue", "darkslateblue", "gold"]
    cmap = ListedColormap(custom_colors)

    _, ax = plt.subplots(figsize=(4, 4))
    
    ax.imshow(lake, cmap=cmap, origin="upper", interpolation="nearest", extent=(0, lake.shape[1], 0, lake.shape[0]))
    ax.grid(color="lightgrey", linewidth=0.5)
    ax.set_xticks(np.arange(0, 5, 1))
    ax.set_yticks(np.arange(0, 5, 1))
    
    return ax


def visualize_trajectory(lake: np.ndarray, trajectory: list[tuple[int, int]]) -> None:
    """
    trajectory is a list of tuples. each tuple is x,y coordinate of the lake
    --> we need a transformation from our indexing of the array to actual plot
    """
    ax = visualize_lake()
    offset = 0.5
    for i, point in enumerate(trajectory):
        color = "lightgrey"
        if i == len(trajectory) - 1:
            color = "orangered"
        x = point[1] + offset
        y = len(lake) - 1 - point[0] + offset
        
        ax.scatter(x, y, color=color, s=50, marker="D")

    plt.show()
