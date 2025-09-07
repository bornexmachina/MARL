import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colors import ListedColormap


def visualize_lake(lake) -> Axes:
    """
    plot the frozen lake
    """
    custom_colors = ["white", "lightsteelblue", "darkslateblue", "gold"]
    cmap = ListedColormap(custom_colors)

    _, ax = plt.subplots(figsize=(6, 6))
    
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
    max_idx = len(lake) - 1

    for i, point in enumerate(trajectory):
        color = "lightgrey"
        if i == len(trajectory) - 1:
            color = "orangered"
        x = point[1] + offset
        y = max_idx - point[0] + offset
        
        ax.scatter(x, y, color=color, s=50, marker="D")

    plt.show()


def visualize_V(lake: np.ndarray, V: dict[tuple[int, int], float]):
    ax = visualize_lake(lake)
    offset = 0.5
    max_idx = len(lake) - 1

    for (i, j), value in V.items():
        color = "white"
        if (i, j) == (0, 0) or (i, j) == (max_idx, max_idx):
            color = "black"
        x = j + offset
        y = max_idx - i + offset

        ax.text(x, y, f"{value:.2f}", ha="center", va="center", color=color, fontsize=10)

    plt.show()


def visualize_Q(lake: np.ndarray, Q: dict[tuple[int, int], dict[int, float]]):
    """
    Q is of form {state: {left: 0.x, right: 0.x, up: 0.x, down: 0.x}}
    --> in best case we can plot a 4x4 cell in each large cell
    """
    ax = visualize_lake(lake)
    offset = 0.5
    max_idx = len(lake) - 1

    for (i, j), directions in Q.items():
        color = "white"
        if (i, j) == (0, 0) or (i, j) == (max_idx, max_idx):
            color = "black"
        # Position is (col=j, row=i) because imshow uses (row, col)
        x = j + offset
        y = max_idx - i + offset

        # Write text in 4 positions
        ax.text(x - 0.25, y, f"{directions['left']:.2f}",
                ha="right", va="center", color=color, fontsize=8)
        ax.text(x + 0.25, y, f"{directions['right']:.2f}",
                ha="left", va="center", color=color, fontsize=8)
        ax.text(x, y - 0.25, f"{directions['up']:.2f}",
                ha="center", va="bottom", color=color, fontsize=8)
        ax.text(x, y + 0.25, f"{directions['down']:.2f}",
                ha="center", va="top", color=color, fontsize=8)
        
    plt.show()