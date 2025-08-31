from board import Board
from player import HumanPlayer, PlayerValueIteration
import utils


def main():
    # Training AI players
    vi_player = PlayerValueIteration("Value Iteration", utils.PlayerSymbol.X)
    vi_player.train()
    
    # Play against a human
    human_player = HumanPlayer("Human", utils.PlayerSymbol.Y)
    board = Board(vi_player, human_player)
    board.play()


if __name__ == "__main__":
    main()