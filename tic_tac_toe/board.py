import numpy as np
import utils


class Board:
    def __init__(self, player_1, player_2):
        self.board = np.zeros((3, 3), dtype=int)
        self.has_ended = False
        self.player_1 = player_1
        self.player_2 = player_2
        self.current_player = utils.PlayerSymbol.X
    
    def switch_players(self):
        self.current_player = utils.PlayerSymbol.X if self.current_player == utils.PlayerSymbol.Y else utils.PlayerSymbol.Y
    
    def update_state(self, action):
        if action is None:
            raise ValueError("Action is not allowed!")
        if action in utils.get_available_positions(self.board):
            self.board[action] = self.current_player
            winner = utils.check_winner(utils.board_to_state(self.board))

            if winner is not None:
                self.has_ended = True
                utils.print_board(self.board)

                if winner == utils.PlayerSymbol.EMPTY:
                    print("Draw!")
                else:
                    print(f"Winner is Player {utils.get_winner_name(winner)}!")

            self.switch_players()

    def play(self):
        while not self.has_ended:
            player = self.player_1 if self.current_player == utils.PlayerSymbol.X else self.player_2
            action = player.choose_action(self.board)
            self.update_state(action)
            utils.print_board(self.board)