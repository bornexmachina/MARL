import numpy as np
import random
import pickle
from enum import IntEnum
from itertools import product


# Helper functions -> refactor from Board and Player
def _print_board(board):
    symbol_map = {PlayerSymbol.EMPTY: ' ',
                    PlayerSymbol.X: 'X',
                    PlayerSymbol.Y: 'O'}
    
    for i in range(3):
        row = ' | '.join(symbol_map[PlayerSymbol(cell)] for cell in board[i])
        print(row)
        if i < 2:
            print('--+---+--')

    print("***")


def _check_winner(board):
    for player_symbol in [PlayerSymbol.X, PlayerSymbol.Y]:
        _rows = np.any(board.sum(axis=1) == player_symbol * 3)
        _cols = np.any(board.sum(axis=0) == player_symbol * 3)
        _diag = np.trace(board) == player_symbol * 3
        _antidiag = np.trace(np.fliplr(board)) == player_symbol * 3

        if _rows or _cols or _diag or _antidiag:
            return player_symbol
        
    if not (board == PlayerSymbol.EMPTY).any():
        return PlayerSymbol.EMPTY
    
    return None


def _get_winner_name(winner):
    if winner == PlayerSymbol.X:
        return "X"
    elif winner == PlayerSymbol.Y:
        return "O"
    else:
        return "Draw"


def _get_available_positions(state):
        return list(zip(*np.where(_state_to_board(state) == PlayerSymbol.EMPTY)))


def _is_valid_board(board):
    """
    Hard wired who goes first -> we might come up with something better at some point
    valid states assuming X starts first:
        - empty
        - same number of X and Y
        - 1 X more than Y --> X has won the game
        - only X, Y and EMPTY symbols on the board
    """
    assert board.shape == (3, 3)
    x_count = np.sum(board == PlayerSymbol.X)
    y_count = np.sum(board == PlayerSymbol.Y)
    empty_count = np.sum(board == PlayerSymbol.EMPTY)

    # assertion is better -> if we add weird symbols, than the whole game is corrupt
    assert x_count + y_count + empty_count == 9

    return (x_count == y_count or x_count - y_count == 1)


def _is_full(state):
    return len(_get_available_positions(state)) == 0


def _generate_all_states():
    all_states = []
    for state in product([PlayerSymbol.EMPTY, PlayerSymbol.X, PlayerSymbol.Y], repeat=9):
        board = _state_to_board(state)
        if _is_valid_board(board):
            all_states.append(state)
    return all_states


def _board_to_state(board):
    return tuple(board.flatten())


def _state_to_board(state):
    return np.array(state).reshape(3, 3)


class PlayerSymbol(IntEnum):
    EMPTY = 0
    X = 1
    Y = -1


class PlayerBaseClass:
    """
    Virtual class - makes the structure easier
    """
    def __init__(self, name, player_symbol):
        self.name = name
        self.player_symbol = player_symbol

    def choose_action(self, positions, board):
        pass

    def save_policy(self):
        pass

    def load_policy(self, file):
        pass


class Player(PlayerBaseClass):
    """
    Implementation of Q-learning with a state history and backpropagation of the reward
    """
    def __init__(self, name, player_symbol, gamma=1.0):
        super().__init__(name, player_symbol)
        self.gamma = gamma
        self.Q = {}

    def next_state_and_reward(self, state, action, player_symbol):
        board = _state_to_board(state).copy()
        board[action] = player_symbol
        winner = _check_winner(board)
        if winner == player_symbol:
            return _board_to_state(board), 1
        elif winner == PlayerSymbol.EMPTY:
            return _board_to_state(board), 0
        elif winner is None:
            return _board_to_state(board), 0
        else:
            return _board_to_state(board), -1

    def train_value_iteration(self, theta=1e-6):
        """Offline value iteration to compute Q-table."""
        states = _generate_all_states()

        actions_dict = {s: _get_available_positions(s) for s in states}
        self.Q = {s: {a: 0.0 for a in actions_dict[s]} for s in states}

        while True:
            delta = 0
            for s in states:
                for a in actions_dict[s]:
                    ns, reward = self.next_state_and_reward(s, a, self.player_symbol)
                    future_q = 0.0
                    if ns in self.Q and self.Q[ns]:
                        future_q = max(self.Q[ns].values())
                    old_q = self.Q[s][a]
                    self.Q[s][a] = reward + self.gamma * future_q
                    delta = max(delta, abs(old_q - self.Q[s][a]))
            if delta < theta:
                break

    def choose_action(self, positions, board):
        state = _board_to_state(board)
        q_values = self.Q.get(state, {})
        if not q_values:
            return random.choice(positions)
        max_q = max(q_values.values())
        best_actions = [a for a, q in q_values.items() if q == max_q]
        return random.choice(best_actions)

    def save_policy(self):
        with open(f'policy_{self.name}.pkl', 'wb') as fw:
            pickle.dump(self.Q, fw)  # Save Q-table instead of undefined states_value

    def load_policy(self, file):
        with open(file, 'rb') as fr:
            self.Q = pickle.load(fr)  # Load Q-table


class HumanPlayer(PlayerBaseClass):
    def choose_action(self, positions, board):
        while True:
            try:
                row = int(input("Input your action row (0-2):"))
                col = int(input("Input your action col (0-2):"))
                action = (row, col)
                
                if action in positions:
                    return action
                else:
                    print("Invalid position. Please choose an empty cell.")
            except ValueError:
                print("Please enter valid integer coordinates.")


class Board:
    def __init__(self, player_1, player_2):
        self.board = np.zeros((3, 3), dtype=int)
        self.has_ended = False
        self.player_1 = player_1
        self.player_2 = player_2
        self.current_player = PlayerSymbol.X
    
    def switch_players(self):
        self.current_player = PlayerSymbol.X if self.current_player == PlayerSymbol.Y else PlayerSymbol.Y
    
    def update_state(self, action):
        if action in _get_available_positions(self.board):
            self.board[action] = self.current_player
            winner = _check_winner(self.board)

            if winner is not None:
                self.has_ended = True
                _print_board(self.board)

                if winner == PlayerSymbol.EMPTY:
                    print("Draw!")
                else:
                    print(f"Winner is Player {_get_winner_name(winner)}!")

            self.switch_players()
            return True
        return False

    def play(self):
        while not self.has_ended:
            positions = _get_available_positions(self.board)
            player = self.player_1 if self.current_player == PlayerSymbol.X else self.player_2
            action = player.choose_action(positions, self.board)
            self.update_state(action)
            _print_board(self.board)


def main():
    # Training AI players
    player1 = Player("AI Player 1", PlayerSymbol.X)
    player1.train_value_iteration()
    player2 = Player("AI Player 2", PlayerSymbol.Y)
    player2.train_value_iteration()

    board = Board(player1, player2)
    
    # Play against a human
    human_player = HumanPlayer("Human", PlayerSymbol.Y)
    board = Board(player1, human_player)
    board.play()


if __name__ == "__main__":
    main()