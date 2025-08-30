import numpy as np
from enum import IntEnum
from itertools import product

class PlayerSymbol(IntEnum):
    EMPTY = 0
    X = 1
    Y = -1

# Helper functions -> refactor from Board and Player
def print_board(board):
    symbol_map = {PlayerSymbol.EMPTY: ' ',
                    PlayerSymbol.X: 'X',
                    PlayerSymbol.Y: 'O'}
    
    for i in range(3):
        row = ' | '.join(symbol_map[PlayerSymbol(cell)] for cell in board[i])
        print(row, flush=True)
        if i < 2:
            print('--+---+--', flush=True)

    print("***", flush=True)


def check_winner(state):
    board = state_to_board(state)

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


def get_winner_name(winner):
    if winner == PlayerSymbol.X:
        return "X"
    elif winner == PlayerSymbol.Y:
        return "O"
    else:
        return "Draw"


def get_available_positions(state):
        return list(zip(*np.where(state_to_board(state) == PlayerSymbol.EMPTY)))


def is_valid_board(board):
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
    return len(get_available_positions(state)) == 0


def generate_all_states():
    all_states = []
    for state in product([PlayerSymbol.EMPTY, PlayerSymbol.X, PlayerSymbol.Y], repeat=9):
        board = state_to_board(state)
        if is_valid_board(board):
            all_states.append(state)
    return all_states


def board_to_state(board):
    return tuple(board.flatten())


def state_to_board(state):
    return np.array(state).reshape(3, 3)


def get_max_action(dictionary):
    """
    Explicit assumption, we have 1 to 1 key-value relationship
    """
    flat_keys = np.array(list(dictionary.keys()))
    tmp_vals = list(dictionary.values())
    
    # just to make sure that no value is a list itself
    flat_vals = []

    for element in tmp_vals:
        if hasattr(element, '__iter__'):
            for l in element:
                flat_vals.append(l)
        else:
            flat_vals.append(element)

    flat_vals = np.array(flat_vals)

    assert len(flat_keys) == len(flat_vals)

    max_value = max(flat_vals)
    argmax_value = flat_keys[np.where(flat_vals == max_value)[0]]

    return argmax_value


def get_min_action(dictionary):
    """
    Explicit assumption, we have 1 to 1 key-value relationship
    """
    flat_keys = np.array(list(dictionary.keys()))
    tmp_vals = list(dictionary.values())
    
    # just to make sure that no value is a list itself
    flat_vals = []

    for element in tmp_vals:
        if hasattr(element, '__iter__'):
            for l in element:
                flat_vals.append(l)
        else:
            flat_vals.append(element)

    flat_vals = np.array(flat_vals)

    assert len(flat_keys) == len(flat_vals)

    min_value = min(flat_vals)
    argmin_value = flat_keys[np.where(flat_vals == min_value)[0]]

    return argmin_value



def find_max_and_argmax_in_dict(dictionary):
    """
    Explicit assumption, we have 1 to 1 key-value relationship
    """
    flat_keys = np.array(list(dictionary.keys()))
    tmp_vals = list(dictionary.values())
    
    # just to make sure that no value is a list itself
    flat_vals = []

    for element in tmp_vals:
        if hasattr(element, '__iter__'):
            for l in element:
                flat_vals.append(l)
        else:
            flat_vals.append(element)

    flat_vals = np.array(flat_vals)

    assert len(flat_keys) == len(flat_vals)

    max_value = max(flat_vals)
    argmax_value = flat_keys[np.where(flat_vals == max_value)[0]]

    return max_value, argmax_value


def opponent(player_symbol):
    return PlayerSymbol.Y if player_symbol == PlayerSymbol.X else PlayerSymbol.X


def to_move(state):
    board = state_to_board(state)
    x = np.sum(board == PlayerSymbol.X)
    y = np.sum(board == PlayerSymbol.Y)
    return PlayerSymbol.X if x == y else PlayerSymbol.Y