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
        print(row, flush=True)
        if i < 2:
            print('--+---+--', flush=True)

    print("***", flush=True)


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


def _get_max_action(dictionary):
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


def _get_min_action(dictionary):
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



def _find_max_and_argmax_in_dict(dictionary):
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


def _opponent(player_symbol):
    return PlayerSymbol.Y if player_symbol == PlayerSymbol.X else PlayerSymbol.X


def _to_move(state):
    board = _state_to_board(state)
    x = np.sum(board == PlayerSymbol.X)
    y = np.sum(board == PlayerSymbol.Y)
    return PlayerSymbol.X if x == y else PlayerSymbol.Y


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
        self.V = {}
        self.policy = {}

    def next_state(self, state, action, player_symbol):
        board = _state_to_board(state).copy()
        board[action] = player_symbol
        return _board_to_state(board)
    
    def reward(self, state, player_symbol):
        winner = _check_winner(_state_to_board(state))
        if winner == player_symbol:
            return 1
        elif winner == PlayerSymbol.EMPTY:
            return 0
        elif winner is None:
            return 0
        else:
            return -1

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
        """
        Theory from EPFL - https://www.epfl.ch/labs/lions/wp-content/uploads/2024/08/Lecture-7_-Markov-Games.pdf
        Bellman operators in two-player zero-sum Markov games
        T_{pi_1} V(s) = max_{pi_1} min_{pi_2} [r(s, pi_1(s), pi_2(s)) + gamma * sum_{s'} P(s' | s, pi_1(s), pi_2(s)) * V(s')]
        T_{pi_2} V(s) = min_{pi_2} max_{pi_1} [r(s, pi_1(s), pi_2(s)) + gamma * sum_{s'} P(s' | s, pi_1(s), pi_2(s)) * V(s')]

        Operators are equivalent --> T_{pi_1} = T_{pi_2} = T
        and in the fixed point T = V*

        Just to be clear, pi_1(s) is always an action --> best action in state s
        Also once again, P(s' | ...) = 1 as we have deterministic transitions
        """
        states = _generate_all_states()

        actions_of_x = {s: _get_available_positions(s) for s in states}
        self.V = {s: 0.0 for s in states}
        self.policy_player_x = {}
        self.policy_player_y = {}


        while True:
            delta = 0
            for state in states:
                max_value_x = {}

                for action_of_x in actions_of_x[state]:
                    # by taking action action_of_x from policy pi_x we arrive at state s'
                    # in s' the opponent will have a new set of reachable sets and thus will take an action action_of_y from policy pi_y
                    state_after_one_turn = self.next_state(state, action_of_x, PlayerSymbol.X)

                    # the game might be over after action_of_x --> check for it
                    if _check_winner(_state_to_board(state_after_one_turn)) is None:
                        actions_of_y = _get_available_positions(state_after_one_turn)
                        min_value_y = {}

                        for action_of_y in actions_of_y:
                            state_after_two_turns = self.next_state(state_after_one_turn, action_of_y, PlayerSymbol.Y)
                            instantaneous_reward = self.reward(state_after_two_turns, player_symbol=PlayerSymbol.Y)
                            
                            current_value = instantaneous_reward + self.gamma * self.V.get(state_after_two_turns, 0.0)
                            
                            min_value_y[action_of_y] = current_value

                        max_value_x[action_of_x] = min(min_value_y.values())

                    else:
                        instantaneous_reward = self.reward(state_after_one_turn, player_symbol=PlayerSymbol.X)
                        current_value = instantaneous_reward + self.gamma * self.V.get(state_after_one_turn, 0.0)
                        max_value_x[action_of_x] = current_value
                        min_value_y = {}

                v_old = self.V[state]

                if max_value_x:
                    self.V[state] = max(max_value_x.values())
                    # after we have run through all actions of x and all actions of y we find what the best policies are
                    self.policy_player_x[state] = tuple(random.choice(_get_max_action(max_value_x)))
                    if min_value_y:
                        self.policy_player_y[state] = tuple(random.choice(_get_min_action(min_value_y)))

                delta = max(delta, abs(v_old - self.V[state]))

            if delta < theta:
                break

    def choose_action(self, board):
        state = _board_to_state(board) 
        return self.policy_player_x[state]

    def save_policy(self):
        with open(f'policy_{self.name}.pkl', 'wb') as fw:
            pickle.dump(self.policy_player_x, fw)

    def load_policy(self, file):
        with open(file, 'rb') as fr:
            self.policy_player_x = pickle.load(fr)


class HumanPlayer(PlayerBaseClass):
    def choose_action(self, board):
        positions = _get_available_positions(_board_to_state(board))
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
        if action is None:
            raise ValueError("Action is not allowed!")
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

    def play(self):
        while not self.has_ended:
            player = self.player_1 if self.current_player == PlayerSymbol.X else self.player_2
            action = player.choose_action(self.board)
            self.update_state(action)
            _print_board(self.board)


def main():
    # Training AI players
    player1 = Player("AI Player 1", PlayerSymbol.X)
    player1.train_value_iteration()
    
    # Play against a human
    human_player = HumanPlayer("Human", PlayerSymbol.Y)
    board = Board(player1, human_player)
    board.play()


if __name__ == "__main__":
    main()