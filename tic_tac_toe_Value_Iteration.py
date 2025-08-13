import numpy as np
import random
import pickle
from enum import IntEnum
from itertools import product

class PlayerSymbol(IntEnum):
    EMPTY = 0
    X = 1
    Y = -1

class Player:
    """
    Implementation of Q-learning with a state history and backpropagation of the reward
    """
    def __init__(self, name, player_symbol, epsilon=0.1, gamma=0.9, alpha=0.1):
        self.name = name
        self.player_symbol = player_symbol
        self.epsilon = epsilon
        self.gamma = gamma
        self.alpha = alpha
        self.Q = {}

    def _check_winner(self, board):
        for symbol in [PlayerSymbol.X, PlayerSymbol.Y]:
            if np.any(board.sum(axis=0) == symbol * 3):  # cols
                return symbol
            if np.any(board.sum(axis=1) == symbol * 3):  # rows
                return symbol
            if np.trace(board) == symbol * 3:            # diag
                return symbol
            if np.trace(np.fliplr(board)) == symbol * 3: # anti-diag
                return symbol
        if not (board == PlayerSymbol.EMPTY).any():
            return PlayerSymbol.EMPTY  # draw
        return None  # not finished

    def _is_valid_state(self, state):
        board = np.array(state)
        x_count = np.sum(board == PlayerSymbol.X)
        o_count = np.sum(board == PlayerSymbol.Y)
        return not (o_count > x_count or x_count - o_count > 1)

    def _get_available_positions(self, state):
        return list(zip(*np.where(np.array(state).reshape(3,3) == PlayerSymbol.EMPTY)))

    def _next_state_and_reward(self, state, action, player_symbol):
        board = np.array(state).reshape(3,3).copy()
        board[action] = player_symbol
        winner = self._check_winner(board)
        if winner == player_symbol:
            return tuple(board.flatten()), 1
        elif winner == PlayerSymbol.EMPTY:
            return tuple(board.flatten()), 0
        elif winner is None:
            return tuple(board.flatten()), 0
        else:
            return tuple(board.flatten()), -1

    def _generate_all_states(self):
        all_states = []
        for cells in product([PlayerSymbol.EMPTY, PlayerSymbol.X, PlayerSymbol.Y], repeat=9):
            if self._is_valid_state(cells):
                all_states.append(tuple(cells))
        return all_states

    def train_value_iteration(self, theta=1e-6):
        """Offline value iteration to compute Q-table."""
        states = self._generate_all_states()
        actions_dict = {s: self._get_available_positions(s) for s in states}
        self.Q = {s: {a: 0.0 for a in actions_dict[s]} for s in states}

        while True:
            delta = 0
            for s in states:
                for a in actions_dict[s]:
                    ns, reward = self._next_state_and_reward(s, a, self.player_symbol)
                    future_q = 0.0
                    if ns in self.Q and self.Q[ns]:
                        future_q = max(self.Q[ns].values())
                    old_q = self.Q[s][a]
                    self.Q[s][a] = reward + self.gamma * future_q
                    delta = max(delta, abs(old_q - self.Q[s][a]))
            if delta < theta:
                break

    def choose_action(self, positions, board):
        state = tuple(board.flatten())
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


class HumanPlayer:
    def __init__(self, name):
        self.name = name

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

    def update_Q(self, state, action, reward, next_state):
        pass

    def reward(self, winner):
        pass

    def reset(self):
        pass

    def save_policy(self):
        pass

    def load_policy(self, file):
        pass


class Board:
    def __init__(self, player_1, player_2):
        self.board = np.zeros((3, 3), dtype=int)
        self.has_ended = False
        self.player_1 = player_1
        self.player_2 = player_2
        self.current_player = PlayerSymbol.X

    def generate_all_states():
        all_states = []

        for cells in product([PlayerSymbol.EMPTY, PlayerSymbol.X, PlayerSymbol.Y], repeat=9):
            board = np.array(cells).reshape(3, 3)

            all_states.append(tuple(board.flatten()))
        
        return all_states
    
    def check_winning_key(self, key):
        _rows = np.any(self.board.sum(axis=1) == key * 3)
        _cols = np.any(self.board.sum(axis=0) == key * 3)
        _diag = np.trace(self.board) == key * 3
        _antidiag = np.trace(np.fliplr(self.board)) == key * 3

        if _rows or _cols or _diag or _antidiag:
            self.has_ended = True
            return True
        
        return False
    
    def is_full(self):
        return len(self.get_available_positions()) == 0
    
    def get_winner(self):
        if self.check_winning_key(PlayerSymbol.X):
            return PlayerSymbol.X
        if self.check_winning_key(PlayerSymbol.Y):
            return PlayerSymbol.Y
        if self.is_full():
            self.has_ended = True
            return PlayerSymbol.EMPTY
        return None
    
    def get_available_positions(self):
        return list(zip(*np.nonzero(self.board == PlayerSymbol.EMPTY)))
    
    def switch_players(self):
        self.current_player = PlayerSymbol.X if self.current_player == PlayerSymbol.Y else PlayerSymbol.Y
    
    def update_state(self, position_tuple):
        # with the instantaneous rewards we have to call 
        # player.reward() each time --> reward(self, state, action, next_state, winner)
        # so we need current state, action equals position_tuple, next state is where
        # we arrive after taking the action, and the winner is to be decided after the move
        current_state = tuple(self.board.flatten())

        if position_tuple in self.get_available_positions():
            self.board[position_tuple] = self.current_player
            winner = self.get_winner()

            if winner is not None:
                self.has_ended = True
                self.print_board()

                if winner == PlayerSymbol.EMPTY:
                    print("Draw!")
                else:
                    print(f"Winner is Player {self.get_winner()}!")

            self.switch_players()
            return True
        return False
    
    def reward_players(self, winner):
        self.player_1.reward(winner)
        self.player_2.reward(winner)
        
    def reset_board(self):
        self.board = np.zeros((3, 3), dtype=int)
        self.has_ended = False
        self.current_player = PlayerSymbol.X

    def reset_game(self):
        winner = self.get_winner()
        self.reward_players(winner)
        self.player_1.reset()
        self.player_2.reset()
        self.reset_board()

    def play(self):
        while not self.has_ended:
            positions = self.get_available_positions()
            player = self.player_1 if self.current_player == PlayerSymbol.X else self.player2
            action = player.choose_action(positions, self.board)
            self.update_state(action)

    def get_winner_name(self, winner):
        if winner == PlayerSymbol.X:
            return f"{self.player_1.name} (X)"
        elif winner == PlayerSymbol.Y:
            return f"{self.player_2.name} (O)"
        else:
            return "Draw"

    def print_board(self):
        symbol_map = {PlayerSymbol.EMPTY: ' ',
                      PlayerSymbol.X: 'X',
                      PlayerSymbol.Y: 'O'}
        
        for i in range(3):
            row = ' | '.join(symbol_map[PlayerSymbol(cell)] for cell in self.board[i])
            print(row)
            if i < 2:
                print('--+---+--')

        print("***")

def main():
    # Training AI players
    player1 = Player("AI Player 1", PlayerSymbol.X)
    player1.train_value_iteration()
    player2 = Player("AI Player 2", PlayerSymbol.Y)
    player2.train_value_iteration()

    board = Board(player1, player2)
    
    
    # Play against a human
    human_player = HumanPlayer("Human")
    board = Board(player1, human_player)
    board.play()

if __name__ == "__main__":
    main()