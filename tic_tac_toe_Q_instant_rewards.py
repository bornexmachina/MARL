import numpy as np
import random
import pickle
from enum import IntEnum

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

    def update_Q(self, state, action, reward, next_state):
        if state not in self.Q:
            self.Q[state] = {}

        if action not in self.Q[state]:
            self.Q[state][action] = 0.0

        future_rewards = 0.0
        if next_state in self.Q and self.Q[next_state]:
            future_rewards = max(self.Q[next_state].values())

        current_q = self.Q[state][action]
        self.Q[state][action] += self.alpha * (reward + self.gamma * future_rewards - current_q)

    def choose_action(self, positions, board):
        state = tuple(board.flatten())

        if np.random.rand() < self.epsilon:
            action = random.choice(positions)
        else:
            q_values = self.Q.get(state, {})
            max_q = -float('inf')
            best_actions = []

            for action in positions:
                q = q_values.get(action, 0.0)
                if q > max_q:
                    max_q = q
                    best_actions = [action]
                elif q == max_q:
                    best_actions.append(action)

            action = random.choice(best_actions) if best_actions else random.choice(positions)

        return action

    # now we try instantaneous rewards
    #   if the game is won by the Player -> get 1
    #   if the game is won by the Opponent -> get -1
    #   if draw -> get 0
    #   if NOT FINISHED -> get 0
    def reward(self, state, action, next_state, winner):
        if winner is None:
            reward = 0
        elif winner == self.player_symbol:
            reward = 1
        elif winner == PlayerSymbol.EMPTY:
            reward = 0
        else:
            reward = -1

        self.update_Q(state, action, reward, next_state)

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
            new_state = tuple(self.board.flatten())

            # not sexy - we need to know which player is currently at his turn
            if self.current_player == PlayerSymbol.X:
                self.player_1.reward(state=current_state, action=position_tuple, next_state=new_state, winner=winner)
            else:
                self.player_2.reward(state=current_state, action=position_tuple, next_state=new_state, winner=winner)

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
        
    def train(self, rounds=100):
        for i in range(rounds):
            if i % 10000 == 0:
                print(f"Current round: {i}")
            while not self.has_ended:
                positions = self.get_available_positions()
                if not positions:
                    self.reset_game()
                    break
                player_1_action = self.player_1.choose_action(positions, self.board)

                if not self.update_state(position_tuple=player_1_action):
                    print("ERROR: Invalid state update")
                    break

                if self.get_winner() is not None:
                    self.reset_game()
                    break

                positions = self.get_available_positions()
                if not positions:
                    self.reset_game()
                    break
                
                player_2_action = self.player_2.choose_action(positions, self.board)
                if not self.update_state(position_tuple=player_2_action):
                    print("ERROR: Invalid state update")
                    break

                if self.get_winner() is not None:
                    self.reset_game()
                    break

    def play(self):
        while not self.has_ended:
            positions = self.get_available_positions()
            if not positions:
                self.reset_game()
                break
            player_1_action = self.player_1.choose_action(positions, self.board)

            if not self.update_state(position_tuple=player_1_action):
                print("ERROR: Invalid state update")
                break

            self.print_board()

            winner = self.get_winner()
            if winner is not None:
                self.print_board()
                print(f"Winner: {self.get_winner_name(winner)}")
                self.reset_game()
                break

            positions = self.get_available_positions()
            if not positions:
                self.reset_game()
                break
            
            player_2_action = self.player_2.choose_action(positions, self.board)
            if not self.update_state(position_tuple=player_2_action):
                print("ERROR: Invalid state update")
                break

            self.print_board()

            winner = self.get_winner()
            if winner is not None:
                print(f"Winner: {self.get_winner_name(winner)}")
                self.reset_game()
                break

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
    player2 = Player("AI Player 2", PlayerSymbol.Y)
    board = Board(player1, player2)
    
    # Train the AI players
    print("Training AI players...")
    board.train(rounds=100000)
    
    # Play against a human
    human_player = HumanPlayer("Human")
    board = Board(player1, human_player)
    board.play()

if __name__ == "__main__":
    main()