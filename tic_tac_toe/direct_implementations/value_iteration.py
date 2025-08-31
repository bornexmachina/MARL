import numpy as np
import random
import pickle
import utils

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

    def next_state_and_reward(self, state, action, player_symbol):
        board = utils.state_to_board(state).copy()
        board[action] = player_symbol
        winner = utils.check_winner(board)
        if winner == player_symbol:
            return utils.board_to_state(board), 1
        elif winner == utils.PlayerSymbol.EMPTY:
            return utils.board_to_state(board), 0
        elif winner is None:
            return utils.board_to_state(board), 0
        else:
            return utils.board_to_state(board), -1

    def train_value_iteration(self, theta=1e-6):
        """
        Offline value iteration to compute V-table.
        Value iteration updates V(s) --> only depends on the state

        First set V to arbitrary value e.g. V(s) = 0 for all s 
        and define the policy in each state as None --> we will update the policy for policy extraction
        
        Bellman equation says:
            V'(s) <-- max_{a in A} sum_{s' in S} P_{a}(s' | s) [r(s, a, s') + gamma * V(s')]
        
        The transition is deterministic --> P_{a}(s' | s) is always one.
        Also for now just ignore the discountinuation i.e. gamma = 1

        for direct implementation we use the algorithm as outlined in
        https://gibberblot.github.io/rl-notes/single-agent/value-iteration.html
        """
        states = utils.generate_all_states()

        actions_dict = {s: utils.get_available_positions(s) for s in states}
        self.V = {s: 0.0 for s in states}
        self.policy = {s: None for s in states}

        Q = {s: {a: 0.0 for a in actions_dict[s]} for s in states}

        while True:
            delta = 0
            for s in states:
                to_move = utils.to_move(s)

                for a in actions_dict[s]:
                    if to_move == self.player_symbol:
                        # per action a we arrive deterministically at position s'
                        # the sum over s' in S is thus exactly one term
                        s_, reward = self.next_state_and_reward(s, a, self.player_symbol)
                    else:
                        s_, reward_opp = self.next_state_and_reward(s, a, utils.opponent(self.player_symbol))
                        reward = - reward_opp

                    if utils.is_valid_board(utils.state_to_board(s_)):
                        gamma_V = 0.0 if utils.check_winner(utils.state_to_board(s_)) is not None else self.gamma * self.V[s_]
                        Q[s][a] = reward + gamma_V

                if Q[s]:
                    if to_move == self.player_symbol:
                        # after the iteration we have to update delta as
                        # delta <-- max(delta, | max_{a in A} Q(s, a) - V(s) |)
                        max_a, argmax_a = utils.find_max_and_argmax_in_dict(Q[s])
                        delta = max(delta, np.abs(max_a - self.V[s]))

                        self.V[s] = max_a
                        self.policy[s] = tuple(random.choice(argmax_a))
                    else:
                        min_val = min(Q[s].values())
                        delta = max(delta, abs(min_val - self.V[s]))
                        self.V[s] = min_val
                        self.policy[s] = None
            
            if delta < theta:
                break

    def choose_action(self, board):
        state = utils.board_to_state(board)

        to_move = utils.to_move(utils.board_to_state(board))

        if to_move != self.player_symbol:
            return None
    
        return self.policy[state]

    def save_policy(self):
        with open(f'policy_{self.name}.pkl', 'wb') as fw:
            pickle.dump(self.policy, fw)

    def load_policy(self, file):
        with open(file, 'rb') as fr:
            self.policy = pickle.load(fr)


class HumanPlayer(PlayerBaseClass):
    def choose_action(self, board):
        positions = utils.get_available_positions(utils.board_to_state(board))
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
        self.current_player = utils.PlayerSymbol.X
    
    def switch_players(self):
        self.current_player = utils.PlayerSymbol.X if self.current_player == utils.PlayerSymbol.Y else utils.PlayerSymbol.Y
    
    def update_state(self, action):
        if action is None:
            raise ValueError("Action is not allowed!")
        if action in utils.get_available_positions(self.board):
            self.board[action] = self.current_player
            winner = utils.check_winner(self.board)

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


def main():
    # Training AI players
    player1 = Player("AI Player 1", utils.PlayerSymbol.X)
    player1.train_value_iteration()
    
    # Play against a human
    human_player = HumanPlayer("Human", utils.PlayerSymbol.Y)
    board = Board(player1, human_player)
    board.play()


if __name__ == "__main__":
    main()