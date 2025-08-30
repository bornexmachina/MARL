from abc import ABC, abstractmethod
import utils
import random
import pickle


class PlayerBaseClass(ABC):
    @abstractmethod
    def next_state(self, state, action, player_symbol):
        pass

    @abstractmethod
    def reward(self, state, player_symbol):
        pass

    @abstractmethod
    def choose_action(self, board):
        pass

    @abstractmethod
    def train(self, theta=1e-6):
        pass

    @abstractmethod
    def save_policy(self):
        pass

    @abstractmethod
    def load_policy(self, file):
        pass


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


class PlayerMaxMin(PlayerBaseClass):
    """
    Implementation of Q-learning with a state history and backpropagation of the reward
    """
    def __init__(self, name, player_symbol, gamma=1.0):
        super().__init__(name, player_symbol)
        self.gamma = gamma
        self.V = {}
        self.policy = {}

    def next_state(self, state, action, player_symbol):
        board = utils.state_to_board(state).copy()
        board[action] = player_symbol
        return utils.board_to_state(board)
    
    def reward(self, state, player_symbol):
        winner = utils.check_winner(state)
        if winner == player_symbol:
            return 1
        elif winner == utils.PlayerSymbol.EMPTY:
            return 0
        elif winner is None:
            return 0
        else:
            return -1

    def train(self, theta=1e-6):
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
        states = utils.generate_all_states()

        actions_of_x = {s: utils.get_available_positions(s) for s in states}
        self.V = {s: 0.0 for s in states}
        self.policy_player_x = {}
        self.policy_player_y = {}


        while True:
            delta = 0
            for state in states:
                # if we already have a terminal state, we do not make any actions
                winner = utils.check_winner(state)

                if winner is not None:
                    # in a terminal state there is no s' as we DO NOT transition
                    # by convention we assume V[s'] is then 0 --> gamma * V[s'] = 0
                    self.V[state] = self.reward(state, utils.PlayerSymbol.X)
                    continue
                
                max_value_x = {}

                for action_of_x in actions_of_x[state]:
                    # by taking action action_of_x from policy pi_x we arrive at state s'
                    # in s' the opponent will have a new set of reachable sets and thus will take an action action_of_y from policy pi_y
                    state_after_one_turn = self.next_state(state, action_of_x, utils.PlayerSymbol.X)

                    # the game might be over after action_of_x --> check for it
                    if utils.check_winner(state_after_one_turn) is not None:
                        value_x = self.reward(state_after_one_turn, utils.PlayerSymbol.X)
                    # game is still going on
                    else:
                        actions_of_y = utils.get_available_positions(state_after_one_turn)
                        min_value_y = {}

                        for action_of_y in actions_of_y:
                            state_after_two_turns = self.next_state(state_after_one_turn, action_of_y, utils.PlayerSymbol.Y)
                            
                            if utils.check_winner(state_after_two_turns) is not None:
                                value_y = self.reward(state_after_one_turn, utils.PlayerSymbol.Y)
                            else:
                                value_y = self.reward(state_after_one_turn, utils.PlayerSymbol.Y) + self.gamma * self.V.get(state_after_two_turns, 0.0)

                            min_value_y[action_of_y] = value_y
                            
                        value_x = min(min_value_y.values())
                        self.policy_player_y[state_after_one_turn] = value_y

                    max_value_x[action_of_x] = value_x

                v_old = self.V[state]
                self.V[state] = max(max_value_x.values())
                self.policy_player_x[state] = tuple(random.choice(utils.get_max_action(max_value_x)))

                delta = max(delta, abs(v_old - self.V[state]))

            if delta < theta:
                break

    def choose_action(self, board):
        state = utils.board_to_state(board) 
        return self.policy_player_x[state]

    def save_policy(self):
        with open(f'policy_{self.name}.pkl', 'wb') as fw:
            pickle.dump(self.policy_player_x, fw)

    def load_policy(self, file):
        with open(file, 'rb') as fr:
            self.policy_player_x = pickle.load(fr)