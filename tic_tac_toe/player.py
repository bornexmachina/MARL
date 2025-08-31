from abc import ABC, abstractmethod
import utils
import random
import pickle
import numpy as np


class PlayerBaseClass(ABC):
    @abstractmethod
    def next_state(self, state, action, player_symbol):
        pass

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

    @abstractmethod
    def choose_action(self, board):
        pass

    @abstractmethod
    def train(self):
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

    def train(self):
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
        theta=1e-6

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


class PlayerValueIteration(PlayerBaseClass):
    """
    Implementation of Value Iteration
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

    def train(self):
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
        theta=1e-6

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
                        reward = self.next_state(s, a, self.player_symbol)
                        s_ = self.reward(s, self.player_symbol)
                    else:
                        reward_opp = self.next_state(s, a, utils.opponent(self.player_symbol))
                        s_ = self.reward(s, utils.opponent(self.player_symbol))
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