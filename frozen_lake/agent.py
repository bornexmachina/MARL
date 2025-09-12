import random
from enums import Actions
from environment import Environment
import constants as c


class Agent:
    def __init__(self,
                 name: str="AI_Agent", 
                 gamma: float=0.9,
                 theta: float=1E-6,
                 env: Environment=Environment(c.LAKE_SMALL, c.REWARD_MAP)):
        self.name = name
        self.gamma = gamma
        self.theta = theta
        self.env = env

    def initialize_V(self) -> None:
        all_states = self.env.all_states()
        self.V = {state: 0 for state in all_states}

    def initialize_Q(self) -> None:
        all_states = self.env.all_states()
        all_actions = Actions.get_actions()
        self.Q = {state: {action: 0.0 for action in all_actions} for state in all_states}

    def initialize_policy(self) -> None:
        all_states = self.env.all_states()
        self.policy = {state: None for state in all_states}

    def optimal_trajectory(self, position: tuple[int, int]=(0, 0)) -> list[tuple[int, int]]:
        trajectory = [position]
        while not self.env.is_terminal(position):
            action = self.policy[position]
            position = self.env.take_action(action, position)
            trajectory.append(position)
        return trajectory

    def value_iteration(self) -> None:
        self.initialize_V()
        self.initialize_policy()
        delta = float('inf')

        all_actions = Actions.get_actions()

        while delta > self.theta:
            delta = 0
            V_new = self.V.copy()

            for position in self.V.keys():
                old_v = self.V[position]  # Store the old value

                if self.env.is_terminal(position):
                    new_v = self.env.get_instantaneous_reward(position)
                else:
                    max_action_value = float('-inf')
                    best_action = None
                    
                    for action in all_actions:
                        new_position = self.env.take_action(action, position)
                        new_value_from_action = 0 + self.gamma * self.V[new_position]

                        if new_value_from_action > max_action_value:
                            max_action_value = new_value_from_action
                            best_action = action
                    
                    new_v = max_action_value
                    self.policy[position] = best_action

                # Update the new value and calculate delta here
                V_new[position] = new_v
                delta = max(delta, abs(old_v - new_v))
            
            self.V = V_new
        
    def action_value_iteration(self) -> None:
        self.initialize_V()
        self.initialize_Q()
        self.initialize_policy()
        delta = float('inf')

        while delta > self.theta:
            delta = 0
            for position in self.Q.keys():
                old_v = self.V[position]

                if self.env.is_terminal(position):
                    new_v = self.env.get_instantaneous_reward(position)
                    for action in self.Q[position].keys():
                        self.Q[position][action] = None
                        self.policy[position] = None
                else:
                    for action in self.Q[position].keys():
                        new_position = self.env.take_action(action, position)
                        self.Q[position][action] = 0 + self.gamma * self.V[new_position]
                    new_v = max(self.Q[position].values())
                    self.policy[position] = max(self.Q[position], key=self.Q[position].get)

                self.V[position] = new_v
                delta = max(delta, abs(old_v - new_v))
                

    def epsilon_greedy_action_value_iteration(self, eps=0.1, alpha=0.1) -> None:
        self.initialize_Q()
        self.initialize_policy()
        delta = float('inf')

        while delta > self.theta:
            delta = 0
            for position in self.Q.keys():
                current_q = max(self.Q[position].values())
                # in terminal state there are no actions to perform
                # just collect the reward
                if self.env.is_terminal(position):
                    new_q = self.env.get_instantaneous_reward(position)
                    for action in self.Q[position].keys():
                        self.Q[position][action] = None
                        self.policy[position] = None
                else:
                    # with the probability epsilon we take a random action
                    if random.random() < eps:
                        action = Actions.sample()
                    # with the probability 1 - epsilon we take the arg max action
                    else:
                        action = max(self.Q[position], key=self.Q[position].get)

                    new_position = self.env.take_action(action, position)
                    new_q = self.Q[position][action] + alpha * (0.0 + self.gamma * max(self.Q[new_position].values()) - self.Q[position][action])

                self.Q[position][action] = new_q
                self.policy[position] = max(self.Q[position], key=self.Q[position].get)
                delta = max(delta, abs(current_q - max(self.Q[position].values())))
                