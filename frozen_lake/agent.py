from environment import Environment as env, Actions
import constants as c


class Agent:
    def __init__(self, name="AI_Agent", gamma=0.9, theta=1E-6):
        self.name = name
        self.gamma = gamma
        self.theta = theta

    def initialize_V(self):
        all_states = env.all_states()
        self.V = {state: 0 for state in all_states}

    def initialize_Q(self):
        all_states = env.all_states()
        all_actions = Actions.get_actions()
        self.Q = {state: {action: 0.0 for action in all_actions} for state in all_states}

    def initialize_policy(self):
        all_states = env.all_states()
        self.policy = {state: None for state in all_states}

    def value_iteration(self):
        self.initialize_V()
        self.initialize_policy()
        delta = float('inf')

        while delta > self.theta:
            for position in self.V.keys():
                # if the state is terminal there are no actions anymore
                # --> just give the final reward
                if env.is_terminal(position):
                    self.V[position] = env.get_instantaneous_reward(position)
                else:
                    all_actions = Actions.get_actions()
                    max_action = float('-inf')
                    best_action = None

                    for action in all_actions:
                        new_position = env.take_action(action, position)
                        new_value = env.get_instantaneous_reward(new_position) + self.gamma * self.V[new_position]

                        if new_value > max_action:
                            max_action = new_value
                            best_action = action

                    delta = max(delta, abs(self.V[position] - max_action))
                    self.policy[position] = best_action

    def action_value_iteration(self):
        self.initialize_V()
        self.initialize_Q()
        self.initialize_policy()
        delta = float('inf')

        while delta > self.theta:
            for position in self.Q.keys():
                if env.is_terminal(position):
                    reward = env.get_instantaneous_reward(position)
                    for action in self.Q[position].keys():
                        self.Q[position][action] = reward
                else:
                    for action in self.Q[position].keys():
                        new_position = env.take_action(action, position)
                        self.Q[position][action] = env.get_instantaneous_reward(new_position) + self.gamma * self.V[new_position]
                delta = max(delta, max(self.Q[position].values()) - self.V[position])
                self.policy[position] = max(self.Q[position], key=self.Q[position].get)