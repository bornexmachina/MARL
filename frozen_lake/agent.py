import random
import numpy as np
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
                

    def epsilon_greedy_action_value_iteration(self, init_position=(0,0), eps=0.1, alpha=0.1, num_episodes=100_000) -> None:
        self.initialize_Q()
        self.initialize_policy()
        
        for i in range(num_episodes):
            position = init_position
            while not self.env.is_terminal(position):
                if random.random() < eps:
                    action = Actions.sample()
                else:
                    action = max(self.Q[position], key=self.Q[position].get)

                new_position = self.env.take_action(action, position)

                current_q = self.Q[position][action]
                reward = self.env.get_instantaneous_reward(new_position)
                max_next_q = max(self.Q[new_position].values())

                self.Q[position][action] = current_q + alpha * (reward + self.gamma * max_next_q - current_q)

                position = new_position

        for state in self.Q.keys():
            if not self.env.is_terminal(state):
                self.policy[state] = max(self.Q[state], key=self.Q[state].get)
            else:
                self.policy[state] = None

    def has_policy_changed(policy_old, policy_new):
        return policy_old == policy_new
    
    def policy_evaluation(self, policy, V):
        delta = float('inf')
        while delta > self.theta:
            delta = 0
            V_local = V.copy()
            for position in V_local.keys():
                old_v = V_local[position]

                action = policy[position]

                if self.env.is_terminal(position):
                    new_v = self.env.get_instantaneous_reward(position)
                    policy[position] = None
                else:
                    new_position = self.env.take_action(action, position)
                    new_v = 0 + self.gamma * V_local[new_position]

                V_local[position] = new_v
                delta = max(delta, abs(old_v - new_v))
            V = V_local
        
        return V

    def policy_iteration(self, max_iters=100):
        self.initialize_V()
        self.initialize_Q()
        # Policy evaluation
        # for each state do
        # V[s] = Sum_{a}P(a|s) Sum_{s'} [r + gamma * V[s']]
        # let the initial policy be uniform
        policy = {s: Actions.sample() for s in self.env.all_states()}

        for i in range(max_iters):            
            V = self.value_iteration(policy, self.V)

            # Policy improvement
            # for each non-terminal state
            # Q[s, a] = Sum_{s'} P[s'|s, a] [r + gamma * V[s']]
            # pick argmax a
            # update the policy
            # this time make stochastic actions if argmax returns more than one
            new_policy = policy.copy()

            for position in V.keys():
                if self.env.is_terminal(position):
                    new_policy[position] = None
                else:
                    for action in policy[position]:
                        new_position = self.env.take_action(action, position)
                        self.Q[position][action] = 0 + self.gamma * V[new_position]


                    max_val = max(self.Q[position].values())
                    max_actions = [k for k, v in self.Q[position].items() if v == max_val]
                    new_policy[position] = max_actions

            # check if the policy is stable
            policy_stable = self.has_policy_changed(policy, new_policy)
            if policy_stable:
                print(f"*** Policy Iteration converged at iteration: {i + 1}", flush=True)
                break

            policy = new_policy
