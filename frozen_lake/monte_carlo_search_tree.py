# implement monte carlo search tree used in RL
# we follow the implementation of gibberblot
# we will need a node for the trees
# and MCTS itself
import random
from enums import Actions
from environment import Environment as env


class Node:
    def __init__(self):
        pass

    def is_expanded(self):
        pass

    def is_terminal(self, state):
        pass

    def select(self, eps=0.1):
        """
        While state s is fully expanded do:
            Select action a to apply in s using a multi-armed bandit algorithm
            Choose one outcome s' according to P(s'|s)
            s <- s'
        return s
        """
        if not self.is_expanded() or self.is_terminal():
            return self
        else:
            action = Actions.sample()
            state = env.take_action(action, state)
            return state


    def expand(self, state):
        """
        if state s is fully expanded then
            randomly select action a to apply in s
            expand one outcome state s' according to P(s'|s) and observe reward r
        return s'
        """
        if not env.is_terminal(state):
            action = Actions.sample()
            state = env.take_action(action, state)
        return state

    def back_propagate(self):
        pass

    def get_visits(self):
        pass

class MCTS:
    """
    1) Q-value function approximation
    2) ExpectiMax search tree
    3) Computational budget as hard limit
    4) Best performing action within the budget
    """
    def __init__(self):
        pass

    def mcts(self):
        """
        while current_time < T do:
            selected_node <- Select(s_0)
            child <- Expand(selected_node)
            G <- Simulate(child)
            Backpropagate(selected_node, child, Q, G)
        """
        pass

    def create_root_node(self):
        pass

    def simulate(self):
        pass