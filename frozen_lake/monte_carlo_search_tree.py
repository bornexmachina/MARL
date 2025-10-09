# implement monte carlo search tree used in RL
# we follow the implementation of gibberblot
# we will need a node for the trees
# and MCTS itself
import random
from enums import Actions
from environment import Environment as env


class Node:
    def __init__(self, state, env, parent):
        """
        we can take a short cut --> our actions will deterministically transition to a state
        thus instead of actual children we could simply track which actions we have already taken
        WOULD NOT WORK IN PROBABILISTIC SETTING
        e.g. if you select UP and with 20% you stay in the state you have NOT explored the child
        """
        self.state = state
        self.env = env
        self.parent = parent
        self.explored_actions = set()
        self.children = set()

    def is_expanded(self):
        """
        A node is fully expanded if all of its actions has been taken
        In my implementation a node can be its own child, as an action stepping out of the environment,
        will reset the agent to the same position
        """
        return len(self.children) == len(self.env.all_actions_in_state())

    def select(self, eps=0.1):
        """
        If the node has not explored all options, i.e. is not expanded or it is a leaf node we stay in the node
        Otherwise we choose to EXPLOIT --> we check which actions have already been taken, i.e. which children we have visited
        and then take either random or epsilon greedy action
        """
        if not self.is_expanded() or self.env.is_terminal(self):
            return self
        else:
            action = Actions.sample()
            state = self.env.take_action(action, state)
            return Node(state, env, self)

    def expand(self, state):
        """
        If the node is a leaf node we do not expand - simply return the node
        Otherwise we choose to EXPLORE 
        This means, check which of the children have not been visited
        """
        if not self.env.is_terminal(state):
            not_explored_actions = self.env.all_actions_in_state() - self.explored_actions
            action = random.choice(tuple(not_explored_actions))
            state = self.env.take_action(action, state)
            return Node(state, env, self)
        return self

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