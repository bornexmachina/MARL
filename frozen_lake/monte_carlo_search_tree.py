# implement monte carlo search tree used in RL
# we follow the implementation of gibberblot
# we will need a node for the trees
# and MCTS itself

class Node:
    def __init__(self):
        pass

    def select(self):
        pass

    def expand(self):
        pass

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