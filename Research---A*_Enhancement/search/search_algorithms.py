import numpy as np
import time

from utilities import structures, utils



""" Search Auxiliary Class """


class Node:
    """A node in a search tree"""

    def __init__(self, state, parent=None, action=None, path_cost=0):
        self.state = state
        self.state_str = str(sorted(list(state)))
        self.parent = parent
        self.action = action
        self.path_cost = path_cost
        self.depth = 0
        self.order = 0
        if parent is not None:
            self.depth = parent.depth + 1
        self.unreached = set()

    def child_node(self, problem, action):
        next_state = problem.result(self.state, action)
        return Node(next_state, self, action, problem.path_cost(self.path_cost, self.state, action, next_state))

    def expand(self, problem):
        """List the nodes reachable in one step from this node."""
        return [self.child_node(problem, action) for action in problem.actions(self.state)]

    def path(self):
        """Return a list of nodes forming the path from the root to this node."""
        node, path_back = self, []
        while node is not None:
            path_back.append(node)
            node = node.parent
        return list(reversed(path_back))

    def solution(self):
        """Return the sequence of actions to go from the root to this node."""
        return [node.action for node in self.path()[1:]]

    def __repr__(self):
        return "<Node {}>".format(self.state)

    def __lt__(self, node):
        return self.state_str < node.state_str

    def __eq__(self, other):
        return isinstance(other, Node) and self.state_str == other.state_str

    def __hash__(self):
        return hash(self.state_str)


""" Search Algorithms """

search_time_limit = 1.5 * 60


def GBFS(problem, h):
    start_time = time.time()

    frontier = structures.PriorityQueue(f=h)
    frontier.push(Node(problem.I))
    explored = set()

    while frontier and time.time() - start_time < search_time_limit:
        # best node to expand
        node = frontier.pop()
        # goal check
        if problem.is_goal(node.state):
            return node
        # node expansion
        for child in node.expand(problem):
            if child not in explored and child not in frontier:
                frontier.push(child)
        explored.add(node)
    return None


def A_star(problem, f):
    start_time = time.time()

    frontier = structures.PriorityQueue(order=min, f=f)
    frontier.push(Node(problem.I))
    explored = set()

    while frontier and time.time() - start_time < search_time_limit:
        # best node to expand
        node = frontier.pop()
        # goal check
        if problem.is_goal(node.state):
            return node, len(explored), len(frontier)
        # node expansion
        for child in node.expand(problem):
            if child not in explored and child not in frontier:
                frontier.push(child)
            # avoid duplicates - we already reach this state, possibly in longer path
            elif child in frontier:
                existing_node = frontier[child]
                if child.path_cost < existing_node.path_cost:
                    del (frontier[existing_node])
                    frontier.push(child)
        explored.add(node)

    # no solution
    return None, None, None


def lazy_Astar(problem, f_functions):
    """ f_function[0] is cheaper than f_function[1]"""
    start_time = time.time()

    frontier = structures.LazyAstarQueue()
    explored = set()
    node = Node(problem.I)
    node.fs = f_functions[0](node)
    node.order = 0
    frontier.push(node)
    used_2 = 0

    while frontier and time.time() - start_time < search_time_limit:
        # best node to expand
        node = frontier.pop()
        # goal check
        if problem.is_goal(node.state):
            return node, len(explored), len(frontier), used_2

        # improve f
        if node.order == 0:
            node.fs = f_functions[1](node)
            node.order = 1
            frontier.push(node)
            used_2 += 1

        # node expansion
        else:
            for child in node.expand(problem):
                if child not in explored and child not in frontier:
                    child.fs = f_functions[0](child)
                    child.order = 0
                    frontier.push(child)
                elif child in frontier:
                    existing_node = frontier[child]
                    if child.path_cost < existing_node.path_cost:
                        del (frontier[existing_node])
                        if child.order == 0:
                            child.fs = f_functions[0](child)
                            child.order = 0
                        else:
                            child.fs = f_functions[1](child)
                            child.order = 1
                        frontier.push(child)
            explored.add(node)

    # no solution
    return None, None, None, None


def rational_lazy_Astar(problem, f_functions, params):
    """ f_function[0] is cheaper than f_function[1]"""
    start_time = time.time()

    p_init = params[0]
    k = params[1]
    ts_ratio = params[2][0] / params[2][1]  # t1/t2
    bf = params[3]  # problem's branching factor

    frontier = structures.LazyAstarQueue()
    explored = set()
    node = Node(problem.I)
    node.fs = f_functions[0](node)
    node.order = 0
    node.h2 = False

    A = 0  # number of nodes that used h2 and are still in frontier
    B = 0  # number of nodes that used h2 (total)

    frontier.push(node)

    while frontier and time.time() - start_time < search_time_limit:
        # best node to expand
        node = frontier.pop()
        # goal check
        if problem.is_goal(node.state):
            return node, len(explored), len(frontier)

        if node.h2 is True:
            A -= 1

        # improve f
        if node.order == 0:
            # decision rule for computing h2
            updated_p = (A + p_init*k)/(B+k)
            if ts_ratio < (updated_p*bf)/(1-updated_p*bf):
                A += 1
                B += 1
                node.fs = f_functions[1](node)
                node.order = 1
                frontier.push(node)
                continue

        # node expansion
        for child in node.expand(problem):
            if child not in explored and child not in frontier:
                child.fs = f_functions[0](child)
                child.order = 0
                frontier.push(child)
            elif child in frontier:
                existing_node = frontier[child]
                if child.path_cost < existing_node.path_cost:
                    del (frontier[existing_node])
                    if child.order == 0:
                        child.fs = f_functions[0](child)
                        child.order = 0
                    else:
                        child.fs = f_functions[1](child)
                        child.order = 1
                    frontier.push(child)
        explored.add(node)

    # no solution
    return None, None, None


def predictive_lazy_Astar(problem, f_functions, params):
    """ f_function[0] is cheaper than f_function[1]"""
    start_time = time.time()

    model = params[0]
    threshold = params[1]

    def calculate(x):
        p = model.predict_proba(x.reshape(1, -1))
        p = 1 - p[0, 0]
        return p > threshold

    counter = 0
    c_hat = f_functions[1](Node(problem.I))

    frontier = structures.LazyAstarQueue()
    explored = set()
    node = Node(problem.I)
    node.fs = f_functions[1](node)
    node.order = 1
    node.used = 2
    frontier.push(node)
    used_2 = 0

    while frontier and time.time() - start_time < search_time_limit:
        # best node to expand
        node = frontier.pop()
        # goal check
        if problem.is_goal(node.state):
            return node, len(explored), len(frontier), used_2

        # improve f procedure
        if node.order == 0:  # only f[0] was used
            node.order = 1
            parent_h1 = node.parent.fs - node.parent.path_cost
            h1 = node.fs - node.path_cost
            if calculate(np.array([1, c_hat, h1, node.path_cost, parent_h1 - h1])):
                counter += 1
                node.fs = f_functions[1](node)
                node.used = 2
                frontier.push(node)
                used_2 += 1
                continue

        # node expansion
        for child in node.expand(problem):
            if child not in explored and child not in frontier:
                child.fs = f_functions[0](child)
                child.order = 0
                frontier.push(child)
            elif child in frontier:
                existing_node = frontier[child]
                if child.path_cost < existing_node.path_cost:
                    del (frontier[existing_node])
                    if child.order == 0:
                        child.fs = f_functions[0](child)
                        child.order = 0
                    elif child.used == 2:
                        child.fs = f_functions[1](child)
                        child.order = 1
                    frontier.push(child)
        explored.add(node)

    # no solution
    return None, None, None, None




if __name__ == '__main__':
    print("search module . . .")


