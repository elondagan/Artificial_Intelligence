import pickle
import numpy as np
import matplotlib.pyplot as plt

from STRIPS.heuristics.lm_cut import LmCutHeuristic
from search.search_algorithms import Node
from utilities import structures

# _____________________________________________________________________________________________________________________


def train_A_star(problem, f):

    frontier = structures.PriorityQueue(order=min, f=f)
    frontier.push(Node(problem.I))
    explored = set()

    # for training
    # bfs_parameters = [0 for _ in range(50)]
    # bfs_parameters[0] = 1

    while frontier:
        # best node to expand
        node = frontier.pop()

        # bfs_parameters[node.path_cost] -= 1

        # goal check
        if problem.is_goal(node.state):
            return node, explored, frontier  # bfs_parameters
        # node expansion
        children = node.expand(problem)
        # bfs_parameters[node.path_cost + 1] += len(children)

        for child in children:
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


def get_good_bad_nodes(tasks, heuristic1, heuristic2, return_factor=False):
    """
    obtain two lists of GOOD and BAD nodes.
    each node also hold possible parameters for predicting the node type
    """

    good_nodes, bad_nodes = [], []
    factors = []

    for task in tasks:
        print("000")

        h1, h2 = heuristic1(task), heuristic2(task)
        f1 = lambda n: n.path_cost + h1(n)
        f2 = lambda n: n.path_cost + h2(n)

        if h2(Node(task.I)) == np.inf:
            continue

        res, explored, _ = train_A_star(task, f1)
        if res is None:
            continue
        real_cost = len(res.solution())

        h2_of_init = h2(Node(task.I))
        for n in explored:
            # fill nodes attributes (n.path_cost already exist)
            n.c_hat = h2_of_init
            n.h1 = int(h1(n))
            n.diff = int(h1(n.parent) - n.h1) if n.parent is not None else 0
            # split to good and bad and add node label
            if f2(n) >= real_cost:
                n.type = "good"
                n_f1 = n.h1 + n.path_cost
                factors.append(f2(n)-n_f1)
                good_nodes.append(n)
            else:
                n.type = "bad"
                bad_nodes.append(n)

    if return_factor:
        return good_nodes, bad_nodes, factors
    return good_nodes, bad_nodes


def plot_parameters(tasks, heuristic1, heuristic2):

    GOOD, BAD = get_good_bad_nodes(tasks, heuristic1, heuristic2)

    # plot 'path_cost' bar
    pc_labels = [str(i) for i in range(50)]
    pc_values = [[] for _ in range(50)]
    for n in GOOD:
        pc_values[n.path_cost].append(1)
    for n in BAD:
        pc_values[n.path_cost].append(0)

    empty_indexes = []
    for i in range(50):
        if len(pc_values[i]) == 0:
            empty_indexes.append(i)
    counter = 0
    for i in empty_indexes:
        pc_values.pop(i-counter)
        pc_labels.pop(i-counter)
        counter += 1

    pc_values = [np.mean(pv) for pv in pc_values]

    plt.bar(pc_labels, pc_values)
    plt.title("GOOD nodes ratio based on path cost")
    plt.xlabel("path cost from initial state")
    plt.ylabel("GOOD/BAD ratio")
    plt.show()

    #########################

    # plot 'h1 value' bar

    hv_labels = [str(i) for i in range(50)]
    hv_values = [[] for _ in range(50)]
    for n in GOOD:
        hv_values[n.h1].append(1)
    for n in BAD:
        hv_values[n.h1].append(0)

    empty_indexes = []
    for i in range(50):
        if len(hv_values[i]) == 0:
            empty_indexes.append(i)
    counter = 0
    for i in empty_indexes:
        hv_values.pop(i-counter)
        hv_labels.pop(i-counter)
        counter += 1

    hv_values = [np.mean(pv) for pv in hv_values]

    plt.bar(hv_labels, hv_values)
    plt.title("GOOD nodes ratio based on h1 value")
    plt.xlabel("h1 estimation")
    plt.ylabel("GOOD/BAD ratio")
    plt.show()

    # plot 'diff' bar
    diff_labels = [str(i) for i in range(-2, 3, 1)]
    diff_values = [[] for _ in range(-2, 3, 1)]
    # addon = int((len(diff_values)-1)/2)
    for n in GOOD:
        diff_values[n.diff+2].append(1)
    for n in BAD:
        diff_values[n.diff+2].append(0)

    empty_indexes = []
    for i in range(-2, 3, 1):
        if len(diff_values[i+2]) == 0:
            empty_indexes.append(i+2)
    counter = 0
    for i in empty_indexes:
        diff_values.pop(i-counter)
        diff_labels.pop(i-counter)
        counter += 1

    diff_values = [np.mean(dv) for dv in diff_values]

    plt.bar(diff_labels, diff_values)
    plt.title("GOOD nodes ratio based on parent/child h1 difference")
    plt.xlabel("h1(parent)-h1(child)")
    plt.ylabel("GOOD/BAD ratio")
    plt.show()



if __name__ == "__main__":
    print("good_bad_nodes module...")

    h_simaple = [lambda t: t.h_all_blank, lambda t: t.h_blocks, lambda t: t.h_trucks, lambda t: t.h_vacuum,
                 lambda t: t.h_elevator]
    h_lmc = LmCutHeuristic
    # h_lm = LandmarkHeuristic
    # h_m = hMaxHeuristic

    task_files_names = ['8-puzzle_tasks.pkl', '7-blocks_tasks.pkl', '3-2-3-trucks_tasks.pkl', '6-vacuum_tasks.pkl',
                        '10-5-3-elevator_tasks.pkl']
    models_names = ['8puzzle_lmc-blanky', '7blocks_lmc-h', '323trucks_lmc-h', '6vacuum_lmc_h', '10-5-3-elevator']
    domains_names = ['8-puzzle', '7-blocks', '3-2-3-trucks', '6-vacuum', '10-5-3-elevator']


    "plot model parameters"
    with open(f'D:/Desktop/AI_Research/STRIPS/tasks/train_tasks/6-vacuum_tasks.pkl', 'rb') as file:
        puzzle8_tasks = pickle.load(file)
    tasks_20to25 = []
    for i in [0, 6]:
        tasks_20to25.append(puzzle8_tasks[i])
    plot_parameters(tasks_20to25, h_simaple[3], h_lmc)


