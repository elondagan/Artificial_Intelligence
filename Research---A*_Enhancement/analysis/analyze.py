import ast
import pickle
import random
import time

import numpy as np
from sklearn.model_selection import train_test_split

from STRIPS import domains
from search.search_algorithms import GBFS, A_star, Node
from STRIPS.heuristics.relaxation_heuristics import hMaxHeuristic
from STRIPS.heuristics.lm_cut import LmCutHeuristic
from STRIPS.heuristics.landmarks import LandmarkHeuristic
from utilities import structures
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import confusion_matrix
# _____________________________________________________________________________________________________________________


def train_A_star(problem, f):
    """ A* algorithm fit for training """
    limit = 1.5 * 60
    st = time.time()
    bf = []

    frontier = structures.PriorityQueue(order=min, f=f)
    frontier.push(Node(problem.I))
    explored = set()
    while frontier and time.time()-st < limit:
        # best node to expand
        node = frontier.pop()
        # goal check
        if problem.is_goal(node.state):
            return node, explored, frontier, bf
        # node expansion
        children = node.expand(problem)
        bf.append(len(children))
        for child in children:
            if child not in explored and child not in frontier:
                frontier.push(child)
            elif child in frontier:
                existing_node = frontier[child]
                if child.path_cost < existing_node.path_cost:
                    del (frontier[existing_node])
                    frontier.push(child)
        explored.add(node)
    # no solution
    return None, None, None, None


# General Analysis

def heuristics_computation_time(tasks_files_names, heuristics, t=1):
    times = [[] for _ in range(len(heuristics))]

    for i, file_name in enumerate(tasks_files_names):

        with open(f"D:\Desktop\AI_Research\STRIPS/tasks/train_tasks/{file_name}", 'rb') as file:
            tasks = pickle.load(file)

        for j in range(5):
            cur_task = tasks[j]
            cur_h = heuristics[i](cur_task)
            _, _, _ = A_star(cur_task, cur_h)
            if t==1:
                times[i] += cur_task.times
            else:
                times[i] += cur_h.times
        times[i] = np.average(times[i])

    return times


# PLA* pre-analysis

def analyze_and_prepare_data(tasks, heuristic1, heuristic2, only_test=False):

    good_nodes, bad_nodes = [], []
    great_ratio = []
    factors = []
    bfs = []

    # find  GOOD & BAD nodes, fill parameters of each node, collect nodes factors
    for task in tasks:
        h1, h2 = heuristic1(task), heuristic2(task)
        f1, f2 = lambda n: n.path_cost + h1(n), lambda n: n.path_cost + h2(n)

        if h2(Node(task.I)) == np.inf:
            continue
        res, explored, frontier, cur_bf = train_A_star(task, f1)
        if res is None:
            continue
            print("none")
        print("...")
        bfs += cur_bf
        real_cost = len(res.solution())
        h2_of_init = h2(Node(task.I))

        explored_size, frontier_size = len(explored), len(frontier)
        great_ratio.append(frontier_size / (explored_size + frontier_size))

        for n in explored:
            h1_val, h2_val = h1(n), h2(n)
            f1_val, f2_val = n.path_cost + h1_val, n.path_cost + h2_val

            # fill nodes attributes (n.path_cost already exist)
            n.c_hat = h2_of_init
            n.h1 = h1_val  # --------- was int(h1(n)) for ploting parameters
            n.diff = int(h1(n.parent) - h1_val) if n.parent is not None else 0    # int(h1(n.parent) same as above

            # split to good and bad and add node label
            if f2_val >= real_cost:
                n.type = "good"
                good_nodes.append(n)
                factors.append(f2_val - f1_val)
            else:
                n.type = "bad"
                bad_nodes.append(n)

    # convert to train and test data
    num_of_attributes = 5
    train_data = good_nodes + bad_nodes
    X = np.zeros((len(train_data), num_of_attributes))
    y = np.zeros(len(train_data))

    for i, s in enumerate(train_data):
        X[i, 0] = 1  # bias
        X[i, 1] = s.c_hat  # h2 estimate to c*
        X[i, 2] = s.h1  # cheap heuristic value
        # X[i, 3] = s.path_cost  # g value
        X[i, 3] = 1  # g value
        X[i, 4] = s.diff  # h1 difference between n and n.parent
        y[i] = 1 if s.type == "good" else 0

    if only_test is True:
        return X, y

    tot_size = len(good_nodes) + len(bad_nodes)
    good_bad_ratio = len(good_nodes) / tot_size
    q_25, q_50, q_75 = np.quantile(factors, 0.25), np.quantile(factors, 0.5), np.quantile(factors, 0.75)

    print("")
    print(f"-"*5, f"Domain Results on {tot_size} nodes:")
    print(f"    good_nodes/all_nodes_ratio = {good_bad_ratio}")
    print(f"    factor:  0.25_quantile = {q_25},  0.5_quantile = {q_50},  0.75_quantile = {q_75},  mean = {np.mean(factors)}")
    print(f"    average branching factor = {np.mean(bfs)}")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return [X_train, y_train], [X_test, y_test], good_bad_ratio, np.mean(factors), np.mean(bfs), np.median(great_ratio)


def train_model(data):
    X_train, y_train = data[0], data[1]
    model = LogisticRegression()
    model.fit(X_train, y_train)
    return model


def test_model(model, X_test, y_test, threshold=0.5):

    if threshold == 0.5:
        y_pred = model.predict(X_test)
    else:
        proba_predictions = model.predict_proba(X_test)
        y_pred = (proba_predictions[:, 1] >= threshold).astype(int)

    # model results
    accuracy = metrics.accuracy_score(y_test, y_pred)
    precision = metrics.precision_score(y_test, y_pred)
    recall = metrics.recall_score(y_test, y_pred)

    confusion_mat = confusion_matrix(y_test, y_pred)
    test_size = len(y_test)
    TN, FP, FN, TP = confusion_mat.ravel()
    size_1 = sum([s == 1 for s in y_test])
    size_0 = sum([s == 0 for s in y_test])
    TP, FN = TP/size_1, FN/size_1
    TN, FP = TN/size_0, FP/size_0

    # TN, FP, FN, TP = TN/test_size, FP/test_size, FN/test_size, TP/test_size

    if threshold == 0.5:
        print(f"\n ----- Model Results (threshold={threshold})-----")
        print("    accuracy = ", accuracy)
        print("    precision = ", precision)
        print("    recall = ", recall)
        print(f"    TP={TP}, TN={TN}, FP={FP}, FN={FN}")
        print('    Coefficients:', model.coef_)
        print('    Intercept:', model.intercept_)
        print(f"    total train set size = {test_size}")
        print("\n")

    return [TP, TN, FP, FN]


# choose model threshold

def choose_threshold(model, X_test, y_test, domain_properties):

    possible_threshold = [np.round(1 - i * 0.05, 2) for i in range(1, 20)]
    can_win = False
    best_th = -1
    vs_results = [-1000]


    ts = domain_properties[0]
    bf = domain_properties[1]
    bf_exp = domain_properties[2]
    ratio = domain_properties[3]
    gr = domain_properties[4]

    for th in possible_threshold:
        properties = test_model(model, X_test, y_test, threshold=th)
        vs_h1, vs_h2, vs_LAstar = check_potential(ratio, properties, ts, bf, bf_exp, gr)
        print(vs_h1, vs_h2, vs_LAstar)
        # print(vs_h1, vs_h2, vs_LAstar)
        # print(f"sum = {vs_h1 + vs_h2 + vs_LAstar}")

        if vs_h1 > 0 and vs_h2 > 0 and vs_LAstar > 0:

            if can_win is False:  # first time that th can win, save it
                vs_results = [vs_h1, vs_h2, vs_LAstar]
                best_th = th
            else:   # won before, choose better one
                if min(vs_h1, vs_h2, vs_LAstar) > min(vs_results):
                    vs_results = [vs_h1, vs_h2, vs_LAstar]
                    best_th = th
            can_win = True

        elif can_win is False:

            cur_wins = sum([vs_h1 > 0, vs_h2 > 0, vs_LAstar > 0])
            prev_wins = sum([vsr > 0 for vsr in vs_results])

            if cur_wins > prev_wins or (cur_wins == prev_wins and vs_h1 + vs_h2 + vs_LAstar > sum(vs_results)):
                vs_results = [vs_h1, vs_h2, vs_LAstar]
                best_th = th


    print("-"*5, f" Best Predicted Results for th={best_th}:", "-"*5)
    print(f"    VS h1 = {vs_results[0]:.4f}")
    print(f"    VS h2 = {vs_results[1]:.4f}")
    print(f"    VS LA* = {vs_results[2]:.4f}")
    # return best_th, vs_results


def check_potential(good_bad_ratio, model_properties, ts, bf, bf_exp, great_ratio):
    """
    :param good_bad_ratio: #good_nodes / #good_nodes+#bad_nodes
    :param model_properties: [TP, TN, FP, FN]
    :param ts: [t1, t2] - heuristics computation time
    :param bf: domain average branching factor
    :param bf_exp:
    :return:
    """

    p = good_bad_ratio
    TP, TN, FP, FN = model_properties[0], model_properties[1], model_properties[2], model_properties[3]
    t1, t2 = ts[0], ts[1]

    vs_LAstar = p * TP * (0) + p * FN * (t2 - bf * t1) + (1 - p) * FP * (0) + (1 - p) * TN * (t2)

    return 1, 1, vs_LAstar


def heuristics_wins(domain, h_functions):

    count_limit = 0
    scores = [0, 0, 0]

    while count_limit < 1500:
        print(count_limit)

        # define random problem
        if domain == 'Npuzzle':
            task = domains.Npuzzle(8, '123456780')
            new_i = task.I
            for _ in range(50):
                new_i = task.result(new_i, random.choice(task.actions(new_i)))
            task.I = new_i
        elif domain == 'Npuzzle-2b':
            size = 8
            init = ''
            g = []
            for i in range(1, size):
                init += str(i) + '.'
                g.append(f'at({i},{i})')
            init += '0.0'
            task = domains.Npuzzle(size, init)
            task.G = g
            new_i = task.I
            for _ in range(50):
                new_i = task.result(new_i, random.choice(task.actions(new_i)))
            task.I = new_i
        elif domain == 'BlocksWorld':
            task = domains.BlocksWorld(6)
        elif domain == 'VacuumCleaner':
            task = domains.VacuumCleaner(6)
        else:  # elif domain == 'Trucks':
            task = domains.Trucks(4, 1, 3)

        # define heuristics
        hs = [h(task) for h in h_functions]

        # find path from I to G
        res = GBFS(task, lambda n: hs[0](n))
        if res is None:
            continue

        for node in res.path():
            count_limit += 1

            h_1, h_2 = hs[0](node), hs[1](node)
            if h_1 > h_2:
                scores[0] += 1
            elif h_1 == h_2:
                scores[1] += 1
            else:
                scores[2] += 1

    return scores


def find_expanding_factor(domain, heuristic):
    """ use LM_cut heuristic to find shortest path from random state, then using this path to
        create a histogram for each 'heuristic' estimation that keep the real distance"""

    hist = {f"h={h}": [] for h in range(1, 30)}
    ef = {f"h={h}": [] for h in range(1, 30)}

    samples_counter = 0
    while samples_counter < 600:

        # create random task
        if domain == 'Npuzzle':
            task = domains.Npuzzle(8, '123456780')
            new_i = task.I
            for _ in range(50):
                new_i = task.result(new_i, random.choice(task.actions(new_i)))
            task.I = new_i
        elif domain == 'Npuzzle-2b':
            size = 15
            init = ''
            g = []
            for i in range(1, size):
                init += str(i) + '.'
                g.append(f'at({i},{i})')
            init += '0.0'
            task = domains.Npuzzle(size, init)
            task.G = g
            new_i = task.I
            for _ in range(50):
                new_i = task.result(new_i, random.choice(task.actions(new_i)))
            task.I = new_i
        elif domain == 'BlocksWorld':
            task = domains.BlocksWorld(6)
        elif domain == 'VacuumCleaner':
            task = domains.VacuumCleaner(6)
        else:  # elif domain == 'Trucks':
            task = domains.Trucks(4, 1, 3)

        # define f function and h
        h_lmc = LmCutHeuristic(task)
        f = lambda x: x.path_cost + h_lmc(x)
        h = heuristic(task)

        # find shorted path to solution
        res, _, _, _ = A_star(task, f)

        c_star = len(res.solution())  # real solution cost
        for n in res.path():
            real_cost_left = c_star - n.path_cost
            if real_cost_left == 0:
                continue
            estimated_cost_left_lmc = h_lmc(n)
            estimated_cost_left_blank = h(n)

            hist[f'h={int(estimated_cost_left_lmc)}'].append(real_cost_left / estimated_cost_left_lmc)
            ef[f'h={int(estimated_cost_left_blank)}'].append(estimated_cost_left_lmc / estimated_cost_left_blank)
            samples_counter += 1

        print(samples_counter)

    # compute the 'expanding factor'
    for c, key in enumerate(hist):
        if len(ef[key]) == 0:
            ef[key] = [1]
        ef[key] = [np.quantile(ef[key], 0.25), np.quantile(ef[key], 0.5), np.quantile(ef[key], 0.75)]
        if len(hist[key]) == 0:
            hist[key] = [1]
        hist[key] = [np.quantile(hist[key], 0.25), np.quantile(hist[key], 0.5), np.quantile(hist[key], 0.75)]


    print(hist)
    print(ef)


if __name__ == '__main__':

    task_files_names = ['8-puzzle_tasks.pkl', '7-blocks_tasks.pkl', '3-2-3-trucks_tasks.pkl', '6-vacuum_tasks.pkl',
                        '10-5-3-elevator_tasks.pkl']
    domains_names = ['8-puzzle', '7-blocks', '3-2-3-trucks', '6-vacuum', '10-5-3-elevator']

    h_lmc = LmCutHeuristic
    h_lm = LandmarkHeuristic
    h_m = hMaxHeuristic
    h_simaple = [lambda t: t.h_all_blank, lambda t: t.h_blocks, lambda t: t.h_trucks, lambda t: t.h_vacuum,
                 lambda t: t.h_elevator]
    # h_simaple = [hMaxHeuristic for _ in range(5)]
    # h_simaple = [LandmarkHeuristic for _ in range(5)]
    h2 = LmCutHeuristic

    " heuristic computation time"
    h1s_ct = heuristics_computation_time(task_files_names, h_simaple, t=1)
    h2s_ct = heuristics_computation_time(task_files_names, [h2 for _ in range(len(domains_names))], t=2)
    ts = []
    for i in range(len(domains_names)):
        ts.append([h1s_ct[i], h2s_ct[i]])
        # ts.append([1, h2s_ct[i] / h1s_ct[i]])
        print(f"{domains_names[i]}: {ts[i]}]   | t2/t1 = {ts[i][1]/ts[i][0]}")
        break

    " Train "
    for i, (fn, dn) in enumerate(zip(task_files_names, domains_names)):

        print("="*20, f"{domains_names[i]}", "="*20)

        # create data
        with open(f'D:/Desktop/AI_Research/STRIPS/tasks/train_tasks/{fn}', 'rb') as file:
            tasks = pickle.load(file)
        train_data, test_data, gbr, factor, bf, gr = analyze_and_prepare_data(tasks[:15], h_simaple[i], h2)
        factor = max(1, factor)

        # train and save model
        model = train_model(train_data)
        with open(f'LogisticRegression_model__{dn}.pkl', 'wb') as file:
            pickle.dump(model, file)

        # test and choose threshold
        choose_threshold(model, test_data[0], test_data[1], [ts[i], bf, factor, gbr, gr])

        print("="*50)


