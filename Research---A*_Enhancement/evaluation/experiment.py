import os
import time
import csv
import pickle
import numpy as np

from utilities.utils import create_csv_file
from STRIPS.heuristics.lm_cut import LmCutHeuristic
from STRIPS.heuristics.landmarks import LandmarkHeuristic
from STRIPS.heuristics.relaxation_heuristics import hMaxHeuristic
from search.search_algorithms import A_star, lazy_Astar, predictive_lazy_Astar, Node
from STRIPS import domains



def evaluate(filename, tasks, heuristics, param, include_variant=False):

    h_names = [heuristics[0][0], heuristics[1][0]]
    if include_variant:
        h_names.append("LA")
        h_names.append("PLA")

    # create csv file for the results (if not already existed)
    headline = ['problem I', 'problem G', 'sol len']
    for f in h_names:
        headline.append(f'{f}_time')
        headline.append(f'{f}_explored')
        headline.append(f'{f}_frontier')
        if f == "LA" or f == "PLA":
            headline.append(f'{f}_used2')
    if not os.path.exists(f'{filename}.csv'):
        create_csv_file(filename, headline)


    # iterate all tasks
    for ite, task in enumerate(tasks):

        h1 = heuristics[0][1](task)
        h2 = heuristics[1][1](task)
        f1 = lambda n: n.path_cost + h1(n)
        f2 = lambda n: n.path_cost + h2(n)

        f_functions = [f1, f2, [f1, f2]]

        # skip unsolvable tasks
        if f_functions[1](Node(task.I)) == np.inf:
            continue

        # solve with A*
        new_row = task.task_rep()
        for i, f in enumerate(f_functions[:-1]):
            # solve task with given f
            st = time.time()
            res, ex, fr = A_star(task, f)
            st = time.time() - st
            # update
            if res is None:
                if i == 0:
                    new_row.append(None)
                new_row.append(None)
                new_row.append(None)
                new_row.append(None)
            else:
                if i == 0:
                    new_row.append(str(len(res.solution())))
                else:
                    if new_row[2] is not None and new_row[2] != str(len(res.solution())):
                        print(new_row[2])
                        print(str(len(res.solution())))
                        print("oops")
                        exit(0)

                new_row.append(str(st))
                new_row.append(str(ex))
                new_row.append(str(fr))

        # solve with LA*
        if include_variant is True:
            st = time.time()
            res, ex, fr, u2 = lazy_Astar(task, f_functions[-1])
            st = time.time() - st
            # update
            if res is None:
                new_row.append(None)
                new_row.append(None)
                new_row.append(None)
                new_row.append(None)
            else:
                if new_row[2] is not None and new_row[2] != str(len(res.solution())):
                    exit(0)
                new_row.append(str(st))
                new_row.append(str(ex))
                new_row.append(str(fr))
                new_row.append(str(u2))

        # solve with my A* variant ---
        if include_variant is True:
            st = time.time()
            res, ex, fr, u2 = predictive_lazy_Astar(task, f_functions[-1], param)
            st = time.time() - st
            # update
            if res is None:
                new_row.append(None)
                new_row.append(None)
                new_row.append(None)
                new_row.append(None)
            else:
                if new_row[2] is not None and new_row[2] != str(len(res.solution())):
                    exit(0)
                new_row.append(str(st))
                new_row.append(str(ex))
                new_row.append(str(fr))
                new_row.append(str(u2))

        # save results
        with open(f'{filename}.csv', mode="a", newline="") as file:
            csv_writer = csv.writer(file)
            csv_writer.writerow(new_row)
        print(ite+1)



if __name__ == '__main__':

    ths = [0.95, 0.95, 0.6, 0.75, 0.25]   # p choosing

    " PLA* evaluation "
    domain_names = ['8puzzle', '7blocks', '323trucks', '6vacuum', '1053elevator']
    tasks_files_names = ['8-puzzle_tasks', '7-blocks_tasks', '3-2-3-trucks_tasks', '6-vacuum_tasks',
                         '10-5-3-elevator_tasks']
    model_files_names = ['8-puzzle', '7-blocks', '3-2-3-trucks', '6-vacuum', '10-5-3-elevator']

    h_simple = [lambda t: t.h_all_blank, lambda t: t.h_blocks, lambda t: t.h_trucks, lambda t: t.h_vacuum
        , lambda t: t.h_elevator]
    heuristic_2 = LmCutHeuristic

    for i, (tfn, mfn) in enumerate(zip(tasks_files_names, model_files_names)):

        heuristic_1 = h_simple[i]

        with open(f'models/LogisticRegression_model__{mfn}.pkl', 'rb') as file:
            model = pickle.load(file)

        with open(f"D:/Desktop/AI_Research/STRIPS/tasks/test_tasks/{tfn}.pkl", 'rb') as file:
            tasks = pickle.load(file)


        params = [model, ths[i]]
        evaluate(f'{domain_names[i]}__lmc_custom_PLA', tasks[60:], [["custom", heuristic_1], ["lmc", heuristic_2]],
                 params, include_variant=True)



