import pickle

import csv
import os
import numpy as np
import time

from search.search_algorithms import Node, A_star
from utilities.utils import create_csv_file

from STRIPS.heuristics.lm_cut import LmCutHeuristic
from STRIPS.heuristics.landmarks import LandmarkHeuristic
from STRIPS.heuristics.relaxation_heuristics import hMaxHeuristic


def compare(filename, tasks, heuristics):
    " first heuristic is best "

    h_names = [heuristics[i][0] for i in range(len(heuristics))]



    # create csv file for the results (if not already existed)
    headline = ['problem I', 'problem G', 'sol len']
    for f in h_names:
        headline.append(f'{f}_time')
        headline.append(f'{f}_explored')
        headline.append(f'{f}_frontier')
    if not os.path.exists(f'{filename}.csv'):
        create_csv_file(filename, headline)

    # iterate all tasks
    for ite, task in enumerate(tasks):

        hs = [heuristics[i][1](task) for i in range(len(heuristics))]
        f_functions = [lambda n: n.path_cost + hs[0](n), lambda n: n.path_cost + hs[1](n), lambda n: n.path_cost + hs[2](n)]

        # skip unsolvable tasks
        if f_functions[0](Node(task.I)) == np.inf:
            continue

        # solve with A*
        new_row = task.task_rep()
        for i, f in enumerate(f_functions):
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

        # save results
        with open(f'{filename}.csv', mode="a", newline="") as file:
            csv_writer = csv.writer(file)
            csv_writer.writerow(new_row)
        print(ite+1)


if __name__ == "__main__":

    domain_names = ['8puzzle', '7blocks', '323trucks', '6vacuum', '1053elevator']
    tasks_files_names = ['8-puzzle_tasks', '7-blocks_tasks', '3-2-3-trucks_tasks', '6-vacuum_tasks',
                         '10-5-3-elevator_tasks']

    hs = [["lmc", LmCutHeuristic], ["lm", LandmarkHeuristic], ["m", hMaxHeuristic]]

    for j, tfn in enumerate(tasks_files_names):

        with open(f"D:/Desktop/AI_Research/STRIPS/tasks/train_tasks/{tfn}.pkl", 'rb') as file:
            tasks = pickle.load(file)

        compare(f'{domain_names[j]}__lmc_lm_m', tasks, hs)
