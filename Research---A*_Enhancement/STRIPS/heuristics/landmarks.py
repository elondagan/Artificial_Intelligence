
import copy
import random
from collections import defaultdict
import time

"""
pyperplan
"""


def get_relaxed_task(task):
    relaxed_task = copy.deepcopy(task)
    for action_name in relaxed_task.A.keys():
        relaxed_task.A[action_name]['del'] = []
    return relaxed_task


def get_landmarks(task):

    task = get_relaxed_task(task)
    landmarks = set(task.G)
    possible_landmarks = set(task.V) - set(task.G)

    for fact in possible_landmarks:
        current_state = set(task.I)

        while not task.is_goal(current_state):
            previous_state = current_state

            for action_name in task.A.keys():
                if action_name in task.actions(current_state) and fact not in task.A[action_name]['add']:

                    # current_state = task.result(current_state, action_name)
                    current_state = task.result(current_state, action_name)


                    if task.is_goal(current_state):
                        break
            if previous_state == current_state and not task.is_goal(current_state):
                landmarks.add(fact)
                break

    return landmarks


def compute_landmark_costs(task, landmarks):
    op_to_lm = defaultdict(set)
    for action_name in task.A.keys():
        for landmark in landmarks:
            if landmark in task.A[action_name]['add']:
                op_to_lm[action_name].add(landmark)
    min_cost = defaultdict(lambda: float("inf"))
    for action_name, landmarks in op_to_lm.items():
        landmarks_achieving = len(landmarks)
        for landmark in landmarks:
            min_cost[landmark] = min(min_cost[landmark], 1 / landmarks_achieving)

    return min_cost


class LandmarkHeuristic:

    def __init__(self, task):
        self.times = []

        task = copy.deepcopy(task)
        task.I = task.I
        self.task = task
        self.landmarks = get_landmarks(task)
        assert set(self.task.G) <= set(self.landmarks)
        self.costs = compute_landmark_costs(task, self.landmarks)

    def __call__(self, node):
        st = time.time()

        use_state = node.state
        if node.parent is None:
            node.unreached = set(self.landmarks) - set(self.task.I)
        else:
            node.unreached = set(node.parent.unreached) - set(self.task.A[node.action]['add'])
        # unreached = node.unreached | (set(self.task.G) - set(node.state))
        unreached = node.unreached | (set(self.task.G) - set(use_state))
        h = sum(self.costs[landmark] for landmark in unreached)

        self.times.append(time.time() - st)

        return h

    def get_class_name(self):
        return "landmarks"

    def __name__(self):
        return self.__class__.__name__




