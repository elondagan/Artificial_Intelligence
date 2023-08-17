import search
import random
import math

name = 'Elon'

class TaxiProblem(search.Problem):

    def __init__(self, initial):

        self.map = initial['map']
        self.map_size = (len(initial['map']), len(initial['map'][0]))
        self.taxis = [t for t in list(initial['taxis'].keys())]
        self.customers = initial['passengers'].copy()

        # keep in memory distance from Gs
        g_locations = []
        for i in range(self.map_size[0]):
            for j in range(self.map_size[1]):
                if self.map[i][j] == 'G':
                    g_locations.append([i, j])
        g_min_distances = [[0 for _ in range(self.map_size[1])] for _ in range(self.map_size[0])]
        for i in range(self.map_size[0]):
            for j in range(self.map_size[1]):
                if self.map[i][j] == 'P':
                    nearest_g = 10000000
                    for loc in g_locations:
                        temp = TaxiProblem.manhattan_dist([i, j], loc)
                        if nearest_g > temp:
                            nearest_g = temp
                    g_min_distances[i][j] = nearest_g
        self.g_min_distances = g_min_distances

        # create first state
        s0 = []
        for taxi in initial['taxis']:
            loc = list(initial['taxis'][taxi]['location'])
            fuel = [initial['taxis'][taxi]['fuel'], initial['taxis'][taxi]['fuel']]
            max_pas = [initial['taxis'][taxi]['capacity']]
            s0.append([loc, fuel, [], max_pas])
        s0.append(list(initial['passengers'].keys()))

        search.Problem.__init__(self, TaxiProblem.list_to_str(s0))

    def actions(self, state):
        """Returns all the actions that can be executed in the given state"""
        state = TaxiProblem.str_to_list(state)
        actions_per_taxi = []

        for i, taxi in enumerate(state[:-1]):
            cur_actions = []

            # wait
            cur_actions.append(("wait", str(self.taxis[i])))

            # refuel
            if self.map[taxi[0][0]][taxi[0][1]] == 'G':
                cur_actions.append(("refuel", str(self.taxis[i])))

            # pick up
            for psngr in state[-1]:
                if list(self.customers[psngr]['location']) == taxi[0] and len(taxi[2]) < taxi[3][0]:
                    cur_actions.append(("pick up", str(self.taxis[i]), str(psngr)))

            # drop off
            for psngr in taxi[2]:
                if list(self.customers[psngr]['destination']) == taxi[0]:
                    cur_actions.append(("drop off", str(self.taxis[i]), str(psngr)))

            # movement
            if taxi[1][0] > 0:  # have fuel
                cur_x_loc, cur_y_loc = taxi[0][0], taxi[0][1]
                if cur_x_loc + 1 < self.map_size[0] and self.map[cur_x_loc + 1][cur_y_loc] != 'I':
                    cur_actions.append(("move", str(self.taxis[i]), (cur_x_loc + 1, cur_y_loc)))

                if cur_x_loc - 1 >= 0 and self.map[cur_x_loc - 1][cur_y_loc] != 'I' and taxi[1][0] > 0:
                    cur_actions.append(("move", str(self.taxis[i]), (cur_x_loc - 1, cur_y_loc)))

                if cur_y_loc + 1 < self.map_size[1] and self.map[cur_x_loc][cur_y_loc + 1] != 'I':
                    cur_actions.append(("move", str(self.taxis[i]), (cur_x_loc, cur_y_loc + 1)))

                if cur_y_loc - 1 >= 0 and self.map[cur_x_loc][cur_y_loc - 1] != 'I':
                    cur_actions.append(("move", str(self.taxis[i]), (cur_x_loc, cur_y_loc - 1)))

            actions_per_taxi.append(cur_actions)

        if len(self.taxis) == 1:
            return actions_per_taxi[0]

        # disturb matrix
        state_len = len(state[:-1])
        can_disturb = [[False for _ in range(state_len)] for _ in range(state_len)]
        for i in range(state_len):
            for j in range(i + 1, state_len):
                if TaxiProblem.manhattan_dist(state[i][0], state[j][0]) < 3:
                    can_disturb[i][j] = True
                    can_disturb[j][i] = True

        # find all possible action combination
        possible_action = []

        def legal(built_action, action):
            cur_len = len(built_action)

            # action type contradiction
            flag = action[0] != 'move' and sum([a[0] != 'move' for a in built_action]) == cur_len
            if flag is True:
                return True

            # location contradiction
            for k in range(cur_len):
                if can_disturb[k][cur_len] is True:
                    if action[0] == 'move':
                        if built_action[k][0] == 'move':
                            if action[2] == built_action[k][2]:
                                return False
                        else:
                            if list(action[2]) == state[k][0]:  # move to location vs current taxi location
                                return False
                    # want to add 'stay' action
                    else:
                        if built_action[k][0] == 'move':
                            if list(built_action[k][2]) == state[cur_len][0]:
                                return False
            return True

        def rec_loop(lists, cur, size, res_list, built_action):

            if cur == size:
                res_list.append(built_action.copy())
                return
            else:
                for action in lists[cur]:
                    if legal(built_action, action):
                        built_action.append(action)
                        rec_loop(lists, cur + 1, size, res_list, built_action)
                        built_action.remove(action)

        rec_loop(actions_per_taxi, 0, len(self.taxis), possible_action, [])

        return tuple(possible_action)

    def result(self, state, action):
        """Return the state that results from executing the given
        action in the given state."""
        state = TaxiProblem.str_to_list(state)

        # only 1 taxi
        if len(self.taxis) == 1:

            if action[0] == 'move':
                state[0][0] = list(action[2])
                state[0][1][0] -= 1

            if action[0] == 'refuel':
                state[0][1][0] = state[0][1][1]

            if action[0] == 'pick up':
                state[0][2].append(action[2])
                state[-1].remove(action[2])

            if action[0] == 'drop off':
                state[0][2].remove(action[2])

        # more than 1 taxi
        else:
            for i, a in enumerate(action):
                if a[0] == 'move':
                    state[i][0] = list(a[2])
                    state[i][1][0] -= 1

                if a[0] == 'pick up':
                    state[i][2].append(a[2])
                    state[-1].remove(a[2])

                if a[0] == 'drop off':
                    state[i][2].remove(a[2])

                if a[0] == 'refuel':
                    state[i][1][0] = state[i][1][1]

        return TaxiProblem.list_to_str(state)

    def goal_test(self, state):
        """ Given a state, checks if this is the goal state.
         Returns True if it is, False otherwise."""
        state = TaxiProblem.str_to_list(state)
        flag = True
        for taxi in state[:-1]:
            flag = flag and len(taxi[2]) == 0
        return flag and len(state[-1]) == 0

    def h(self, node):
        """ This is the heuristic. It gets a node (not a state)
        and returns a goal distance estimate"""
        state = TaxiProblem.str_to_list(node.state)
        tot_approx = 0
        inf = 10000000

        # check if state has enough fuel
        if node.parent is not None:
            prev_state = TaxiProblem.str_to_list(node.parent.state)
            for i, taxi in enumerate(state[:-1]):

                no_fuel_for_drop, no_fuel_to_g = False, False

                no_fuel_to_g = (self.g_min_distances[taxi[0][0]][taxi[0][1]] > taxi[1][0])

                if len(taxi[2]) == 0:
                    no_possible_pd, taxi_moved, not_disturbed = False, False, False

                    # check 1
                    min_dist = 0
                    for p in state[-1]:
                        temp = TaxiProblem.manhattan_dist(taxi[0],
                                                          self.customers[p]['location']) + TaxiProblem.manhattan_dist(
                            self.customers[p]['location'], self.customers[p]['destination'])
                        if temp < min_dist:
                            min_dist = temp
                    if min_dist > taxi[1][0]:
                        no_possible_pd = True

                    # check 2
                    taxi_moved = (taxi[0] == prev_state[i][0])

                    # check 3
                    temp_flag = True
                    for j, other_taxi in enumerate(state[:-1]):
                        if j != i:
                            if other_taxi[0] == prev_state[i][0]:
                                temp_flag = False
                    not_disturbed = temp_flag

                    no_fuel_for_drop = (no_possible_pd and taxi_moved and not_disturbed)

                else:
                    max_dist = 0
                    for p in taxi[2]:
                        temp = TaxiProblem.manhattan_dist(taxi[0], self.customers[p]['destination'])
                        if temp > max_dist:
                            max_dist = temp

                    no_fuel_for_drop = (max_dist > taxi[1][0])

                if no_fuel_for_drop and no_fuel_to_g:
                    return inf

        # in progress passengers - mean distances
        for taxi in state[:-1]:
            if len(taxi[2]) != 0:
                temp = 0
                for p in taxi[2]:
                    temp += TaxiProblem.manhattan_dist(taxi[0], self.customers[p]['destination'])
                tot_approx += temp / len(taxi[2])

        # if need to pickup, add min location-taxi dist
        all_taxis_free = True
        for taxi in state[:-1]:
            if len(taxi[2]) != 0:
                all_taxis_free = False
                break
        if all_taxis_free and len(state[-1]) != 0:
            min_min_dist = inf
            for taxi in state[:-1]:
                min_dist = inf
                for p in state[-1]:
                    temp = TaxiProblem.manhattan_dist(taxi[0], self.customers[p]['location'])
                    if temp < min_dist:
                        min_dist = temp
                if min_dist < min_min_dist:
                    min_min_dist = min_dist
            tot_approx += min_min_dist

        # punish for no movement where state[-1] != 0
        if node.parent is not None and len(state[-1]) != 0:
            prev_state = TaxiProblem.str_to_list(node.parent.state)
            for cur_taxi, prev_taxi in zip(state[:-1], prev_state[:-1]):
                if len(cur_taxi[2]) != 0 or len(prev_taxi[2]):
                    continue
                if cur_taxi[0] == prev_taxi[0]:
                    tot_approx += 1

        # hardest passenger from waiting
        waiting_max = 0
        for passenger in state[-1]:
            temp = TaxiProblem.manhattan_dist(self.customers[passenger]['location'],
                                              self.customers[passenger]['destination'])
            if temp > waiting_max:
                waiting_max = temp
        tot_approx += waiting_max

        # unpicked passengers pickup + drop-off time
        tot_approx += 2 * len(state[-1])

        # in-progress passengers drop-off time
        for taxi in state[:-1]:
            tot_approx += len(taxi[2])

        return tot_approx

    def h_1(self, node):
        """
        This is a simple heuristic
        """
        state = TaxiProblem.str_to_list(node.state)

        unpicked_passengers = len(state[-1])
        in_progress_passengers = 0
        for taxi in state[:-1]:
            in_progress_passengers += len(taxi[2])

        return (unpicked_passengers * 2 + in_progress_passengers) / len(self.taxis)

    def h_2(self, node):
        """
        This is a slightly more sophisticated Manhattan heuristic
        """
        state = TaxiProblem.str_to_list(node.state)

        D = []
        for p in state[-1]:
            D.append(TaxiProblem.manhattan_dist(self.customers[p]['location'], self.customers[p]['destination']))

        T = []
        for taxi in state[:-1]:
            for p in taxi[2]:
                T.append(TaxiProblem.manhattan_dist(taxi[0], self.customers[p]['destination']))

        return (sum(D) + sum(T)) / len(self.taxis)

    @staticmethod
    def list_to_str(list_of_lists):
        res = ''
        for alist in list_of_lists:
            res += '_'
            for cur in alist:
                res += '.'
                for val in cur:
                    res += str(val)
                    res += '#'
        return res

    @staticmethod
    def str_to_list(a_string):
        list_of_lists = []

        for l in a_string.split('_')[1:]:
            t1 = []
            for a in l.split('.')[1:]:
                t2 = []
                for b in a.split('#')[:-1]:
                    if b != '':
                        if b.isdigit():
                            t2.append(int(b))
                        else:
                            t2.append(b)
                    else:
                        t2.append([])
                t1.append(t2)
            list_of_lists.append(t1)

        temp = []
        for name in list_of_lists[-1]:
            temp.append(''.join(name))
        list_of_lists[-1] = temp

        return list_of_lists

    @staticmethod
    def manhattan_dist(loc_1, loc_2):
        return abs(loc_2[0] - loc_1[0]) + abs(loc_2[1] - loc_1[1])


def create_taxi_problem(game):
    return TaxiProblem(game)
