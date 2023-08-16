import copy
from ast import literal_eval
import networkx as nx
import numpy as np


class OptimalTaxiAgent:
    def __init__(self, initial):
        self.initial = initial
        self.handle_too_big_input()
        self.initial_as_state = OptimalTaxiAgent.state_representation(self.initial)
        self.map_size = (len(self.initial['map']), len(self.initial['map'][0]))
        self.taxi_amount = len(self.initial['taxis'])
        self.taxi_names = list(self.initial['taxis'].keys())
        self.passengers_amount = len(self.initial['passengers'])
        self.passengers_names = list(self.initial['passengers'].keys())
        self.states = self.create_all_states()
        self.states_info = self.set_states_info()
        self.value_iteration()

    def handle_too_big_input(self):
        # check only for input with one taxi and one passenger
        if len(self.initial['taxis']) > 1:
            return
        size = len(self.initial['map']) * len(self.initial['map'][0]) * self.initial['taxis']['taxi 1']['fuel']
        size *= self.initial['turns to go']
        for psngr in self.initial['passengers'].values():
            size *= len(psngr['possible_goals'])
            break

        if size > 15*15*20*100:
            self.initial['turns to go'] = 10
            for taxi in self.initial['taxis'].values():
                taxi['fuel'] = 10
                taxi['capacity'] = 1

    def list_to_dict(self, state):
        cur_dict = {}
        taxis_dict = {}
        for i, t in enumerate(range(self.taxi_amount)):
            taxis_dict[f'taxi {i + 1}'] = {
                'location': state[1 + i],
                'fuel': state[self.taxi_amount + 1][i],
                'capacity': state[self.taxi_amount + 2][i]
            }
        passengers_dict = {}
        for i, p in enumerate(range(self.passengers_amount)):
            passengers_dict[self.passengers_names[i]] = {
                'location': state[-1][i][0],
                'destination': state[-1][i][1]
            }
        cur_dict['taxis'] = taxis_dict
        cur_dict['passengers'] = passengers_dict

        return cur_dict

    def is_legal_state(self, state):
        state = self.list_to_dict(state)
        # no two taxis in the same location
        loc = state['taxis']['taxi 1']['location']
        for i, taxi in enumerate(state['taxis'].values()):
            if i == 0:
                continue
            if taxi['location'] == loc:
                return False

        # each taxi contain correct number of pas
        suppose_to_contain = [0 for _ in range(self.taxi_amount)]
        for psngr in state['passengers'].values():
            if isinstance(psngr['location'], str):
                suppose_to_contain[int(psngr['location'][-1]) - 1] += 1
        for i, taxi in enumerate(state['taxis'].values()):
            if taxi['capacity'] + suppose_to_contain[i] != self.initial['taxis'][self.taxi_names[i]]['capacity']:
                return False

        # no passenger on I tile (unless his on I in original input)
        for psngr, psngr_org in zip(state['passengers'].values(), self.initial['passengers'].values()):
            if isinstance(psngr['location'], str):
                continue
            elif self.initial['map'][psngr['location'][0]][psngr['location'][1]] == 'I' and \
                                        self.initial['map'][psngr_org['location'][0]][psngr_org['location'][1]] != 'I':
                return False

        return True

    def create_all_states(self):  # can add function to filter impossible results
        """
        create dictionary of all state, each state is a key
        """
        taxis = list(self.initial['taxis'].values())
        passengers = list(self.initial['passengers'].values())

        all_possible_time_left = [[t] for t in range(self.initial['turns to go'] + 1)]
        all_locs = [[r, c] for r in range(self.map_size[0]) for c in range(self.map_size[1])]
        all_possible_taxi_loc = all_locs.copy()
        for loc in all_locs:
            if self.initial['map'][loc[0]][loc[1]] == 'I':
                all_possible_taxi_loc.remove(loc)
        all_possible_fuel_comb = OptimalTaxiAgent.custom_loop([taxis[i]['fuel'] + 1 for i in range(self.taxi_amount)],
                                                              use_values=False)
        all_possible_capacity_comb = OptimalTaxiAgent.custom_loop(
            [taxis[i]['capacity'] + 1 for i in range(self.taxi_amount)], use_values=False)

        all_possible_passengers_loc_dest = []
        for p in passengers:
            pos_locations = list(self.initial['taxis'].keys())
            pos_destinations = [list(d) for d in p['possible_goals']]
            # adding current destination to possible destination
            if list(p['destination']) not in pos_destinations:
                pos_destinations += [list(p['destination'])]
            pos_locations += pos_destinations
            if list(p['location']) not in pos_locations:
                pos_locations += [list(p['location'])]
            all_possible_passengers_loc_dest.append(
                OptimalTaxiAgent.custom_loop([pos_locations, pos_destinations], use_values=True))
        all_possible_passengers_loc_dest = OptimalTaxiAgent.custom_loop(all_possible_passengers_loc_dest,
                                                                        use_values=True)

        temp_states = list()
        temp_states.append(all_possible_time_left)
        for _ in range(self.taxi_amount):
            temp_states.append(copy.deepcopy(all_possible_taxi_loc))
        temp_states.append(all_possible_fuel_comb)
        temp_states.append(all_possible_capacity_comb)
        temp_states.append(all_possible_passengers_loc_dest)
        temp_states = OptimalTaxiAgent.custom_loop(temp_states, use_values=True, with_legal_check=self.is_legal_state)

        # represent state as dictionaries
        states = [{} for _ in range(self.initial['turns to go'] + 1)]
        for state in temp_states:
            iter = state[0][0]
            cur_dict = self.list_to_dict(state)
            states[iter][str(cur_dict)] = {'value': 0, 'pi': []}

        return states

    def set_states_info(self):
        info_dict = {}
        for state in self.states[1].keys():
            info_dict[state] = {}
            possible_actions = self.all_possible_actions(literal_eval(state))
            info_dict[state]['possible_actions'] = possible_actions
            for action in possible_actions[:-1]:
                next_possible_states, probs = self.next_possible_states(literal_eval(state), action)
                info_dict[state][str(action)] = [next_possible_states, probs]
            info_dict[state][str(possible_actions[-1])] = [[str(self.initial_as_state)], [1]]

        return info_dict

    def all_possible_actions(self, from_state):
        """
        Returns all the actions that can be executed in the given state
        :param from_state: state as dictionary
        :return: tuple of all possible actions
        """
        actions_per_taxi = []
        possible_action = []

        # all actions for each taxi (without effects or other taxis)
        for i, taxi in enumerate(from_state['taxis'].values()):
            cur_actions = list()

            # wait
            cur_actions.append(("wait", f'taxi {i + 1}'))

            # refuel
            if self.initial['map'][taxi['location'][0]][taxi['location'][1]] == 'G':
                cur_actions.append(("refuel", f'taxi {i + 1}'))

            # pick up
            for j, psngr in enumerate(from_state['passengers'].values()):
                if psngr['location'] == taxi['location'] and psngr['location'] != psngr['destination'] and taxi[
                    'capacity'] != 0:
                    cur_actions.append(("pick up", f'taxi {i + 1}', self.passengers_names[j]))

            # drop off
            for j, psngr in enumerate(from_state['passengers'].values()):
                if psngr['location'] == f'taxi {i + 1}' and taxi['location'] == psngr['destination']:
                    cur_actions.append(("drop off", f'taxi {i + 1}', self.passengers_names[j]))

            # movement
            if taxi['fuel'] >= 1:
                cur_x_loc, cur_y_loc = taxi['location'][0], taxi['location'][1]

                if cur_x_loc + 1 < self.map_size[0] and self.initial['map'][cur_x_loc + 1][cur_y_loc] != 'I':
                    cur_actions.append(("move", f'taxi {i + 1}', [cur_x_loc + 1, cur_y_loc]))

                if cur_x_loc - 1 >= 0 and self.initial['map'][cur_x_loc - 1][cur_y_loc] != 'I':
                    cur_actions.append(("move", f'taxi {i + 1}', [cur_x_loc - 1, cur_y_loc]))

                if cur_y_loc + 1 < self.map_size[1] and self.initial['map'][cur_x_loc][cur_y_loc + 1] != 'I':
                    cur_actions.append(("move", f'taxi {i + 1}', [cur_x_loc, cur_y_loc + 1]))

                if cur_y_loc - 1 >= 0 and self.initial['map'][cur_x_loc][cur_y_loc - 1] != 'I':
                    cur_actions.append(("move", f'taxi {i + 1}', [cur_x_loc, cur_y_loc - 1]))

            cur_actions.append('reset')
            actions_per_taxi.append(cur_actions)

        if self.taxi_amount == 1:
            return actions_per_taxi[0]

            # creating disturb matrix - which taxis effect each other when choosing an action
        state_len = self.passengers_amount
        can_disturb = [[False for _ in range(state_len)] for _ in range(state_len)]
        for i in range(state_len):
            for j in range(i + 1, state_len):
                taxi_i_loc = from_state['taxis'][f'taxi {i + 1}']['location']
                taxi_j_loc = from_state['taxis'][f'taxi {i + 1}']['location']
                if OptimalTaxiAgent.manhattan_dist(taxi_i_loc, taxi_j_loc) < 3:
                    can_disturb[i][j] = True
                    can_disturb[j][i] = True

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
                            # move to location vs current taxi location
                            if list(action[2]) == list(from_state['taxis'].values())[k]['location']:
                                return False
                    # want to add 'stay' action
                    else:
                        if built_action[k][0] == 'move':
                            if list(built_action[k][2]) == list(from_state['taxis'].values())[cur_len]['location']:
                                return False
            return True

        def rec_loop(lists, cur, size, res_list, built_action):
            if cur == size:
                res_list.append(copy.deepcopy(built_action))
                return
            else:
                for action in lists[cur]:
                    if legal(built_action, action):
                        built_action.append(action)
                        rec_loop(lists, cur + 1, size, res_list, built_action)
                        built_action.remove(action)

        rec_loop(actions_per_taxi, 0, self.taxi_amount, possible_action, [])
        return tuple(possible_action)

    def next_possible_states(self, from_state, with_action):
        """
        :param from_state: state as dictionary
        :param with_action:
        :return:
        """
        if with_action == 'reset':
            return [self.initial_as_state]

        next_states = list()
        template_new_state = copy.deepcopy(from_state)

        if self.taxi_amount == 1:  # assume for multi taxis i will get [(),()]
            with_action = [with_action]

        for i, sub_action in enumerate(with_action):

            if sub_action[0] == 'move':
                template_new_state['taxis'][f'taxi {i + 1}']['location'] = sub_action[-1]
                template_new_state['taxis'][f'taxi {i + 1}']['fuel'] -= 1

            elif sub_action[0] == 'pick up':
                template_new_state['taxis'][f'taxi {i + 1}']['capacity'] -= 1
                template_new_state['passengers'][sub_action[-1]]['location'] = f'taxi {i + 1}'

            elif sub_action[0] == 'drop off':
                template_new_state['taxis'][f'taxi {i + 1}']['capacity'] += 1
                template_new_state['passengers'][sub_action[-1]]['location'] = \
                    from_state['taxis'][f'taxi {i + 1}']['location']

            elif sub_action[0] == 'refuel':
                template_new_state['taxis'][f'taxi {i + 1}']['fuel'] = self.initial['taxis'][f'taxi {i + 1}']['fuel']

            elif sub_action[0] == 'terminate':
                pass

        # create the stochastic states based on template state
        possible_goals_per_passenger = []
        for p in range(self.passengers_amount):
            cur_ps = self.initial['passengers'][self.passengers_names[p]]['possible_goals']
            cur_ps = [list(c) for c in cur_ps]
            possible_goals_per_passenger.append(cur_ps)

        possible_goals_combs = OptimalTaxiAgent.custom_loop(possible_goals_per_passenger, use_values=True)

        for pos in possible_goals_combs:
            another_state = copy.deepcopy(template_new_state)
            for p in range(self.passengers_amount):
                another_state['passengers'][self.passengers_names[p]]['destination'] = pos[p]
            next_states.append(copy.deepcopy(another_state))

        probs = []
        for next_state in next_states:
            probs.append(self.transition_probability(next_state, from_state))

        return next_states, probs

    def transition_probability(self, stag, s, a=None):
        """
        return transition probability to next_state given current_state and action
        """
        if a == 'reset':
            return 1
        probability = 1
        for p in self.passengers_names:
            change_p = self.initial['passengers'][p]['prob_change_goal']
            change_ops = len(self.initial['passengers'][p]['possible_goals'])
            if stag['passengers'][p]['destination'] != s['passengers'][p]['destination']:
                probability *= (change_p / change_ops)
            else:
                probability *= 1 - ((change_p * (change_ops - 1)) / change_ops)
        return probability

    def get_best_action_and_expected_reward(self, state, iteration):
        """
        :param state: state as dictionary
        :param iteration: current turns to go
        :return: best action to take and its expected value
        """
        best_action, best_sum = [], -1

        possible_actions = self.states_info[state]['possible_actions']
        for action in possible_actions:  # without reset action
            next_possible_states = self.states_info[state][str(action)][0]
            next_states_probs = self.states_info[state][str(action)][1]
            cur_sum = 0
            for i, s in enumerate(next_possible_states):
                cur_sum += next_states_probs[i] * self.states[iteration - 1][str(s)]['value']
            cur_sum += self.reward(action)
            if cur_sum >= best_sum:
                best_sum = cur_sum
                best_action = action

        return best_action, best_sum

    def reward(self, action):
        """
        calculate received reward for performing given action
        :param action: tuple of action for each taxi
        :return: reward earned
        """
        reward = 0
        if self.taxi_amount == 1:
            action = [action]
        for a in action:
            if a[0] == 'drop off':
                reward += 100
            elif a[0] == 'refuel':
                reward -= 10
            elif a == 'reset':
                reward -= 50
        return reward

    def value_iteration(self):
        for t in range(1, self.initial['turns to go'] + 1):
            for s in self.states[t]:
                current_best_action, current_expected_reward = self.get_best_action_and_expected_reward(s, t)
                self.states[t][s]['value'] = current_expected_reward
                self.states[t][s]['pi'].insert(0, current_best_action)

    def act(self, state):
        fitted_state = OptimalTaxiAgent.state_representation(state)
        time_left = state['turns to go']
        action = self.states[time_left][str(fitted_state)]['pi']

        # convert action to required format
        if action[0] == 'reset':
            return 'reset'
        new_action = []
        for a in action:
            temp = list(a)
            if isinstance(temp[-1], list):
                temp[-1] = tuple(temp[-1])
            new_action.append(tuple(temp))
        new_action = tuple(new_action)
        return new_action

    @staticmethod
    def state_representation(a_dict):
        as_state = copy.deepcopy(a_dict)
        del as_state['optimal']
        del as_state['map']
        del as_state['turns to go']

        for taxi in as_state['taxis']:
            as_state['taxis'][taxi]['location'] = list(as_state['taxis'][taxi]['location'])

        for psngr in as_state['passengers']:
            if isinstance(as_state['passengers'][psngr]['location'], tuple):
                as_state['passengers'][psngr]['location'] = list(as_state['passengers'][psngr]['location'])
            else:
                as_state['passengers'][psngr]['location'] = as_state['passengers'][psngr]['location']
            as_state['passengers'][psngr]['destination'] = list(as_state['passengers'][psngr]['destination'])
            del as_state['passengers'][psngr]['possible_goals']
            del as_state['passengers'][psngr]['prob_change_goal']

        return as_state

    @staticmethod
    def manhattan_dist(loc_1, loc_2):
        return abs(loc_2[0] - loc_1[0]) + abs(loc_2[1] - loc_1[1])

    @staticmethod
    def custom_loop(ranges, use_values, with_legal_check=lambda state: True):
        """
        create all possible combinations in given ranges
        :param ranges: value range for each position in inner list
        :param use_values: False = index combinations, True = value combination
        :param with_legal_check =
        :return: list of lists
        """
        is_legal = with_legal_check

        def recursive_index_loop(loop_size, ranges1, cur_index, temp_output, output):
            if cur_index == loop_size:
                if is_legal(temp_output):
                    output.append(temp_output.copy())
                return
            for i in range(ranges1[cur_index]):
                temp_output.append(i)
                recursive_index_loop(loop_size, ranges1, cur_index + 1, temp_output, output)
                temp_output.pop()

        def recursive_value_loop(lists, lists_size, cur_index, res_list, built_action):
            if cur_index == lists_size:
                if is_legal(built_action):
                    res_list.append(built_action.copy())
                return
            else:
                for action in lists[cur_index]:
                    built_action.append(action)
                    recursive_value_loop(lists, lists_size, cur_index + 1, res_list, built_action)
                    built_action.pop()

        result = []
        if use_values is True:
            recursive_value_loop(ranges, len(ranges), 0, result, [])

        else:
            recursive_index_loop(len(ranges), ranges, 0, [], result)

        return result


class TaxiAgent:
    def __init__(self, initial):
        self.initial = initial
        self.map_size = (len(self.initial['map']), len(self.initial['map'][0]))
        self.taxi_amount = len(self.initial['taxis'])
        self.passengers_names = list(self.initial['passengers'].keys())
        self.passengers_amount = len(self.passengers_names)
        self.real_distances = self.get_real_distances()
        self.assignment = self.set_assignment()
        self.agents = self.get_agents()
        self.restart_flags = [False for _ in range(self.taxi_amount)]
        self.score = 0

    def get_real_distances(self):

        def is_adjacent(place1, place2):
            if place1[0] == place2[0] and place1[1] == place2[1] + 1:
                return True

            if place1[0] == place2[0] and place1[1] + 1 == place2[1]:
                return True

            if place1[0] + 1 == place2[0] and place1[1] == place2[1]:
                return True

            if place1[0] == place2[0] + 1 and place1[1] == place2[1]:
                return True

            return False

        def create_graph():
            g = nx.Graph()
            for i in range(self.map_size[0]):
                for j in range(self.map_size[1]):
                    if self.initial['map'][i][j] != 'I':
                        g.add_node((i, j), type=self.initial['map'][i][j])
            for node1 in g.nodes:
                for node2 in g.nodes:
                    if is_adjacent(node1, node2):
                        g.add_edge(node1, node2)
            return g

        def find_distances(graph):
            res0 = {}
            for node in graph.nodes:
                res0[node] = nx.shortest_path_length(graph, source=node)
            return res0

        def fill_missing(adict):
            all_keys = []
            for i in range(self.map_size[0]):
                for j in range(self.map_size[1]):
                    all_keys.append((i, j))

            for d in adict:
                d_keys = list(adict[d].keys())
                for k in all_keys:
                    if k not in d_keys:
                        adict[d][k] = 10000000
            return adict

        g = create_graph()
        real_distances = fill_missing(find_distances(g))
        return real_distances

    def set_assignment(self):
        """
        check the size of the states space and dicide how to act
        :return: 'optimal' or list of lists of names
        """
        # check states sizes
        states_size = self.initial['turns to go']
        ms = sum([self.initial['map'][i][j] != 'I' for i in range(self.map_size[0]) for j in range(self.map_size[1])])
        for taxi in self.initial['taxis'].values():
            states_size *= ms * taxi['fuel'] * taxi['capacity']
        for psngr in self.initial['passengers'].values():
            states_size *= (len(psngr['possible_goals']) + 2) * (len(psngr['possible_goals']) + 1)
        if states_size < 10 ** 5:
            return 'optimal'

        # check how many passengers each taxi can handle optimally
        psngrs_per_taxi = 0
        max_fuel = max([taxi['fuel'] for taxi in self.initial['taxis'].values()])
        current_state_size = self.initial['turns to go'] * ms * max_fuel
        max_psngr_loc_dest_size = 0
        for psngr in self.initial['passengers'].values():
            cur = (len(psngr['possible_goals']) + 2) * (len(psngr['possible_goals']) + 1)
            if cur > max_psngr_loc_dest_size:
                max_psngr_loc_dest_size = cur

        for _ in range(self.passengers_amount):
            current_state_size *= max_psngr_loc_dest_size
            if current_state_size < 10 ** 5:
                psngrs_per_taxi += 1

        # assign passengers to taxis
        taxi_assignment = []
        for i, taxi in enumerate(self.initial['taxis'].values()):
            if i  == self.passengers_amount:
                break
            best_psngr = -1
            min_tot = 100000000000
            for i, psngr in enumerate(self.initial['passengers'].values()):
                try:
                    taxi_to_loc = self.real_distances[taxi['location']][psngr['location']]
                    loc_to_some_dest = min([self.real_distances[psngr['location']][b] for b in psngr['possible_goals']]) / \
                                       psngr['prob_change_goal']
                    tot = taxi_to_loc + loc_to_some_dest
                    # punish for not enough fuel to reach psngr location
                    if taxi['fuel'] < taxi_to_loc:
                        tot += taxi_to_loc
                except:
                    tot = np.inf
                if tot < min_tot and [self.passengers_names[i]] not in taxi_assignment:
                    min_tot = tot
                    best_psngr = i

            taxi_assignment.append([self.passengers_names[best_psngr]])

        return taxi_assignment

    def divide_state(self, state):
        if self.assignment == 'optimal':
            return [state]

        new_states = list()
        for i, taxi in enumerate(state['taxis'].values()):
            new_initial = copy.deepcopy(state)
            new_initial['turns to go'] = min(50, state['turns to go'])
            new_initial['taxis'] = {'taxi 1': taxi}
            for psngr in self.assignment[i]:
                new_initial['passengers'] = {psngr: state['passengers'][psngr]}
                # handle location
                if isinstance(state['passengers'][psngr]['location'], str):
                    new_initial['passengers'][psngr]['location'] = 'taxi 1'

            new_states.append(copy.deepcopy(new_initial))
            # handle case when not all taxis are used
            if i + 1 >= len(self.assignment):
                break

        return new_states

    def get_agents(self):
        """
        create optimal agents to sub-problems
        :return: list of agents for each taxi
        """
        if self.assignment == 'optimal':
            return [OptimalTaxiAgent(self.initial)]

        else:
            agents = []
            state_per_taxi = self.divide_state(self.initial)
            for i in range(len(state_per_taxi)):
                agents.append(OptimalTaxiAgent(state_per_taxi[i]))
            return agents

    def check_and_legalize_actions(self, actions, original_state):
        final_actions = actions.copy()

        if self.taxi_amount == 1:
            if final_actions[0] == 'reset':
                return 'reset'
            else:
                return tuple(final_actions)

        # handle more taxis than passengers case
        for i in range(len(self.assignment), self.taxi_amount):
            self.restart_flags[i] = True
            final_actions.append('reset')

        # check and handle reset requests
        for i, sub_action in enumerate(final_actions):
            if sub_action == 'reset':
                self.restart_flags[i] = True
                final_actions[i] = ('wait', f'taxi {i + 1}')
        if sum(self.restart_flags) == self.taxi_amount:
            self.restart_flags = [False for _ in range(self.taxi_amount)]
            return 'reset'

        # check and fix actions contradiction
        def get_wanted_locations():
            res_location = []
            for i, sub_action in enumerate(final_actions):
                if sub_action[0] == 'move':
                    res_location.append(sub_action[-1])
                else:
                    res_location.append(original_state['taxis'][f'taxi {i + 1}']['location'])
            return res_location

        wanted_locations = get_wanted_locations()
        flag_changed = True
        while flag_changed:
            flag_changed = False
            for i in range(self.taxi_amount):
                for j in range(i + 1, self.taxi_amount):
                    if wanted_locations[i] == wanted_locations[j]:
                        flag_changed = True
                        # case 1 - both want to move to the same tile
                        if final_actions[i][0] == final_actions[j][0]:
                            final_actions[i] = ('wait', f'taxi {i + 1}')
                        # case 2 - one want to enter to the other tile
                        else:
                            if final_actions[i][0] == 'move':
                                if final_actions[j][0] == 'wait' or final_actions[j][0] == 'reset':
                                    if original_state['taxis'][f'taxi {j + 1}']['fuel'] > 0:
                                        final_actions[j] = ('move', f'taxi {j + 1}',
                                                            original_state['taxis'][f'taxi {i + 1}']['location'])
                                    else:
                                        ### can try to send him to other tile, for now tell him to wait
                                        final_actions[i] = ('wait', f'taxi {i + 1}')
                                else:
                                    final_actions[i] = ('wait', f'taxi {i + 1}')
                            else:
                                if final_actions[i][0] == 'wait' or final_actions[i][0] == 'reset':
                                    if original_state['taxis'][f'taxi {i + 1}']['fuel'] > 0:
                                        final_actions[i] = ('move', f'taxi {i + 1}',
                                                            original_state['taxis'][f'taxi {j + 1}']['location'])
                                    else:
                                        ### can try to send him to other tile, for now tell him to wait
                                        final_actions[j] = ('wait', f'taxi {j + 1}')
                                else:
                                    final_actions[j] = ('wait', f'taxi {j + 1}')
                    wanted_locations = get_wanted_locations()

        # rename taxi names
        for i, sub_action in enumerate(final_actions):
            temp = list(sub_action)
            temp[1] = f'taxi {i + 1}'
            final_actions[i] = temp

        return tuple(final_actions)

    def act(self, state):

        if self.assignment == 'optimal':
            return self.agents[0].act(state)

        final_actions = []
        state_per_taxi = self.divide_state(state)
        for i, agent in enumerate(self.agents):
            action = agent.act(state_per_taxi[i])
            if action == 'reset':
                final_actions.append(action)
            else:
                final_actions.append(action[0])  # assuming each taxi handle one passenger
        final_actions = self.check_and_legalize_actions(final_actions, state)

        # terminate game in case of possible negative score
        if final_actions == 'reset':
            if self.score > 50:
                self.score -= 50
                return final_actions
            else:
                return 'terminate'

        # keep track of score
        for action in final_actions:
            if action[0] == 'drop off':
                self.score += 100
            elif action[0] == 'refuel':
                self.score -= 10
        return final_actions
