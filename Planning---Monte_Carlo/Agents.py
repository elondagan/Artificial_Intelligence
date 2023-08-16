import numpy as np
import copy
import time
import networkx as nx
import random

Name = 'Elon'
tl = 4.95


class Tree:
    def __init__(self, root):
        self.root = root


class Node:
    def __init__(self, actions, of_player, parent=None):
        self.actions = actions
        self.parent = parent
        self.kids = []
        self.of_player = of_player
        self.n = 0
        self.x = 0

    def add_kid(self, node):
        self.kids.append(node)
        node.parent = self


class Agent:

    def __init__(self, initial_state, player_number):
        self.Name = Name
        self.initial = initial_state
        self.player_number = player_number
        # environment information
        self.map_size = (len(initial_state['map']), len(initial_state['map'][0]))
        self.taxi_amount, self.taxi_names = self.get_player_taxis_info()
        self.passengers_amount = len(initial_state['passengers'])
        self.passengers_names = list(initial_state['passengers'].keys())
        self.distances = self.find_real_distances()   # only from not I to some tile
        self.impassible_location = self.get_impassible_locations()

    # ---------- initial auxiliary functions ----------

    def get_player_taxis_info(self):
        """
        find number of taxis in their names for each player
        :return: list of #taxis for each player, list of lists containing the names of the taxis for each player
        """
        amounts, names = [], []
        for player in range(1, 3):
            counter = 0
            cur_names = []
            for i, taxi in enumerate(self.initial['taxis'].values()):
                if taxi['player'] == player:
                    counter += 1
                    cur_names.append(list(self.initial['taxis'].keys())[i])
            amounts.append(counter)
            names.append(cur_names)

        return amounts, names

    def find_real_distances(self):

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

        def create_graph(amap):
            g = nx.Graph()
            for i in range(self.map_size[0]):
                for j in range(self.map_size[1]):
                    if amap[i][j] != 'I':
                        g.add_node((i, j), type=amap[i][j])
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

        def fill_missing(amap, adict):
            all_keys = []
            for i in range(len(amap)):
                for j in range(len(amap[0])):
                    all_keys.append((i, j))

            for d in adict:
                d_keys = list(adict[d].keys())
                for k in all_keys:
                    if k not in d_keys:
                        adict[d][k] = np.inf
            return adict

        g = create_graph(self.initial['map'])
        return fill_missing(self.initial['map'], find_distances(g))

    def get_impassible_locations(self):
        locations = []
        for i in range(self.map_size[0]):
            for j in range(self.map_size[1]):
                if self.initial['map'][i][j] == 'I':
                    locations.append([i, j])
                    locations.append((i,j))
        return locations

    # ---------- act auxiliary functions ----------

    def all_possible_actions(self, from_state, of_player):
        """
        Returns all the actions that "of_player" can execute in the given state
        :param from_state: state as dictionary
        :param of_player: player ID
        :return: tuple of all possible actions
        """
        actions_per_taxi = []
        possible_action = []

        # find occupied (by other taxis) locations
        occupied_locations = []
        for taxi in from_state['taxis'].values():
            if taxi['player'] != of_player:
                occupied_locations.append(taxi['location'])

        k = -1
        # all actions for each taxi of player (without effects of other taxis)
        for i, taxi in enumerate(from_state['taxis'].values()):
            if taxi['player'] != of_player:
                continue
            k += 1
            cur_actions = list()

            # pick up
            for j, psngr in enumerate(from_state['passengers'].values()):
                if psngr['location'] == taxi['location'] and psngr['location'] != psngr['destination'] and taxi[
                                  'capacity'] != 0 and psngr['destination'] not in self.impassible_location:
                    cur_actions.append(("pick up", self.taxi_names[of_player - 1][k], self.passengers_names[j]))

            # drop off
            for j, psngr in enumerate(from_state['passengers'].values()):
                if psngr['location'] == self.taxi_names[of_player - 1][k] and taxi['location'] == psngr['destination']:
                    cur_actions.append(("drop off", self.taxi_names[of_player - 1][k], self.passengers_names[j]))

            # movement
            cur_x_loc, cur_y_loc = taxi['location'][0], taxi['location'][1]

            if cur_x_loc + 1 < self.map_size[0] and from_state['map'][cur_x_loc + 1][cur_y_loc] != 'I' and tuple(
                    [cur_x_loc + 1, cur_y_loc]) not in occupied_locations:
                cur_actions.append(("move", self.taxi_names[of_player - 1][k], [cur_x_loc + 1, cur_y_loc]))

            if cur_x_loc - 1 >= 0 and from_state['map'][cur_x_loc - 1][cur_y_loc] != 'I' and tuple(
                    [cur_x_loc - 1, cur_y_loc]) not in occupied_locations:
                cur_actions.append(("move", self.taxi_names[of_player - 1][k], [cur_x_loc - 1, cur_y_loc]))

            if cur_y_loc + 1 < self.map_size[1] and from_state['map'][cur_x_loc][cur_y_loc + 1] != 'I' and tuple(
                    [cur_x_loc, cur_y_loc + 1]) not in occupied_locations:
                cur_actions.append(("move", self.taxi_names[of_player - 1][k], [cur_x_loc, cur_y_loc + 1]))

            if cur_y_loc - 1 >= 0 and from_state['map'][cur_x_loc][cur_y_loc - 1] != 'I' and tuple(
                    [cur_x_loc, cur_y_loc - 1]) not in occupied_locations:
                cur_actions.append(("move", self.taxi_names[of_player - 1][k], [cur_x_loc, cur_y_loc - 1]))

            # wait
            cur_actions.append(("wait", self.taxi_names[of_player - 1][k]))

            actions_per_taxi.append(cur_actions)

        if self.taxi_amount[of_player - 1] == 1:
            return actions_per_taxi[0]

        # creating disturb matrix - which taxis effect each other when choosing an action
        state_len = self.taxi_amount[of_player - 1]
        can_disturb = [[False for _ in range(state_len)] for _ in range(state_len)]
        for i in range(state_len):
            for j in range(i + 1, state_len):
                taxi_i_loc = from_state['taxis'][self.taxi_names[of_player - 1][k]]['location']
                taxi_j_loc = from_state['taxis'][self.taxi_names[of_player - 1][k]]['location']
                if self.distances[taxi_i_loc][taxi_j_loc] < 3:
                    can_disturb[i][j] = True
                    can_disturb[j][i] = True

        def legal(built_action, action):
            cur_len = len(built_action)

            if self.player_number == 1:
                step = 0
            else:
                step = self.taxi_amount[0]

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
                            if tuple(action[2]) == list(from_state['taxis'].values())[step+k]['location']:
                                return False
                    # want to add 'stay' action
                    else:
                        if built_action[k][0] == 'move':
                            if tuple(built_action[k][2]) == list(from_state['taxis'].values())[step+cur_len]['location']:
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

        rec_loop(actions_per_taxi, 0, self.taxi_amount[of_player - 1], possible_action, [])
        return tuple(possible_action)

    def choose_best_action(self, actions_arr, state):

        def choose_target():
            max_val, best_psngr = 0, None
            for psngr in state['passengers'].values():
                if psngr in used_psngrs or type(psngr['location']) == str or psngr['destination'] in self.impassible_location:
                    continue
                cur_val = psngr['reward'] / (self.distances[taxi['location']][psngr['location']] + 0.001)
                if cur_val > max_val:
                    max_val = cur_val
                    best_psngr = psngr
            return best_psngr

        def choose_action(psngr, use='location'):

            min_dist, minimizer = 1000, None
            for action in actions_arr:
                if fleet_size == 1:
                    action = [action]

                if action[index][0] == 'move':
                    cur_dist = self.distances[tuple(action[index][-1])][psngr[use]]
                    if cur_dist < min_dist:
                        min_dist = cur_dist
                        minimizer = action[index]
            return minimizer

        def find_closest_action():
            max_similarity, maximizer = -1, 0
            for action in actions_arr:
                if fleet_size == 1:
                    action = [action]
                cur_similarity = 0
                for sub_action, req_sub_action in zip(action, requested_actions):
                    cur_similarity += int(sub_action == req_sub_action)
                if cur_similarity > max_similarity:
                    max_similarity = cur_similarity
                    maximizer = action

            return maximizer

        def target_drop():
            min_dist, minimizer = 10000, None
            on_taxi = []
            for psngr in state['passengers'].values():
                if psngr['location'] == taxi_name:
                    on_taxi.append(psngr)

            if len(on_taxi) == 0:
                return ('wait', taxi_name)
            else:
                for psngr in on_taxi:
                    dest = psngr['destination']
                    cur_dist = self.distances[taxi['location']][dest]
                    if cur_dist < min_dist:
                        min_dist = cur_dist
                        minimizer = psngr
                return choose_action(minimizer, use='destination')

        fleet_size = self.taxi_amount[self.player_number - 1]
        can_drop_off = [False for _ in range(fleet_size)]
        can_pick_up = [False for _ in range(fleet_size)]

        # check for each taxi if pick up / drop off is possible
        for action in actions_arr:
            # fit case of one taxi to general case
            if fleet_size == 1:
                action = [action]
            for t in range(fleet_size):
                if action[t][0] == 'pick up':
                    can_pick_up[t] = action[t]
                elif action[t][0] == 'drop off':
                    can_drop_off[t] = action[t]

        used_psngrs = []
        requested_actions = []
        for index, taxi_name in enumerate(self.taxi_names[self.player_number - 1]):

            if can_drop_off[index] is not False:
                requested_actions.append(can_drop_off[index])
                continue
            if can_pick_up[index] is not False:
                requested_actions.append(can_pick_up[index])
                continue

            taxi = state['taxis'][taxi_name]
            if taxi['capacity'] != 0:
                targeted_psngr = choose_target()

                if taxi['capacity'] != self.initial['taxis'][taxi_name]['capacity'] and targeted_psngr is not None \
                                                  and self.distances[taxi['location']][targeted_psngr['location']] > 3:
                    targeted_psngr = None

                if targeted_psngr is not None:
                    used_psngrs.append(targeted_psngr)
                    wanted_action = choose_action(targeted_psngr)

                else:
                    wanted_action = target_drop()

                requested_actions.append(wanted_action)

            else:
                wanted_action = target_drop()
                requested_actions.append(wanted_action)

        if fleet_size == 1:
            if requested_actions[0] in actions_arr:
                return requested_actions[0]
            # some actions contradict, find most similar action possible
            else:
                return find_closest_action()[0]

        else:
            if requested_actions in actions_arr:
                return requested_actions
            # some actions contradict, find most similar action possible
            else:
                return find_closest_action()

    # ---------- main function ----------

    def act(self, state):
        self.passengers_amount = len(state['passengers'])
        self.passengers_names = list(state['passengers'].keys())

        actions = self.all_possible_actions(state, self.player_number)
        best_action = self.choose_best_action(actions, state)

        # convert output to required format
        if len(self.taxi_names[self.player_number-1]) == 1:
            best_action = (best_action,)
        best_action = list(best_action)
        for i, action in enumerate(best_action):
            best_action[i] = list(best_action[i])
            if action[0] == 'move':
                best_action[i][-1] = tuple(best_action[i][-1])
            best_action[i] = tuple(best_action[i])
        best_action = tuple(best_action)

        return best_action


class UCTAgent:

    def __init__(self, initial_state, player_number):
        self.Name = Name
        self.initial = initial_state
        self.player_number = player_number
        # environment information
        self.map_size = (len(initial_state['map']), len(initial_state['map'][0]))
        self.taxi_amount, self.taxi_names = self.get_player_taxis_info()
        self.passengers_amount = len(initial_state['passengers'])
        self.passengers_names = list(initial_state['passengers'].keys())
        self.impassible_location = self.get_impassible_locations()

    # ---------- initial auxiliary functions ----------

    def get_player_taxis_info(self):
        """
        find number of taxis in their names for each player
        :return: list of #taxis for each player, list of lists containing the names of the taxis for each player
        """
        amounts, names = [], []
        for player in range(1, 3):
            counter = 0
            cur_names = []
            for i, taxi in enumerate(self.initial['taxis'].values()):
                if taxi['player'] == player:
                    counter += 1
                    cur_names.append(list(self.initial['taxis'].keys())[i])
            amounts.append(counter)
            names.append(cur_names)

        return amounts, names

    def get_impassible_locations(self):
        locations = []
        for i in range(self.map_size[0]):
            for j in range(self.map_size[1]):
                if self.initial['map'][i][j] == 'I':
                    locations.append([i, j])
                    locations.append((i,j))
        return locations

    # ---------- MCTS auxiliary functions ----------

    def all_possible_actions(self, from_state, of_player):
        """
        Returns all the actions that "of_player" can execute in the given state
        :param from_state: state as dictionary
        :param of_player: player ID
        :return: tuple of all possible actions
        """
        actions_per_taxi = []
        possible_action = []

        # find occupied (by other taxis) locations
        occupied_locations = []
        for taxi in from_state['taxis'].values():
            if taxi['player'] != of_player:
                occupied_locations.append(taxi['location'])

        k = -1
        # all actions for each taxi of player (without effects of other taxis)
        for i, taxi in enumerate(from_state['taxis'].values()):
            if taxi['player'] != of_player:
                continue
            k += 1
            cur_actions = list()

            # pick up
            for j, psngr in enumerate(from_state['passengers'].values()):
                if psngr['location'] == taxi['location'] and psngr['location'] != psngr['destination'] and taxi[
                                  'capacity'] != 0 and psngr['destination'] not in self.impassible_location:
                    cur_actions.append(("pick up", self.taxi_names[of_player - 1][k], self.passengers_names[j]))

            # drop off
            for j, psngr in enumerate(from_state['passengers'].values()):
                if psngr['location'] == self.taxi_names[of_player - 1][k] and taxi['location'] == psngr['destination']:
                    cur_actions.append(("drop off", self.taxi_names[of_player - 1][k], self.passengers_names[j]))

            # movement
            cur_x_loc, cur_y_loc = taxi['location'][0], taxi['location'][1]

            if cur_x_loc + 1 < self.map_size[0] and from_state['map'][cur_x_loc + 1][cur_y_loc] != 'I' and tuple(
                    [cur_x_loc + 1, cur_y_loc]) not in occupied_locations:
                cur_actions.append(("move", self.taxi_names[of_player - 1][k], [cur_x_loc + 1, cur_y_loc]))

            if cur_x_loc - 1 >= 0 and from_state['map'][cur_x_loc - 1][cur_y_loc] != 'I' and tuple(
                    [cur_x_loc - 1, cur_y_loc]) not in occupied_locations:
                cur_actions.append(("move", self.taxi_names[of_player - 1][k], [cur_x_loc - 1, cur_y_loc]))

            if cur_y_loc + 1 < self.map_size[1] and from_state['map'][cur_x_loc][cur_y_loc + 1] != 'I' and tuple(
                    [cur_x_loc, cur_y_loc + 1]) not in occupied_locations:
                cur_actions.append(("move", self.taxi_names[of_player - 1][k], [cur_x_loc, cur_y_loc + 1]))

            if cur_y_loc - 1 >= 0 and from_state['map'][cur_x_loc][cur_y_loc - 1] != 'I' and tuple(
                    [cur_x_loc, cur_y_loc - 1]) not in occupied_locations:
                cur_actions.append(("move", self.taxi_names[of_player - 1][k], [cur_x_loc, cur_y_loc - 1]))

            # wait
            cur_actions.append(("wait", self.taxi_names[of_player - 1][k]))

            actions_per_taxi.append(cur_actions)

        if self.taxi_amount[of_player - 1] == 1:
            return actions_per_taxi[0]

        # creating disturb matrix - which taxis effect each other when choosing an action
        state_len = self.taxi_amount[of_player - 1]
        can_disturb = [[False for _ in range(state_len)] for _ in range(state_len)]
        for i in range(state_len):
            for j in range(i + 1, state_len):
                taxi_i_loc = from_state['taxis'][self.taxi_names[of_player - 1][k]]['location']
                taxi_j_loc = from_state['taxis'][self.taxi_names[of_player - 1][k]]['location']
                if UCTAgent.manhattan_dist(taxi_i_loc, taxi_j_loc) < 3:
                    can_disturb[i][j] = True
                    can_disturb[j][i] = True

        def legal(built_action, action):
            cur_len = len(built_action)

            if self.player_number == 1:
                step = 0
            else:
                step = self.taxi_amount[0]

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
                            if tuple(action[2]) == list(from_state['taxis'].values())[step+k]['location']:
                                return False
                    # want to add 'stay' action
                    else:
                        if built_action[k][0] == 'move':
                            if tuple(built_action[k][2]) == list(from_state['taxis'].values())[step+cur_len]['location']:
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

        rec_loop(actions_per_taxi, 0, self.taxi_amount[of_player - 1], possible_action, [])
        return tuple(possible_action)

    def run_actions(self, from_state, actions, starting_player):
        """
        find the resulting state by performing series of actions from given state
        :param from_state: initial state
        :param actions: action to perform from initial state
        :param starting_player: the player who play first in initial state
        :return: resulting state
        """
        next_state = copy.deepcopy(from_state)
        cur_player = starting_player
        for action in actions:
            next_state, _ = self.perform_action(next_state, action, cur_player)
            cur_player = int(cur_player % 2 + 1)

        return next_state

    def perform_action(self, from_state, with_action, of_plyer):
        """
        given a state and an action, return the next state and the reward earned
        :param from_state: current state
        :param with_action: current action to take
        :param of_plyer: the player taking the action
        :return: the new state after the action, the reward the player earned
        """
        reward = 0
        new_state = copy.deepcopy(from_state)

        # fit with action to work in for loop when there is only one taxi
        if self.taxi_amount[of_plyer - 1] == 1:
            with_action = [with_action]

        for i, action in enumerate(with_action):
            if action[0] == 'wait':
                continue
            if action[0] == 'move':
                new_state['taxis'][self.taxi_names[of_plyer - 1][i]]['location'] = tuple(action[-1])
                continue
            if action[0] == 'pick up':
                new_state['taxis'][self.taxi_names[of_plyer - 1][i]]['capacity'] -= 1
                new_state['passengers'][action[-1]]['location'] = self.taxi_names[of_plyer - 1][i]
                continue
            if action[0] == 'drop off':
                new_state['taxis'][self.taxi_names[of_plyer - 1][i]]['capacity'] += 1
                new_state['passengers'][action[-1]]['location'] = new_state['taxis'][self.taxi_names[of_plyer - 1][i]][
                    'location']
                reward += new_state['passengers'][action[-1]]['reward']
        # update time left
        new_state['turns to go'] -= 0.5  # each turn includes 2 actions

        return new_state, reward

    # ---------- MCTS functions ----------

    def selection(self, UCT_tree, root_state):
        """
        move from root to expandable node using UCB1 for choosing path
        :param UCT_tree: root to a tree where each node represent series of actions
        :param root_state: the state fit to the root (no actions) of the tree
        :return: node to expand
        """
        cur_node = UCT_tree.root
        while True:
            # get the state corresponding to cur_node
            cur_state = self.run_actions(root_state, cur_node.actions, self.player_number) # also should work: UCT_tree.root.of_player
            # get possible actions to take from that state
            cur_possible_actions = self.all_possible_actions(cur_state, cur_node.of_player)
            # if more actions than kids then need to add kid
            if len(cur_possible_actions) > len(cur_node.kids):
                return cur_node
            # all action have been explored, choose next node based on UCB1
            else:
                max_ucb1, maximize_node = -np.inf, None
                for kid in cur_node.kids:
                    cur_ucb1 = UCTAgent.UCB1(kid.n, kid.parent.n, kid.x)
                    if cur_ucb1 > max_ucb1:
                        max_ucb1 = cur_ucb1
                        maximize_node = kid
                cur_node = maximize_node

    def expansion(self, UCT_tree, state, parent_node):
        """
        :param UCT_tree:
        :param state: the initial state corresponding to the tree root
        :param parent_node: the parent of the node (that still doesn't exist) which we next simulate from
        :return: a node to expand (to add to the tree)
        """
        cur_state = self.run_actions(state, parent_node.actions, self.player_number) # was parent_node.of_player
        cur_possible_actions = self.all_possible_actions(cur_state, parent_node.of_player)

        # randomize actions
        cur_possible_actions = list(cur_possible_actions)
        random.shuffle(cur_possible_actions)
        cur_possible_actions = tuple(cur_possible_actions)


        # find and choose node that does not exist to expand
        found = False
        for action in cur_possible_actions:
            path = parent_node.actions + [action]
            for kid in parent_node.kids:
                if kid.actions == path:
                    found = True
                    break
            if found is False:
                # if got here, no node with current path of actions
                new_node = Node(path, int(parent_node.of_player % 2 + 1), parent=parent_node)
                parent_node.kids.append(new_node)
                return new_node
            else:
                found = False

    def simulation(self, state, from_node, time_limit=0.9*tl):
        """
        :param state: initial state corresponding to tree root
        :param from_node: node to simulate from
        :param time_limit: time allowed
        :return: estimated reward
        """
        start_time = time.time()
        reward_player_1, reward_player_2 = 0, 0

        # get the state corresponding to from_node
        current_state = self.run_actions(state, from_node.actions, self.player_number)
        cur_state = copy.deepcopy(current_state)
        cur_player = from_node.of_player

        # run simulation
        while time.time() - start_time < time_limit and cur_state['turns to go'] != 0:

            possible_actions = self.all_possible_actions(cur_state, cur_player)
            current_action = UCTAgent.random_choose(possible_actions)
            cur_state, cur_reward = self.perform_action(cur_state, current_action, cur_player)

            # update reward earned
            if cur_player == 1:
                reward_player_1 += cur_reward
            else:
                reward_player_2 += cur_reward

            # update which player plays next
            cur_player = int(cur_player % 2 + 1)

        # if game not finished, evaluate state using heuristic
        if cur_state['turns to go'] != 0:
            reward_player_1 += self.heuristic(cur_state)

        return [reward_player_1, reward_player_2]

    def backpropagation(self, simulation_result, from_node):

        cur_node = from_node
        while cur_node.parent is not None:

            if cur_node.of_player == 1:
                reward = simulation_result[1]
            else:
                reward = simulation_result[0]

            cur_node.x = (cur_node.x * cur_node.n + reward) / (cur_node.n + 1)
            cur_node.n += 1
            cur_node = cur_node.parent

        cur_node.n += 1

    # ---------- main function ----------

    def act(self, state):
        start_time = time.time()

        self.passengers_names = list(state['passengers'].keys())
        uct_tree = Tree(Node([], self.player_number))

        # perform simulation to choose next action
        while time.time() - start_time < 0.95*tl:

            node_to_expand = self.selection(uct_tree, state)

            node_to_simulate = self.expansion(uct_tree, state, node_to_expand)

            estimated_reward = self.simulation(state, node_to_simulate, time_limit=tl - time.time() + start_time)

            self.backpropagation(estimated_reward, node_to_simulate)

        # find best estimated action based on the simulation
        max_reward, best_action = -1, None
        for kid in uct_tree.root.kids:
            if kid.x > max_reward:
                max_reward = kid.x
                best_action = kid.actions[0]

        # convert output to required format
        if len(self.taxi_names[self.player_number-1]) == 1:
            best_action = (best_action,)
        best_action = list(best_action)
        for i, action in enumerate(best_action):
            best_action[i] = list(best_action[i])
            if action[0] == 'move':
                best_action[i][-1] = tuple(best_action[i][-1])
            best_action[i] = tuple(best_action[i])
        best_action = tuple(best_action)

        return best_action

    # ---------- Agent auxiliary functions ----------
    @staticmethod
    def UCB1(n, t, x):
        """
        given a state n,t,x parameters, choose which arm to pull
        :param n: number of times arm i has been pulled
        :param t: number of times the parent of arm i has been pulled
        :param x: average observed reward so far for arm i
        :return: UCB1 score
        """
        return x + np.sqrt((2 * np.log(t)) / n)

    @staticmethod
    def heuristic(from_state):
        return 0

    @staticmethod
    def random_choose(actions_arr):
        return actions_arr[np.random.randint(0, len(actions_arr))]

    @staticmethod
    def manhattan_dist(loc_1, loc_2):
        return abs(loc_2[0] - loc_1[0]) + abs(loc_2[1] - loc_1[1])
