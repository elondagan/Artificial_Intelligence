import copy
import math
import time
import numpy as np
import random
import ast

from utilities import utils


class STRIPS:

    def __init__(self, V, I, A, G):
        """
        param V: set of state variables (atoms), each state variable is true of false
        param I: initial state - subset over V (atoms that are True)
        param A: dictionaries of dictionaries of the form {action: {pre, add, del}, ... }
        param G: goal state - subset over V (atoms that are True)
        """
        self.V = V
        self.I = I
        self.A = A
        self.G = G

    def actions(self, state):
        """
        param state: set of atoms that are true
        return: all possible actions (as keys names) to take from the given state
        """
        possible_actions = []
        for action_name in self.A.keys():
            if all(atom in state for atom in self.A[action_name]['pre']):
                possible_actions.append(action_name)
        return possible_actions

    def result(self, state, action):
        """
        :param state: set of atoms that are true
        :param action: dictionary with pre, add, del keys
        :return: resulting state
        """
        next_state = copy.deepcopy(state)
        # apply 'delete effects'
        for atom in state:
            if atom in self.A[action]['del']:
                next_state.remove(atom)
        # apply 'add effects'
        for atom in self.A[action]['add']:
            next_state.add(atom)
        return next_state

    def is_goal(self, state):
        for atom in self.G:
            if atom not in state:
                return False
        return True

    def path_cost(self, c, state1, action, state2):
        return c + 1


""" STRIPS Domains """


class Npuzzle(STRIPS):

    def __init__(self, N, initial=None):
        self.times = []

        self.N = N + 1
        # infer puzzle's width and height (for now, assume that width=height)
        self.width = int(math.sqrt(self.N))
        self.height = int(math.sqrt(self.N))

        # create state variables (atoms)
        V = set()
        for place in range(1, self.width * self.height + 1):
            for num in range(1, self.width * self.height):
                V.add(f"at({num},{place})")
            V.add(f"at(0,{place})")

        # create action set
        A = {}
        for i in range(self.height):
            for j in range(self.width):
                for num in range(1, self.width * self.height):
                    blank_location = i * self.width + j + 1
                    # move up
                    if 0 < i:
                        A[f"move_b({blank_location},{blank_location - self.width},{num})"] = {
                            "pre": [f"at(0,{blank_location})", f"at({num},{blank_location - self.width})"],
                            "add": [f"at(0,{blank_location - self.width})", f"at({num},{blank_location})"],
                            "del": [f"at(0,{blank_location})", f"at({num},{blank_location - self.width})"]}
                    # move down
                    if i < self.height - 1:
                        A[f"move_b({blank_location},{blank_location + self.width},{num})"] = {
                            "pre": [f"at(0,{blank_location})", f"at({num},{blank_location + self.width})"],
                            "add": [f"at(0,{blank_location + self.width})", f"at({num},{blank_location})"],
                            "del": [f"at(0,{blank_location})", f"at({num},{blank_location + self.width})"]}
                    # move left
                    if 0 < j:
                        A[f"move_b({blank_location},{blank_location - 1},{num})"] = {
                            "pre": [f"at(0,{blank_location})", f"at({num},{blank_location - 1})"],
                            "add": [f"at(0,{blank_location - 1})", f"at({num},{blank_location})"],
                            "del": [f"at(0,{blank_location})", f"at({num},{blank_location - 1})"]}
                    # move right
                    if j < self.width - 1:
                        A[f"move_b({blank_location},{blank_location + 1},{num})"] = {
                            "pre": [f"at(0,{blank_location})", f"at({num},{blank_location + 1})"],
                            "add": [f"at(0,{blank_location + 1})", f"at({num},{blank_location})"],
                            "del": [f"at(0,{blank_location})", f"at({num},{blank_location + 1})"]}

        # create random initial state
        I = set()
        if initial is None:
            unique_numbers = random.sample(range(1, self.N + 1), self.width * self.height)
            for k, un in enumerate(unique_numbers):
                if un != self.N:
                    I.add(f"at({un},{k + 1})")
                else:
                    I.add(f"at(0,{k + 1})")
        else:
            initial = initial.split('.')
            if len(initial) == 1:
                initial = initial[0]
            for i, num in enumerate(initial):
                I.add(f"at({num},{i + 1})")
        # create goal state
        G = set()
        for num in range(self.N - 1):
            G.add(f"at({num + 1},{num + 1})")

        super().__init__(V, I, A, G)

    def print_state(self, state):
        """ print state as array """
        arr = [[None for _ in range(self.width)] for _ in range(self.height)]
        for atom in state:
            loc = ast.literal_eval(atom[2:])
            i, j = utils.row_col_of_number(loc[1], self.width, self.height)
            arr[i][j] = loc[0]
        for row in arr:
            print(row)

    def flat_state(self, state):
        """ convert array to single string"""
        fs = ''
        arr = [[None for _ in range(self.width)] for _ in range(self.height)]
        for atom in state:
            nums = ast.literal_eval(atom[2:])
            i, j = utils.row_col_of_number(nums[1], self.width, self.height)
            arr[i][j] = nums[0]
        for row in arr:
            for v in row:
                fs += str(v) + '.'
        return fs

    def task_rep(self):
        """represent task as string"""
        fs = ''
        arr = [[None for _ in range(self.width)] for _ in range(self.height)]
        for atom in self.I:
            nums = ast.literal_eval(atom[2:])
            i, j = utils.row_col_of_number(nums[1], self.width, self.height)
            arr[i][j] = nums[0]
        for row in arr:
            for v in row:
                fs += str(v) + '.'
        return [fs, '-']

    """ Npuzzle heuristic """

    def h_all_blank(self, node, revers=False):
        """ sum of euclidian distances between each tile and its location (blank not included) """

        st = time.time()

        def find_number_location(s, number):
            for atom in s:
                try:
                    atom_literal = ast.literal_eval(atom[2:])
                    if atom_literal[0] == number:
                        row = math.ceil(atom_literal[1] / self.height) - 1
                        col = atom_literal[1] - 1 - row * self.width
                        return row, col
                except:
                    continue

        # def state_to_array(state_to_convert):
        #     res_array = [[None for _ in range(self.width)] for _ in range(self.height)]
        #     for atom in state_to_convert:
        #         tuple_atom = ast.literal_eval(atom[2:])
        #         loc = find_number_location(state_to_convert, tuple_atom[0])
        #         res_array[loc[0]][loc[1]] = tuple_atom[0]
        #     return res_array


        state = node.state
        max_num = self.height * self.width  # for puzzle-2b add: -1

        # Manhattan distance based
        goal_state = copy.deepcopy(self.G)
        h = 0
        if revers is False:
            a, b = 1, int(max_num)
        else:
            a, b = int(max_num), 1
        for num in range(a, b):
            loc1 = find_number_location(goal_state, num)
            loc2 = find_number_location(state, num)
            h += utils.euclidian_distance(loc1, loc2)

        self.times.append(time.time() - st)

        if h > 20:
            h -= 5
        elif h > 15:
            h -= 4
        elif h > 10:
            h -= 3
        elif h > 5:
            h -= 2
        elif h > 2:
            h -= 1

        return h

    def h_tiles(self, node):
        st = time.time()
        h = 0
        for a in node.state:
            tile = a[3]
            loc = a[5]
            for a2 in self.G:
                if a2[3] == tile:
                    if a2[5] != loc:
                        h += 1
        self.times.append(time.time() - st)
        return h



class BlocksWorld(STRIPS):

    def __init__(self, num_of_blocks, initial=None):
        self.times = []

        def from_arr_to_state(arr):
            state = set()
            for sub_arr in arr:
                arr_len = len(sub_arr)
                for i in range(arr_len):
                    if i == 0:
                        if arr_len > 1:
                            state.add(f"{sub_arr[i]}_on_Table")
                        else:
                            state.add(f"{sub_arr[i]}_on_Table")
                            state.add(f"{sub_arr[i]}_free")
                    elif i == arr_len - 1 and arr_len > 1:
                        state.add(f"{sub_arr[i]}_on_{sub_arr[i - 1]}")
                        state.add(f"{sub_arr[i]}_free")
                    elif 0 < i < arr_len - 1:
                        state.add(f"{sub_arr[i]}_on_{sub_arr[i - 1]}")
            return state

        V = set()
        blocks = [chr(ord('A') + i) for i in range(num_of_blocks)]
        for bi in blocks:
            for bottom in blocks + ['Table']:
                if bottom != bi:
                    V.add(f"{bi}_on_{bottom}")
                    V.add(f"{bi}_free")

        A = {}
        for bi in blocks:
            for old_bottom in blocks:
                if bi == old_bottom:
                    continue
                for new_bottom in blocks:
                    if bi == new_bottom or new_bottom == old_bottom:
                        continue
                    # move block from one pile to the other
                    A[f"move_{bi}_from_{old_bottom}_to_{new_bottom}"] = {
                        "pre": [f"{bi}_on_{old_bottom}", f"{bi}_free", f"{new_bottom}_free"],
                        "add": [f"{bi}_on_{new_bottom}", f"{old_bottom}_free"],
                        "del": [f"{bi}_on_{old_bottom}", f"{new_bottom}_free"]
                    }
                # move block from pile to table
                A[f"move_{bi}_from_{old_bottom}_to_Table"] = {
                    "pre": [f"{bi}_on_{old_bottom}", f"{bi}_free"],
                    "add": [f"{bi}_on_Table", f"{old_bottom}_free"],
                    "del": [f"{bi}_on_{old_bottom}"]
                }
                # move block from table to pile
                A[f"move_{bi}_from_Table_to_{old_bottom}"] = {
                    "pre": [f"{bi}_on_Table", f"{bi}_free", f"{old_bottom}_free"],
                    "add": [f"{bi}_on_{old_bottom}"],
                    "del": [f"{bi}_on_Table", f"{old_bottom}_free"]
                }

        # fixed to 2 initial piles and 2 final piel

        self.arr_I = None
        if initial is None:
            shuffled_blocks = copy.deepcopy(blocks)
            random.shuffle(shuffled_blocks)
            num_of_piles = 2  # random.randrange(1, 4)
            i = [[] for _ in range(num_of_piles)]
            while shuffled_blocks:
                block = shuffled_blocks.pop()
                i[random.randrange(0, num_of_piles)].append(block)
            self.arr_I = i
            I = from_arr_to_state(i)
        else:
            self.arr_I = initial[0]
            I = from_arr_to_state(initial[0])

        self.arr_G = None
        if initial is None:
            shuffled_blocks = copy.deepcopy(blocks)
            random.shuffle(shuffled_blocks)
            num_of_piles = 1  # random.randrange(1, 3)
            g = [[] for _ in range(num_of_piles)]
            while shuffled_blocks:
                block = shuffled_blocks.pop()
                g[random.randrange(0, num_of_piles)].append(block)
            self.arr_G = g
            G = from_arr_to_state(g)
        else:
            self.arr_G = initial[1]
            G = from_arr_to_state(initial[1])

        super().__init__(V, I, A, G)

    def task_rep(self):
        return [self.arr_I, self.arr_G]

    def h_blocks(self, node):
        st = time.time()
        h = 0
        for atom in node.state:
            if atom[-4:] != 'free' and atom not in self.G:
                h += 1
        self.times.append(time.time() - st)
        return h


class Trucks(STRIPS):

    def __init__(self, size, num_of_trucks, num_of_packs, initial=None):
        self.times = []

        self.num_of_trucks = num_of_trucks
        self.num_of_packs = num_of_packs

        def from_arr_to_I(arr):
            I = set()
            t = 0
            for i in range(len(arr)):
                for j in range(len(arr)):
                    if arr[i][j] != 0:
                        if arr[i][j][0] == 't':
                            I.add(f"t{arr[i][j][1]}_at_({i},{j})")
                            I.add(f"t{t + 1}_empty")
                            t += 1
                        elif arr[i][j][0] == 'p':
                            I.add(f"p{arr[i][j][1]}_at_({i},{j})")
            return I

        V = set()
        for t in range(num_of_trucks):
            V.add(f"t{t + 1}_full")
            V.add(f"t{t + 1}_empty")
        for p in range(num_of_packs):
            for t in range(num_of_trucks):
                V.add(f"p{p + 1}_at_t{t + 1}")
        for i in range(size):
            for j in range(size):
                for t in range(num_of_trucks):
                    V.add(f"t{t + 1}_at_({i},{j})")
                for p in range(num_of_packs):
                    V.add(f"p{p + 1}_at_({i},{j})")

        A = {}
        # move up
        for i in range(size):
            for j in range(size):
                for t in range(num_of_trucks):
                    # move up
                    if 0 < i:
                        A[f"move_t{t + 1}_from_({i},{j})_to_({i - 1},{j})"] = {
                            "pre": [f"t{t + 1}_at_({i},{j})", f"t{t + 1}_empty"],
                            "add": [f"t{t + 1}_at_({i - 1},{j})"],
                            "del": [f"t{t + 1}_at_({i},{j})"]
                        }
                    # move down
                    if i < size - 1:
                        A[f"move_t{t + 1}_from_({i},{j})_to_({i + 1},{j})"] = {
                            "pre": [f"t{t + 1}_at_({i},{j})", f"t{t + 1}_empty"],
                            "add": [f"t{t + 1}_at_({i + 1},{j})"],
                            "del": [f"t{t + 1}_at_({i},{j})"]
                        }
                    # move left
                    if 0 < j:
                        A[f"move_t{t + 1}_from_({i},{j})_to_({i},{j - 1})"] = {
                            "pre": [f"t{t + 1}_at_({i},{j})", f"t{t + 1}_empty"],
                            "add": [f"t{t + 1}_at_({i},{j - 1})"],
                            "del": [f"t{t + 1}_at_({i},{j})"]
                        }
                    # move right
                    if j < size - 1:
                        A[f"move_t{t + 1}_from_({i},{j})_to_({i},{j + 1})"] = {
                            "pre": [f"t{t + 1}_at_({i},{j})", f"t{t + 1}_empty"],
                            "add": [f"t{t + 1}_at_({i},{j + 1})"],
                            "del": [f"t{t + 1}_at_({i},{j})"]
                        }

                    for p in range(num_of_packs):
                        # carry up
                        if 0 < i:
                            A[f"carry_p{p + 1}_with_t{t + 1}_from_({i},{j})_to_({i - 1},{j})"] = {
                                "pre": [f"t{t + 1}_at_({i},{j})", f"p{p + 1}_at_t{t + 1}"],
                                "add": [f"t{t + 1}_at_({i - 1},{j})"],
                                "del": [f"t{t + 1}_at_({i},{j})"]
                            }
                        # carry down
                        if i < size - 1:
                            A[f"carry_p{p + 1}_with_t{t + 1}_from_({i},{j})_to_({i + 1},{j})"] = {
                                "pre": [f"t{t + 1}_at_({i},{j})", f"p{p + 1}_at_t{t + 1}"],
                                "add": [f"t{t + 1}_at_({i + 1},{j})"],
                                "del": [f"t{t + 1}_at_({i},{j})"]
                            }
                        # carry left
                        if 0 < j:
                            A[f"carry_p{p + 1}_with_t{t + 1}_from_({i},{j})_to_({i},{j - 1})"] = {
                                "pre": [f"t{t + 1}_at_({i},{j})", f"p{p + 1}_at_t{t + 1}"],
                                "add": [f"t{t + 1}_at_({i},{j - 1})"],
                                "del": [f"t{t + 1}_at_({i},{j})"]
                            }
                        # carry right
                        if j < size - 1:
                            A[f"carry_p{p + 1}_with_t{t + 1}_from_({i},{j})_to_({i},{j + 1})"] = {
                                "pre": [f"t{t + 1}_at_({i},{j})", f"p{p + 1}_at_t{t + 1}"],
                                "add": [f"t{t + 1}_at_({i},{j + 1})"],
                                "del": [f"t{t + 1}_at_({i},{j})"]
                            }

                        # pick up
                        A[f"pick_p{p + 1}_with_t{t + 1}_at_({i},{j})"] = {
                            "pre": [f"t{t + 1}_at_({i},{j})", f"p{p + 1}_at_({i},{j})", f"t{t + 1}_empty"],
                            "add": [f"p{p + 1}_at_t{t + 1}", f"t{t + 1}_full"],
                            "del": [f"p{p + 1}_at_({i},{j})", f"t{t + 1}_empty"]
                        }

                        # drop off
                        A[f"drop_p{p + 1}_with_t{t + 1}_at_({i},{j})"] = {
                            "pre": [f"t{t + 1}_at_({i},{j})", f"p{p + 1}_at_t{t + 1}"],
                            "add": [f"p{p + 1}_at_({i},{j})", f"t{t + 1}_empty"],
                            "del": [f"p{p + 1}_at_t{t + 1}", f"t{t + 1}_full"]
                        }

        self.arr_I = None
        if initial is None:
            arr_I = [[0 for _ in range(size)] for _ in range(size)]
            used = []

            t = 0
            while t < self.num_of_trucks:
                i, j = random.randrange(0, size), random.randrange(0, size)
                if (i, j) in used:
                    continue
                arr_I[i][j] = f"t{t + 1}"
                used.append((i, j))
                t += 1
            p = 0
            while p < self.num_of_packs:
                i, j = random.randrange(0, size), random.randrange(0, size)
                if (i, j) in used:
                    continue
                arr_I[i][j] = f"p{p + 1}"
                used.append((i, j))
                p += 1
            self.arr_I = arr_I
            I = from_arr_to_I(arr_I)
        else:
            self.arr_I = initial[0]
            I = from_arr_to_I(initial[0])

        G = set()
        self.arr_G = None
        if initial is None:
            arr_g = []
            for p in range(num_of_packs):
                i, j = random.randrange(0, size), random.randrange(0, size)
                arr_g.append([i, j])
                G.add(f"p{p + 1}_at_({i},{j})")
            self.arr_G = arr_g
        else:
            self.arr_G = initial[1]
            for p, loc in enumerate(initial[1]):
                G.add(f"p{p + 1}_at_({loc[0]},{loc[1]})")

        super().__init__(V, I, A, G)

    def task_rep(self):
        return [self.arr_I, self.arr_G]

    def h_trucks(self, node):
        st = time.time()
        # find packages and trucks destinations
        ps_locations = {}
        ps_destinations = {}
        for p in range(self.num_of_packs):
            ps_locations[f"p{p + 1}"] = None
            ps_destinations[f"p{p + 1}"] = None
        ts_locations = {}
        ts_picked = []
        for t in range(self.num_of_trucks):
            ts_locations[f"t{t + 1}"] = None
        for atom in node.state:
            if atom[0] == 't' and atom[-1] == ')':
                ts_locations[atom[0:2]] = ast.literal_eval(atom[-5:])

        for atom in node.state:
            if atom[0] == 'p':
                if atom[-2] == 't':
                    ts_picked.append(f"p{atom[1]}")
                    ps_locations[atom[0:2]] = ts_locations[atom[-2:]]
                else:
                    ps_locations[atom[0:2]] = ast.literal_eval(atom[-5:])
        for atom in self.G:
            ps_destinations[atom[0:2]] = ast.literal_eval(atom[-5:])

        # arrived packages
        ps_arrived = 0
        for p in ps_locations.keys():
            if ps_locations[p] == ps_destinations[p] and p not in ts_picked:
                ps_arrived += 1

        # picked packages
        ps_picked = len(ts_picked)

        h = 0
        # distance between package location to package destination
        for p in range(1, self.num_of_packs + 1):
            h += utils.euclidian_distance(ps_locations[f"p{p}"], ps_destinations[f"p{p}"])

        if ps_picked == 0:
            t_to_p = np.inf
            for t in ts_locations.values():
                for p_loc, p_dest in zip(ps_locations.values(), ps_destinations.values()):
                    if p_loc == p_dest:
                        continue
                    d = utils.euclidian_distance(t, p_loc)
                    if d < t_to_p:
                        t_to_p = d
            if t_to_p == np.inf:
                t_to_p = 0
            h += t_to_p

        h += (ps_picked + (self.num_of_packs - ps_arrived - ps_picked) * 2)

        self.times.append(time.time() - st)

        if h > 20:
            h -= 6
        elif h > 15:
            h -= 5
        elif h > 10:
            h -= 4
        elif h > 5:
            h -= 3
        elif h > 2:
            h -= 1

        return h


class VacuumCleaner(STRIPS):

    def __init__(self, size, initial=None):

        self.times = []

        def from_arr_to_I(arr):
            """
            0 = blocked
            1 = clear and clean
            2 = clear and dirty
            3 = clear and clean with vacuum
            """
            state = set()
            for i in range(len(arr)):
                for j in range(len(arr[0])):
                    if arr[i][j] == 0:
                        # continue
                        state.add(f"clean_at_({i},{j})")
                        # state.add(f"tile_({i},{j})_free")
                    elif arr[i][j] == 1:
                        state.add(f"tile_({i},{j})_free")
                        state.add(f"clean_at_({i},{j})")
                    elif arr[i][j] == 2:
                        state.add(f"tile_({i},{j})_free")
                        state.add(f"dirt_at_({i},{j})")
                    elif arr[i][j] == 3:
                        state.add(f"tile_({i},{j})_free")
                        state.add(f"at({i},{j})")
                        state.add(f"clean_at_({i},{j})")
            return state

        V = set()
        for i in range(size):
            for j in range(size):
                V.add(f"at({i},{j})")
                V.add(f"dirt_at_({i},{j})")
                V.add(f"clean_at_({i},{j})")
                V.add(f"tile_({i},{j})_free")

        A = {}
        for i in range(size):
            for j in range(size):
                # clean
                A[f"clean({i},{j})"] = {
                    "pre": [f"at({i},{j})", f"dirt_at_({i},{j})"],
                    "add": [f"clean_at_({i},{j})"],
                    "del": [f"dirt_at_({i},{j})"]
                }
                # move up
                if 0 < i:
                    A[f"move_from_({i},{j})_to_({i - 1},{j})"] = {
                        "pre": [f"at({i},{j})", f"tile_({i - 1},{j})_free"],
                        "add": [f"at({i - 1},{j})"],
                        "del": [f"at({i},{j})"]
                    }
                # move down
                if i < size - 1:
                    A[f"move_from_({i},{j})_to_({i + 1},{j})"] = {
                        "pre": [f"at({i},{j})", f"tile_({i + 1},{j})_free"],
                        "add": [f"at({i + 1},{j})"],
                        "del": [f"at({i},{j})"]
                    }
                # move left
                if 0 < j:
                    A[f"move_from_({i},{j})_to_({i},{j - 1})"] = {
                        "pre": [f"at({i},{j})", f"tile_({i},{j - 1})_free"],
                        "add": [f"at({i},{j - 1})"],
                        "del": [f"at({i},{j})"]
                    }
                # move right
                if j < size - 1:
                    A[f"move_from_({i},{j})_to_({i},{j + 1})"] = {
                        "pre": [f"at({i},{j})", f"tile_({i},{j + 1})_free"],
                        "add": [f"at({i},{j + 1})"],
                        "del": [f"at({i},{j})"]
                    }

        self.arr_I = None
        if initial is None:
            arr_i = [[1 for _ in range(size)] for _ in range(size)]
            v_i, v_j = random.randrange(0, size), random.randrange(0, size)
            arr_i[v_i][v_j] = 3
            for i in range(size):
                for j in range(size):
                    if arr_i[i][j] == 3:
                        continue
                    r = random.random()
                    if r < 0.2:
                        arr_i[i][j] = 2
                    elif r < 0.3:
                        arr_i[i][j] = 0
            self.arr_I = arr_i
            I = from_arr_to_I(arr_i)
        else:
            self.arr_I = initial
            I = from_arr_to_I(initial)

        G = set()
        for i in range(size):
            for j in range(size):
                G.add(f"clean_at_({i},{j})")

        super().__init__(V, I, A, G)

    def task_rep(self):
        return [self.arr_I, '-']

    def h_vacuum(self, node):
        st = time.time()

        state = copy.deepcopy(node.state)
        r, dirt = None, []
        for atom in state:
            if atom[0] == 'a':
                r = [(int(atom[3]), int(atom[5]))]
            elif atom[0] == 'd':
                dirt.append((int(atom[-4]), int(atom[-2])))

        dists = [utils.euclidian_distance(r[0], dirt_loc) for dirt_loc in dirt]

        # print(dists)
        if len(dists) == 0:
            h = 0
        else:
            # h = np.average(dists)
            h = max(dists)
            h += len(dirt)



        self.times.append(time.time() - st)
        return h


class Elevator(STRIPS):

    def __init__(self, size, num_of_passengers, capacity, initial=None):
        self.size = size
        self.num_of_passengers = num_of_passengers
        self.times = []

        V = set()
        for i in range(size):
            V.add(f"elevator_at_{i}")
            for p in range(num_of_passengers):
                V.add(f"p{p + 1}_at_{i}")
        for p in range(num_of_passengers):
            V.add(f"p{p + 1}_in_elevator")
        for c in range(capacity + 1):
            V.add(f"elevator_hold_{c}")

        A = {}
        for i in range(size):
            # move up
            if i < size - 1:
                A[f"up_to_{i + 1}"] = {
                    "pre": [f"elevator_at_{i}"],
                    "add": [f"elevator_at_{i + 1}"],
                    "del": [f"elevator_at_{i}"]
                }
            # move down
            if i > 0:
                A[f"down_to_{i - 1}"] = {
                    "pre": [f"elevator_at_{i}"],
                    "add": [f"elevator_at_{i - 1}"],
                    "del": [f"elevator_at_{i}"]
                }
            # pick ups
            for i in range(size):
                for p in range(num_of_passengers):
                    for c in range(1, capacity + 1):
                        A[f"pick_p{p + 1}_at_{i}_hold_{c}"] = {
                            "pre": [f"elevator_at_{i}", f"p{p + 1}_at_{i}", f"elevator_hold_{c - 1}"],
                            "add": [f"p{p + 1}_in_elevator", f"elevator_hold_{c}"],
                            "del": [f"p{p + 1}_at_{i}", f"elevator_hold_{c - 1}"]
                        }
            # drop ofs
            for i in range(size):
                for p in range(num_of_passengers):
                    for c in range(capacity):
                        A[f"drop_p{p + 1}_at_{i}_hold_{c}"] = {
                            "pre": [f"elevator_at_{i}", f"p{p + 1}_in_elevator", f"elevator_hold_{c + 1}"],
                            "add": [f"p{p + 1}_at_{i}", f"elevator_hold_{c}"],
                            "del": [f"p{p + 1}_in_elevator", f"elevator_hold_{c + 1}"]
                        }

        I = set()
        I.add(f"elevator_hold_0")
        I.add(f"elevator_at_{random.randrange(size)}")
        for p in range(num_of_passengers):
            I.add(f"p{p + 1}_at_{random.randrange(size)}")

        G = set()
        for p in range(num_of_passengers):
            G.add(f"p{p + 1}_at_{random.randrange(size)}")



        super().__init__(V, I, A, G)

    def task_rep(self):
        return [str(self.I), str(self.G)]

    def h_elevator(self, node):
        st = time.time()

        h = 0

        p_locations = [0 for _ in range(self.num_of_passengers)]
        p_destinations = [0 for _ in range(self.num_of_passengers)]
        elevator_location = 0


        for atom in node.state:
            if atom[0] == 'p':
                p_num = int(atom[1])
                if atom[-1] == 'r':  # in elevator
                    p_locations[p_num-1] = 'e'
                else:
                    p_locations[p_num-1] = int(atom[-2:]) if self.size > 10 else int(atom[-1])
            elif atom[9] == 'a':
                elevator_location = int(atom[-2:]) if self.size > 10 else int(atom[-1])

        for atom in self.G:
            if atom[0] == 'p':
                p_destinations[int(atom[1]) - 1] = int(atom[-2:]) if self.size > 10 else int(atom[-1])

        min_dist_left = 0
        ele_dist = 0
        for i in range(self.num_of_passengers):
            if p_destinations[i] != p_locations[i]:

                if p_locations[i] == 'e':
                    h += 1
                else:
                    h += 2
                    # if abs(p_destinations[i]-p_locations[i]) > min_dist_left:
                    #     min_dist_left = abs(p_destinations[i]-p_locations[i])

                    if abs(p_locations[i]-elevator_location) > ele_dist:
                        ele_dist = abs(p_locations[i]-elevator_location)

        h += min_dist_left
        h += ele_dist

        # self.times.append(time.time() - st)

        return h

