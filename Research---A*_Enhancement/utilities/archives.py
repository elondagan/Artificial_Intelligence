

""" heuristics declaration """

# h_lmc = LmCutHeuristic(task)
# f_lmc = lambda n: n.path_cost + h_lmc(n)

# h_lm = LandmarkHeuristic(task)
# f_lm = lambda n: n.path_cost + h_lm(n)

# h_m = hMaxHeuristic(task)
# f_m = lambda n: n.path_cost + h_m(n)

# h_blank = task.h_all_blank
# f_blank = lambda n: n.path_cost + h_blank(n)


" consistence check "

# for _ in range(1000):
#     task = domains.Npuzzle(8)
#     # task = domains.Trucks(4, 1, 3)
#
#     # h = hMaxHeuristic(task)
#     # h = LandmarkHeuristic(task)
#     h = LmCutHeuristic(task)
#     # h = task.h_all_blank
#
#     print(".")
#
#     pa = task.actions(task.I)
#     for a in pa:
#         res = task.result(task.I, a)
#         if h(Node(task.I)) > 1 + h(Node(res)):
#             print("not consistent!!!")
#             exit(0)

" obtain two heuristics from LMcut"
# def h_lmc_1(a_task):
#     t1 = copy.deepcopy(a_task)
#     t1_i = set()
#     for atom in t1.I:
#         tup_atom = ast.literal_eval(atom[2:])
#         if tup_atom[0] in [2, 3, 4, 7]:
#             t1_i.add(atom)
#         else:
#             t1_i.add(f'at(0,{tup_atom[1]})')
#     t1.I = t1_i
#     t1.G = {'at(2,2)', 'at(3,3)', 'at(4,4)', 'at(7,7)'}
#     return LmCutHeuristic(t1)
#
# def h_lmc_2(a_task):
#     t2 = copy.deepcopy(a_task)
#     t2_i = set()
#     for atom in t2.I:
#         tup_atom = ast.literal_eval(atom[2:])
#         if tup_atom[0] in [1, 5, 6, 8]:
#             t2_i.add(atom)
#         else:
#             t2_i.add(f'at(0,{tup_atom[1]})')
#     t2.I = t2_i
#     t2.G = {'at(1,1)', 'at(5,5)', 'at(6,6)', 'at(8,8)'}
#     return LmCutHeuristic(t2)
#
# h_lmc_1 = h_lmc_1(task)
# f_lmc_1 = lambda n: n.path_cost + h_lmc_1(n)
# h_lmc_2 = h_lmc_2(task)
# f_lmc_2 = lambda n: n.path_cost + h_lmc_2(n)
# f_max = lambda n: max(f_lmc_1(n), f_lmc_2(n))
# f_functions = [f_lmc_1, f_lmc_2, f_max, [f_lmc_1, f_lmc_2]]