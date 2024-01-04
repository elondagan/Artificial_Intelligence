import pickle
import random

from STRIPS import domains


def create_Npuzzle_tasks(size, amount, reg=True):
    """
    only for 8 puzzle
    for two blanks, for now it has 2 goals
    """
    init = ''
    g = []
    if reg is True:
        for i in range(1, size+1):
            init += str(i) + '.'
            g.append(f'at({i},{i})')
        init += '0'
    else:
        for i in range(1, size):
            init += str(i) + '.'
            g.append(f'at({i},{i})')
        init += '0.0'

    puzzle_tasks = []
    for _ in range(amount):
        cur_task = domains.Npuzzle(size, init) # '123456780'
        if reg is False:  # puzzle with two blanks
            cur_task = domains.Npuzzle(size, init)
            cur_task.G = g  # {'at(6,6)', 'at(3,3)', 'at(1,1)', 'at(7,7)', 'at(4,4)', 'at(2,2)', 'at(5,5)'}
        new_i = cur_task.I
        for _ in range(100):
            new_i = cur_task.result(new_i, random.choice(cur_task.actions(new_i)))
        cur_task.I = new_i
        puzzle_tasks.append(cur_task)
    if reg is True:
        with open(f'{size}-puzzle_tasks.pkl', 'wb') as file:
            pickle.dump(puzzle_tasks, file)
    else:
        with open(f'{size}-puzzle-2b_tasks.pkl', 'wb') as file:
            pickle.dump(puzzle_tasks, file)


def create_blocks_tasks(size, amount):
    blocks_tasks = [domains.BlocksWorld(size) for _ in range(amount)]
    with open(f'{size}-blocks_tasks.pkl', 'wb') as file:
        pickle.dump(blocks_tasks, file)


def create_trucks_tasks(size, amount):
    trucks_tasks = [domains.Trucks(size[0], size[1], size[2]) for _ in range(amount)]
    with open(f'{size[0]}-{size[1]}-{size[2]}-trucks_tasks.pkl', 'wb') as file:
        pickle.dump(trucks_tasks, file)


def create_vacuum_tasks(size, amount):
    vacuum_tasks = [domains.VacuumCleaner(size) for _ in range(amount)]
    with open(f'{size}-vacuum_tasks.pkl', 'wb') as file:
        pickle.dump(vacuum_tasks, file)


def create_elevator_tasks(size, amount):
    elevator_tasks = [domains.Elevator(size[0], size[1], size[2]) for _ in range(amount)]
    with open(f'{size[0]}-{size[1]}-{size[2]}-elevator_tasks.pkl', 'wb') as file:
        pickle.dump(elevator_tasks, file)

# def create_grid_tasks(size, amount):
#     grid_tasks = [STRIPS.Grid(size) for _ in range(amount)]
#     with open(f'{size}-grid_tasks.pkl', 'wb') as file:
#         pickle.dump(grid_tasks, file)


if __name__ == '__main__':
    print("create tasks . . . \n")

    # create_Npuzzle_tasks(15, 100)
    # create_Npuzzle_tasks(15, 100, reg=False)

    create_Npuzzle_tasks(8, 25)
    # create_Npuzzle_tasks(8, 100, reg=False)
    #
    # create_blocks_tasks(7, 25)
    # create_blocks_tasks(8, 100)

    # create_trucks_tasks([3, 2, 3], 25)

    # create_vacuum_tasks(6, 25)
    #
    # create_elevator_tasks([10, 5, 3], 25)
