import ast
import math
import csv


def create_csv_file(file_name, file_headline):
    with open(file_name + '.csv', mode='w', newline="") as file:
        csv_writer = csv.writer(file)
        csv_writer.writerow(file_headline)


def row_col_of_number(number, width, height):
    """
    :param number: number in the range 1 to width*height
    :param width: width of 2D array
    :param height: height of 2D array
    :return: the i,j indexes of the number in array that array[0][0] = 1, array[-1][-1] = width*height
    """
    row = math.ceil(number/height) - 1
    col = number - 1 - row * width
    return row, col


def euclidian_distance(loc1, loc2):
    return abs(loc2[1] - loc1[1]) + abs(loc2[0] - loc1[0])


def sort_and_order(list_1, list_2, revers=False):
    """
    sort list_1 and order list 2 based on the sorted list_1
    """
    if len(list_1) != len(list_2):
        raise "lists size dont match"
    if len(list_1) == 0:
        return list_1, list_2
    pairs = list(zip(list_1, list_2))
    if revers is False:
        sorted_pairs = sorted(pairs, key=lambda x: x[0])
    else:
        sorted_pairs = sorted(pairs, key=lambda x: x[0], reverse=True)
    res_1, res_2 = zip(*sorted_pairs)
    return list(res_1), list(res_2)


def not_in_list(list_of_lists, arr):
    for l in list_of_lists:
        if arr == l:
            return False
    return True