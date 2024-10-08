import numpy as np
import os
from collections import defaultdict
import math
from itertools import islice
import csv
import pandas as pd

import multiprocessing

cpu_count = multiprocessing.cpu_count()
concurrent_processes = cpu_count // 2


comparison_info = defaultdict(dict)
comparison_info['diabetes']["time_dilation"] = 500_000
comparison_info['diabetes']["sample_rate"] = 16_000
comparison_info['diabetes']["threshold"] = 0.0001
comparison_info['diabetes']["volume_normalization"] = True
comparison_info['diabetes']["original_directory"] = "/home/hy/Downloads/diabetes_wing_csvs/wing_csvs"
comparison_info['diabetes']['comparison'] = "dm_comparison/wing_csvs"

comparison_info['alzheimers']["time_dilation"] = 100
comparison_info['alzheimers']["sample_rate"] = 44_100
comparison_info['alzheimers']["threshold"] = 0.01
comparison_info['alzheimers']["volume_normalization"] = False
comparison_info['alzheimers']["original_directory"] = "alzheimers_wing_csvs"
comparison_info['alzheimers']['comparison'] = "alz_comparison/wing_csvs"


disease = 'diabetes'
# disease = 'alzheimers'
# disease = 'control'

# comparison = True
comparison = False

threshold = comparison_info[disease]["threshold"]
time_dilation = comparison_info[disease]['time_dilation']

# wing_csv_dir = os.path.join(comparison_info[disease]["comparison_directory"], "wing_csvs")
if comparison:
    wing_csv_dir = comparison_info[disease]["comparison"]
else:
    wing_csv_dir = comparison_info[disease]["original_directory"]


def get_arguments():
    arguments = []
    for user in os.listdir(wing_csv_dir):
        if user == "passage b" or user == "Recording 1":
            continue
        max_part = get_max_part(user)
        user_wing_csv_dir = os.path.join(wing_csv_dir, user)
        for partition in range(max_part + 1):
            xs, ys = read_data(user_wing_csv_dir, user, partition)
            arguments.append((user, partition, xs, ys))
    return arguments


def get_max_part(user):
    user_vector_csv_dir = os.path.join(wing_csv_dir, user)
    partition = [-1]
    for file in os.listdir(user_vector_csv_dir):
        part = file.split(".")[0].split("_")[1]
        partition.append(int(part))

    return max(partition)


def main():
    pool = multiprocessing.Pool(processes=concurrent_processes)

    arguments = get_arguments()
    for user, partition, xs, ys in arguments:
        pool.apply_async(generate_matrix, args=(user, partition, xs, ys))

    pool.close()
    pool.join()


def read_data(user_folder, user, partition):
    csv_path = os.path.join(user_folder, get_file_name(user, partition))
    df = pd.read_csv(csv_path)
    return list(df["logNorm"]), list(df["angle"])


def check_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)


matrix_output = os.path.join(os.getcwd(), "matrices")
check_dir(matrix_output)


def get_file_name(user, partition):
    match disease:
        case "alzheimers":
            if comparison:
                return f'{user}_{partition}_0_{time_dilation}.csv'
            else:
                return f'{user}_{partition}_{threshold}_{time_dilation}.csv'
        case "diabetes":
            return f'{user.split(" ")[0]}_{partition}_0_{time_dilation}.csv'
        case _:
            raise Exception("Not a possible disease")


x_bin_size = 0.1
y_bin_size = 0.05
x_poles = np.arange(0.6, 21.3 + x_bin_size, x_bin_size)
y_poles = np.arange(-math.pi/2, math.pi/2 + y_bin_size, y_bin_size)


def window(seq, n=2):
    it = iter(seq)
    result = tuple(islice(it, n))
    if len(result) == n:
        yield result
    for elem in it:
        result = result[1:] + (elem,)
        yield result


show_excluded_points = True


def generate_matrix(user, partition, xs, ys):
    histogram_2d = []

    new_filename = get_file_name(user, partition)
    if disease == "diabetes" and not comparison:
        csv_path = os.path.join(matrix_output, f"{user}_{new_filename}")
    else:
        csv_path = os.path.join(matrix_output, new_filename)

    total_in_matrix = 0
    for y1, y2 in window(y_poles):
        layer_histogram = []
        for x1, x2 in window(x_poles):
            count = [1 for x, y in zip(xs, ys) if x1 <= x < x2 and y1 <= y < y2]
            points_in_cell = sum(count)
            layer_histogram.append(points_in_cell)
            total_in_matrix += points_in_cell
        histogram_2d.append(layer_histogram)

    with open(csv_path, 'w') as opened:
        writer = csv.writer(opened)
        writer.writerow(["layer" + str(i) for i in range(len(x_poles) - 1)])
        for row in histogram_2d:
            writer.writerow(row)

    if show_excluded_points:
        lost_points = len(xs) - total_in_matrix
        if lost_points:
            print(min(xs), max(xs), min(ys), max(ys))
            print("Lost", len(xs) - total_in_matrix, "points")


if __name__ == "__main__":
    main()
