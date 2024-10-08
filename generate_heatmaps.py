import numpy as np
import os
import seaborn
import math
from itertools import islice
import csv
import pandas as pd

import multiprocessing

cpu_count = multiprocessing.cpu_count()
concurrent_processes = cpu_count // 2


def get_arguments():
    arguments = []
    for file in os.listdir(matrix_output):
        df = pd.read_csv(os.path.join(matrix_output, file))
        matrix = df.to_numpy()
        arguments.append((file.split(".")[0] + ".png", matrix))
    return arguments


def main():
    pool = multiprocessing.Pool(processes=concurrent_processes)

    arguments = get_arguments()

    for new_file_name, matrix in arguments:
        pool.apply_async(generate_heatmap, args=(new_file_name, matrix))

    pool.close()
    pool.join()


def check_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)


matrix_output = os.path.join(os.getcwd(), "matrices")
heatmap_output = os.path.join(os.getcwd(), "heatmaps")
check_dir(heatmap_output)

matrix_dimensions = []

x_bin_size = 0.1
y_bin_size = 0.05
x_poles = np.arange(0, 11 + x_bin_size, x_bin_size)
y_poles = np.arange(-math.pi/2, math.pi/2 + y_bin_size, y_bin_size)


show_excluded_points = True

max_count = 4000


def generate_heatmap(new_file_name, matrix):
    counting_max = 0
    for row in matrix:
        for val in row:
            if val > counting_max:
                counting_max = val
    if counting_max > max_count:
        print(counting_max)
    ax = seaborn.heatmap(matrix, vmin=0, vmax=max_count)
    ax.set(xlabel="logNorm", ylabel="angle")
    fig = ax.get_figure()
    fig.savefig(os.path.join(heatmap_output, new_file_name), dpi=1600)
    fig.clf()


if __name__ == "__main__":
    main()
