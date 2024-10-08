import pandas as pd
import math
import os
import csv
import matplotlib.pyplot as plt

wing_csv_dir = os.path.join(os.getcwd(), "wing_csvs")


def check_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)


def read_data(file_path):
    df = pd.read_csv(file_path)
    return list(df['logNorm']), list(df['angle'])


def get_file_name(dilation, part):
    return f'{part}_{dilation}.csv'


def delete_ds_store(path):
    for root, sub, files in os.walk(path):
        for file in files:
            if file == ".DS_Store":
                fullpath = os.path.abspath(os.path.join(root, file))
                os.remove(fullpath)


delete_ds_store(os.getcwd())


excluded_data_dir = os.path.join(os.getcwd(), "excluded_data")
excluded_visualization_dir = os.path.join(os.getcwd(), "excluded_plots")

check_dir(excluded_data_dir)
check_dir(excluded_visualization_dir)


def get_title(user, dilation, part=None):
    title = f'{user}_{part + "_" if part else ""}{dilation}'
    return title


def get_excluded_data(user, dilation, part=None):
    title = get_title(user, dilation, part=part)
    csv_path = os.path.join(excluded_data_dir, title + ".csv")
    if not os.path.exists(csv_path):
        return None
    df = pd.read_csv(csv_path)
    return [list(df['logNorm']), list(df['angle'])]


def save_excluded(excluded_data, user, dilation, part=None):
    title = get_title(user, dilation, part=part)
    csv_path = os.path.join(excluded_data_dir, title + ".csv")
    with open(csv_path, "w") as opened:
        writer = csv.writer(opened)

        writer.writerow(["logNorm", "angle"])
        for x, y in zip(excluded_data[0], excluded_data[1]):
            writer.writerow([x, y])


def get_file_paths(dilation):
    all_file_paths = dict()
    for user_folder in os.listdir(wing_csv_dir):
        user_folder_path = os.path.join(wing_csv_dir, user_folder)
        user = user_folder
        subs = []
        for file in os.listdir(user_folder_path):
            if int(file.split(".")[0].split("_")[-1]) == dilation:
                file_name = file.split(".")[0]
                file_path = os.path.join(user_folder_path, file)
                subs.append((file_name, file_path))
        all_file_paths[user] = subs
    return all_file_paths


def visualize_excluded_data(excluded_data, dilation):

    domain_x = [math.inf, -math.inf]
    range_y = [math.inf, -math.inf]

    for user in excluded_data.keys():
        user_data = excluded_data[user]
        if user_data:
            domain_x = [
                min(domain_x[0], min(user_data[0]), min(user_data[0])),
                max(domain_x[1], max(user_data[0]), max(user_data[0]))]
            range_y = [
                min(range_y[0], min(user_data[1]), min(user_data[1])),
                max(range_y[1], max(user_data[1]), max(user_data[1]))]

    for user in excluded_data.keys():
        title = get_title(user, dilation)
        file_path = os.path.join(excluded_visualization_dir, title + ".png")
        if os.path.exists(file_path):
            continue

        plt.figure(figsize=(10, 10))

        xs, ys = excluded_data[user]

        plt.scatter(xs, ys, alpha=0.2)
        plt.ylabel("Angle")
        plt.xlabel("Log Norm")
        plt.xlim(domain_x)
        plt.ylim(range_y)
        plt.title(title, fontsize=20)
        plt.savefig(file_path)
        plt.close()


def transform_data(dilation, domain_x, range_y):
    all_file_paths = get_file_paths(dilation)

    data = dict()
    excluded_data = dict()

    for user in all_file_paths.keys():
        all_parts = all_file_paths[user]
        user_parts = dict()
        excluded_parts = dict()

        for file_name, file_path in all_parts:
            log_norms, angles = read_data(file_path)

            pos_data = [[], []]
            neg_data = [[], []]
            excluded = [[], []]

            if not z_transformed_saved(user, file_name):
                for i, angle in enumerate(angles):
                    if angle == 0:
                        excluded[0].append(log_norms[i])
                        excluded[1].append(angle)
                        continue
                    temp = 0
                    if ztransform:
                        if angle < 0:
                            temp = angle*-1
                        else:
                            temp = angle
                        normalized_y = temp / math.radians(90)
                        transformed_y = 1 / normalized_y - 1
                        if transformed_y < 0:
                            excluded[0].append(log_norms[i])
                            excluded[1].append(angle)
                            continue
                        temp = math.log(transformed_y)
                    if angle > 0:  # > 0 , < 0
                        pos_data[0].append(log_norms[i])
                        pos_data[1].append(temp)
                    else:
                        neg_data[0].append(log_norms[i])
                        neg_data[1].append(temp)
                save_z_transform(pos_data, "up", user, file_name)
                save_z_transform(neg_data, "down", user, file_name)
                #save_excluded(excluded, user, dilation, subsample_and_clip)
            else:
                pos_data, neg_data = get_pos_neg_z(user, file_name)
                # excluded = get_excluded_data(user, dilation, subsample_and_clip)

            domain_x = [
                min(domain_x[0], min(pos_data[0]), min(neg_data[0])),
                max(domain_x[1], max(pos_data[0]), max(neg_data[0]))]
            range_y = [
                min(range_y[0], min(pos_data[1]), min(neg_data[1])),
                max(range_y[1], max(pos_data[1]), max(neg_data[1]))]

            user_parts[file_name] = (pos_data, neg_data)
            # excluded_parts[file_name] = excluded

        data[user] = user_parts
        excluded_data[user] = excluded_parts

    for user, subdict in data.items():
        for file_name in subdict.keys():
            pos_data, neg_data = data[user][file_name]
            plot_ztransform(pos_data, user, "up", domain_x, range_y, file_name)
            plot_ztransform(neg_data, user, "down", domain_x, range_y, file_name)

    save_domain_range(domain_x, range_y, dilation)

    # visualize_excluded_data(excluded_data, dilation)


def get_domain_range_path(dilation):
    return os.path.join(os.getcwd(), f'{dilation}_domain_range.csv')


def save_domain_range(domain_x, range_y, dilation):
    with open(get_domain_range_path(dilation), 'w') as opened:
        writer = csv.writer(opened)
        writer.writerow([domain_x[0], domain_x[1], range_y[0], range_y[1]])


def get_domain_range(dilation):
    if os.path.exists(get_domain_range_path(dilation)):
        df = pd.read_csv(get_domain_range_path(dilation), header=None)
        return [df[0][0], df[1][0]], [df[2][0], df[3][0]]
    return [math.inf, -math.inf], [math.inf, -math.inf]


ztransformed_csv = os.path.join(os.getcwd(), "ztransform_csvs")
check_dir(ztransformed_csv)


def z_transformed_saved(user, filename):
    for direction in ["up", "down"]:
        user_path = os.path.join(ztransformed_csv, user)
        csv_path = os.path.join(user_path, f'{direction}_z_{filename}')
        if not os.path.exists(csv_path):
            return False
    # if not get_excluded_data(user, subsample_and_clip, dilation):
    #     return False
    return True


def get_pos_neg_z(user, file_name):
    user_path = os.path.join(ztransformed_csv, user)
    csv_path = os.path.join(user_path, f'up_z_{file_name}')
    df_up = pd.read_csv(csv_path)
    csv_path = os.path.join(user_path, f'down_z_{file_name}')
    df_down = pd.read_csv(csv_path)

    return [list(df_up['x']), list(df_up['y'])], [list(df_down['x']), list(df_down['y'])]


def save_z_transform(positions, addendum, user, file_name):
    xs, ys = positions

    user_path = os.path.join(ztransformed_csv, user)
    check_dir(user_path)
    csv_path = os.path.join(user_path, f'{addendum}_z_{file_name}')

    with open(csv_path, "w") as opened:
        writer = csv.writer(opened)

        writer.writerow(["x", "y"])
        for x, y in zip(xs, ys):
            writer.writerow([x, y])


domain_range_dir = os.path.join(os.getcwd(), "z_domain_range")
check_dir(domain_range_dir)


ztransform = os.path.join(os.getcwd(), "ztransform")
check_dir(ztransform)


def plot_ztransform(positions, user, direction, domain_x, range_y, file_name):
    title = f'{user}_{file_name}_{direction}'

    user_dir = os.path.join(ztransform, user)
    check_dir(user_dir)
    file_path = os.path.join(user_dir, title + ".png")

    if os.path.exists(file_path):
        return

    plt.figure(figsize=(15, 15))

    xs, ys = positions
    plt.scatter(xs, ys, alpha=0.2)

    plt.ylabel("Z-transformed Angle")
    plt.xlabel("Norm")
    plt.xlim(domain_x)
    plt.ylim(range_y)
    plt.title(title, fontsize=20)
    plt.savefig(file_path)
    plt.close()


# time_dilations = [1, 2, 5, 10, 15, 20, 50, 100, 500, 1_000, 5_000, 10_000, 50_000, 500_000]
time_dilations = [100]


def main():
    for dilation in time_dilations:
        domain_x, range_y = get_domain_range(dilation)
        transform_data(dilation, domain_x, range_y)


if __name__ == '__main__':
    main()
