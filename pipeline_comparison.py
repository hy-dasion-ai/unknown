from numVector import getVectorsNum

import pandas as pd
import math
import os
import numpy as np
import csv
import matplotlib.pyplot as plt
from pydub import AudioSegment
from scipy.io import wavfile
from collections import defaultdict
from itertools import islice

raw_data_dir = os.path.join(os.getcwd(), "data")

raw_plots_dir = os.path.join(os.getcwd(), "raw_plots")

comparison_info = defaultdict(dict)
comparison_info['diabetes']["time_dilation"] = 500_000
comparison_info['diabetes']["sample_rate"] = 16_000
comparison_info['diabetes']["threshold"] = 0.0001
comparison_info['diabetes']["volume_normalization"] = True
comparison_info['alzheimers']["time_dilation"] = 100
comparison_info['alzheimers']["sample_rate"] = 44_100
comparison_info['alzheimers']["threshold"] = 0.01
comparison_info['alzheimers']["volume_normalization"] = False

disease = 'diabetes'
# disease = 'alzheimers'

time_dilation = comparison_info[disease]['time_dilation']
sample_rate = comparison_info[disease]['sample_rate']
genericThreshold = comparison_info[disease]['threshold']
normalize_volume = comparison_info[disease]['volume_normalization']

segment_length = 30 * sample_rate

def check_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)


check_dir(raw_plots_dir)


numVectors = None


curveThreshold = genericThreshold
distThreshold = genericThreshold

nameScheme = numVectors if numVectors else genericThreshold


def delete_ds_store(path):
    for root, sub, files in os.walk(path):
        for file in files:
            if file == ".DS_Store":
                fullpath = os.path.abspath(os.path.join(root, file))
                os.remove(fullpath)


delete_ds_store(os.getcwd())


def load_descriptors():
    gender_mapping = dict()
    phq_mapping = dict()
    for file in os.listdir(raw_data_dir):
        if ".csv" not in file:
            continue
        if "split" not in file:
            continue
        df = pd.read_csv(os.path.join(raw_data_dir, file))
        df.reset_index()
        for index, row in df.iterrows():
            user_id = int(row['Participant_ID'])
            if "PHQ8_Score" in row.keys():
                phq_score = row['PHQ8_Score']
                gender = "female" if int(row['Gender']) == 0 else "male"
            else:
                gender = row['Gender']
                phq_score = row['PHQ_Score']
            phq_score = int(phq_score)
            gender_mapping[user_id] = gender
            phq_mapping[user_id] = phq_score

    return gender_mapping, phq_mapping


genders, phq_scores = load_descriptors()


def underscore(arr):
    return "_".join([str(el) for el in arr])


def calculate_lengths(arrays):
    """
    calculate_lengths
    given an array of tuples, calculates the norm of that vector
    """
    norms = []
    for npArray in arrays:
        norm = np.asarray([np.linalg.norm(pair) for pair in npArray])
        norms.append(norm)

    return norms


def window(seq, n=2):
    """
    Returns a sliding window (of width n) over data from the iterable
    s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ... 
    """                  
    it = iter(seq)
    result = tuple(islice(it, n))
    if len(result) == n:
        yield result
    for elem in it:
        result = result[1:] + (elem,)
        yield result


def calculate_angles_dil(arrays, dilation_factor):
    """
    Calculate angles but with a dilation factor for time dimension.
    """
    angles = []
    for npArray in arrays:
        angle = np.asarray([math.atan(pair[1] / (pair[0] * dilation_factor)) for pair in npArray])
        angles.append(angle)

    return angles


vector_csv_dir = os.path.join(os.getcwd(), "vectors")
check_dir(vector_csv_dir)

vector_plot_dir = os.path.join(os.getcwd(), "vectors_plot")
check_dir(vector_plot_dir)

wing_csv_dir = os.path.join(os.getcwd(), "wing_csvs")
check_dir(wing_csv_dir)


def get_descriptor(user):
    return user.split(" ")[0]


def vectorize(samplerate, all_y, user, continuous_subsample_num, clip_num):
    user_vector_csv_dir = os.path.join(vector_csv_dir, user)
    check_dir(user_vector_csv_dir)

    user_vector_plot_dir = os.path.join(vector_plot_dir, user)
    check_dir(user_vector_plot_dir)

    user_wing_csv_dir = os.path.join(wing_csv_dir, user)
    check_dir(user_wing_csv_dir)

    description = get_descriptor(user)

    title = description + "_" + str(continuous_subsample_num) + "_" + str(clip_num)

    output_file_path = os.path.join(user_vector_csv_dir, title + ".csv")

    vectorized = os.path.exists(output_file_path)

    if test_rounding or not vectorized:
        all_x = [i / samplerate for i in range(len(all_y))]

        d = {'timestamp': all_x, 'value': all_y}
        data = pd.DataFrame(data=d)

        vector_matrix, vector_points, crits = (
            getVectorsNum(data['value'], data['timestamp'], curveThreshold=curveThreshold, distThreshold=distThreshold))
        fig = plt.figure(figsize=(15, 15))
        fig.suptitle(user + " vectorized: " + title, fontsize=20)
        plt.plot(data['timestamp'], data['value'], color="blue", label="raw data")

        # add in the "start" critical
        if vector_points[0][0] != 0:
            vector_points = np.insert(vector_points, 0, np.asarray([all_x[0], all_y[0]]), axis=0)

        save_path = os.path.join(user_vector_plot_dir, title + ".png")
        plt.plot(
            [x[0] for x in vector_points], 
            [x[1] for x in vector_points], 
            marker='o', 
            color='r', 
            label="vectorized")
        plt.savefig(save_path)
        plt.close()

        with open(output_file_path, 'w') as opened:
            writer = csv.writer(opened)
            writer.writerow(["x", "y"])
            for x, y in vector_matrix:
                writer.writerow([x, y])
    else:
        df = pd.read_csv(output_file_path)
        vector_matrix = np.asarray(list(zip(list(df['x']), list(df['y']))))

    title += "_" + str(time_dilation)
    output_file_path = os.path.join(user_wing_csv_dir, title + ".csv")

    if test_rounding or not os.path.exists(output_file_path):
        angles = calculate_angles_dil([vector_matrix], time_dilation)[0]
        norms = calculate_lengths([vector_matrix])[0]
        norms = np.log(norms)

        with open(output_file_path, 'w') as opened:
            writer = csv.writer(opened)
            writer.writerow(["logNorm", "angle"])
            for norm, angle in zip(norms, angles):
                writer.writerow([norm, angle])
    else:
        df = pd.read_csv(output_file_path)
        norms = list(df["logNorm"])
        angles = list(df["angle"])

    return norms, angles


def process_data(file_path):
    samplerate, data = wavfile.read(file_path)

    if len(data.shape) > 1:
        # convert stereo to mono
        sound = AudioSegment.from_wav(file_path)
        sound = sound.set_channels(1)
        sound.export(file_path, format="wav")
        samplerate, data = wavfile.read(file_path)

    print(file_path, samplerate, len(data), len(data)/samplerate, "s")

    return samplerate, data


verbose = True


target_dBFS = -20.0


def match_target_amplitude(sound, target):
    change_in_dbfs = target - sound.dBFS
    return sound.apply_gain(change_in_dbfs)


def process_single_speaker(signal, samplerate, user, clip_num):
    # print(clip_num, time_dilation, user)
    segment_folder = os.path.join(os.getcwd(), str(segment_length) + "_data")
    check_dir(segment_folder)
    segment_wing_folder = os.path.join(os.getcwd(), str(segment_length) + "_wing")
    check_dir(segment_wing_folder)

    user_segment_folder = os.path.join(segment_folder, user)
    check_dir(user_segment_folder)
    user_segment_wing_folder = os.path.join(segment_wing_folder, user)
    check_dir(user_segment_wing_folder)

    chunk_nums = len(signal) // segment_length
    print("Should generate", chunk_nums, "partitions")

    # take segment_length chunks
    for chunk_num in range(chunk_nums):
        next_chunk_num = chunk_num + 1
        current_chunk = signal[segment_length*chunk_num:segment_length*next_chunk_num]

        new_file_name = str(chunk_num) + "_" + str(clip_num) + "_" + str(time_dilation)
        data_segment_name = str(chunk_num) + "_" + str(clip_num)

        if normalize_volume:
            audio_segment = AudioSegment(np.asarray(current_chunk), frame_rate=samplerate, sample_width=2, channels=1)
            normalized_segment = match_target_amplitude(audio_segment, target_dBFS)
            current_chunk = normalized_segment.get_array_of_samples()

        if test_rounding:
            current_chunk = np.array(rounded_chunk(current_chunk, base=10))

        sub_wing_plot_path = os.path.join(user_segment_wing_folder, new_file_name + ".png")
        print(clip_num, time_dilation, user)
        print(sub_wing_plot_path, os.path.exists(sub_wing_plot_path))
        if test_rounding or not os.path.exists(sub_wing_plot_path):
            data_segment_path = os.path.join(user_segment_folder, "part" + data_segment_name + ".png")
            if not os.path.exists(data_segment_path):
                plt.figure(figsize=(10, 10))
                plt.title("Raw " + str(segment_length) + " data plot: " + new_file_name)
                time = np.linspace(0, segment_length / samplerate, segment_length)
                plt.plot(time, current_chunk)
                plt.xlabel("Time [s]")
                plt.ylabel("Amplitude")
                plt.savefig(data_segment_path)
                plt.close()

            norms, angles = vectorize(samplerate, current_chunk, user, chunk_num, clip_num)
            possible_combos = set()
            for norm, angle in zip(norms, angles):
                possible_combos.add((norm, angle))

            plt.figure(figsize=(15, 15))
            plt.title("Wing plot: part " + str(chunk_num))
            plt.scatter(norms, angles, alpha=0.5, s=2)
            plt.xlabel("norms")
            plt.ylabel("angles")
            plt.savefig(sub_wing_plot_path)
            plt.close()


def write_completed_user(user):
    with open("completed.txt", "a") as completed_file:
        completed_file.write(user + str(time_dilation) + "\n")


def get_completed_users():
    completed_file = os.path.join(os.getcwd(), "completed.txt")
    if not os.path.exists(completed_file):
        with open(completed_file, "w"):
            pass

    completed_users = [] 
    with open(completed_file) as completed_file:
        for line in completed_file.readlines():
            if line.strip():
                completed_users.append(line.strip())

    return completed_users


def retrieve_data():
    user_to_filepath = dict()
    for file in os.listdir(raw_data_dir):
        sub_path = os.path.join(raw_data_dir, file)
        # should be directory after prepare_data_dir.py
        if os.path.isdir(sub_path):
            speaker_clips = os.path.join(sub_path, "speaker_clips")
            file_paths = []
            for clip in os.listdir(speaker_clips):
                file_paths.append((os.path.join(speaker_clips, clip), str(clip.split(".")[0].split("_")[1])))
            user_to_filepath[file] = file_paths

    return user_to_filepath


def retrieve_user_data(filepath):
    sr, signal = process_data(filepath)
    if verbose:
        length = signal.shape[0] / sr
        print("Processing", filepath)
        print("samplerate=" + str(sr) + "/s")
        print("length=" + str(length) + "s")
    return signal


test_rounding = False


def rounded_chunk(signal, base=5):
    rounded = [round_to_nearest_x(val, base=base) for val in list(signal)]
    print(len(rounded))
    return rounded


def round_to_nearest_x(x, base = 5):
    return base * round(x/base)


def main():
    print("loading_raw_data")
    user_to_filepath = retrieve_data()
    print("data_loaded")

    completed_users = get_completed_users()

    for user in user_to_filepath.keys():
        if user + str(time_dilation) in completed_users:
            continue
        filepath = user_to_filepath[user]

        for clip_path, clip_num in filepath:
            signal = retrieve_user_data(clip_path)
            process_single_speaker(signal, sample_rate, user, clip_num)

        print("Processed", user)
        write_completed_user(user)


if __name__ == '__main__':
    main()
    