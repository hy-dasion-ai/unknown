from collections import defaultdict

import pandas as pd
import os
import numpy as np
import sklearn.model_selection
from xgboost import XGBRegressor
from itertools import islice

train_percentage = 0.7

time_dilation = 1000

wing_data = os.path.join(os.getcwd(), "wing_csvs")


def underscore(arr):
    return "_".join([str(el) for el in arr])


def get_user_label(user, addendum=""):
    if addendum:
        addendum = "_" + addendum
    return str(user) + addendum


def available_data(skip_users=None, keep_users=None):
    if keep_users is None:
        keep_users = []
    if skip_users is None:
        skip_users = []

    users = []
    for folder in os.listdir(wing_data):
        users.append(folder)
    return users


def main():
    data = load_data()
    run_CNN_iteratively(data)


matrix_output = os.path.join(os.getcwd(), "matrices")


def load_data():
    data = defaultdict(dict)
    for file in os.listdir(matrix_output):
        absPath = os.path.join(matrix_output, file)
        splitInfo = file.split(".")[0].split("_")
        if int(splitInfo[-1]) == time_dilation:
            # user, partition
            data["_".join(splitInfo[:-3])][int(splitInfo[-3])] = absPath
    return data


from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import csv

matrix_dimension = (900, 900)


def get_score(user):
    if user.split("_")[0] == "HC":
        return 0
    return 1


def load_matrix(matrix_path):
    df = pd.read_csv(matrix_path)
    print(matrix_path)
    return df.to_numpy()


def run_CNN_iteratively(data):
    output_results = os.path.join(os.getcwd(), "cnn_wing_results.txt")
    individual_output = os.path.join(os.getcwd(), "individual_cnn_wing_results.txt")
    for user_removed in data.keys():

        print("Running for user", user_removed)
        trainX = []
        trainY = []

        testX = []
        testY = []

        for user in data.keys():
            score = get_score(user)

            for p in data[user].keys():
                x_val = load_matrix(data[user][p])

                if user == user_removed:
                    testX.append(x_val)
                    testY.append(score)
                else:
                    trainX.append(x_val)
                    trainY.append(score)

        trainX = np.asarray(trainX)
        trainY = np.asarray(trainY)

        testX = np.asarray(testX)

        trainYEncoded = np.zeros((trainY.size, 2))
        trainYEncoded[np.arange(trainY.size), trainY] = 1

        # Define the model
        model = Sequential()

        # Add a convolutional layer with 32 filters, each of size 3x3
        model.add(Conv2D(32, (9, 9), activation='relu', input_shape=(matrix_dimension[1], matrix_dimension[0], 1)))

        # Add a max pooling layer with pool size 2x2
        model.add(MaxPooling2D(pool_size=(6, 6)))

        # Add another convolutional layer with 64 filters, each of size 3x3
        # model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(Conv2D(matrix_dimension[0], (6, 6), activation='relu'))

        # Add another max pooling layer with pool size 2x2
        model.add(MaxPooling2D(pool_size=(4, 4)))

        # Flatten the input to a 1D array
        model.add(Flatten())

        # Add a fully connected layer with 64 neurons
        model.add(Dense(64, activation='relu'))

        # Add an output layer with 2 neurons (assuming 2 classes)
        model.add(Dense(2, activation='softmax'))

        # Compile the model
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        # Convert the labels to one-hot encoding
        # one_hot_labels = np.zeros((1000, 10))
        # one_hot_labels[np.arange(1000), labels.flatten()] = 1

        # Train the model
        model.fit(trainX, trainYEncoded, epochs=10, batch_size=2)

        yhat = model.predict(testX)

        use_within_range = False
        correct = 0
        for predicted, actual in zip(yhat, testY):
            # print(predicted, actual)
            rounded = np.argmax(predicted)
            with open(individual_output, "a") as opened:
                writer = csv.writer(opened)
                writer.writerow([rounded, actual])
            if rounded == actual:
                correct += 1

        print("Final:", user_removed, len(yhat), correct, correct/len(yhat), get_score(user_removed))

        with open(output_results, "a") as opened:
            writer = csv.writer(opened)
            writer.writerow([user_removed, len(yhat), correct/len(yhat)])


if __name__ == '__main__':
    main()
