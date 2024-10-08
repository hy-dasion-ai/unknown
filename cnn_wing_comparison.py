from collections import defaultdict
import pandas as pd
import os
import numpy as np

comparison_info = defaultdict(dict)
comparison_info['diabetes']["time_dilation"] = 500_000
comparison_info['diabetes']["sample_rate"] = 16_000
comparison_info['diabetes']["threshold"] = 0.0001
comparison_info['diabetes']["volume_normalization"] = True
comparison_info['diabetes']["original_directory"] = "dm_matrices"
comparison_info['diabetes']['comparison'] = "dm_comparison_matrices"

comparison_info['alzheimers']["time_dilation"] = 100
comparison_info['alzheimers']["sample_rate"] = 44_100
comparison_info['alzheimers']["threshold"] = 0.01
comparison_info['alzheimers']["volume_normalization"] = False
comparison_info['alzheimers']["original_directory"] = "alz_matrices"
comparison_info['alzheimers']['comparison'] = "alz_comparison_matrices"


disease = 'diabetes'
# disease = 'alzheimers'

train_matrices = comparison_info[disease]["original_directory"]
test_matrices = comparison_info[disease]["comparison"]


def load_descriptors():
    df = pd.read_csv(os.path.join("data", "adresso-train-mmse-scores.csv"))
    user_mmse_dx = list(zip(list(df["adressfname"]), list(df["mmse"]), list(df['dx'])))
    user_to_mmse = dict()
    user_to_dx = dict()
    for user, mmse, dx in user_mmse_dx:
        user_to_mmse[user] = mmse
        user_to_dx[user] = dx

    return user_to_mmse, user_to_dx


mmse_mapping, dx_mapping = load_descriptors()


def underscore(arr):
    return "_".join([str(el) for el in arr])


def get_user_label(user, addendum=""):
    if addendum:
        addendum = "_" + addendum
    return str(user) + addendum


def main():
    data = load_data()
    run_CNN_iteratively(data)


def load_data():
    data = defaultdict(dict)
    for file in os.listdir(train_matrices):
        absPath = os.path.join(train_matrices, file)
        splitInfo = file.split(".")[0].split("_")
        if disease == "alzheimers":
            data[splitInfo[0]][int(splitInfo[1])] = absPath
        elif disease == "diabetes":
            data["_".join(splitInfo[:-3])][splitInfo[-3]] = absPath
    return data


from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense


def get_score(user):
    match disease:
        case "diabetes":
            if user[:2] == "DM":
                return 1
            elif user[:4] == "CTRL":
                return 0
            else:
                raise Exception("No class" + user)
        case "alzheimers":
            dx = dx_mapping[user]
            if dx == "ad":
                return 0
            return 1
        case _:
            raise Exception("Unknown disease")


def load_matrix(matrix_path):
    df = pd.read_csv(matrix_path)
    return df.to_numpy()


def run_CNN_iteratively(data):
    trainX = []
    trainY = []

    for user in data.keys():
        score = get_score(user)
        for p in data[user].keys():
            x_val = load_matrix(data[user][p])
            trainX.append(x_val)
            trainY.append(score)

    testX = []
    testY = []
    for file in os.listdir(test_matrices):
        x_val = load_matrix(os.path.join(test_matrices, file))
        testX.append(x_val)
        testY.append(file)

    trainX = np.asarray(trainX)
    trainY = np.asarray(trainY)

    testX = np.asarray(testX)

    trainYEncoded = np.zeros((trainY.size, 2))
    trainYEncoded[np.arange(trainY.size), trainY] = 1

    # Define the model
    model = Sequential()

    # Add a convolutional layer with 32 filters, each of size 3x3
    dimensions = trainX[0].shape
    model.add(Conv2D(32, (9, 9), activation='relu', input_shape=(dimensions[1], dimensions[0], 1)))

    # Add a max pooling layer with pool size 2x2
    model.add(MaxPooling2D(pool_size=(6, 6)))

    # Add another convolutional layer with 64 filters, each of size 3x3
    # model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Conv2D(dimensions[0], (6, 6), activation='relu'))

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

    for predicted, file in zip(yhat, testY):
        print(file, predicted)


if __name__ == '__main__':
    main()
