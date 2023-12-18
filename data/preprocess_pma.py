from data.load_data_pma import *
from tensorflow.keras.utils import to_categorical
from sklearn.utils import shuffle
import numpy as np
import cv2


def load_datasets(path="/home/luan/works/dataset/experiment-i", type_load="17class", preproc=True):
    if type_load == "17class":
        exp_i_data = load_exp_i_new(path, preprocess=preproc)

    datasets = {"Base": exp_i_data}

    return datasets


def preprocess(datasets, num_cls, sub="S1", hist_equal=True, apply_color_map=False, normalize=True):

    subjects = ["S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9", "S10", "S11", "S12", "S13"]
    subjects.remove(sub)
    train_data = Mat_Dataset(datasets, ["Base"], subjects)
    test_data = Mat_Dataset(datasets, ["Base"], [sub])

    x_train = []
    for i in range(len(train_data.samples)):
        if hist_equal:
            train_data.samples[i] = cv2.equalizeHist(train_data.samples[i])
            
        if apply_color_map:
            img = cv2.applyColorMap(train_data.samples[i], cv2.COLORMAP_JET)
        else:
            img = train_data.samples[i][:, :, np.newaxis]

        x_train.append(img)

    x_test = []
    for i in range(len(test_data.samples)):
        if hist_equal:
            test_data.samples[i] = cv2.equalizeHist(test_data.samples[i])
            
        if apply_color_map:
            img = cv2.applyColorMap(test_data.samples[i], cv2.COLORMAP_JET)
        else:
            img = test_data.samples[i][:, :, np.newaxis]

        x_test.append(img)

    x_train = np.array(x_train, dtype=np.float64)
    x_test = np.array(x_test, dtype=np.float64)

    if normalize:
        x_train /= 255
        x_test /= 255

    y_train = to_categorical(train_data.labels, num_cls)
    y_test = to_categorical(test_data.labels, num_cls)

    random.seed(42)
    (x_train, y_train) = shuffle(x_train, y_train)
    random.seed(42)
    (x_test, y_test) = shuffle(x_test, y_test)

    return x_train, y_train, x_test, y_test

