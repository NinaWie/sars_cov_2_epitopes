from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, Conv1D, MaxPooling1D
from tensorflow.keras import optimizers


def get_split_sequences(x, y, test_size=0.1, val_size=0.1):
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=1
    )
    x_train, x_val, y_train, y_val = train_test_split(
        x_train, y_train, test_size=val_size, random_state=1
    )
    return x_train, y_train, x_val, y_val, x_test, y_test


def get_data_from_csv(
    seq_csv_name, pad=True, pad_max_len=400, test_size=0.1, val_size=0.1
):
    df = pd.read_csv(seq_csv_name)
    x = [eval(d) for d in df["data"]]
    y = [eval(lab) for lab in df["labels"]]
    if pad:
        x, y = pad_seq(x, y, max_len=pad_max_len)
    # print(y.shape, [len(y_o) for y_o in y])
    return get_split_sequences(x, y, test_size=test_size, val_size=val_size)


def pad_seq(seq_data, seq_labels, pad_val=-1, max_len=None):
    if max_len is None:
        lens = np.array([len(seq) for seq in seq_data])
        max_len = np.max(lens)
    for i in range(len(seq_data)):
        pad_len = max_len - len(seq_data[i])
        if pad_len > 0:
            padding = np.array(
                [[pad_val for _ in range(5)] for _ in range(pad_len)]
            )
            # print(s.shape, padding.shape)
            seq_data[i] = np.concatenate([seq_data[i], padding], axis=0)
            seq_labels[i] = list(seq_labels[i]) + [-1 for _ in range(pad_len)]
        elif pad_len < 0:
            seq_data[i] = seq_data[i][:max_len]
            seq_labels[i] = seq_labels[i][:max_len]
    return np.array(seq_data), np.array(seq_labels)


def pad_seq_data(seq_data, pad_val=-1, max_len=None):
    if max_len is None:
        lens = np.array([len(seq) for seq in seq_data])
        max_len = np.max(lens)
    new_seq = []
    for i in range(len(seq_data)):
        pad_len = max_len - len(seq_data[i])
        if pad_len > 0:
            padding = np.array(
                [
                    [pad_val for _ in range(len(seq_data[0][0]))]
                    for _ in range(pad_len)
                ]
            )
            # print(s.shape, padding.shape)
            seq_data[i] = np.concatenate([seq_data[i], padding],
                                         axis=0).tolist()
        elif pad_len < 0:
            seq_data[i] = list(seq_data[i][:max_len])
        new_seq.append(seq_data[i])
    return np.array(new_seq)