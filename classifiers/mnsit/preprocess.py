import numpy as np
import re as re
import csv
import pickle
import pandas as pd

train = pd.read_csv('./dataset/train.csv', header = 0)
# test  = pd.read_csv('./dataset/test.csv' , header = 0)
# print ("data loaded, train: %d, test: %d" % (len(train), len(test)))

image_size = 28
n_samples = 42000
n_training = 34000
n_validation = 5000
n_test = 3000

# One hot encoding of labels
num_labels = 10
Y = train['label'].values
Y_ = (np.arange(num_labels) == Y[:,None]).astype(np.float32)
Y_ = Y_.astype(np.float32)

# Normalize the columns
train_dataset = train.drop(['label'], 1).values / 255
train_dataset = train_dataset.astype(np.float32)

# split input to train, validation and test
X_train = train_dataset[:n_training, :]
X_validation = train_dataset[n_training + 1: n_training + n_validation, :]
X_test = train_dataset[n_training + n_validation + 1:, :]

# split labels to train, validation and test
train_labels = Y_[:,:]
Y_train = train_labels[:n_training, :]
Y_validation = train_labels[n_training + 1: n_training + n_validation, :]
Y_test = train_labels[n_training + n_validation + 1:, :]

with open("dataset/d.train.pkl", "wb") as ofp:
    pickle.dump([X_train, Y_train], ofp)

with open("dataset/d.test.pkl", "wb") as ofp:
    pickle.dump([X_test, Y_test], ofp)

with open("dataset/d.validation.pkl", "wb") as ofp:
    pickle.dump([X_validation, Y_validation], ofp)

