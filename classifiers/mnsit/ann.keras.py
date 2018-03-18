import numpy as np
import pickle
import json

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils import np_utils

X_train = None
Y_train = None
X_test = None
Y_test = None
X_validation = None
Y_validation = None

with open("dataset/d.train.pkl", "rb") as ifp:
    X_train, Y_train = pickle.load(ifp)

with open("dataset/d.test.pkl", "rb") as ifp:
    X_test, Y_test = pickle.load(ifp)

with open("dataset/d.validation.pkl", "rb") as ifp:
    X_validation, Y_validation = pickle.load(ifp)


# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

# flatten 28*28 images to a 784 vector for each image
num_pixels = X_train.shape[1]
X_train = X_train.reshape(X_train.shape[0], num_pixels).astype('float32')
X_test = X_test.reshape(X_test.shape[0], num_pixels).astype('float32')
num_classes = Y_test.shape[1]

print ("Number of pixels", num_pixels)
print ("X_Train", X_train.shape)
print ("X_test", X_test.shape)

# define baseline model
def baseline_model():
    # create model
    model = Sequential()
    model.add(Dense(num_pixels, input_dim=num_pixels, kernel_initializer='normal', activation='relu'))
    model.add(Dense(512, kernel_initializer='normal', activation='relu'))
    model.add(Dense(256, kernel_initializer='normal', activation='relu'))
    model.add(Dense(num_classes, kernel_initializer='normal', activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# build the model
model = baseline_model()
# Fit the model
model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=10, batch_size=200, verbose=2)
# Final evaluation of the model
scores = model.evaluate(X_test, Y_test, verbose=0)
print("Baseline Error: %.2f%%" % (100-scores[1]*100))
print (scores)
