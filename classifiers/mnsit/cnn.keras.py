import numpy as np
import pickle
import json
from time import time

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
K.set_image_dim_ordering('th')

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


startTime = time()

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

# flatten 28*28 images to a 784 vector for each image
num_pixels = X_train.shape[1]
X_train = X_train.reshape(X_train.shape[0], 1, 28, 28).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 1, 28, 28).astype('float32')
num_classes = Y_test.shape[1]

print ("Number of pixels", num_pixels)
print ("X_Train", X_train.shape)
print ("X_test", X_test.shape)

# define baseline model
def baseline_model():
    # create model
	model = Sequential()
	model.add(Conv2D(30, (5, 5), input_shape=(1, 28, 28), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	# model.add(Conv2D(15, (3, 3), activation='relu'))
	# model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.2))
	model.add(Flatten())
	# model.add(Dense(128, activation='relu'))
	model.add(Dense(50, activation='relu'))
	model.add(Dense(num_classes, activation='softmax'))
    # Compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model


# build the model
model = baseline_model()
# Fit the model
model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=2, batch_size=200, verbose=2)
# Final evaluation of the model
scores = model.evaluate(X_test, Y_test, verbose=0)
timeTaken = time() - startTime

print("CNN Error: %.2f%%" % (100-scores[1]*100))
print (scores)
print("Time Taken: %.3f s" % timeTaken)
