import numpy as np
import pickle
import json

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers.recurrent import SimpleRNN, LSTM, GRU 
from keras.utils import np_utils
from keras import backend as K
K.set_image_dim_ordering('th')


from distutils.version import LooseVersion as LV
from keras import __version__

from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

print('Using Keras version:', __version__, 'backend:', K.backend())
assert(LV(__version__) >= LV("2.0.0"))

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
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
num_classes = Y_test.shape[1]

print ("Number of pixels", num_pixels)
print ("X_Train", X_train.shape)
print ("X_test", X_test.shape)

# define baseline model
def baseline_model(nb_units=50):
    img_rows = 28
    img_cols = 28

    # create model
    model = Sequential()

    # Recurrent layers supported: SimpleRNN, LSTM, GRU:
    model.add(SimpleRNN(nb_units, input_shape=(img_rows, img_cols)))

    # To stack multiple RNN layers, all RNN layers except the last one need
    # to have "return_sequences=True".  An example of using two RNN layers:
    #model.add(SimpleRNN(16,
    #                    input_shape=(img_rows, img_cols),
    #                    return_sequences=True))
    #model.add(SimpleRNN(32))

    model.add(Dense(units=num_classes))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])

    print(model.summary())
    return model


# build the model
model = baseline_model()
# SVG(model_to_dot(model, show_shapes=True).create(prog='dot', format='svg'))

# Fit the model
model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=10, batch_size=200, verbose=2)
# Final evaluation of the model
scores = model.evaluate(X_test, Y_test, verbose=0)
print("CNN Error: %.2f%%" % (100-scores[1]*100))
print (scores)

plt.figure(figsize=(5,3))
plt.plot(history.epoch,history.history['loss'])
plt.title('loss')

plt.figure(figsize=(5,3))
plt.plot(history.epoch,history.history['acc'])
plt.title('accuracy');
