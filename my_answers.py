import numpy as np
import keras
import string

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Activation



# TODO: fill out the function below that transforms the input series 
# and window-size into a set of input/output pairs for use with our RNN model
def window_transform_series(series, window_size):
    # containers for input/output pairs
    X = []
    y = []

    for cnt in range(0,len(series)-window_size):
        X.append(series[cnt: cnt + window_size])
        y.append(series[cnt + window_size])

        # reshape each
    X = np.asarray(X)
    X.shape = (np.shape(X)[0:2])
    y = np.asarray(y)
    y.shape = (len(y),1)

    return X,y

# TODO: build an RNN to perform regression on our time series input/output data
def build_part1_RNN(window_size):
    #layer 1 uses an LSTM module with 5 hidden units (note here the input_shape = (window_size,1))
    model = Sequential()
    model.add(LSTM(5, input_shape=(window_size, 1)))

    #layer 2 uses a fully connected module with one unit
    model.add(Dense(1))
    return model


### TODO: return the text input with only ascii lowercase and the punctuation given below included.
def cleaned_text(text):

    punctuation = ['!', ',', '.', ':', ';', '?']
    alphaplusspace = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z',' ']
    validchars = alphaplusspace + punctuation

    # replace invalid characters with blanks
    inputtext = list(text)
    for i, char in enumerate(inputtext):
        if char not in validchars:
            inputtext[i] = ''

    return ''.join(inputtext)


### TODO: fill out the function below that transforms the input text and window-size into a set of input/output pairs for use with our RNN model
def window_transform_text(text, window_size, step_size):
    # containers for input/output pairs
    inputs = []
    outputs = []

    for i in range(0, len(text)-window_size, step_size):
        inputs.append(text[i: i + window_size])
        outputs.append(text[i + window_size])

    return inputs,outputs

# TODO build the required RNN model: 
# a single LSTM hidden layer with softmax activation, categorical_crossentropy loss 
def build_part2_RNN(window_size, num_chars):


    model = Sequential()
    #LSTM with 200 hidden units
    model.add(LSTM(200, input_shape=(window_size, num_chars)))
    # fully connected layer
    model.add(Dense(num_chars))
    # softmax activation layer
    model.add(Activation('softmax'))

    return model
