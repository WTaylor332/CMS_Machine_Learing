import tensorflow as tf
from tensorflow import keras
import numpy as np

def multiLayerPerceptron(form, op, lossFunc):
    # model = keras.models.Sequential([
    #     keras.Input(shape=form),
    #     keras.layers.Dense(100, activation="relu"),
    #     # keras.layers.Dropout(rate=0.3),
    #     keras.layers.Dense(50, activation="relu"),
    #     # keras.layers.Dropout(rate=0.3),
    #     keras.layers.Flatten(),
    #     keras.layers.Dense(1)
    # ])
    # model = keras.models.Sequential([
    #     keras.Input(shape=form),
    #     keras.layers.Dense(80, activation="relu"),
    #     # keras.layers.Dropout(rate=0.3),
    #     keras.layers.Dense(40, activation="relu"),
    #     keras.layers.Dense(20, activation="relu"),
    #     # keras.layers.Dropout(rate=0.3),
    #     keras.layers.Flatten(),
    #     keras.layers.Dense(1)
    # ])
    # model = keras.models.Sequential([
    #     keras.Input(shape=form),
    #     keras.layers.Dense(60, activation="relu"),
    #     # keras.layers.Dropout(rate=0.3),
    #     keras.layers.Dense(40, activation="relu"),
    #     keras.layers.Dense(20, activation="relu"),
    #     keras.layers.Dense(10, activation="relu"),
    #     # keras.layers.Dropout(rate=0.3),
    #     keras.layers.Flatten(),
    #     keras.layers.Dense(1)
    # ])
    # model = keras.models.Sequential([
    #     keras.Input(shape=form),
    #     keras.layers.Dense(50, activation="relu"),
    #     # keras.layers.Dropout(rate=0.3),
    #     keras.layers.Dense(40, activation="relu"),
    #     keras.layers.Dense(30, activation="relu"),
    #     keras.layers.Dense(20, activation="relu"),
    #     keras.layers.Dense(10, activation="relu"),
    #     # keras.layers.Dropout(rate=0.3),
    #     keras.layers.Flatten(),
    #     keras.layers.Dense(1)
    # ])
    model = keras.models.Sequential([
        keras.Input(shape=form),
        keras.layers.Dense(10, activation="relu"),
        # keras.layers.Dropout(rate=0.3),
        keras.layers.Dense(8, activation="relu"),
        keras.layers.Dense(6, activation="relu"),
        keras.layers.Dense(4, activation="relu"),
        keras.layers.Dense(2, activation="relu"),
        # keras.layers.Dropout(rate=0.3),
        keras.layers.Flatten(),
        keras.layers.Dense(1)
    ])
    model.compile(optimizer=op, loss=lossFunc)
    model.summary()
    return model, 'mlp'


def convModel(shape, op, lossFunc):
    # 2D model
    # A3
    # model = keras.models.Sequential([
    #     # convolutional layers
    #     keras.layers.Conv2D(12, kernel_size=(1,8), activation='relu', input_shape=(shape)),
    #     keras.layers.MaxPool2D(pool_size=(1,4)),

    #     keras.layers.Conv2D(12, kernel_size=(1,8), activation='relu'),
    #     keras.layers.MaxPool2D(pool_size=(1,4)),

    #     keras.layers.Conv2D(12, kernel_size=(1,8), activation='relu'),
    #     keras.layers.MaxPool2D(pool_size=(1,2)),

    #     # multi later perceptron
    #     keras.layers.Flatten(),
    #     keras.layers.Dense(15, activation="relu"),
    #     # keras.layers.Dropout(rate=0.3),
    #     keras.layers.Dense(10, activation="relu"),
    #     # keras.layers.Dropout(rate=0.3),
    #     keras.layers.Dense(5, activation="relu"),
    #     # keras.layers.Dropout(rate=0.3),
    #     keras.layers.Dense(1)
    # ])
    # A4
    # model = keras.models.Sequential([
    #     # convolutional layers
    #     keras.layers.Conv2D(12, kernel_size=(1,8), activation='relu', input_shape=(shape)),
    #     keras.layers.MaxPool2D(pool_size=(1,4)),

    #     keras.layers.Conv2D(12, kernel_size=(1,8), activation='relu'),
    #     keras.layers.MaxPool2D(pool_size=(1,4)),

    #     keras.layers.Conv2D(12, kernel_size=(1,8), activation='relu'),
    #     keras.layers.MaxPool2D(pool_size=(1,2)),

    #     # multi later perceptron
    #     keras.layers.Flatten(),
    #     keras.layers.Dense(12, activation="relu"),
    #     # keras.layers.Dropout(rate=0.3),
    #     keras.layers.Dense(8, activation="relu"),
    #     keras.layers.Dense(8, activation="relu"),
    #     # keras.layers.Dropout(rate=0.3),
    #     keras.layers.Dense(8, activation="relu"),
    #     # keras.layers.Dropout(rate=0.3),
    #     keras.layers.Dense(1)
    # ])
    # A5
    # model = keras.models.Sequential([
    #     # convolutional layers
    #     keras.layers.Conv2D(10, kernel_size=(1,8), activation='relu', input_shape=(shape)),
    #     keras.layers.MaxPool2D(pool_size=(1,4)),

    #     keras.layers.Conv2D(10, kernel_size=(1,8), activation='relu'),
    #     keras.layers.MaxPool2D(pool_size=(1,4)),

    #     keras.layers.Conv2D(10, kernel_size=(1,8), activation='relu'),
    #     keras.layers.MaxPool2D(pool_size=(1,2)),

    #     # multi later perceptron
    #     keras.layers.Flatten(),
    #     keras.layers.Dense(12, activation="relu"),
    #     # keras.layers.Dropout(rate=0.3),
    #     keras.layers.Dense(8, activation="relu"),
    #     keras.layers.Dense(8, activation="relu"),
    #     # keras.layers.Dropout(rate=0.3),
    #     keras.layers.Dense(8, activation="relu"),
    #     # keras.layers.Dropout(rate=0.3),
    #     keras.layers.Dense(1)
    # ])
    # A7
    # model = keras.models.Sequential([
    #     # convolutional layers
    #     keras.layers.Conv2D(50, kernel_size=(1,8), activation='relu', input_shape=(shape)),
    #     keras.layers.MaxPool2D(pool_size=(1,4)),

    #     keras.layers.Conv2D(50, kernel_size=(1,8), activation='relu'),
    #     keras.layers.MaxPool2D(pool_size=(1,4)),

    #     keras.layers.Conv2D(50, kernel_size=(1,8), activation='relu'),
    #     keras.layers.MaxPool2D(pool_size=(1,2)),

    #     # multi later perceptron
    #     keras.layers.Flatten(),
    #     keras.layers.Dense(100, activation="relu"),
    #     keras.layers.Dense(50, activation="relu"),
    #     keras.layers.Dense(40, activation="relu"),
    #     keras.layers.Dense(30, activation="relu"),
    #     keras.layers.Dense(20, activation="relu"),
    #     keras.layers.Dense(10, activation="relu"),
    #     keras.layers.Dense(5, activation="relu"),
    #     keras.layers.Dense(1)
    # ])
    # A8
    # model = keras.models.Sequential([
    #     # convolutional layers
    #     keras.layers.Conv2D(10, kernel_size=(1,8), activation='relu', input_shape=(shape)),
    #     keras.layers.MaxPool2D(pool_size=(1,4)),

    #     keras.layers.Conv2D(10, kernel_size=(1,8), activation='relu'),
    #     keras.layers.MaxPool2D(pool_size=(1,4)),

    #     keras.layers.Conv2D(10, kernel_size=(1,8), activation='relu'),
    #     keras.layers.MaxPool2D(pool_size=(1,2)),

    #     # multi later perceptron
    #     keras.layers.Flatten(),
    #     keras.layers.Dense(8, activation="relu"),
    #     keras.layers.Dense(8, activation="relu"),
    #     keras.layers.Dense(8, activation="relu"),
    #     keras.layers.Dense(8, activation="relu"),
    #     keras.layers.Dense(8, activation="relu"),
    #     keras.layers.Dense(8, activation="relu"),
    #     keras.layers.Dense(1)
    # ])
    # A9
    model = keras.models.Sequential([
        # convolutional layers
        keras.layers.Conv2D(10, kernel_size=(1,8), activation='relu', input_shape=(shape)),
        keras.layers.MaxPool2D(pool_size=(1,4)),

        keras.layers.Conv2D(10, kernel_size=(1,8), activation='relu'),
        keras.layers.MaxPool2D(pool_size=(1,4)),

        keras.layers.Conv2D(10, kernel_size=(1,8), activation='relu'),
        keras.layers.MaxPool2D(pool_size=(1,2)),

        # multi later perceptron
        keras.layers.Flatten(),
        keras.layers.Dense(6, activation="relu"),
        keras.layers.Dense(6, activation="relu"),
        keras.layers.Dense(6, activation="relu"),
        keras.layers.Dense(6, activation="relu"),
        keras.layers.Dense(6, activation="relu"),
        keras.layers.Dense(6, activation="relu"),
        keras.layers.Dense(6, activation="relu"),
        keras.layers.Dense(6, activation="relu"),
        keras.layers.Dense(6, activation="relu"),
        keras.layers.Dense(6, activation="relu"),
        keras.layers.Dense(1)
    ])
    # A10
    # model = keras.models.Sequential([
    #     # convolutional layers
    #     keras.layers.Conv2D(10, kernel_size=(1,8), activation='relu', input_shape=(shape)),
    #     keras.layers.MaxPool2D(pool_size=(1,4)),

    #     keras.layers.Conv2D(10, kernel_size=(1,8), activation='relu'),
    #     keras.layers.MaxPool2D(pool_size=(1,4)),

    #     keras.layers.Conv2D(10, kernel_size=(1,8), activation='relu'),
    #     keras.layers.MaxPool2D(pool_size=(1,2)),

    #     # multi later perceptron
    #     keras.layers.Flatten(),
    #     keras.layers.Dense(5, activation="relu"),
    #     keras.layers.Dense(5, activation="relu"),
    #     keras.layers.Dense(5, activation="relu"),
    #     keras.layers.Dense(5, activation="relu"),
    #     keras.layers.Dense(5, activation="relu"),
    #     keras.layers.Dense(5, activation="relu"),
    #     keras.layers.Dense(5, activation="relu"),
    #     keras.layers.Dense(5, activation="relu"),
    #     keras.layers.Dense(5, activation="relu"),
    #     keras.layers.Dense(5, activation="relu"),
    #     keras.layers.Dense(5, activation="relu"),
    #     keras.layers.Dense(5, activation="relu"),
    #     keras.layers.Dense(5, activation="relu"),
    #     keras.layers.Dense(1)
    # ])

    model.compile(optimizer=op, loss=lossFunc)
    return model, 'conv'


def pureCNN(form, op, lossFunc):
    if len(form) < 3:
        #1D model - FIX IT
        model = keras.models.Sequential([
            # convolutional layers
            keras.layers.Conv1D(12, kernel_size=8, activation='relu', input_shape=(form)),
            keras.layers.MaxPool1D(pool_size=4),

            keras.layers.Conv1D(12, kernel_size=8, activation='relu'),
            keras.layers.MaxPool1D(pool_size=4),

            keras.layers.Conv1D(12, kernel_size=8, activation='relu'),
            keras.layers.MaxPool1D(pool_size=2),

            keras.layers.Flatten()
        ])
    else:
        # 2D model
        # A1
        # model = keras.models.Sequential([
        #     # convolutional layers
        #     keras.layers.Conv2D(12, kernel_size=(1,8), activation='relu', input_shape=(form)),
        #     keras.layers.MaxPool2D(pool_size=(1,4)),

        #     keras.layers.Conv2D(12, kernel_size=(1,8), activation='relu'),
        #     keras.layers.MaxPool2D(pool_size=(1,4)),

        #     keras.layers.Conv2D(12, kernel_size=(1,8), activation='relu'),
        #     keras.layers.MaxPool2D(pool_size=(1,4)),

        #     keras.layers.Conv2D(1, kernel_size=(2,2), activation='relu'),

        #     keras.layers.Flatten()
        # ])

        # A2
        # model = keras.models.Sequential([
        #     # convolutional layers
        #     keras.layers.Conv2D(12, kernel_size=(1,8), activation='relu', input_shape=(form)),
        #     keras.layers.MaxPool2D(pool_size=(1,2)),

        #     keras.layers.Conv2D(12, kernel_size=(1,8), activation='relu'),
        #     keras.layers.MaxPool2D(pool_size=(1,2)),

        #     keras.layers.Conv2D(12, kernel_size=(1,8), activation='relu'),
        #     keras.layers.MaxPool2D(pool_size=(1,2)),

        #     keras.layers.Conv2D(12, kernel_size=(1,8), activation='relu'),
        #     keras.layers.MaxPool2D(pool_size=(1,2)),

        #     keras.layers.Conv2D(12, kernel_size=(1,8), activation='relu'),
        #     keras.layers.MaxPool2D(pool_size=(1,2)),

        #     keras.layers.Conv2D(1, kernel_size=(2,2), activation='relu'),

        #     keras.layers.Flatten()
        # ])

        # # A3
        # model = keras.models.Sequential([
        #     # convolutional layers
        #     keras.layers.Conv2D(12, kernel_size=(1,8), activation='relu', input_shape=(form)),
        #     keras.layers.Conv2D(12, kernel_size=(1,8), activation='relu'),
        #     keras.layers.MaxPool2D(pool_size=(1,4)),

        #     keras.layers.Conv2D(12, kernel_size=(1,8), activation='relu'),
        #     keras.layers.Conv2D(12, kernel_size=(1,8), activation='relu'),
        #     keras.layers.MaxPool2D(pool_size=(1,4)),

        #     keras.layers.Conv2D(12, kernel_size=(1,8), activation='relu'),
        #     keras.layers.Conv2D(1, kernel_size=(2,7), activation='relu'),

        #     keras.layers.Flatten()
        # ])

        # A4
        model = keras.models.Sequential([
            # convolutional layers
            keras.layers.Conv2D(12, kernel_size=(2,6), activation='relu', input_shape=(form)),
            keras.layers.MaxPool2D(pool_size=(1,4)),

            keras.layers.Conv2D(12, kernel_size=(1,6), activation='relu'),
            keras.layers.MaxPool2D(pool_size=(1,4)),

            keras.layers.Conv2D(12, kernel_size=(1,6), activation='relu'),
            keras.layers.MaxPool2D(pool_size=(1,4)),

            keras.layers.Conv2D(1, kernel_size=(1,3), activation='relu'),

            keras.layers.Flatten()
        ])

    model.compile(optimizer=op, loss=lossFunc)
    return model, 'pconv'


def rnn(form, op, lossFunc, maskNo):

    # model = keras.models.Sequential([
    #     keras.Input(shape=form),
    #     keras.layers.Bidirectional(keras.layers.GRU(20, return_sequences=True, activation='relu')),
    #     keras.layers.GRU(20, activation='relu'),
    #     keras.layers.Dense(1)
    # ])

    # masking model
    # model = keras.models.Sequential([
    #     keras.Input(shape=form),
    #     # keras.layers.Masking(mask_value=-9999),
    #     # keras.layers.GRU(100, return_sequences=True, activation='relu'),
    #     keras.layers.Masking(mask_value=None),
    #     keras.layers.GRU(50, activation='relu'),
    #     keras.layers.Dense(1)
    # ])

    # LSTM Masking
    # model = keras.models.Sequential([
    #     keras.Input(shape=form),
    #     keras.layers.Masking(mask_value=maskNo),
    #     keras.layers.LSTM(20, return_sequences=True, activation='relu'),
    #     keras.layers.LSTM(20, activation='relu'),
    #     keras.layers.Dense(1)
    # ])
    
    # model = keras.models.Sequential([
    #     keras.Input(shape=form),
    #     keras.layers.Masking(mask_value=maskNo),
    #     keras.layers.SimpleRNN(20, return_sequences=True, activation='tanh'),
    #     keras.layers.SimpleRNN(20, activation='tanh'),
    #     keras.layers.Dense(1)
    # ])

    # model = keras.models.Sequential([
    #     keras.Input(shape=form),
    #     keras.layers.Masking(mask_value=maskNo),
    #     keras.layers.GRU(40, return_sequences=True, activation='relu'),
    #     keras.layers.GRU(40, activation='relu'),
    #     keras.layers.Dense(1)
    # ])

    # Ragged RNN model - not working
    # model = keras.models.Sequential([
    #     keras.layers.InputLayer(shape=[None, 3], dytpe=tf.float64, ragged=True),
    #     keras.layers.GRU(20, use_bias=False, return_sequences=True, activation='relu'),
    #     keras.layers.GRU(20, use_bias=False, activation='relu'),
    #     keras.layers.Dense(1)
    # ])

    # raw bin data
    # model = keras.models.Sequential([
    #      keras.layers.Masking(mask_value=maskNo, input_shape=form),
    #      keras.layers.GRU(20, return_sequences=True),
    #      keras.layers.GRU(20),
    #      keras.layers.Dense(1)
    #  ])
    # model = keras.models.Sequential([
    #      keras.layers.Masking(mask_value=maskNo, input_shape=form),
    #      keras.layers.SimpleRNN(20, return_sequences=True),
    #      keras.layers.SimpleRNN(20),
    #      keras.layers.Dense(1)
    #  ])
    model = keras.models.Sequential([
         keras.layers.Masking(mask_value=maskNo, input_shape=form),
         keras.layers.GRU(20, return_sequences=True),
         keras.layers.GRU(20),
         keras.layers.Dense(1, activation='softmax')
     ])

    model.compile(optimizer=op, loss=lossFunc)
    return model, 'rnn'
    

def wavenet(form, op, lossFunc):
    if form[1] < 2:
        #1D model
        model = keras.models.Sequential([
            # convolutional layers
            keras.layers.Conv1D(12, kernel_size=8, activation='relu', input_shape=(form), dilation_rate=2),
            keras.layers.MaxPool1D(pool_size=4),

            # multi later perceptron
            keras.layers.Flatten(),
            keras.layers.Dense(15, activation="relu"),
            # keras.layers.Dropout(rate=0.3),
            keras.layers.Dense(5, activation="relu"),
            # keras.layers.Dropout(rate=0.3),
            keras.layers.Dense(1)
        ])
    else:
        # 2D model

        model = keras.models.Sequential([
            # convolutional layers
            keras.layers.Conv2D(12, kernel_size=(1,8), activation='relu', input_shape=(form), dilation_rate=(1,2)),
            keras.layers.MaxPool2D(pool_size=(1,4)),

            keras.layers.Conv2D(12, kernel_size=(1,8), activation='relu'), 
            keras.layers.MaxPool2D(pool_size=(1,4)),

            keras.layers.Conv2D(12, kernel_size=(1,8), activation='relu'), 
            keras.layers.MaxPool2D(pool_size=(1,2)),

            # multi later perceptron
            keras.layers.Flatten(),
            keras.layers.Dense(15, activation="relu"),
            keras.layers.Dense(10, activation="relu"),
            # keras.layers.Dropout(rate=0.3),
            keras.layers.Dense(5, activation="relu"),
            # keras.layers.Dropout(rate=0.3),
            keras.layers.Dense(1)
        ])

        # model = keras.models.Sequential([
        #     # convolutional layers
        #     keras.layers.Conv2D(12, kernel_size=(1,8), activation='relu', input_shape=(form), dilation_rate=(1,3)),
        #     keras.layers.MaxPool2D(pool_size=(1,4)),

        #     keras.layers.Conv2D(12, kernel_size=(1,8), activation='relu'), 
        #     keras.layers.MaxPool2D(pool_size=(1,4)),

        #     keras.layers.Conv2D(12, kernel_size=(1,8), activation='relu'),
        #     keras.layers.MaxPool2D(pool_size=(1,2)),

        #     # multi later perceptron
        #     keras.layers.Flatten(),
        #     keras.layers.Dense(15, activation="relu"),
        #     keras.layers.Dense(10, activation="relu"),
        #     keras.layers.Dense(5, activation="relu"),
        #     # keras.layers.Dropout(rate=0.3),
        #     keras.layers.Dense(1)
        # ])

        # model = keras.models.Sequential([
        #     # convolutional layers
        #     keras.layers.Conv2D(12, kernel_size=(1,8), activation='relu', input_shape=(form), dilation_rate=(1,2)),
        #     keras.layers.MaxPool2D(pool_size=(1,4)),

        #     keras.layers.Conv2D(12, kernel_size=(1,8), activation='relu', dilation_rate=(1,2)),
        #     keras.layers.MaxPool2D(pool_size=(1,4)),

        #     keras.layers.Conv2D(12, kernel_size=(1,8), activation='relu'),
        #     keras.layers.MaxPool2D(pool_size=(1,2)),

        #     # multi later perceptron
        #     keras.layers.Flatten(),
        #     keras.layers.Dense(15, activation="relu"),
        #     # keras.layers.Dropout(rate=0.3),
        #     keras.layers.Dense(10, activation="relu"),
        #     # keras.layers.Dropout(rate=0.3),
        #     keras.layers.Dense(5, activation="relu"),
        #     # keras.layers.Dropout(rate=0.3),
        #     keras.layers.Dense(1)
        # ])

    model.compile(optimizer=op, loss=lossFunc)
    return model, 'wavenet'



# RNN with attention

# shape of data is (no_of_events, no_parameters, tracklength)
# treat each event as a point in time and then we can treat this as time series data
# data is already in vector form - a 2D array of shape (no_parameters, tracklength)

# Encoder - data is the context

# Decoder - pv values are the target 

def rnnAttention(form, op, lossFunc, units):
    model = keras.Sequential([
        keras.layers.Bidirectional(merge_mode='sum',
                            layer=tf.keras.layers.GRU(units, return_sequences=True,recurrent_initializer='glorot_uniform')),
        keras.layers.MultiHeadAttention(key_dim=units, num_heads=1),
        keras.layers.LayerNomalization()


    ])
