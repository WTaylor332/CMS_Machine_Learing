import tensorflow as tf
from tensorflow import keras
import numpy as np

def multiLayerPerceptron(form, op, lossFunc):
    model = keras.models.Sequential([
        keras.Input(shape=form),
        keras.layers.Dense(100, activation="relu"),
        keras.layers.Dropout(rate=0.3),
        keras.layers.Dense(50, activation="relu"),
        keras.layers.Dropout(rate=0.3),
        keras.layers.Dense(1)
    ])
    model.compile(optimizer=op, loss=lossFunc)
    model.summary()
    return model


def convModel(shape, op, lossFunc):
    if shape[1] < 2: # 1D model
        # A1
        # model = keras.models.Sequential([
        #     # convolutional layers
        #     keras.layers.Conv1D(12, kernel_size=8, activation='relu', input_shape=(shape)),
        #     keras.layers.MaxPool1D(pool_size=4),

        #     keras.layers.Conv1D(12, kernel_size=8, activation='relu'),
        #     keras.layers.MaxPool1D(pool_size=4),

        #     keras.layers.Conv1D(12, kernel_size=8, activation='relu'),
        #     keras.layers.MaxPool1D(pool_size=2),

        #     # multi later perceptron
        #     keras.layers.Flatten(),
        #     keras.layers.Dense(30, activation="relu"),
        #     # keras.layers.Dropout(rate=0.3),
        #     keras.layers.Dense(1)
        # ])
        # A2
        model = keras.models.Sequential([
            # convolutional layers
            keras.layers.Conv1D(12, kernel_size=8, activation='relu', input_shape=(shape)),
            keras.layers.MaxPool1D(pool_size=4),

            keras.layers.Conv1D(12, kernel_size=8, activation='relu'),
            keras.layers.MaxPool1D(pool_size=4),

            keras.layers.Conv1D(12, kernel_size=8, activation='relu'),
            keras.layers.MaxPool1D(pool_size=2),

            # multi later perceptron
            keras.layers.Flatten(),
            keras.layers.Dense(15, activation="relu"),
            # keras.layers.Dropout(rate=0.3),
            keras.layers.Dense(5, activation="relu"),
            # keras.layers.Dropout(rate=0.3),
            keras.layers.Dense(1)
        ])
        # A3
        # model = keras.models.Sequential([
        #     # convolutional layers
        #     keras.layers.Conv1D(12, kernel_size=8, activation='relu', input_shape=(shape)),
        #     keras.layers.MaxPool1D(pool_size=4),

        #     keras.layers.Conv1D(12, kernel_size=8, activation='relu'),
        #     keras.layers.MaxPool1D(pool_size=4),

        #     keras.layers.Conv1D(12, kernel_size=8, activation='relu'),
        #     keras.layers.MaxPool1D(pool_size=2),

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
        
    else:
        # 2D model
        # A1
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
        #     keras.layers.Dense(30, activation="relu"),
        #     # keras.layers.Dropout(rate=0.3),
        #     keras.layers.Dense(1)
        # ])
        # A2
        model = keras.models.Sequential([
            # convolutional layers
            keras.layers.Conv2D(12, kernel_size=(1,8), activation='relu', input_shape=(shape)),
            keras.layers.MaxPool2D(pool_size=(1,4)),

            keras.layers.Conv2D(12, kernel_size=(1,8), activation='relu'),
            keras.layers.MaxPool2D(pool_size=(1,4)),

            keras.layers.Conv2D(12, kernel_size=(1,8), activation='relu'),
            keras.layers.MaxPool2D(pool_size=(1,2)),

            # multi later perceptron
            keras.layers.Flatten(),
            keras.layers.Dense(15, activation="relu"),
            # keras.layers.Dropout(rate=0.3),
            keras.layers.Dense(5, activation="relu"),
            # keras.layers.Dropout(rate=0.3),
            keras.layers.Dense(1)
        ])
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

    model.compile(optimizer=op, loss=lossFunc)
    return model


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

        # A3
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
    return model


def rnn(form, op, lossFunc):
    # model = keras.models.Sequential([
    #     keras.Input(shape=form),
    #     keras.layers.GRU(100, return_sequences=True, activation='relu'),
    #     keras.layers.GRU(50, activation='relu'),
    #     keras.layers.Dense(1)
    # ])

    # model = keras.models.Sequential([
    #     keras.Input(shape=form),
    #     keras.layers.Bidirectional(keras.layers.GRU(100, return_sequences=True, activation='relu')),
    #     keras.layers.GRU(50, activation='relu'),
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
    model = keras.models.Sequential([
        keras.Input(shape=form),
        # keras.layers.Masking(mask_value=-9999),
        # keras.layers.GRU(100, return_sequences=True, activation='relu'),
        keras.layers.Masking(mask_value=-1000),
        keras.layers.LSTM(50, activation='relu'),
        keras.layers.Dense(1)
    ])

    model.compile(optimizer=op, loss=lossFunc)
    return model
    

def wavenet(form):
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
            keras.layers.Conv2D(12, kernel_size=(8,1), activation='relu', input_shape=(form), dilation_rate=(2,2)),
            keras.layers.MaxPool2D(pool_size=(4,4)),

            # multi later perceptron
            keras.layers.Flatten(),
            keras.layers.Dense(15, activation="relu"),
            # keras.layers.Dropout(rate=0.3),
            keras.layers.Dense(5, activation="relu"),
            # keras.layers.Dropout(rate=0.3),
            keras.layers.Dense(1)
        ])
    return model


def transformer(form):
    print()