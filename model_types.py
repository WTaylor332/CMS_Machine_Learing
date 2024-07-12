import tensorflow as tf
from tensorflow import keras

def multiLayerPerceptron(form):
    model = keras.models.Sequential([
        keras.Input(shape=form),
        keras.layers.Dense(100, activation="relu"),
        keras.layers.Dropout(rate=0.3),
        keras.layers.Dense(50, activation="relu"),
        keras.layers.Dropout(rate=0.3),
        keras.layers.Dense(1)
    ])
    return model


def convModel(shape):
    if shape[1] < 2:
        #1D model
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
    else:
        # 2D model
        model = keras.models.Sequential([
            # convolutional layers
            keras.layers.Conv2D(12, kernel_size=(8,1), activation='relu', input_shape=(shape)),
            keras.layers.MaxPool2D(pool_size=(4,4)),

            keras.layers.Conv2D(12, kernel_size=(8,1), activation='relu'),
            keras.layers.MaxPool2D(pool_size=(4,4)),

            keras.layers.Conv2D(12, kernel_size=(8,1), activation='relu'),
            keras.layers.MaxPool2D(pool_size=(2,2)),

            # multi later perceptron
            keras.layers.Flatten(),
            keras.layers.Dense(15, activation="relu"),
            # keras.layers.Dropout(rate=0.3),
            keras.layers.Dense(5, activation="relu"),
            # keras.layers.Dropout(rate=0.3),
            keras.layers.Dense(1)
        ])
    return model


def rnn(form):
    print()
    


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