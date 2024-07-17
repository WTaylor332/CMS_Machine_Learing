import tensorflow as tf
from tensorflow import keras

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
        #     keras.layers.Conv2D(12, kernel_size=(8,1), activation='relu', input_shape=(shape)),
        #     keras.layers.MaxPool2D(pool_size=(4,1)),

        #     keras.layers.Conv2D(12, kernel_size=(8,1), activation='relu'),
        #     keras.layers.MaxPool2D(pool_size=(4,1)),

        #     keras.layers.Conv2D(12, kernel_size=(8,1), activation='relu'),
        #     keras.layers.MaxPool2D(pool_size=(2,1)),

        #     # multi later perceptron
        #     keras.layers.Flatten(),
        #     keras.layers.Dense(30, activation="relu"),
        #     # keras.layers.Dropout(rate=0.3),
        #     keras.layers.Dense(1)
        # ])
        # A2
        model = keras.models.Sequential([
            # convolutional layers
            keras.layers.Conv2D(12, kernel_size=(8,1), activation='relu', input_shape=(shape)),
            keras.layers.MaxPool2D(pool_size=(4,1)),

            keras.layers.Conv2D(12, kernel_size=(8,1), activation='relu'),
            keras.layers.MaxPool2D(pool_size=(4,1)),

            keras.layers.Conv2D(12, kernel_size=(8,1), activation='relu'),
            keras.layers.MaxPool2D(pool_size=(2,1)),

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
        #     keras.layers.Conv2D(12, kernel_size=(8,1), activation='relu', input_shape=(shape)),
        #     keras.layers.MaxPool2D(pool_size=(4,1)),

        #     keras.layers.Conv2D(12, kernel_size=(8,1), activation='relu'),
        #     keras.layers.MaxPool2D(pool_size=(4,1)),

        #     keras.layers.Conv2D(12, kernel_size=(8,1), activation='relu'),
        #     keras.layers.MaxPool2D(pool_size=(2,1)),

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
        #1D model
        ps = 4
        featNo = 12
        ks = 8
        model = keras.models.Sequential([
            # convolutional layers
            keras.layers.Conv1D(12, kernel_size=8, activation='relu', input_shape=(form)),
            keras.layers.MaxPool1D(pool_size=4),

            keras.layers.Conv1D(12, kernel_size=8, activation='relu'),
            keras.layers.MaxPool1D(pool_size=4),

            keras.layers.Conv1D(12, kernel_size=8, activation='relu'),
            keras.layers.MaxPool1D(pool_size=2),

            keras.layers.Conv1D(12, kernel_size=8, activation='relu'),
            keras.layers.MaxPool1D(pool_size=2),

            keras.layers.Conv1D(12, kernel_size=8, activation='relu'),
            keras.layers.MaxPool1D(pool_size=2),

            keras.layers.Flatten()
        ])
    else:
        # 2D model
        # A1
        # model = keras.models.Sequential([
        #     # convolutional layers
        #     keras.layers.Conv2D(12, kernel_size=(8,1), activation='relu', input_shape=(form)),
        #     keras.layers.MaxPool2D(pool_size=(4,1)),

        #     keras.layers.Conv2D(12, kernel_size=(8,1), activation='relu'),
        #     keras.layers.MaxPool2D(pool_size=(4,1)),

        #     keras.layers.Conv2D(12, kernel_size=(8,1), activation='relu'),
        #     keras.layers.MaxPool2D(pool_size=(4,1)),

        #     keras.layers.Conv2D(1, kernel_size=(2,2), activation='relu'),

        #     keras.layers.Flatten()
        # ])

        # A2
        model = keras.models.Sequential([
            # convolutional layers
            keras.layers.Conv2D(12, kernel_size=(8,1), activation='relu', input_shape=(form)),
            keras.layers.MaxPool2D(pool_size=(2,1)),

            keras.layers.Conv2D(12, kernel_size=(8,1), activation='relu'),
            keras.layers.MaxPool2D(pool_size=(2,1)),

            keras.layers.Conv2D(12, kernel_size=(8,1), activation='relu'),
            keras.layers.MaxPool2D(pool_size=(2,1)),

            keras.layers.Conv2D(12, kernel_size=(8,1), activation='relu'),
            keras.layers.MaxPool2D(pool_size=(2,1)),

            keras.layers.Conv2D(12, kernel_size=(8,1), activation='relu'),
            keras.layers.MaxPool2D(pool_size=(2,1)),

            keras.layers.Conv2D(1, kernel_size=(2,2), activation='relu'),

            keras.layers.Flatten()
        ])

        # A3
        # model = keras.models.Sequential([
        #     # convolutional layers
        #     keras.layers.Conv2D(12, kernel_size=(8,1), activation='relu', input_shape=(form)),
        #     keras.layers.Conv2D(12, kernel_size=(8,1), activation='relu'),
        #     keras.layers.MaxPool2D(pool_size=(4,1)),

        #     keras.layers.Conv2D(12, kernel_size=(8,1), activation='relu'),
        #     keras.layers.Conv2D(12, kernel_size=(8,1), activation='relu'),
        #     keras.layers.MaxPool2D(pool_size=(4,1)),

        #     keras.layers.Conv2D(12, kernel_size=(8,1), activation='relu'),
        #     keras.layers.Conv2D(1, kernel_size=(7,2), activation='relu'),

        #     keras.layers.Flatten()
        # ])

    model.compile(optimizer=op, loss=lossFunc)
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