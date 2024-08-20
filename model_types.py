import tensorflow as tf
from tensorflow import keras

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

    model.compile(optimizer=op, loss=lossFunc)
    return model, 'conv'


def rnn(form, op, lossFunc, maskNo):
    # raw bin data
    # model = keras.models.Sequential([
    #      keras.layers.Masking(mask_value=maskNo, input_shape=form),
    #      keras.layers.SimpleRNN(20, return_sequences=True),
    #      keras.layers.SimpleRNN(20),
    #      keras.layers.Dense(1, activation='sigmoid')
    #  ])

    # model = keras.models.Sequential([
    #      keras.layers.Masking(mask_value=maskNo, input_shape=form),
    #      keras.layers.GRU(20, return_sequences=True),
    #      keras.layers.GRU(20),
    #      keras.layers.Dense(1, activation='sigmoid')
    #  ])
    
    # model = keras.models.Sequential([
    #      keras.layers.Masking(mask_value=maskNo, input_shape=form),
    #      keras.layers.SimpleRNN(20, return_sequences=True),
    #      keras.layers.SimpleRNN(20, return_sequences=True),
    #      keras.layers.SimpleRNN(20, return_sequences=True),
    #      keras.layers.SimpleRNN(20),
    #      keras.layers.Dense(1, activation='sigmoid')
    #  ])

    # FPGA 
    model = keras.models.Sequential([
         keras.layers.Masking(mask_value=maskNo, input_shape=form),
         keras.layers.SimpleRNN(2, return_sequences=True),
         keras.layers.SimpleRNN(2, return_sequences=True),
         keras.layers.SimpleRNN(2, return_sequences=True),
         keras.layers.SimpleRNN(2),
         keras.layers.Dense(1, activation='sigmoid')
     ])

    model.compile(optimizer=op, loss=lossFunc)
    return model, 'rnn'


def wavenet(form, op, lossFunc):
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