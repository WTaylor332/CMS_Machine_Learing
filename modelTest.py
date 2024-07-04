import numpy as np 
print()
import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras.callbacks import TensorBoard
print()
import matplotlib.pyplot as plt 
from matplotlib import cm 
from accessData import *
from sklearn.preprocessing import StandardScaler


def loading(name):
    if name == "TTbar.root":
        eventZ, eventPT, eventsPV = loadData(name)
        rawDataAll = rawPaddedData(eventZ, eventPT)
        binDataAll = histogramData(rawData)
    else:
        eventsAll = np.array([[]])
        eventsPV = np.array([])
        rawDataAll = np.array([])
        binDataAll = np.array([])

        for i in range(100,110):
            name = 'GTTObjects_ttbar200PU_{}.root'.format(i)
            eventZ, eventPT, pv = loadData(name)
            print(len(eventZ))
            print(len(eventPT))
            rawData = rawPaddedData(eventZ, eventPT)
            rawDataAll = np.append(rawDataAll, rawData)
            binData = histogramData(rawData)
            binDataAll = np.append(binDataAll, binData)
            pv = np.array(pv).flatten()
            print(pv.shape)
            eventsPV = np.append(eventsPV, [pv])

        print()
        print(eventsAll.shape)

    return rawDataAll, binDataAll, eventsPV.reshape(-1)


TTbar = np.load("TTbar.npz")
zRaw, ptRaw, ptBin, pvRaw = TTbar['z'], TTbar['pt'], TTbar['ptB'], TTbar['pv']


def binModel(pt, pv):

    t = len(pt)//10
    v = len(pt)//5
    xTest, xValid, xTrain = pt[:t], pt[t:v], pt[v:]
    yTest, yValid, yTrain = pv[:t], pv[t:v], pv[v:]

    model = keras.models.Sequential([
        keras.layers.Flatten(input_shape=(300,)),
        keras.layers.Dense(200, activation="relu"),
        keras.layers.Dense(100, activation="relu"),
        keras.layers.Dense(1)
    ])

    model.summary()

    model.compile(optimizer='adam',
    loss = "mean_absolute_error",
    metrics=["accuracy"])

    history = model.fit(xTrain, yTrain, epochs=10)

    model.evaluate(xValid, yValid)

    yPredicted = model.predict(xTest)
    yPredictedLabels = [np.argmax(i) for i in yPredicted]
    print(yPredicted.shape)
    print(yPredicted[:5])

    return history



def rawModel(z, pt, pv):
    z[z == float("nan")] = 9999
    pt[pt == float("nan")] = 9999
    binDataAll = np.stack((z,pt), axis=1)
    t = len(binDataAll)//10
    v = len(binDataAll)//5
    xTest, xValid, xTrain = binDataAll[:t], binDataAll[t:v], binDataAll[v:]
    yTest, yValid, yTrain = pv[:t], pv[t:v], pv[v:]

    model = keras.models.Sequential([
        keras.layers.Flatten(input_shape=(2,300)),
        keras.layers.Dense(200, activation="relu"),
        keras.layers.Dense(100, activation="relu"),
        keras.layers.Dense(50, activation="relu"),
        keras.layers.Dense(25, activation="relu"),
        keras.layers.Dense(1, activation="softmax")
    ])

    model.summary()

    model.compile(optimizer='adam',
    loss = "mean_absolute_error",
    metrics=["accuracy"])

    history = model.fit(xTrain, yTrain, epochs=5, validation_data=(xValid, yValid))

    return history


# hist = binModel(pt, pvRaw.flatten())

# loss = hist.history['loss']
# val_loss = hist.history['val_loss']

# epochs = range(1, len(loss) + 1)

# plt.plot(epochs, loss, 'bo', label='Training Loss')
# plt.plot(epochs, val_loss, 'b', color='red', label='Validation Loss')
# plt.title('Training and Validation Loss')
# plt.legend()
# plt.savefig("binModel_loss2.png")


# print()
# hist = rawModel(zRaw, ptRaw, pvRaw)

# loss = hist.history['loss']
# val_loss = hist.history['val_loss']

# epochs = range(1, len(loss) + 1)

# plt.plot(epochs, loss, 'bo', label='Training Loss')
# plt.plot(epochs, val_loss, 'b', color='red', label='Validation Loss')
# plt.title('Training and Validation Loss')
# plt.legend()
# plt.savefig("rawModel_loss1.png")
