import numpy as np 
import pandas as pd
import time
import os
import tqdm
print()
import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras.callbacks import TensorBoard
print()
import matplotlib.pyplot as plt 
from sklearn.preprocessing import StandardScaler

class haltCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if (logs.get('val_loss') <= 0.13):
            print('\n\nValuation loss reach 0.1232 so training stopped.\n\n')
            self.model.stop_training = True


def binModelSplit(pt, pv):
    # scaling 
    columnPT = pt.reshape(pt.shape[0]*pt.shape[1], 1)
    scaler = StandardScaler().fit(columnPT)
    ptScale = scaler.transform(columnPT)
    pt = ptScale.reshape(pt.shape[0], pt.shape[1])

    # splitting data into test, validation and training data
    t = len(pt)//10
    v = (len(pt)//10) * 3
    xTest, xValid, xTrain = pt[:t], pt[t:v], pt[v:]
    yTest, yValid, yTrain = pv[:t], pv[t:v], pv[v:]

    return xTrain, yTrain, xValid, yValid, xTest, yTest


def convModel(shape):
    model = keras.models.Sequential([
        # convolutional layer
        keras.layers.Conv1D(5, kernel_size=8, activation='relu', input_shape=(shape)),
        keras.layers.MaxPool1D(pool_size=4),

        keras.layers.Conv1D(5, kernel_size=8, activation='relu'),
        keras.layers.MaxPool1D(pool_size=2),

        # keras.layers.Conv1D(5, kernel_size=4, activation='relu'),
        # keras.layers.MaxPool1D(pool_size=2),


        # multi later perceptron
        keras.layers.Flatten(),
        keras.layers.Dense(25, activation="relu"),
        # keras.layers.Dropout(rate=0.3),
        # keras.layers.Dense(10, activation="relu"),
        # keras.layers.Dropout(rate=0.3),
        keras.layers.Dense(1)
    ])
    return model


def binModel(xTrain, yTrain, xValid, yValid, xTest, yTest):
    form = (xTrain.shape[1],1)

    # creating model
    # model = keras.models.Sequential([
    #     
    #     # multi later perceptron
    #     keras.Input(shape=form),
    #     keras.layers.Dense(100, activation="relu"),
    #     keras.layers.Dropout(rate=0.3),
    #     keras.layers.Dense(50, activation="relu"),
    #     keras.layers.Dropout(rate=0.3),
    #     keras.layers.Dense(1)
    # ])
    model = convModel(form)
    model.summary()

    op = keras.optimizers.Adam()
    lossFunc = keras.losses.Huber()

    # saving the model and best weights
    weights = "Bin_model_conv_weights_{t}.h5".format(t=clock)
    modelDirectory = "models"
    modelName = "Bin_model_conv_{o}_{l}_{t}".format(o='adam', l=lossFunc.name, t=clock)
    
    # callbacks
    checkpointCallback = keras.callbacks.ModelCheckpoint(filepath=weights, monitor="val_loss", save_weights_only=True, save_best_only=True, verbose=1)
    lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.4, patience=2, cooldown = 1, min_lr=0.000001, verbose=1)
    csvLogger = keras.callbacks.CSVLogger("training_{}.log".format(modelName), separator=',', append=False)
    stopTraining = haltCallback()
    earlyStop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)

    model.compile(optimizer=op,
    loss=lossFunc)
    
    epochNo = 10
    history = model.fit(xTrain, yTrain, epochs=epochNo, validation_data=(xValid, yValid), callbacks=[lr, checkpointCallback, csvLogger, stopTraining, earlyStop])

    checkpointFilename = os.path.join(modelDirectory, weights)
    check = os.path.isdir(modelDirectory)
    if not check:
        os.makedirs(modelDirectory)
        print("Created directory:" , modelDirectory)

    # saves full model
    modelName = "Bin_model_conv_{o}_{l}_{t}".format(o='adam', l=lossFunc.name, t=clock)
    modelFilename = os.path.join(modelDirectory, modelName)
    model.save(modelName)

    return model, history, modelName


def rawModelSplit(z, pt, pv):
    # scaling z
    columnZ = z.reshape(z.shape[0]*z.shape[1], 1)
    scaler = StandardScaler().fit(columnZ)
    columnZ = scaler.transform(columnZ)
    z = columnZ.reshape(pt.shape[0], pt.shape[1])

    z = np.nan_to_num(z, nan=-9999)
    z = z[:,:250]
    pt = np.nan_to_num(pt, nan=-9999)
    pt = pt[:,:250]
    binDataAll = np.stack((z,pt), axis=1)

    # splitting data into test, validation and training data
    t = len(binDataAll)//10
    v = len(binDataAll)//5
    xTest, xValid, xTrain = binDataAll[:t], binDataAll[t:v], binDataAll[v:]
    yTest, yValid, yTrain = pv[:t], pv[t:v], pv[v:]

    return xTrain, yTrain, xValid, yValid, xTest, yTest


def rawModel(xTrain, yTrain, xValid, yValid, xTest, yTest):
    form = (xTrain.shape[1],1)

    # creating model
    model = keras.models.Sequential([
        # convolutional layer
        # keras.layers.Conv1D(14, kernel_size = (1,8), input_shape=(2,z.shape[1]), stride=1),
        # keras.layers.MaxPool2D(pool_size=(,2)),

        # multi layer perceptron
        keras.layers.Flatten(input_shape=(2,xTrain.shape[1])),
        keras.layers.Dense(50, activation="relu"),
        keras.layers.Dense(25, activation="relu"),
        keras.layers.Dense(1)
    ])
    model.summary()
    
    # saving the model and best weights
    weights = "Raw_model_weights_{t}.h5".format(t=int(time.time()))
    modelDirectory = "models"
    modelName = "Raw_model_{o}_{l}_{t}".format(o='adam', l='huber', t=clock)

    checkpointFilename = os.path.join(modelDirectory, weights)
    check = os.path.isdir(modelDirectory)
    if not check:
        os.makedirs(modelDirectory)
        print("Created directory:" , modelDirectory)

    op = keras.optimizers.Adam(learning_rate=0.01)
    lossFunc = keras.losses.Huber()

    model.compile(optimizer=op,
    loss = lossFunc)

    checkpointCallback = keras.callbacks.ModelCheckpoint(filepath=weights, monitor="val_loss", save_weights_only=True, save_best_only=True, verbose=1)
    lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, cooldown = 1, min_lr=0.000001, verbose=1)
    csvLogger = keras.callbacks.CSVLogger("training_{}.log".format(modelName), separator=',', append=False)

    history = model.fit(xTrain, yTrain, epochs=10, validation_data=(xValid, yValid), callbacks=[checkpointCallback, lr, csvLogger])

    # saves full model
    modelName = "Raw_model_{o}_{l}_{t}".format(o=model.optimizer.name, l=model.loss.name, t=clock)
    modelFilename = os.path.join(modelDirectory, modelName)
    model.save(modelName)

    return model, history, modelName


def testing(model, hist, xValid, yValid, xTest, yTest, name):
    print()
    model.evaluate(xValid, yValid)

    yPredicted = model.predict(xTest)

    # plot of epochs against training and validation loss
    print()
    loss = hist.history['loss']
    val_loss = hist.history['val_loss']
    epochs = range(1, len(loss) + 1)

    plt.clf()
    plt.plot(epochs, loss, 'bo', label='Training Loss')
    plt.plot(epochs, val_loss, 'b', color='red', label='Validation Loss')
    plt.grid(which='major', color='#DDDDDD', linewidth=0.8)
    plt.grid(which='minor', color='#EEEEEE', linestyle=':', linewidth=0.6)
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig("Train_valid_loss_{}.png".format(name))

    # histogram of difference on test sample
    print()
    plt.clf()
    diff = abs(yPredicted.flatten() - yTest.flatten())
    print(diff[:5])
    print(max(diff), min(diff))
    print(np.std(diff), np.mean(diff))
    plt.hist(diff, bins=200)
    plt.title('Loss of predicted vs test Histogram')
    plt.savefig("Hist_loss_{}.png".format(name))

    # plotting % of predictions vs loss
    print()
    plt.clf()
    percent = (np.arange(0,len(diff),1)*100)/len(diff)
    percentile = np.zeros(len(diff)) + 90
    tolerance = np.zeros(len(diff)) + 0.2
    sortedDiff = np.sort(diff)
    index = np.where(sortedDiff <= 0.2)
    print(len(sortedDiff))
    print(index[0])
    print(index[0][-1])
    print(percent[index[0][-1]])
    fig, ax = plt.subplots()
    plt.plot(sortedDiff, percent, color="green")
    plt.plot(sortedDiff, percentile, color='blue')
    plt.plot(tolerance, percent, color='red')
    ax.minorticks_on()
    ax.grid(which='major', color='#CCCCCC', linewidth=0.8)
    ax.grid(which='minor', color='#DDDDDD', linestyle='--', linewidth=0.6)
    plt.title("Percentage of values vs loss")
    plt.savefig("Percentage_vs_loss_{}.png".format(name))


def comparison(models, xTest, yTest):
    # Percentage vs difference plot comparsion
    plt.clf()
    fig, ax = plt.subplots()
    ax.minorticks_on()
    ax.grid(which='major', color='#CCCCCC', linewidth=0.8)
    ax.grid(which='minor', color='#DDDDDD', linestyle='--', linewidth=0.6)
    yPredicted = np.zeros((len(models), len(yTest)))
    for i in range(len(models)):
        modelLoaded = loadModel(models[i])
        yPredicted[i] = modelLoaded.predict(xTest).flatten()

        diff = abs(yPredicted[i].flatten() - yTest.flatten())
        sortedDiff = np.sort(diff)
        percent = (np.arange(0,len(diff),1)*100)/len(diff)

        percentile = np.zeros(len(diff)) + 90
        tolerance = np.zeros(len(diff)) + 0.1

        plt.plot(sortedDiff, percent)

    plt.title("Percentage of values vs loss")
    # plt.plot(sortedDiff, percentile, color='blue')
    # plt.plot(tolerance, percent, color='red')
    name = "Bin_model_comparison_{t}".format(t=clock)
    plt.savefig("Percentage_vs_loss_{}.png".format(name))


def loadModel(name):
    loadedModel = keras.models.load_model(name)
    loadedModel.summary()
    return loadedModel


def loadWeights(name):
    loadedModel = loadModel(name)
    weights = loadedModel.load_weights(name)
    loadedModel.summary()
    return weights


def testLoadedModel(model, xTest, yTest):
    modelLoaded = loadModel(model)
    history = pd.read_csv('training.log', sep=',', engine='python')

    yPredicted = modelLoaded.predict(xTest)

    # plot of epochs against training and validation loss
    print()
    loss = hist.history['loss']
    val_loss = hist.history['val_loss']
    epochs = range(1, len(loss) + 1)

    plt.clf()
    plt.plot(epochs, loss, 'bo', label='Training Loss')
    plt.plot(epochs, val_loss, 'b', color='red', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig("Train_valid_loss_{}.png".format(name))

    # histogram of loss on test sample
    print()
    plt.clf()
    diff = abs(yPredicted.flatten() - yTest.flatten())
    print(max(diff), min(diff))
    print(np.std(diff), np.mean(diff))
    plt.hist(diff, bins=200)
    plt.title('Loss of predicted vs test Histogram')
    plt.savefig("Hist_loss_{}.png".format(name))

    # plotting % of predictions vs loss
    print()
    plt.clf()
    percent = (np.arange(0,len(diff),1)*100)/len(diff)
    percentile = np.zeros(len(diff)) + 90
    tolerance = np.zeros(len(diff)) + 0.1
    sortedDiff = np.sort(diff)
    fig, ax = plt.subplots()
    plt.plot(sortedDiff, percent, color="green")
    plt.plot(sortedDiff, percentile, color='blue')
    plt.plot(tolerance, percent, color='red')
    ax.minorticks_on()
    ax.grid(which='major', color='#CCCCCC', linewidth=0.8)
    ax.grid(which='minor', color='#DDDDDD', linestyle='--', linewidth=0.6)
    plt.title("Percentage of values vs loss")
    plt.savefig("Percentage_vs_loss_{}.png".format(name))


# ----------------------------------------------------- main --------------------------------------------------------------------------------

# loading numpy arrays of data
rawD = np.load('TTbarRaw3.npz')
binD = np.load('TTbarBin3.npz')
zRaw, ptRaw, pvRaw = rawD['z'], rawD['pt'], rawD['pv']
ptBin = binD['ptB']
# trackLength, maxTrack = rawD['tl'], rawD['maxValue']

clock = int(time.time())

# plt.hist(trackLength, bins=100, color='red')
# plt.plot()
# plt.savefig("TTbarTrackDistribution.png")

print()
xTrain, yTrain, xValid, yValid, xTest, yTest = binModelSplit(ptBin, pvRaw.flatten())
model, history, name = binModel(xTrain, yTrain, xValid, yValid, xTest, yTest)
testing(model, history, xValid, yValid, xTest, yTest, name)

# print()
# xTrain, yTrain, xValid, yValid, xTest, yTest = rawModelSplit(zRaw, ptRaw, pvRaw.flatten())
# model, history, name = rawModel(xTrain, yTrain, xValid, yValid, xTest, yTest)
# testing(model, history, xValid, yValid, xTest, yTest, name)



# Loaded model test and comparison to other models

# print()
# xTrain, yTrain, xValid, yValid, xTest, yTest = binModelSplit(ptBin, pvRaw.flatten())
# xTrain, yTrain, xValid, yValid, xTest, yTest = rawModelSplit(zRaw, ptRaw, pvRaw.flatten())

# model = "Bin_model_1720443577.h5"
# testLoadedModel(model, xTest, yTest)

# models = np.array(['Bin_model_dropout_adam_huber_loss_1720518813',\
#         'Bin_model_<keras.optimizers.optimizer_v2.adam.Adam object at 0x7f9600370b80>_huber_1720449668',\
#         'Bin_model_<keras.optimizers.optimizer_v2.adam.Adam object at 0x7f707c7e8b50>_huber_1720452269',\
#         'Bin_model_adam_<keras.losses.Huber object at 0x7faaf059d2e0>_1720514485',\
#         'Bin_model_adam_<keras.losses.Huber object at 0x7f4c5413e2e0>_1720514906',\
#         'Bin_model_adam_huber_1720464056',\
#         'Bin_model_adam_huber_1720455533'])
# comparison(models, xTest, yTest)
