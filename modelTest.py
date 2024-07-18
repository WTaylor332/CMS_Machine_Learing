import numpy as np 
import pandas as pd
import time
import os
from tqdm import tqdm
print()
import tensorflow as tf 
from tensorflow import keras
print()
import matplotlib.pyplot as plt 
from sklearn.preprocessing import StandardScaler
from model_types import convModel as cnn, pureCNN as pcnn, rnn

class haltCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if (logs.get('val_loss') < 0.05):
            print('\n\nValuation loss reach 0.1 so training stopped.\n\n')
            self.model.stop_training = True


def binModelSplit(pt, pv, track=np.array([])):
    # scaling 
    columnPT = pt.reshape(pt.shape[0]*pt.shape[1], 1)
    scaler = StandardScaler().fit(columnPT)
    ptScale = scaler.transform(columnPT)
    pt = ptScale.reshape(pt.shape[0], pt.shape[1])

    if len(track) != 0:
        columnT = track.reshape(track.shape[0]*track.shape[1], 1)
        scaler = StandardScaler().fit(columnT)
        tScale = scaler.transform(columnT)
        track = tScale.reshape(pt.shape[0], pt.shape[1])
        binDataAll = np.stack((track,pt), axis=1)
    else:
        binDataAll = pt

    # splitting data into test, validation and training data
    t = len(pt)//10
    v = (len(pt)//10) * 3
    xTest, xValid, xTrain = binDataAll[:t], binDataAll[t:v], binDataAll[v:]
    yTest, yValid, yTrain = pv[:t], pv[t:v], pv[v:]

    return xTrain, yTrain, xValid, yValid, xTest, yTest


def binModel(xTrain, yTrain, xValid, yValid):

    if len(xTrain.shape) > 2:
        form = (xTrain.shape[1], xTrain.shape[2], 1)
        num = 2
    else:
        form = (xTrain.shape[1], 1)
        num = 1

    op = keras.optimizers.Adam()
    lossFunc = keras.losses.Huber()
    model = pcnn(form, op, lossFunc)
    model.summary()
    
    # saving the model and best weights
    weights = "Bin_model_{n}inputs_pconv_weights_{o}_{l}_{t}.weights.h5".format(n=num, o='adam', l=lossFunc.name, t=clock)
    modelDirectory = "models"
    modelName = "Bin_model_{n}inputs_pconv_{o}_{l}_{t}".format(n=num, o='adam', l=lossFunc.name, t=clock)
    
    # callbacks
    checkpointCallback = keras.callbacks.ModelCheckpoint(filepath=weights, monitor="val_loss", save_weights_only=True, save_best_only=True, verbose=1)
    lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, cooldown = 1, min_lr=0.000001, verbose=1)
    csvLogger = keras.callbacks.CSVLogger("training_{}.log".format(modelName), separator=',', append=False)
    stopTraining = haltCallback()
    earlyStop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=100)

    epochNo = 500
    print(modelName)
    history = model.fit(xTrain, yTrain, epochs=epochNo,\
                        validation_data=(xValid, yValid),\
                        callbacks=[lr, checkpointCallback, csvLogger, stopTraining, earlyStop])

    checkpointFilename = os.path.join(modelDirectory, weights)
    check = os.path.isdir(modelDirectory)
    if not check:
        os.makedirs(modelDirectory)
        print("Created directory:" , modelDirectory)

    # saves full model
    modelFilename = os.path.join(modelDirectory, modelName)
    model.save(modelName+".keras")

    return model, history, modelName


def rawModelSplit(z, pt, eta, pv):
    # scaling z
    columnZ = z.reshape(z.shape[0]*z.shape[1], 1)
    scaler = StandardScaler().fit(columnZ)
    columnZ = scaler.transform(columnZ)
    z = columnZ.reshape(pt.shape[0], pt.shape[1])

    print(z.shape, pt.shape, eta.shape)

    # z = z[:,:150]
    # pt = pt[:,:150]
    # eta = eta[:,:150]
    z = np.nan_to_num(z, nan=-9999)
    pt = np.nan_to_num(pt, nan=-9999)
    eta = np.nan_to_num(eta, nan=-9999)

    binDataAll = np.stack((z,pt,eta), axis=1)
    print(binDataAll.shape)

    # splitting data into test, validation and training data
    t = len(binDataAll)//10
    v = len(binDataAll)//5
    xTest, xValid, xTrain = binDataAll[:t], binDataAll[t:v], binDataAll[v:]
    yTest, yValid, yTrain = pv[:t], pv[t:v], pv[v:]

    return xTrain, yTrain, xValid, yValid, xTest, yTest


def rawModel(xTrain, yTrain, xValid, yValid):
    if xTrain.shape[1] > 2:
        num = 3
    else:
        num = 2
    form = (xTrain.shape[1], xTrain.shape[2])
    # creating model
    op = keras.optimizers.Adam(learning_rate=0.01)
    lossFunc = keras.losses.Huber()

    model = rnn(form, op, lossFunc)
    model.summary()
    
    # saving the model and best weights
    weights = "Raw_model_{n}inputs_rnn_weights_{o}_{l}_{t}.weights.h5".format(n=num, o='adam', l=lossFunc.name, t=clock)
    modelDirectory = "models"
    modelName = "Raw_model_{n}inputs_rnn_{o}_{l}_{t}".format(n=num, o='adam', l=lossFunc.name, t=clock)

    checkpointCallback = keras.callbacks.ModelCheckpoint(filepath=weights, monitor="val_loss", save_weights_only=True, save_best_only=True, verbose=1)
    lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, cooldown = 1, min_lr=0.000001, verbose=1)
    csvLogger = keras.callbacks.CSVLogger("training_{}.log".format(modelName), separator=',', append=False)
    stopTraining = haltCallback()
    earlyStop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=100)

    epochNum = 500
    print(modelName)
    history = model.fit(xTrain, yTrain, epochs=epochNum,\
                        validation_data=(xValid, yValid),\
                        callbacks=[checkpointCallback, lr, csvLogger, stopTraining, earlyStop])

    checkpointFilename = os.path.join(modelDirectory, weights)
    check = os.path.isdir(modelDirectory)
    if not check:
        os.makedirs(modelDirectory)
        print("Created directory:" , modelDirectory)

    # saves full model
    modelFilename = os.path.join(modelDirectory, modelName)
    model.save(modelName+'.keras')

    return model, history, modelName


def testing(model, hist, xValid, yValid, xTest, yTest, name):
    print()
    model.evaluate(xValid, yValid)

    yPredicted = model.predict(xTest)
    diff = abs(yPredicted.flatten() - yTest.flatten())
    print(max(diff), min(diff))
    print(np.std(diff), np.mean(diff))

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
    plt.hist(diff, bins=200)
    plt.title('Loss of predicted vs test Histogram')
    plt.savefig("Hist_loss_{}.png".format(name))

    # plotting % of predictions vs difference
    print()
    plt.clf()
    per = 90
    tol = 0.15
    shortenedDiff = diff[diff<2]
    percent = (np.arange(0,len(shortenedDiff),1)*100)/len(shortenedDiff)
    percentile = np.zeros(len(shortenedDiff)) + per
    tolerance = np.zeros(len(shortenedDiff)) + tol
    sortedDiff = np.sort(shortenedDiff)
    tolIndex = np.where(sortedDiff <= tol)
    perIndex = np.where(percent <= per)
    print('Percentage where difference is <=', tol, ":", percent[tolIndex[0][-1]])
    print('Value of', per, 'th percentil:', sortedDiff[perIndex[0][-1]])
    fig, ax = plt.subplots()
    plt.plot(sortedDiff, percent, color="green", label=name)
    plt.plot(sortedDiff, percentile, color='blue', label=str(per)+"th percentile")
    plt.plot(tolerance, percent, color='red', label=str(tol)+" tolerance")
    ax.minorticks_on()
    ax.grid(which='major', color='#CCCCCC', linewidth=0.8)
    ax.grid(which='minor', color='#DDDDDD', linestyle='--', linewidth=0.6)
    plt.title("Percentage of values vs loss")
    plt.legend()
    plt.savefig("Percentage_vs_loss_{}.png".format(name), dpi=1200)


def comparison(models, train, xTest, yTest):
    print()
    # Percentage vs difference plot comparsion
    plt.clf()
    fig, ax = plt.subplots()
    ax.minorticks_on()
    ax.grid(which='major', color='#CCCCCC', linewidth=0.8)
    ax.grid(which='minor', color='#DDDDDD', linestyle='--', linewidth=0.6)
    # yPredicted = np.zeros((len(models), len(yTest)))
    labels = np.array(['A2', 'A2', 'A3', 'A1'])
    for i in range(0, len(models)):
        print(i)
        if i == 1:
            print(xTest.shape)
            xTest = xTest[:,:,0,:]
            print(xTest.shape)
        
        if models[i][-2:] == 'h5':
            modelLoaded = loadWeights(models[i], xTest)
        else:
            modelLoaded = loadModel(models[i], xTest)
        
        hist = pd.read_csv(train[i], sep=',', engine='python')
        val_loss = hist['val_loss']
        print(np.sort(val_loss)[:5])
        yPredicted = modelLoaded.predict(xTest).flatten()
        per = 90
        tol = 0.15
        diff = abs(yPredicted - yTest.flatten())
        print(max(diff), min(diff), np.mean(diff))
        print(np.std(diff), np.mean(diff))
        sortedDiff = np.sort(diff[diff<2])
        percent = (np.arange(0,len(sortedDiff),1)*100)/len(sortedDiff)
        tolIndex = np.where(sortedDiff <= tol)
        perIndex = np.where(percent <= per)
        print('Percentage where difference is <=', tol, ":", percent[tolIndex[0][-1]])
        print('Value of', per, 'th percentil:', sortedDiff[perIndex[0][-1]])

        percentile = np.zeros(len(sortedDiff)) + 90
        tolerance = np.zeros(len(sortedDiff)) + 0.1

        plt.plot(sortedDiff, percent, label=labels[i])
        print()
    plt.legend()
    plt.title("Percentage of values vs loss")
    # plt.plot(sortedDiff, percentile, color='blue')
    # plt.plot(tolerance, percent, color='red')
    name = "Bin_model_comparison_of_architectures_{t}".format(t=clock)
    plt.savefig("Percentage_vs_loss_{}.png".format(name), dpi=1200)


def loadModel(name):
    mod = name # + '/saved_model'
    loadedModel = tf.keras.models.load_model(mod)
    loadedModel.summary()
    return loadedModel


def loadWeights(name, x):
    if len(x.shape) > 3:
        form = (x.shape[1], 2, 1)
    else:
        form = (x.shape[1], 1)
    print(form)
    model = cnn(form, op=keras.optimizers.Adam(), lossFunc=keras.losses.Huber())
    model.load_weights(name)
    model.summary()
    return model


def testLoadedModel(model, xTest, yTest, name):
    modelLoaded = loadModel(model)
    hist = pd.read_csv('training_{}.log'.format(name), sep=',', engine='python')

    yPredicted = modelLoaded.predict(xTest)

    # plot of epochs against training and validation loss
    print()
    loss = hist['loss']
    val_loss = hist['val_loss']
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
rawD = np.load('TTbarRaw5.npz')
# binD = np.load('TTbarBin4.npz')
zRaw, ptRaw, etaRaw, pvRaw = rawD['z'], rawD['pt'], rawD['eta'], rawD['pv']
# ptBin, trackBin = binD['ptB'], binD['tB']
# trackLength, maxTrack = rawD['tl'], rawD['maxValue']



clock = int(time.time())

# plt.hist(trackLength, bins=100, color='red')
# plt.plot()
# plt.savefig("TTbarTrackDistribution.png")

# print()
# xTrain, yTrain, xValid, yValid, xTest, yTest = binModelSplit(pt=ptBin, pv=pvRaw.flatten(), track=trackBin)
# xTrain = xTrain.reshape(xTrain.shape[0], xTrain.shape[1], xTrain.shape[2], 1)
# xValid = xValid.reshape(xValid.shape[0], xValid.shape[1], xValid.shape[2], 1)
# xTest = xTest.reshape(xTest.shape[0], xTest.shape[1], xTest.shape[2], 1)
# model, history, name = binModel(xTrain, yTrain, xValid, yValid)
# testing(model, history, xValid, yValid, xTest, yTest, name)

# print()
xTrain, yTrain, xValid, yValid, xTest, yTest = rawModelSplit(zRaw, ptRaw, etaRaw, pvRaw.flatten())
model, history, name = rawModel(xTrain, yTrain, xValid, yValid)
testing(model, history, xValid, yValid, xTest, yTest, name)


# Loaded model test and comparison to other models
# xTrain, yTrain, xValid, yValid, xTest, yTest = binModelSplit(ptBin, pvRaw.flatten(), track=trackBin)
# xTrain, yTrain, xValid, yValid, xTest, yTest = rawModelSplit(zRaw, ptRaw, pvRaw.flatten())

# testLoadedModel(model, xTest, yTest)

# models = np.array(['Bin_model_2inputs_conv_weights_adam_huber_loss_1721056475.weights.h5',\
#                    'Bin_model_conv_weights_1720614426.h5',\
#                    'Bin_model_2inputs_conv_adam_huber_loss_1721144270.keras',\
#                    'Bin_model_2inputs_conv_adam_huber_loss_1721145153.keras'])


# training = np.array(['training_Bin_model_2inputs_conv_adam_huber_loss_1721056475.log',\
#                     'training_Bin_model_conv_adam_huber_loss_1720614426.log'
#                     'training_Bin_model_2inputs_conv_adam_huber_loss_1721144270.log',\
#                     'training_Bin_model_2inputs_conv_adam_huber_loss_1721145153.log'])

# xTest = xTest.reshape(xTest.shape[0], xTest.shape[2], xTest.shape[1], 1)
# comparison(models, training, xTest, yTest)

# finding architecture of model from weights file - didn't work
# f = h5py.File(models[1], 'r')
# print(f)
# print(f.attrs.get('keras_version'))
# f.attrs.values()
