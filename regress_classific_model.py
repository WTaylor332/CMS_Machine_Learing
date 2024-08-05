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
from customFunction import welsch, learningRate, power_decay, piecewise_constant_fn, OneCycleLr

def cnn(form, op, lossFunc, bins):
    # conv model with regression and classification combines
    inp = keras.Input(shape=form)
    conv1 = keras.layers.Conv2D(10, kernel_size=(1,8), activation='relu')(inp)
    pool1 = keras.layers.MaxPool2D(pool_size=(1,4))(conv1)

    conv2 = keras.layers.Conv2D(10, kernel_size=(1,8), activation='relu')(pool1)
    pool2 = keras.layers.MaxPool2D(pool_size=(1,4))(conv2)

    conv3 = keras.layers.Conv2D(10, kernel_size=(1,8), activation='relu')(pool2)
    pool3 = keras.layers.MaxPool2D(pool_size=(1,2))(conv3)

    flatten =  keras.layers.Flatten()(pool3)
    hidden1 = keras.layers.Dense(6, activation="relu")(flatten)
    hidden2 = keras.layers.Dense(6, activation="relu")(hidden1)
    hidden3 = keras.layers.Dense(6, activation="relu")(hidden2)
    hidden4 = keras.layers.Dense(6, activation="relu")(hidden3)
    hidden5 = keras.layers.Dense(6, activation="relu")(hidden4)
    hidden6 = keras.layers.Dense(6, activation="relu")(hidden5)
    hidden7 = keras.layers.Dense(6, activation="relu")(hidden6)
    hidden8 = keras.layers.Dense(6, activation="relu")(hidden7)
    hidden9 = keras.layers.Dense(6, activation="relu")(hidden8)
    hidden10 = keras.layers.Dense(6, activation="relu")(hidden9)

    outReg = keras.layers.Dense(1)(hidden10)
    outClass = keras.layers.Dense(bins, activation='softmax')(hidden10)

    model = keras.Model(inputs=inp, outputs=[outReg, outClass])
    model.compile(optimizer=op, loss=lossFunc)
    return model, 'conv'


def rnn(form, op, lossFunc, maskNo, bins):
    inp = keras.Input(shape=form)
    mask = keras.layers.Masking(mask_value=maskNo)(inp)
    gru1 = keras.layers.GRU(20, return_sequences=True, activation='tanh')(mask)
    gru2 = keras.layers.GRU(20, activation='tanh')(gru1)

    outReg = keras.layers.Dense(1)(gru2)
    outClass = keras.layers.Dense(bins, activation='softmax')(gru2)

    model = keras.Model(inputs=inp, outputs=[outReg, outClass])
    model.compile(optimizer=op, loss=lossFunc)
    return model, 'rnn'


def binModelSplit(pt, track, vertBin, prob, pv):
    # scaling 
    columnPT = pt.reshape(pt.shape[0]*pt.shape[1], 1)
    scaler = StandardScaler().fit(columnPT)
    ptScale = scaler.transform(columnPT)
    pt = ptScale.reshape(pt.shape[0], pt.shape[1])

    columnT = track.reshape(track.shape[0]*track.shape[1], 1)
    scaler = StandardScaler().fit(columnT)
    tScale = scaler.transform(columnT)
    track = tScale.reshape(pt.shape[0], pt.shape[1])
    binDataAll = np.stack((track,pt), axis=1)

    # columnV = vertBin.reshape(vertBin.shape[0]*vertBin.shape[1], 1)
    # scaler = StandardScaler().fit(columnV)
    # vScale = scaler.transform(columnV)
    # vertBin = vScale.reshape(prob.shape[0], prob.shape[1])

    # print(vertBin.shape)
    print(prob.shape)


    # output = np.stack((vertBin, prob), axis=1)
    # print()
    # print(output.shape)
    # print(output[0])
    # print()

    # splitting data into test, validation and training data
    t = len(pt)//10
    v = (len(pt)//10) * 3
    xTest, xValid, xTrain = binDataAll[:t], binDataAll[t:v], binDataAll[v:]
    yTestReg, yValidReg, yTrainReg = pv[:t], pv[t:v], pv[v:]
    yTestClass, yValidClass, yTrainClass = prob[:t], prob[t:v], prob[v:]
    yTrain = [yTrainReg, yTrainClass]
    yValid = [yValidReg, yValidClass]
    yTest = [yTestReg, yTestClass]

    print(xTrain.shape)
    print(yTest[0].shape)
    print(yTest[1].shape)

    return xTrain, yTrain, xValid, yValid, xTest, yTest


def binModel(xTrain, yTrain, xValid, yValid):

    form = (xTrain.shape[1], xTrain.shape[2], 1)
    num = 2
    epochNo = 500
    bSize = None

    op = keras.optimizers.Adam()
    lossFunc = [keras.losses.Huber(delta=0.1, name='modified01_huber_loss'), keras.losses.CategoricalCrossentropy()]
    # lossFunc = keras.losses.Huber()
    # lossFunc = keras.losses.MeanAbsoluteError()
    # lossFunc = welsch
    model, typeM = cnn(form, op, lossFunc, bins=yTrain[1].shape[1])
    model.summary()
    
    # saving the model and best weights
    weights = "{d}_Bin_model_{n}inputs_{type}_{o}_{l}_{t}.weights.h5".format(n=num, type=typeM, o=op.name, l=lossFunc[0].name+'_and_'+lossFunc[1].name, d=nameData, t=clock)
    modelDirectory = "models"
    modelName = weights[:-11]
    print(modelName)
    start =[i for i, letter in enumerate(modelName) if letter == '_']

    # callbacks
    checkpointCallback = keras.callbacks.ModelCheckpoint(filepath=weights, monitor="val_dense_10_loss", save_weights_only=True, save_best_only=True, verbose=1)
    lr = keras.callbacks.ReduceLROnPlateau(monitor='val_dense_10_loss', factor=0.5, patience=30, cooldown = 1, min_lr=0.000001, verbose=1)
    # lr = OneCycleLr(max_lr=0.001, steps_per_epoch=len(xTrain), epochs=epochNo)
    # lr = keras.callbacks.LearningRateScheduler(piecewise_constant_fn)
    csvLogger = keras.callbacks.CSVLogger(f"{nameData}_training_{modelName[start[0]+1:]}.log", separator=',', append=False)
    earlyStop = keras.callbacks.EarlyStopping(monitor='val_dense_10_loss', patience=500)

    history = model.fit(xTrain, yTrain, epochs=epochNo, batch_size=bSize,\
                        validation_data=(xValid, yValid),\
                        callbacks=[lr, checkpointCallback, csvLogger, earlyStop])

    checkpointFilename = os.path.join(modelDirectory, weights)
    check = os.path.isdir(modelDirectory)
    if not check:
        os.makedirs(modelDirectory)
        print("Created directory:" , modelDirectory)

    # saves full model
    modelFilename = os.path.join(modelDirectory, modelName)
    model.save(modelName+".keras")

    return model, history, modelName, lossFunc


def rawModelSplit(z, pt, eta, pv, prob):

    # scaling z
    columnZ = z.reshape(z.shape[0]*z.shape[1], 1)
    scaler = StandardScaler().fit(columnZ)
    columnZ = scaler.transform(columnZ)
    z = columnZ.reshape(pt.shape[0], pt.shape[1])

    z = np.nan_to_num(z, nan=MASK_NO)
    pt = np.nan_to_num(pt, nan=MASK_NO)
    eta = np.nan_to_num(eta, nan=MASK_NO)

    rawDataAll = np.stack((z,pt,eta), axis=1)
    print(rawDataAll.shape)

    # splitting data into test, validation and training data
    t = len(pv)//10
    v = len(pv)//5

    # padded data split
    rawDataAll = rawDataAll.swapaxes(1,2)
    xTest, xValid, xTrain = rawDataAll[:t], rawDataAll[t:v], rawDataAll[v:]

    # desired values
    yTestReg, yValidReg, yTrainReg = pv[:t], pv[t:v], pv[v:]
    yTestClass, yValidClass, yTrainClass = prob[:t], prob[t:v], prob[v:]
    yTrain = [yTrainReg, yTrainClass]
    yValid = [yValidReg, yValidClass]
    yTest = [yTestReg, yTestClass]

    return xTrain, yTrain, xValid, yValid, xTest, yTest


def rawModel(xTrain, yTrain, xValid, yValid):

    num = xTrain.shape[2]
    form = (xTrain.shape[1], xTrain.shape[2])

    # creating model
    op = keras.optimizers.Adam()
    lossFunc = [keras.losses.Huber(delta=0.1, name='modified01_huber_loss'), keras.losses.CategoricalCrossentropy()] 
    model, typeM = rnn(form, op, lossFunc, MASK_NO, bins=yTrain[1].shape[1])
    
    # saving the model and best weights
    weights = "{d}_Raw_model_{n}inputs_{m}_{o}_{l}_{t}.weights.h5".format(n=num, m=typeM, o='adam', l=lossFunc.name, d=nameData, t=clock)
    modelDirectory = "models"
    modelName = weights[:-11]
    start =[i for i, letter in enumerate(modelName) if letter == '_']
    print(modelName)
    print()

    # callbacks
    checkpointCallback = keras.callbacks.ModelCheckpoint(filepath=weights, monitor="val_loss", save_weights_only=True, save_best_only=True, verbose=1)
    lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=30, cooldown = 1, min_lr=0.000001, verbose=1)
    csvLogger = keras.callbacks.CSVLogger(f"{nameData}_training_{modelName[start[0]+1:]}.log", separator=',', append=False)
    earlyStop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=500)

    epochNum = 10
    batchNo = 256
    history = model.fit(xTrain, yTrain, epochs=epochNum, batch_size=batchNo,\
                        validation_data=(xValid, yValid),\
                        callbacks=[checkpointCallback, lr, csvLogger, earlyStop])

    checkpointFilename = os.path.join(modelDirectory, weights)
    check = os.path.isdir(modelDirectory)
    if not check:
        os.makedirs(modelDirectory)
        print("Created directory:" , modelDirectory)

    # saves full model
    modelFilename = os.path.join(modelDirectory, modelName)
    model.save(modelName+'.keras')

    return model, history, modelName



def testing(model, hist, xTest, yTest, name):
    print()
    print(name)
    print(hist.params)
    print(hist.history.keys())
    print()
    yRegPred, yClassPred = model.predict(xTest)
    yRegPred = yRegPred.flatten()
    print(len(yRegPred))
    print(len(yClassPred))
    diff = abs(yRegPred - yTest[0].flatten())
    print()
    print(max(diff), min(diff))
    print(np.std(diff), np.mean(diff))
    start =[i for i, letter in enumerate(name) if letter == '_']

    # plot of epochs against training and validation loss
    print()
    loss = hist.history['loss']
    val_loss = hist.history['val_loss']
    epochs = range(1, len(loss) + 1)

    plt.clf()
    plt.plot(epochs, loss, color='blue', label='Training Loss', linewidth=0.7)
    plt.plot(epochs, val_loss, color='red', label='Validation Loss', linewidth=0.7)
    plt.grid(which='major', color='#DDDDDD', linewidth=0.8)
    plt.grid(which='minor', color='#EEEEEE', linestyle=':', linewidth=0.6)
    minX = np.argmin(val_loss) + 1
    minY = np.min(val_loss)
    plt.scatter(minX, minY, color='green', label='minimum '+str(round(minY, 5)), s=6)
    plt.xlabel('Epoch number')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig(f"{name[:start[0]]}_Train_valid_loss_{name[start[0]+1:]}.png", dpi=1200)
    print('min val loss:', min(val_loss))
    print('At epoch number:',np.argmin(val_loss)+1)
    print('min loss:', min(loss))
    print('At epoch number:',np.argmin(loss)+1)

    # plotting % of predictions vs difference
    plt.clf()
    per = 90
    tol = 0.15
    shortenedDiff = diff[diff<2]
    percent = (np.arange(0,len(shortenedDiff),1)*100)/len(diff)
    percentile = np.zeros(len(shortenedDiff)) + per
    tolerance = np.zeros(len(diff)) + tol
    tolPercent = (np.arange(0,len(diff),1)*100)/len(diff)
    sortedDiff = np.sort(shortenedDiff)
    tolIndex = np.where(sortedDiff <= tol)
    perIndex = np.where(tolPercent <= per)
    print('Percentage where difference is <=', tol, ":", percent[tolIndex[0][-1]])
    print('Value of', per, 'th percentile:', np.sort(diff)[perIndex[0][-1]])

    fig, ax = plt.subplots()
    plt.plot(sortedDiff, percent, color="green", label=name[start[3]+1:start[-1]], linewidth=0.7)
    plt.plot(sortedDiff, percentile, color='blue', linestyle=':', label=str(per)+"th percentile")
    plt.plot(tolerance, tolPercent, color='red', linestyle=':', label=str(tol)+" tolerance")
    plt.scatter(tol, percent[tolIndex[0][-1]], color='red', label=str(tol)+' tolerance: '+str(round(percent[tolIndex[0][-1]], 3)))
    if np.sort(diff)[perIndex[0][-1]] < 2:
        plt.scatter(np.sort(diff)[perIndex[0][-1]], per, color='blue', label=str(per)+' percentile: '+str(round(np.sort(diff)[perIndex[0][-1]], 3)))
    ax.minorticks_on()
    ax.grid(which='major', color='#CCCCCC', linewidth=0.8)
    ax.grid(which='minor', color='#DDDDDD', linestyle='--', linewidth=0.6)
    plt.xlabel('Difference between predicted and true value')
    plt.ylabel('Percentage')
    plt.title("Percentage of values vs Difference")
    plt.legend()
    plt.savefig(f"{name[:start[0]]}_Percentage_vs_loss_{name[start[0]+1:]}.png", dpi=1200)

    # plot of scattered train and validation data
    print()
    yRegPredTrain = model.predict(xTrain)[0].flatten()
    plt.clf()
    fig, ax = plt.subplots(1, 2, figsize=(12,6), sharey=True)
    ax[0].axis('equal')
    ax[0].scatter(yTrain[0].flatten(), yRegPredTrain.flatten(), marker='^', color='r', edgecolor='k')
    line = np.array([-15, 15])
    ax[0].plot(line, line, color='black')
    ax[0].plot(line, line+max(line)*0.2, '--', c='orange')
    ax[0].plot(line, line-max(line)*0.2, '--', c='orange')
    ax[0].plot(line, line+max(line)*0.1, '--', c='pink')
    ax[0].plot(line, line-max(line)*0.1, '--', c='pink')
    ax[0].set_title('Test Set')
    ax[0].set_xlabel('True values')
    ax[0].set_ylabel('Predicted values')
    ax[0].set_ylim(-15,15)
    ax[0].minorticks_on()
    ax[0].grid(which='both', alpha=0.7, c='#DDDDDD')

    ax[1].axis('equal')
    ax[1].scatter(yTest[0].flatten(), yRegPred.flatten(), marker='^', color='r', edgecolor='k')
    ax[1].plot([-15,15], [-15,15], color='black')
    ax[1].plot(line, line+max(line)*0.2,'--', c='orange')
    ax[1].plot(line, line-max(line)*0.2, '--', c='orange')
    ax[1].plot(line, line+max(line)*0.1, '--', c='pink')
    ax[1].plot(line, line-max(line)*0.1, '--', c='pink')
    ax[1].set_title('Validation Set')
    ax[1].set_xlabel('True values')
    ax[1].set_ylabel('Predicted values')
    ax[1].set_ylim(-15,15)
    ax[1].minorticks_on()
    ax[1].grid(which='both', alpha=0.7, c='#DDDDDD')
    plt.savefig(f'{name[:start[0]]}_True_vs_predicted_scatter_{name[start[0]+1:]}.png', dpi=1000)
    print('scatter plot made')

    # plot of scattered train and validation data
    print()
    plt.clf()
    fig, ax = plt.subplots(1, 2, figsize=(12,6), sharey=True)
    ax[0].axis('equal')
    extent = np.array([[min(yTrain[0]), max(yTrain[0])], [min(yRegPredTrain), max(yRegPredTrain)]])
    heatmap = ax[0].hist2d(yTrain[0], yRegPredTrain, bins=20, cmap='hot_r', range=extent)
    fig.colorbar(heatmap[3], ax=ax[0])
    line = np.array([-15, 15])
    ax[0].plot(line, line, color='black')
    ax[0].plot(line, line+max(line)*0.2, '--', c='orange')
    ax[0].plot(line, line-max(line)*0.2, '--', c='orange')
    ax[0].plot(line, line+max(line)*0.1, '--', c='pink')
    ax[0].plot(line, line-max(line)*0.1, '--', c='pink')
    ax[0].set_title('Test Set')
    ax[0].set_xlabel('True values')
    ax[0].set_ylabel('Predicted values')
    ax[0].set_ylim(-15,15)
    ax[0].grid(which='both', alpha=0.7, c='#DDDDDD')

    ax[1].axis('equal')
    extent = np.array([[min(yTest[0]), max(yTest[0])], [min(yRegPred), max(yRegPred)]])
    heatmap = ax[1].hist2d(yTest[0], yRegPred, bins=20, cmap='hot_r', range=extent)
    fig.colorbar(heatmap[3], ax=ax[1])
    ax[1].plot([-15,15], [-15,15], color='black')
    ax[1].plot(line, line+max(line)*0.2,'--', c='orange')
    ax[1].plot(line, line-max(line)*0.2, '--', c='orange')
    ax[1].plot(line, line+max(line)*0.1, '--', c='pink')
    ax[1].plot(line, line-max(line)*0.1, '--', c='pink')
    ax[1].set_title('Validation Set')
    ax[1].set_xlabel('True values')
    ax[1].set_ylabel('Predicted values')
    ax[1].set_ylim(-15,15)
    ax[1].grid(which='both', alpha=0.7, c='#DDDDDD')
    plt.savefig(f'{name[:start[0]]}_True_vs_predicted_map_{name[start[0]+1:]}.png')
    print('map plot made')

    # plotting learning rate against epochs
    print()
    lr = hist.history['lr']
    plt.clf()
    plt.plot(epochs, lr, color='b', linewidth=0.7)
    plt.grid(which='major', color='#DDDDDD', linewidth=0.8)
    plt.grid(which='minor', color='#EEEEEE', linestyle=':', linewidth=0.6)
    plt.xlabel('Epoch number')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate against epochs')
    plt.savefig(f"{name[:start[0]]}_Learning_rate_{name[start[0]+1:]}.png")
    print('learning rate plot made')



# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------- MAIN ------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------

MASK_NO = -9999.99

# loading numpy arrays of data

nameData = 'Merged'
rawD = np.load('Merged_deacys_Raw.npz')
binD = np.load('Merged_decays_Bin.npz')
vert = np.load('Hard_Vertex_Probability_30_bins.npz')

print(nameData)

zRaw, ptRaw, etaRaw, pvRaw = rawD['z'], rawD['pt'], rawD['eta'], rawD['pv']
ptBin, trackBin = binD['ptB'], binD['tB']
probability, vertBin = vert['prob'], vert['bins']

clock = int(time.time())

# xTrain, yTrain, xValid, yValid, xTest, yTest = binModelSplit(pt=ptBin, track=trackBin, vertBin=vertBin, prob=probability, pv=pvRaw.flatten())
# model, history, name, lossFunc = binModel(xTrain, yTrain, xValid, yValid)
# testing(model, history, xTest, yTest, name)

xTrain, yTrain, xValid, yValid, xTest, yTest = rawModelSplit(zRaw, ptRaw, etaRaw, pvRaw.flatten(), prob=probability)
model, history, name = rawModel(xTrain, yTrain, xValid, yValid)
testing(model, history, xTest, yTest, name)