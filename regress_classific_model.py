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
from model_types import convModel as cnn, pureCNN as pcnn, rnn, wavenet, multiLayerPerceptron as mlp
from customFunction import welsch, learningRate, power_decay, piecewise_constant_fn, OneCycleLr
from modelTest import testing

def binModelSplit(pt, vertBin, track, prob):
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

    columnV = vertBin.reshape(vertBin.shape[0]*vertBin.shape[1], 1)
    scaler = StandardScaler().fit(columnV)
    vScale = scaler.transform(columnV)
    vertBin = vScale.reshape(prob.shape[0], prob.shape[1])

    print(vertBin.shape)
    print(prob.shape)


    output = np.stack((vertBin, prob), axis=1)
    print()
    print(output.shape)
    print(output[0])
    print()

    # splitting data into test, validation and training data
    t = len(pt)//10
    v = (len(pt)//10) * 3
    xTest, xValid, xTrain = binDataAll[:t], binDataAll[t:v], binDataAll[v:]
    yTest, yValid, yTrain = output[:t], output[t:v], output[v:]

    return xTrain, yTrain, xValid, yValid, xTest, yTest

def binModel(xTrain, yTrain, xValid, yValid):

    form = (xTrain.shape[1], xTrain.shape[2], 1)
    num = 2
    epochNo = 500
    bSize = 256

    # op = keras.optimizers.Adam()
    op = keras.optimizers.Adadelta()
    lossFunc = keras.losses.Huber(delta=0.1, name='modified01_huber_loss')
    # lossFunc = keras.losses.Huber()
    # lossFunc = keras.losses.MeanAbsoluteError()
    # lossFunc = welsch
    model, typeM = cnn(form, op, lossFunc)
    model.summary()
    
    # saving the model and best weights
    weights = "{d}_Bin_model_{n}inputs_{type}_{o}_{l}_{t}.weights.h5".format(n=num, type=typeM, o=op.name, l=lossFunc.name, d=nameData, t=clock)
    modelDirectory = "models"
    modelName = weights[:-11]
    print(modelName)
    start =[i for i, letter in enumerate(modelName) if letter == '_']

    # callbacks
    checkpointCallback = keras.callbacks.ModelCheckpoint(filepath=weights, monitor="val_loss", save_weights_only=True, save_best_only=True, verbose=1)
    lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=30, cooldown = 1, min_lr=0.000001, verbose=1)
    # lr = OneCycleLr(max_lr=0.001, steps_per_epoch=len(xTrain), epochs=epochNo)
    # lr = keras.callbacks.LearningRateScheduler(piecewise_constant_fn)
    csvLogger = keras.callbacks.CSVLogger(f"{nameData}_training_{modelName[start[0]+1:]}.log", separator=',', append=False)
    stopTraining = haltCallback()
    earlyStop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=500)

    history = model.fit(xTrain, yTrain, epochs=epochNo, batch_size=bSize,\
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

    return model, history, modelName, lossFunc


# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------- MAIN ------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# loading numpy arrays of data
# nameData = 'TTbar'
# rawD = np.load('TTbarRaw5.npz')
# binD = np.load('TTbarBin4.npz')

# nameData = 'QCD'
# rawD = np.load('QCD_Pt-15To3000.npz')
# binD = np.load('QCD_Pt-15To3000_Bin.npz')

nameData = 'Merged'
rawD = np.load('Merged_deacys_Raw.npz')
binD = np.load('Merged_decays_Bin.npz')
vert = np.load('Hard_Vertex_Probability_30_bins.npz')

# nameData = 'WJets'
# rawD = np.load('WJetsToLNu.npz')
# binD = np.load('WJetsToLNu_Bin.npz')

print(nameData)

zRaw, ptRaw, etaRaw, pvRaw = rawD['z'], rawD['pt'], rawD['eta'], rawD['pv']
ptBin, trackBin = binD['ptB'], binD['tB']
prob, vertBin = vert['prob'], vert['bins']
print(zRaw.shape, ptRaw.shape, etaRaw.shape, pvRaw.shape)

clock = int(time.time())

xTrain, yTrain, xValid, yValid, xTest, yTest = binModelSplit(pt=ptBin, vertBin=vertBin, track=trackBin, prob=prob)

# model, history, name, lossFunc = binModel(xTrain, yTrain, xValid, yValid)
# testing(model, history, xTest, yTest, name, lossFunc)