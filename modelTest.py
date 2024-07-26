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
from model_types import convModel as cnn, pureCNN as pcnn, rnn, wavenet

class haltCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if (logs.get('val_loss') < 0.01):
            print('\n\nValuation loss reach 0.01 so training stopped.\n\n')
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

    form = (xTrain.shape[1], xTrain.shape[2], 1)
    num = 2
    op = keras.optimizers.Adam()
    lossFunc = keras.losses.Huber()
    model = cnn(form, op, lossFunc)
    model.summary()
    
    # saving the model and best weights
    weights = "Bin_model_{n}inputs_conv_weights_{o}_{l}_{d}_{t}.weights.h5".format(n=num, o='adam', l=lossFunc.name, d=nameData, t=clock)
    modelDirectory = "models"
    modelName = "Bin_model_{n}inputs_conv_{o}_{l}_{d}_{t}".format(n=num, o='adam', l=lossFunc.name, d=nameData, t=clock)
    
    # callbacks
    checkpointCallback = keras.callbacks.ModelCheckpoint(filepath=weights, monitor="val_loss", save_weights_only=True, save_best_only=True, verbose=1)
    lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, cooldown = 1, min_lr=0.000001, verbose=1)
    csvLogger = keras.callbacks.CSVLogger("training_{}.log".format(modelName), separator=',', append=False)
    stopTraining = haltCallback()
    earlyStop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=500)

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

    z = np.nan_to_num(z, nan=0)
    pt = np.nan_to_num(pt, nan=0)
    eta = np.nan_to_num(eta, nan=0)

    z = z[:,:150]
    pt = pt[:,:150]
    eta = eta[:,:150]

    # getting jagged data
    # print(int(sum(trackLength)))
    # allJag = np.zeros((z.shape[0], z.shape[1], 3))
    # print(len(allJag))
    # for i in tqdm(range(z.shape[0])):
    #     track = np.zeros((z.shape[1],3))
    #     for j in range(0, z.shape[1]):
    #         track[j] = [z[i,j], pt[i,j], eta[i,j]]
    #     allJag[i] = track
    #     trackLength[i] = int(trackLength[i])

    # print()
    # allJag = tf.RaggedTensor.from_tensor(allJag, lengths=trackLength)
    # print(allJag.shape)

    rawDataAll = np.stack((z,pt,eta), axis=1)
    print(rawDataAll.shape)

    # splitting data into test, validation and training data
    t = len(pv)//10
    v = len(pv)//5

    # padded data split
    xTest, xValid, xTrain = rawDataAll[:t], rawDataAll[t:v], rawDataAll[v:]
    # jagged data split
    # xTest, xValid, xTrain = allJag[:t], allJag[t:v], allJag[v:]

    # desired values
    yTest, yValid, yTrain = pv[:t], pv[t:v], pv[v:]

    return xTrain, yTrain, xValid, yValid, xTest, yTest


def rawModel(xTrain, yTrain, xValid, yValid): 
    num = 3
    form = (xTrain.shape[1], xTrain.shape[2])
    print(xTrain.shape)
    # creating model
    op = keras.optimizers.Adam(learning_rate=0.01)
    lossFunc = keras.losses.Huber()

    model = rnn(form, op, lossFunc, xTrain.shape[0])
    model.summary()
    
    # saving the model and best weights
    weights = "Raw_model_{n}inputs_rnn_weights_{o}_{l}_{d}_{t}.weights.h5".format(n=num, o='adam', l=lossFunc.name, d=nameData, t=clock)
    modelDirectory = "models"
    modelName = "Raw_model_{n}inputs_rnn_{o}_{l}_{d}_{t}".format(n=num, o='adam', l=lossFunc.name, d=nameData, t=clock)

    checkpointCallback = keras.callbacks.ModelCheckpoint(filepath=weights, monitor="val_loss", save_weights_only=True, save_best_only=True, verbose=1)
    lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, cooldown = 1, min_lr=0.000001, verbose=1)
    csvLogger = keras.callbacks.CSVLogger("training_{}.log".format(modelName), separator=',', append=False)
    stopTraining = haltCallback()
    earlyStop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=500)

    epochNum = 500
    print()
    print(modelName)
    print(xTrain.shape)
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
    print(name)
    model.evaluate(xValid, yValid)
    yPredicted = model.predict(xTest)
    diff = abs(yPredicted.flatten() - yTest.flatten())
    print()
    print(max(diff), min(diff))
    print(np.std(diff), np.mean(diff))

    # plot of epochs against training and validation loss
    print()
    loss = hist.history['loss']
    val_loss = hist.history['val_loss']
    epochs = range(1, len(loss) + 1)

    plt.clf()
    plt.plot(epochs, loss, 'b', color='blue', label='Training Loss')
    plt.plot(epochs, val_loss, 'b', color='red', label='Validation Loss')
    plt.grid(which='major', color='#DDDDDD', linewidth=0.8)
    plt.grid(which='minor', color='#EEEEEE', linestyle=':', linewidth=0.6)
    minX = np.argmin(val_loss) + 1
    minY = np.min(val_loss)
    plt.scatter(minX, minY, color='green', label='minimum')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig("Train_valid_loss_{}.png".format(name), dpi=1200)
    print('min val loss:', min(val_loss))
    print('At epoch number:',np.argmin(val_loss)+1)
    print('min loss:', min(loss))
    print('At epoch number:',np.argmin(loss)+1)


    # histogram of difference on test sample
    print()
    plt.clf()
    plt.hist(diff, bins=300)
    plt.title('Loss of predicted vs test Histogram')
    plt.savefig("Hist_loss_{}.png".format(name), dpi=1200)

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
    perIndex = np.where(percent <= per)
    print('Percentage where difference is <=', tol, ":", percent[tolIndex[0][-1]])
    print('Value of', per, 'th percentil:', sortedDiff[perIndex[0][-1]])
    fig, ax = plt.subplots()
    plt.plot(sortedDiff, percent, color="green", linestyle=':', label=name)
    plt.plot(sortedDiff, percentile, color='blue', linestyle=':', label=str(per)+"th percentile")
    plt.plot(tolerance, tolPercent, color='red', label=str(tol)+" tolerance")
    ax.minorticks_on()
    ax.grid(which='major', color='#CCCCCC', linewidth=0.8)
    ax.grid(which='minor', color='#DDDDDD', linestyle='--', linewidth=0.6)
    plt.title("Percentage of values vs Difference")
    plt.legend()
    plt.savefig("Percentage_vs_loss_{}.png".format(name), dpi=1200)


def comparison(models, train, xTest, yTest):
    print()
    name = "{start}_comparison_of_conv_architecture_{d}_{t}".format(start=models[0][:27], d=nameData, t=clock)
    print(name)
    # Percentage vs difference plot comparsion
    plt.clf()
    fig, ax = plt.subplots()
    ax.minorticks_on()
    ax.grid(which='major', color='#CCCCCC', linewidth=0.8)
    ax.grid(which='minor', color='#DDDDDD', linestyle='--', linewidth=0.6)
    # labels = np.array(['CCPCCPCC ks=8 ps=4', 'CPCPCPC ks=6 ps=4', 'CPCPCPC ks=8 ps=4', 'CPCPCPCPCPC ks=8 ps=2'])
    labels = ['MAE', 'MSE', 'Huber']
    # labels = ['D30 D1', 'D15 D5 D1', 'D15 D10 D5 D1']
    # for i in range(0, len(models)):    
    #     print()
    #     if models[i][-2:] == 'h5':
    #         modelLoaded = loadWeights(models[i], xTest)
    #     else:
    #         modelLoaded = loadModel(models[i])
    #     # if i == 2:
    #     #     print('\n\n\n\n')
    #     #     xTest = xTest.reshape(xTest.shape[0], xTest.shape[2], xTest.shape[1], 1)
    #     print()
    #     print(models[i])

    #     hist = pd.read_csv(train[i], sep=',', engine='python')
    #     val_loss = hist['val_loss']
    #     loss = hist['loss']

    #     print(np.sort(val_loss)[:5])
    #     yPredicted = modelLoaded.predict(xTest).flatten()

    #     diff = abs(yPredicted - yTest.flatten())
    #     print(max(diff), min(diff))
    #     print(np.std(diff), np.mean(diff))

    #     sortedDiff = np.sort(diff[diff<2])
    #     percent = (np.arange(0,len(sortedDiff),1)*100)/len(diff)
    #     tolPercent = (np.arange(0,len(diff),1)*100)/len(diff)
    #     per = 90
    #     tol = 0.15
    #     tolIndex = np.where(sortedDiff <= tol)
    #     perIndex = np.where(percent <= per)
        
    #     print('Percentage where difference is <=', tol, ":", percent[tolIndex[0][-1]])
    #     print('Value of', per, 'th percentil:', sortedDiff[perIndex[0][-1]])
    #     print('min val loss:', min(val_loss))
    #     print('At epoch number:',np.argmin(val_loss)+1)
    #     print('min loss:', min(loss))
    #     print('At epoch number:',np.argmin(loss)+1)

    #     percentile = np.zeros(len(sortedDiff)) + per
    #     tolerance = np.zeros(len(diff)) + tol
    #     plt.plot(sortedDiff, percent, label=labels[i])
    #     print()
     
    # plt.plot(sortedDiff, percentile, color='blue', linestyle=':', label=str(per)+"th percentile")
    # plt.plot(tolerance, tolPercent, color='red', linestyle=':', label=str(tol)+" tolerance")
    # plt.legend()
    # plt.title("Percentage of values vs Difference")
    # plt.savefig("Percentage_vs_loss_{}.png".format(name), dpi=1200)

    plt.clf()
    for i in range(len(models)):
        print(i)
        hist = pd.read_csv(train[i], sep=',', engine='python')
        loss = hist['loss']
        val_loss = hist['val_loss']
        epochs = range(1, len(loss) + 1)
        print(epochs)
        plt.plot(epochs, val_loss, label='Validation Loss '+labels[i])
        plt.grid(which='major', color='#DDDDDD', linewidth=0.8)
        plt.grid(which='minor', color='#EEEEEE', linestyle=':', linewidth=0.6)
        minX = np.argmin(val_loss) + 1
        minY = np.min(val_loss)
        plt.scatter(minX, minY, edgecolors='black', linewidths=1, label='minimum '+str(round(minY, 5)))
    print('training vs val loss plot made')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig("Train_valid_loss_{}.png".format(name), dpi=1200)


def loadModel(name):
    loadedModel = tf.keras.models.load_model(name)
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


def trainLoadedModel(model, train, xTrain, yTrain, xValid, yValid):
    modelLoaded = model
    hist = pd.read_csv(train, sep=',', engine='python')
    epochs = len(hist['loss'])

    print(epochs)

    weights = model[:-6] + '.weights.h5'
    print(weights)
    time.sleep(5)
    checkpointCallback = keras.callbacks.ModelCheckpoint(filepath=weights, monitor="val_loss", save_weights_only=True, save_best_only=True, verbose=1)
    lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, cooldown = 1, min_lr=0.000001, verbose=1)
    csvLogger = keras.callbacks.CSVLogger(train, separator=',', append=False)
    stopTraining = haltCallback()
    earlyStop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=500)

    epochNo = 500 - epochs
    print('\n'+str(epochNo)+'\n')

    history = modelLoaded.fit(xTrain, yTrain, epochs=epochNo,\
                        validation_data=(xValid, yValid),\
                        callbacks=[lr, checkpointCallback, csvLogger, stopTraining, earlyStop])
    
    modelDirectory = "models"
    checkpointFilename = os.path.join(modelDirectory, weights)
    check = os.path.isdir(modelDirectory)
    if not check:
        os.makedirs(modelDirectory)
        print("Created directory:" , modelDirectory)

    # saves full model
    modelFilename = os.path.join(modelDirectory, model)
    print(model)
    modelLoaded.save(model)


def testLoadedModel(model, train, xTest, yTest):
    modelLoaded = model
    hist = pd.read_csv(train, sep=',', engine='python')
    print()
    print(model)

    name = model[:-6]
    # plot of epochs against training and validation loss
    loss = hist['loss']
    val_loss = hist['val_loss']
    epochs = range(1, len(loss) + 1)
    print()
    print(epochs)
    print(len(loss))
    print('min val loss:', min(val_loss))
    print('At epoch number:',np.argmin(val_loss)+1)
    print('min loss:', min(loss))
    print('At epoch number:',np.argmin(loss)+1)

    plt.clf()
    plt.plot(epochs, loss, color='blue', label='Training Loss')
    plt.plot(epochs, val_loss, color='red', label='Validation Loss')
    minX = np.argmin(val_loss) + 1
    minY = np.min(val_loss)
    plt.scatter(minX, minY, color='green', label='minimum')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig("Train_valid_loss_{}.png".format(name),dpi=1200)

    yPredicted = modelLoaded.predict(xTest)

    # histogram of loss on test sample
    print()
    plt.clf()
    diff = abs(yPredicted.flatten() - yTest.flatten())
    print(max(diff), min(diff))
    print(np.std(diff), np.mean(diff))
    plt.hist(diff, bins=200)
    plt.title('Loss of predicted vs test Histogram')
    plt.savefig("Hist_loss_{}.png".format(name), dpi=1200)

    # plotting % of predictions vs loss
    print()
    plt.clf()
    sortedDiff = np.sort(diff[diff<2])
    percent = (np.arange(0,len(sortedDiff),1)*100)/len(diff)
    percentile = np.zeros(len(sortedDiff)) + 90
    tolerance = np.zeros(len(diff)) + 0.1
    tolPercent = (np.arange(0,len(diff),1)*100)/len(diff)
    per = 90
    tol = 0.15
    tolIndex = np.where(sortedDiff <= tol)
    perIndex = np.where(percent <= per)
    print('Percentage where difference is <=', tol, ":", percent[tolIndex[0][-1]])
    print('Value of', per, 'th percentil:', sortedDiff[perIndex[0][-1]])
    fig, ax = plt.subplots()
    plt.plot(sortedDiff, percent, color="green")
    plt.plot(sortedDiff, percentile, color='blue', linestyle=':', label=str(per)+"th percentile")
    plt.plot(tolerance, tolPercent, color='red', linestyle=':', label=str(tol)+" tolerance")
    ax.minorticks_on()
    ax.grid(which='major', color='#CCCCCC', linewidth=0.8)
    ax.grid(which='minor', color='#DDDDDD', linestyle='--', linewidth=0.6)
    plt.title("Percentage of values vs Difference")
    plt.savefig("Percentage_vs_loss_{}.png".format(name), dpi=1200)

# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------- MAIN -----------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# loading numpy arrays of data
# nameData = 'TTbar'
# rawD = np.load('TTbarRaw5.npz')
# binD = np.load('TTbarBin4.npz')

nameData = 'WJets'
rawD = np.load('WJetsToLNu.npz')
binD = np.load('WJetsToLNu_Bin.npz')

# nameData = 'QCD'
# rawD = np.load('QCD_Pt-15To3000.npz')
# binD = np.load('QCD_Pt-15To3000_Bin.npz')

# nameData = 'Merged'
# rawD = np.load('Merged_deacys_Raw.npz')
# binD = np.load('Merged_decays_Bin.npz')

zRaw, ptRaw, etaRaw, pvRaw = rawD['z'], rawD['pt'], rawD['eta'], rawD['pv']
ptBin, trackBin = binD['ptB'], binD['tB']
trackLength = rawD['tl']
print(zRaw.shape, ptRaw.shape, etaRaw.shape, pvRaw.shape)

clock = int(time.time())

# plt.hist(trackLength, bins=100, color='red')
# plt.plot()
# plt.savefig("TTbarTrackDistribution.png")

print()
# xTrain, yTrain, xValid, yValid, xTest, yTest = binModelSplit(pt=ptBin, pv=pvRaw.flatten(), track=trackBin)
# xTrain = xTrain.reshape(xTrain.shape[0], xTrain.shape[1], xTrain.shape[2], 1)
# xValid = xValid.reshape(xValid.shape[0], xValid.shape[1], xValid.shape[2], 1)
# xTest = xTest.reshape(xTest.shape[0], xTest.shape[1], xTest.shape[2], 1)

# model, history, name = binModel(xTrain, yTrain, xValid, yValid)
# testing(model, history, xValid, yValid, xTest, yTest, name)

# print()
# xTrain, yTrain, xValid, yValid, xTest, yTest = rawModelSplit(zRaw, ptRaw, etaRaw, pvRaw.flatten())
# xTrain = xTrain.reshape(xTrain.shape[0], xTrain.shape[2], xTrain.shape[1])
# xValid = xValid.reshape(xValid.shape[0], xValid.shape[2], xValid.shape[1])
# xTest = xTest.reshape(xTest.shape[0], xTest.shape[2], xTest.shape[1])
# model, history, name = rawModel(xTrain, yTrain, xValid, yValid)
# testing(model, history, xValid, yValid, xTest, yTest, name)


# Loaded model test and comparison to other models

# models = np.array(['Raw_model_3inputs_rnn_adam_huber_loss_1721296121.keras',\
#                    'Raw_model_3inputs_rnn_adam_huber_loss_1721396555.keras',\
#                    'Bin_model_2inputs_rnn_adam_huber_loss_WJets_1721832523.keras',\
#                    'Raw_model_3inputs_rnn_adam_huber_loss_1721296121.keras',\
#                    'Raw_model_3inputs_rnn_adam_huber_loss_1721315255.keras',\
#                    'Bin_model_2inputs_rnn_adam_huber_loss_1721311690.keras',\
#                     'Bin_model_2inputs_conv_adam_huber_loss_WJets_1721659080.keras',\
#                     'Bin_model_2inputs_conv_adam_huber_loss_WJets_1721661172.keras',\
#                     'Bin_model_2inputs_wavenet_adam_huber_loss_1721391189.keras',\
#                     # 'Bin_model_2inputs_wavenet_adam_huber_loss_1721316446.keras',\
#                     'Bin_model_2inputs_pconv_adam_huber_loss_1721227042.keras'
#                     #'Bin_model_2inputs_pconv_adam_huber_loss_1721228818.keras'
#                     ])

# training = np.array(['training_Raw_model_3inputs_rnn_adam_huber_loss_1721296121.log',\
#                      'training_Raw_model_3inputs_rnn_adam_huber_loss_1721396555.log',\
#                      'training_Bin_model_2inputs_rnn_adam_huber_loss_WJets_1721832523.log',\
#                      'training_Raw_model_3inputs_rnn_adam_huber_loss_1721296121.log',\
#                      'training_Raw_model_3inputs_rnn_adam_huber_loss_1721315255.log',\
#                      'training_Bin_model_2inputs_rnn_adam_huber_loss_1721311690.log',\
#                      'training_Bin_model_2inputs_conv_adam_huber_loss_WJets_1721659080.log',\
#                      'training_Bin_model_2inputs_conv_adam_huber_loss_WJets_1721661172.log',\
#                     'training_Bin_model_2inputs_wavenet_adam_huber_loss_1721391189.log',\
#                     # 'training_Bin_model_2inputs_wavenet_adam_huber_loss_1721316446.log',\
#                     'training_Bin_model_2inputs_pconv_adam_huber_loss_1721227042.log',\
#                     #'training_Bin_model_2inputs_pconv_adam_huber_loss_1721228818.log'
#                     ])


xTrain, yTrain, xValid, yValid, xTest, yTest = binModelSplit(ptBin, pvRaw.flatten(), track=trackBin)
# xTrain, yTrain, xValid, yValid, xTest, yTest = rawModelSplit(zRaw, ptRaw, etaRaw, pvRaw.flatten())

# xTrain = xTrain.reshape(xTrain.shape[0], xTrain.shape[1], xTrain.shape[2], 1)
# xValid = xValid.reshape(xValid.shape[0], xValid.shape[1], xValid.shape[2], 1)
xTest = xTest.reshape(xTest.shape[0], xTest.shape[1], xTest.shape[2], 1)
# print(xTrain[0,0])
# print(xTrain.shape)

# trainLoadedModel(mod, training[0], xTrain, yTrain, xValid, yValid)
# testLoadedModel(mod, training[0], xTest, yTest)

# trainLoadedModel(models[1], training[1], xTrain, yTrain, xValid, yValid)
# testLoadedModel(models[1], training[1], xTest, yTest)

# xTrain = xTrain.reshape(xTrain.shape[0], xTrain.shape[2], xTrain.shape[1])
# xValid = xValid.reshape(xValid.shape[0], xValid.shape[2], xValid.shape[1])
# xTest = xTest.reshape(xTest.shape[0], xTest.shape[2], xTest.shape[1], 1)

# Comparing various models
# modelsCompare = ['Bin_model_2inputs_pconv_adam_huber_loss_1721227042.keras',\
#                  'Bin_model_2inputs_pconv_adam_huber_loss_1721228818.keras',\
#                  'Bin_model_2inputs_pconv_adam_huber_loss_TTbar_1721750592.keras',\
#                  'Bin_model_2inputs_pconv_adam_huber_loss_TTbar_1721751238.keras']
# trainingCompare = ['training_Bin_model_2inputs_pconv_adam_huber_loss_1721227042.log',\
#                    'training_Bin_model_2inputs_pconv_adam_huber_loss_1721228818.log',\
#                    'training_Bin_model_2inputs_pconv_adam_huber_loss_TTbar_1721750592.log',\
#                    'training_Bin_model_2inputs_pconv_adam_huber_loss_TTbar_1721751238.log']

modelsCompare = ['',\
                 '',\
                 '']
trainingCompare = ['training_Bin_model_2inputs_conv_adam_mean_absolute_error_1721663273.log',\
                   'training_Bin_model_2inputs_conv_adam_mean_squared_error_1721663286.log',\
                   'training_Bin_model_2inputs_conv_adam_huber_loss_1721663295.log']

# print(modelsCompare[0][:27])
# mod = loadModel(modelsCompare[0])
# config = mod.get_config()
# print(config["layers"][0]["config"])
# mod = loadModel(modelsCompare[1])
# config = mod.get_config()
# print(config["layers"][0]["config"])
# mod = loadModel(modelsCompare[2])
# config = mod.get_config()
# print(config["layers"][0]["config"])

# print(xTest.shape)
comparison(modelsCompare, trainingCompare, xTest, yTest)
