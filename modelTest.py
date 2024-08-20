import numpy as np 
import pandas as pd
import time
import os
from tqdm import tqdm
print()
import tensorflow as tf 
from tensorflow import keras
print()
import seaborn as sn
import matplotlib.pyplot as plt 
from sklearn.preprocessing import StandardScaler
from scipy.stats import gaussian_kde
print()
from model_types import convModel as cnn, rnn, wavenet, multiLayerPerceptron as mlp
from customFunction import welsch, learningRate, power_decay, piecewise_constant_fn


class haltCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if (logs.get('val_loss') < 0.00001):
            print('\n\nValuation loss reach 0.00001 so training stopped.\n\n')
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
    lossFunc = keras.losses.MeanAbsoluteError()
    model, typeM = cnn(form, op, lossFunc)
    model.summary()
    
    # saving the model and best weights
    weights = "{d}_Bin_model_{n}inputs_{type}_{o}_{l}_{t}.weights.h5".format(n=num, type=typeM, o=op.name, l=lossFunc.name, d=nameData, t=CLOCK)
    modelDirectory = "models"
    modelName = weights[:-11]
    print(modelName)
    start =[i for i, letter in enumerate(modelName) if letter == '_']

    # callbacks
    checkpointCallback = keras.callbacks.ModelCheckpoint(filepath=weights, monitor="val_loss", save_weights_only=True, save_best_only=True, verbose=1)
    lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=30, cooldown = 1, min_lr=0.000001, verbose=1)
    csvLogger = keras.callbacks.CSVLogger(f"{nameData}_training_{modelName[start[0]+1:]}.log", separator=',', append=False)
    stopTraining = haltCallback()

    history = model.fit(xTrain, yTrain, epochs=EPOCHS, batch_size=BATCH_SIZE,\
                        validation_data=(xValid, yValid),\
                        callbacks=[lr, checkpointCallback, csvLogger, stopTraining])

    checkpointFilename = os.path.join(modelDirectory, weights)
    check = os.path.isdir(modelDirectory)
    if not check:
        os.makedirs(modelDirectory)
        print("Created directory:" , modelDirectory)

    # saves full model
    modelFilename = os.path.join(modelDirectory, modelName)
    model.save(modelName+".keras")

    return model, history, modelName, lossFunc


def findPVGivenProb(z, modelName, xT, yT):

    model = loadModel(modelName)
    indexTest = np.argwhere(yT[0] != MASK_NO).flatten()

    testPredProb = model.predict(xT).flatten()
    print('test predict done.\n')

    indexPred = np.argmax(testPredProb.reshape(xT.shape[0]//z.shape[1], z.shape[1]), axis=1) # change to take highest prob in each event as the bin with the pv in it
    oneDIndex = (z.shape[1]) * np.arange(indexPred.shape[0]) + indexPred # converts index to the equivalent index position for a 1D equivalent array

    xTestFocus = xT[oneDIndex]
    yTestFocus = yT[0][indexTest]
    indexNan = np.argwhere(yTestFocus == MASK_NO)

    print(100*(len(yTestFocus) - len(indexNan))/len(yTestFocus))
    print()    
    print(np.count_nonzero(yT[1]==MASK_NO))
    print(np.round(100 * np.count_nonzero(yT[1]==MASK_NO)/yT[1].shape[0], 5))

    if len(indexTest) < len(indexPred):
        length = len(indexTest)
    else:
        length = len(indexPred)
    count = 0
    for i in tqdm(range(length)):
        if oneDIndex[i] == indexTest[i]:
            count += 1
    print('\nPercentage of correct predicted bin: ', round(count*100/len(indexTest), 5))

    return xTestFocus, yTestFocus


def pvToProbRNN(form , op, lossFunc, maskNo):

    # modelLoad = loadModel('TTbar_Raw_model_3inputs_rnn_adam_mean_absolute_error_overlap_bins_size1_1723539163.keras') # not overlap
    # modelLoad = loadModel('TTbar_Raw_model_3inputs_rnn_adam_mean_absolute_error_overlap_bins_size2_pv_1723539044.keras') # not overlap just bin size of 2
    # modelLoad = loadModel('TTbar_Raw_model_3inputs_rnn_adam_mean_absolute_error_overlap_bins_size2_pv_1723538801.keras')
    # modelLoad = loadModel('TTbar_Raw_model_3inputs_rnn_adam_mean_absolute_error_overlap_bins_size05_1723540545.keras') # not overlap
    # modelLoad = loadModel('TTbar_Raw_model_3inputs_rnn_adam_mean_absolute_error_bins_size2_1723650181.keras')
    # modelLoad = loadModel('TTbar_Raw_model_3inputs_rnn_adam_mean_absolute_error_bins_size2_1723650091.keras')
    modelLoad = loadModel('TTbar_Raw_model_3inputs_rnn_adam_mean_absolute_error_bins_size2_pv_1723712884.keras')

    inp = keras.Input(shape=form)
    mask = keras.layers.Masking(mask_value=maskNo, trainable=False)(inp)
    rnn1 = keras.layers.SimpleRNN(20, return_sequences=True, activation='tanh', trainable=False)(mask)
    rnn2 = keras.layers.SimpleRNN(20, return_sequences=True, activation='tanh', trainable=False)(rnn1)
    rnn3 = keras.layers.SimpleRNN(20, return_sequences=True, activation='tanh')(rnn2)
    rnn4 = keras.layers.SimpleRNN(20, activation='tanh')(rnn3)
    outClass = keras.layers.Dense(1, activation='sigmoid')(rnn4)

    model = keras.Model(inputs=inp, outputs=[outClass])
    model.layers[1].set_weights(modelLoad.layers[0].get_weights())
    model.layers[2].set_weights(modelLoad.layers[1].get_weights())
    model.layers[3].set_weights(modelLoad.layers[2].get_weights())

    model.compile(optimizer=op, loss=lossFunc)

    return model, 'pv_to_prob_rnn'


def reshapeRawBin(z, pt, eta,):
    zData = z.reshape(z.shape[0]*z.shape[1], z.shape[2])
    ptData = pt.reshape(z.shape[0]*z.shape[1], z.shape[2])
    etaData = eta.reshape(z.shape[0]*z.shape[1], z.shape[2])

    return zData, ptData, etaData


def rawModelSplit(z, pt, eta, pv, pvPr=None, prob=None):
    if len(z.shape) > 2:
        z, pt, eta = reshapeRawBin(z, pt, eta)
        print(z.shape, pt.shape, eta.shape, pv.shape)
        # scaling z
        columnZ = z.reshape(z.shape[0]*z.shape[1], 1)
        scaler = StandardScaler().fit(columnZ)
        columnZ = scaler.transform(columnZ)
        z = columnZ.reshape(pt.shape[0], pt.shape[1])
        zVal, ptVal, etaVal = z, pt, eta
        pvVal = pv

    if prob is None:
        indexNan = np.argwhere(np.isnan(pv))
        print(indexNan.shape)
        z = np.delete(z, indexNan, 0)
        pt = np.delete(pt, indexNan, 0)
        eta = np.delete(eta, indexNan, 0)
        pv = np.delete(pv, indexNan, 0)
        print(z.shape, pt.shape, eta.shape, pv.shape)

    if pvPr is not None:
        pvReshaped = np.zeros((z.shape[0], z.shape[1]))
        pvReshaped[pvReshaped==0] = np.nan
        for i in range(z.shape[0]):
            numNans = np.count_nonzero(~np.isnan(z[i]))
            pvReshaped[i, :numNans] = pvPr[i]
        pvReshaped = np.nan_to_num(pvReshaped, nan=MASK_NO)

    z = np.nan_to_num(z, nan=MASK_NO)
    pt = np.nan_to_num(pt, nan=MASK_NO)
    eta = np.nan_to_num(eta, nan=MASK_NO)
    pv = np.nan_to_num(pv, nan=MASK_NO)

    # columnZ = zVal.reshape(zVal.shape[0]*zVal.shape[1], 1)
    # scaler = StandardScaler().fit(columnZ)
    # columnZ = scaler.transform(columnZ)
    # zVal = columnZ.reshape(ptVal.shape[0], ptVal.shape[1])

    # zVal = np.nan_to_num(zVal, nan=MASK_NO)
    # ptVal = np.nan_to_num(ptVal, nan=MASK_NO)
    # etaVal = np.nan_to_num(etaVal, nan=MASK_NO)
    # pvVal = np.nan_to_num(pvVal, nan=MASK_NO)
    # pvVal = pvVal.flatten()

    print(z.shape, pt.shape, eta.shape, pv.shape)

    if pvPr is not None:
        rawDataAll = np.stack((z,pt,eta, pvReshaped), axis=1)
    else:
        rawDataAll = np.stack((z,pt,eta), axis=1)
    print(rawDataAll.shape)

    # splitting data into test, validation and training data
    t = len(pv)//10
    v = len(pv)//5

    # indexMask = np.argwhere(pvVal != MASK_NO)
    # zMask = zVal[indexMask]

    # padded data split
    rawDataAll = rawDataAll.swapaxes(1,2)
    xTest, xValid, xTrain = rawDataAll[:t], rawDataAll[t:v], rawDataAll[v:]
    # jagged data split
    # xTest, xValid, xTrain = allJag[:t], allJag[t:v], allJag[v:]

    # desired values
    if prob is None:
        yTest, yValid, yTrain = pv[:t], pv[t:v], pv[v:]
    else:
        print('probability')
        yTest, yValid, yTrain = prob[:t], prob[t:v], prob[v:]

    # choosing random bin 10 % of the time
    # rawBinAll_I = np.stack((zVal, ptVal, etaVal), axis=1)
    # rawBinAll_I = rawBinAll_I.swapaxes(1,2)
    # randomBin = np.random.choice(np.arange(zVal.shape[0]), yTest.shape[0]//10, replace=False)
    # print(randomBin.shape)
    # for i in range(0, randomBin.shape[0]):
    #     yTest[i] = pvVal[randomBin[i]]
    #     xTest[i] = rawBinAll_I[randomBin[i]]    
    # print(xTest.shape, yTest.shape)
    # print(np.count_nonzero(yTest==MASK_NO))
    # print(np.round(100 * np.count_nonzero(yTest==MASK_NO)/yTest.shape[0]),5)

    # yTestReg, yValidReg, yTrainReg = pv[:t], pv[t:v], pv[v:] # regression test data
    # yTestClass, yValidClass, yTrainClass = prob[:t], prob[t:v], prob[v:] # probability test data
    # yTrain = [yTrainReg, yTrainClass]
    # yValid = [yValidReg, yValidClass]
    # yTest = [yTestReg, yTestClass]
        
    return xTrain, yTrain, xValid, yValid, xTest, yTest


def rawModel(xTrain, yTrain, xValid, yValid): 
    num = xTrain.shape[2]
    form = (xTrain.shape[1], xTrain.shape[2])

    # creating model
    op = keras.optimizers.Adam(learning_rate=0.001)
    # lossFunc = keras.losses.Huber(delta=0.1, name='modified01_huber_loss')
    lossFunc = keras.losses.BinaryCrossentropy() #from_logits=True)
    # lossFunc = keras.losses.MeanAbsoluteError()

    model, typeM = rnn(form, op, lossFunc, MASK_NO)
    # model, typeM = pvToProbRNN(form, op, lossFunc, MASK_NO)
    model.summary()
    
    # saving the model and best weights
    weights = "{d}_Raw_model_{n}inputs_{m}_{o}_{l}_bins_size1_{t}.weights.h5".format(n=num, m=typeM, o='adam', l=lossFunc.name, d=nameData, t=CLOCK)
    modelDirectory = "models"
    modelName = weights[:-11]
    start =[i for i, letter in enumerate(modelName) if letter == '_']
    print(modelName)
    print()
    # callbacks
    checkpointCallback = keras.callbacks.ModelCheckpoint(filepath=weights, monitor="val_loss", save_weights_only=True, save_best_only=True, verbose=1)
    lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=20, cooldown = 1, min_lr=0.000001, verbose=1)
    csvLogger = keras.callbacks.CSVLogger(f"{nameData}_training_{modelName[start[0]+1:]}.log", separator=',', append=False)
    stopTraining = haltCallback()
    earlyStop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=500)

    history = model.fit(xTrain, yTrain, epochs=EPOCHS, batch_size=BATCH_SIZE,\
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


def testing(model, hist, xT, yT, name):
    print()
    print(name)
    yPredicted = model.predict(xT).flatten()
    diff = abs(yPredicted - yT.flatten())
    print()
    print(max(diff), min(diff))
    print(np.std(diff), np.mean(diff))
    start =[i for i, letter in enumerate(name) if letter == '_']

    # plot of epochs against training and validation loss
    # print()
    # loss = hist['loss']
    # val_loss = hist['val_loss']
    # epochs = range(1, len(loss) + 1)

    # plt.clf()
    # plt.plot(epochs, loss, color='blue', label='Training Loss', linewidth=0.7)
    # plt.plot(epochs, val_loss, color='red', label='Validation Loss', linewidth=0.7)
    # plt.grid(which='major', color='#DDDDDD', linewidth=0.8)
    # plt.grid(which='minor', color='#EEEEEE', linestyle=':', linewidth=0.6)
    # minX = np.argmin(val_loss) + 1
    # minY = np.min(val_loss)
    # plt.scatter(minX, minY, color='green', label='minimum '+str(round(minY, 5)), s=6)
    # plt.xlabel('Epoch number')
    # plt.ylabel('Loss')
    # plt.title('Training and Validation Loss')
    # plt.legend()
    # plt.savefig(f"{name[:start[0]]}_Train_valid_loss_{name[start[0]+1:]}.png", dpi=1200)
    # print('min val loss:', min(val_loss))
    # print('At epoch number:',np.argmin(val_loss)+1)
    # print('min loss:', min(loss))
    # print('At epoch number:',np.argmin(loss)+1)

    # # plotting % of predictions vs difference
    # plt.clf()
    # per = 90
    # tol = 0.15
    # sortedDiff = np.sort(diff)
    # shortenedDiff = sortedDiff[sortedDiff<2]
    # percent = (np.arange(0,len(shortenedDiff),1)*100)/len(diff)
    # percentile = np.zeros(len(shortenedDiff)) + per
    # tolerance = np.zeros(len(diff)) + tol
    # tolPercent = (np.arange(0,len(diff),1)*100)/len(diff)
    # tolIndex = np.where(shortenedDiff <= tol)
    # perIndex = np.where(tolPercent <= per)
    # print('Percentage where difference is <=', tol, ":", percent[tolIndex[0][-1]])
    # print('Value of', per, 'th percentile:', sortedDiff[perIndex[0][-1]])

    # fig, ax = plt.subplots()
    # plt.plot(shortenedDiff, percent, color="green", label=name[start[3]+1:start[-1]], linewidth=0.7)
    # plt.plot(shortenedDiff, percentile, color='blue', linestyle=':', label=str(per)+"th percentile")
    # plt.plot(tolerance, tolPercent, color='red', linestyle=':', label=str(tol)+" tolerance")
    # plt.scatter(tol, percent[tolIndex[0][-1]], color='red', label=str(tol)+' tolerance: '+str(round(percent[tolIndex[0][-1]], 3)))
    # if sortedDiff[perIndex[0][-1]] < 2:
    #     plt.scatter(sortedDiff[perIndex[0][-1]], per, color='blue', label=str(per)+' percentile: '+str(round(sortedDiff[perIndex[0][-1]], 3)))
    # ax.minorticks_on()
    # ax.grid(which='major', color='#CCCCCC', linewidth=0.8)
    # ax.grid(which='minor', color='#DDDDDD', linestyle='--', linewidth=0.6)
    # ax.set_xlim(0,2)
    # plt.xlabel('Difference between predicted and true value')
    # plt.ylabel('Percentage')
    # plt.title("Percentage of values vs Difference")
    # plt.legend()
    # plt.savefig(f"{name[:start[0]]}_Percentage_vs_loss_up_to_2_{name[start[0]+1:]}.png", dpi=1200)


    # plt.clf()
    # fig, ax = plt.subplots()
    # shortenedDiff = sortedDiff[sortedDiff<10]
    # percent = (np.arange(0,len(shortenedDiff),1)*100)/len(diff)
    # percentile = np.zeros(len(shortenedDiff)) + per
    # plt.plot(shortenedDiff, percent, color="green", label=name[start[3]+1:start[-1]], linewidth=0.7)
    # plt.plot(shortenedDiff, percentile, color='blue', linestyle=':', label=str(per)+"th percentile")
    # plt.plot(tolerance, tolPercent, color='red', linestyle=':', label=str(tol)+" tolerance")
    # plt.scatter(tol, percent[tolIndex[0][-1]], color='red', label=str(tol)+' tolerance: '+str(round(percent[tolIndex[0][-1]], 3)))
    # if sortedDiff[perIndex[0][-1]] < 2:
    #     plt.scatter(sortedDiff[perIndex[0][-1]], per, color='blue', label=str(per)+' percentile: '+str(round(sortedDiff[perIndex[0][-1]], 3)))
    # ax.minorticks_on()
    # ax.grid(which='major', color='#CCCCCC', linewidth=0.8)
    # ax.grid(which='minor', color='#DDDDDD', linestyle='--', linewidth=0.6)
    # ax.set_xlim(0,10)
    # plt.xlabel('Difference between predicted and true value')
    # plt.ylabel('Percentage')
    # plt.title("Percentage of values vs Difference")
    # plt.legend()
    # plt.savefig(f"{name[:start[0]]}_Percentage_vs_loss_up_to_10_{name[start[0]+1:]}.png", dpi=1200)

    # plt.clf()
    # percent = (np.arange(0,len(sortedDiff),1)*100)/len(diff)
    # percentile = np.zeros(len(sortedDiff)) + per
    # fig, ax = plt.subplots()
    # plt.plot(sortedDiff, percent, color="green", label=name[start[3]+1:start[-1]], linewidth=0.7)
    # plt.plot(sortedDiff, percentile, color='blue', linestyle=':', label=str(per)+"th percentile")
    # plt.plot(tolerance, tolPercent, color='red', linestyle=':', label=str(tol)+" tolerance")
    # plt.scatter(tol, percent[tolIndex[0][-1]], color='red', label=str(tol)+' tolerance: '+str(round(percent[tolIndex[0][-1]], 3)))
    # if sortedDiff[perIndex[0][-1]] < 2:
    #     plt.scatter(sortedDiff[perIndex[0][-1]], per, color='blue', label=str(per)+' percentile: '+str(round(sortedDiff[perIndex[0][-1]], 3)))
    # ax.minorticks_on()
    # ax.grid(which='major', color='#CCCCCC', linewidth=0.8)
    # ax.grid(which='minor', color='#DDDDDD', linestyle='--', linewidth=0.6)
    # plt.xlabel('Difference between predicted and true value')
    # plt.ylabel('Percentage')
    # plt.title("Percentage of values vs Difference")
    # plt.legend()
    # plt.savefig(f"{name[:start[0]]}_Percentage_vs_loss_up_to_maxdiff_{name[start[0]+1:]}.png", dpi=1200)

    print('Integration plot made')

    # plotting histogram of difference
    diff = yPredicted.flatten() - yT.flatten()
    diffDataset = pd.DataFrame(dict(error=diff))

    plt.clf()
    fig, ax2 = plt.subplots()
    ax2.minorticks_on()
    ax2.grid(which='major', color='#CCCCCC', linewidth=0.8)
    ax2.grid(which='minor', color='#DDDDDD', linestyle='--', linewidth=0.6)
    ax2.set_yscale('symlog')
    ax2.set_ylabel('Log count')
    ax2.set_xlim(-20,20)
    sn.histplot(data=diffDataset, x='error', bins=300, kde=True, ax=ax2)
    ax2.set_title('Distribution of errors')
    ax2.set_xlabel('Difference between predicted and true PV [cm]')
    plt.savefig(f"{nameData}_Hist_loss_log_{name[start[0]+1:]}.png", dpi=1200)
    print('Hist plot made')

    plt.clf()
    fig, ax1 = plt.subplots()
    ax1.minorticks_on()
    ax1.grid(which='major', color='#CCCCCC', linewidth=0.8)
    ax1.grid(which='minor', color='#DDDDDD', linestyle='--', linewidth=0.6)
    ax1.set_ylabel('Count')
    ax1.set_xlim(-20,20)
    sn.histplot(data=diffDataset, x='error', bins=300, kde=True, ax=ax1)
    ax1.set_title('Distribution of errors')
    ax1.set_xlabel('Difference between predicted and true PV [cm]')
    plt.savefig(f"{nameData}_Hist_loss_{name[start[0]+1:]}.png", dpi=1200)
    print('Hist plot made')

    shortenedDiff = diff[(diff<2) & (diff>-2)]
    shortDiffDataset = pd.DataFrame(dict(error=shortenedDiff))
    plt.clf()
    fig, ax1 = plt.subplots()
    ax1.minorticks_on()
    ax1.grid(which='major', color='#CCCCCC', linewidth=0.8)
    ax1.grid(which='minor', color='#DDDDDD', linestyle='--', linewidth=0.6)
    ax1.set_ylabel('Count')
    sn.histplot(data=shortDiffDataset, x='error', bins=300, kde=True, ax=ax1)
    ax1.set_title('Distribution of errors')
    ax1.set_xlabel('Difference between predicted and true PV [cm]')
    ax1.set_xlim(-2,2)
    plt.savefig(f"{nameData}_Hist_loss_shortened_{name[start[0]+1:]}.png", dpi=1200)
    print('Hist plot made')

    # # plot of scattered train and validation data
    # print()
    # yPredTrain = model.predict(xTrain).flatten()
    # plt.clf()
    # fig, ax = plt.subplots(1, 2, figsize=(12,6), sharey=True)
    # ax[0].axis('equal')
    # ax[0].scatter(yTrain.flatten(), yPredTrain.flatten(), marker='^', color='r', edgecolor='k')
    # line = np.array([-15, 15])
    # ax[0].plot(line, line, color='black')
    # ax[0].plot(line, line+max(line)*0.2, '--', c='orange')
    # ax[0].plot(line, line-max(line)*0.2, '--', c='orange')
    # ax[0].plot(line, line+max(line)*0.1, '--', c='pink')
    # ax[0].plot(line, line-max(line)*0.1, '--', c='pink')
    # ax[0].set_title('Test Set')
    # ax[0].set_xlabel('True values')
    # ax[0].set_ylabel('Predicted values')
    # ax[0].set_ylim(-20,20)
    # ax[0].set_xlim(-20,20)
    # ax[0].minorticks_on()
    # ax[0].grid(which='both', alpha=0.7, c='#DDDDDD')

    # ax[1].axis('equal')
    # ax[1].scatter(yT.flatten(), yPredicted.flatten(), marker='^', color='r', edgecolor='k')
    # ax[1].plot([-15,15], [-15,15], color='black')
    # ax[1].plot(line, line+max(line)*0.2,'--', c='orange')
    # ax[1].plot(line, line-max(line)*0.2, '--', c='orange')
    # ax[1].plot(line, line+max(line)*0.1, '--', c='pink')
    # ax[1].plot(line, line-max(line)*0.1, '--', c='pink')
    # ax[1].set_title('Validation Set')
    # ax[1].set_xlabel('True values')
    # ax[1].set_ylabel('Predicted values')
    # ax[1].set_ylim(-20,20)
    # ax[1].set_xlim(-20,20)
    # ax[1].minorticks_on()
    # ax[1].grid(which='both', alpha=0.7, c='#DDDDDD')
    # plt.savefig(f'{name[:start[0]]}_True_vs_predicted_scatter_{name[start[0]+1:]}.png', dpi=1000)
    # print('scatter plot made')

    # # plotting learning rate against epochs
    # print()
    # if 'lr' in hist.columns[0]:
    #     lr = hist['lr']
    #     plt.clf()
    #     plt.plot(epochs, lr, color='b', linewidth=0.7)
    #     plt.grid(which='major', color='#DDDDDD', linewidth=0.8)
    #     plt.grid(which='minor', color='#EEEEEE', linestyle=':', linewidth=0.6)
    #     plt.xlabel('Epoch number')
    #     plt.ylabel('Learning Rate')
    #     plt.title('Learning Rate against epochs')
    #     plt.savefig(f"{name[:start[0]]}_Learning_rate_{name[start[0]+1:]}.png")
    #     print('learning rate plot made')
    # else:
    #     print('No learning rate')

    # % values that predicted the correct bin
    # yPredicted = yPredicted.reshape(xT.shape[0]//zRaw.shape[1], zRaw.shape[1])
    # indexPred = np.argmax(yPredicted, axis=1).flatten()
    # indexTest = np.argwhere(yT.flatten() == 1).flatten()
    # indexTest = indexTest%zRaw.shape[1]
    # count = 0
    # print(indexTest.shape)
    # print(indexTest[:5])
    # print(indexPred.shape)
    # print(indexPred[:5])
    # print(np.round(yPredicted[:10]))
    # print(yT[:10])
    # print(yT.shape)
    # print(yPredicted[:10])
    # print(yPredicted.shape)
    # if len(indexTest) < len(indexPred):
    #     length = len(indexTest)
    # else:
    #     length = len(indexPred)
    # for i in tqdm(range(length)):
    #     if indexPred[i] == indexTest[i]:
    #         count += 1
    # print()
    # print('Percentage of correct predicted bin: ', round(count*100/len(indexTest), 5))

    # # confunstion matrix
    # print()
    # plt.clf()
    # plt.figure(figsize=(30,20))
    # plt.rcParams.update({'font.size': 40})
    # yClassPredLabels = np.round(yPredicted)
    # print(yT.shape)
    # print(yClassPredLabels.shape)
    # cm = tf.math.confusion_matrix(labels=yT, predictions=yClassPredLabels)
    # sn.heatmap(cm, annot=True, fmt='d')
    # plt.xlabel('Predicted')
    # plt.ylabel('True')
    # if nameData != name[:start[0]]:
    #     plt.savefig(f"{nameData}_cm_probability_{name}.png", dpi=1000)
    # else:
    #     plt.savefig(f'{nameData}_cm_probability_{name[start[0]+1:]}.png')
    # print('cm plot made')
    

def comparison(models, train, xTest, yT):
    print()
    endStart =[i for i, letter in enumerate(models[0]) if letter == '_']
    name = "{start}_comparison_of_rnn_models_{t}".format(start=models[0][endStart[0]+1:endStart[7]], t=CLOCK)
    print(name)
    time.sleep(5)
    # Percentage vs difference plot comparsion
    plt.clf()
    fig, ax = plt.subplots()
    ax.minorticks_on()
    ax.grid(which='major', color='#CCCCCC', linewidth=0.8)
    ax.grid(which='minor', color='#DDDDDD', linestyle='--', linewidth=0.6)
    colours = ['green', 'red', 'blue', 'purple', 'goldenrod']
    # labels = ['MAE', 'Huber delta=1','MSE', 'Huber delta=0.1']
    labels = ['Simple RNN', 'GRU']
    # labels = ['4112 Parameters', '5377 Parameters', '2727 Parameters']
    # labels = ['Mixed model', 'TTbar model']
    # labels = ['D30 D1', 'D15 D5 D1', 'D15 D10 D5 D1']
    # labels = ['MLP', 'RNN', 'CNN + MLP']
    # labels = ['GRU100 GRU50 D1', 'GRU20 GRU20 D1']
    # labels = ['T150 GRU100 GRU50', 'BiGRU20 GRU20', 'MASK GRU50', 'MASK GRU20 GRU20', 'MASK LSTM20 LSTM20']
    # labels = ['dr(1,2) dr(1,2)', 'dr(1,2)', 'dr(1,3)']
    # labels = ['None', '32', '512', '2048', str(len(xTrain))]
    for i in range(0, len(models)):    
        print()
        # if i == 2:
        #     print('\n\n\n\n')
        #     xTest = xTest.reshape(xTest.shape[0], xTest.shape[1], xTest.shape[2], 1)
        if models[i][-2:] == 'h5':
            if i == 0:
                modelLoaded = loadWeightsSimple(models[i], xTest)
            elif i == 1:
                modelLoaded = loadWeightsGRU(models[i], xTest)
            else:
                break
        else:
            modelLoaded = loadModel(models[i])

        print()
        print(models[i])
        print(xTest.shape)

        hist = pd.read_csv(train[i], sep=',', engine='python')
        val_loss = hist['val_loss']
        loss = hist['loss']

        yPredicted = modelLoaded.predict(xTest).flatten()
        diff = abs(yPredicted - yT.flatten())
        print(max(diff), min(diff))
        print(np.std(diff), np.mean(diff))

        sortedDiff = np.sort(diff[diff<2])
        percent = (np.arange(0,len(sortedDiff),1)*100)/len(diff)
        tolPercent = (np.arange(0,len(diff),1)*100)/len(diff)
        per = 90
        tol = 0.15
        tolIndex = np.where(sortedDiff <= tol)
        perIndex = np.where(tolPercent <= per)
        
        print('Percentage where difference is <=', tol, ":", percent[tolIndex[0][-1]])
        print('Value of', per, 'th percentile:', np.sort(diff)[perIndex[0][-1]])
        print('min val loss:', min(val_loss))
        print('At epoch number:',np.argmin(val_loss)+1)
        print('min loss:', min(loss))
        print('At epoch number:',np.argmin(loss)+1)

        percentile = np.zeros(len(sortedDiff)) + per
        tolerance = np.zeros(len(diff)) + tol
        plt.plot(sortedDiff, percent, label=labels[i], color=colours[i], linewidth=0.8)
        plt.scatter(tol, percent[tolIndex[0][-1]], color='c', label=str(tol)+' tolerance: '+str(round(percent[tolIndex[0][-1]], 3)), s=10)
        if np.sort(diff)[perIndex[0][-1]] < 2:
            plt.scatter(np.sort(diff)[perIndex[0][-1]], per, color='orange', label=str(per)+'th percentile: '+str(round(np.sort(diff)[perIndex[0][-1]], 3)), s=10)
        print()
 
    plt.plot(sortedDiff, percentile, color='orange', linestyle=':', label=str(per)+"th percentile")
    plt.plot(tolerance, tolPercent, color='c', linestyle=':', label=str(tol)+" tolerance")
    
    plt.legend()
    plt.xlabel('Difference between predicted and true value [cm]')
    plt.ylabel('Percentage')
    plt.title("Percentage of values vs Difference")
    plt.savefig(f"{nameData}_Percentage_vs_loss_{name}.png", dpi=1200)
    print('percentage vs difference plot made')

    plt.clf()
    fig, ax = plt.subplots()
    ax.minorticks_on()
    ax.grid(which='major', color='#CCCCCC', linewidth=0.8)
    ax.grid(which='minor', color='#DDDDDD', linestyle='--', linewidth=0.6)
    ax.set_yscale('log')
    for i in range(len(models)):
        # if i == 2:
        #     print('\n\n\n\n')
        #     xTest = xTest.reshape(xTest.shape[0], xTest.shape[1], xTest.shape[2], 1)
        if models[i][-2:] == 'h5':
            if i == 0:
                modelLoaded = loadWeightsSimple(models[i], xTest)
            elif i == 1:
                modelLoaded = loadWeightsGRU(models[i], xTest)
            else:
                break
        else:
            modelLoaded = loadModel(models[i])
        hist = pd.read_csv(train[i], sep=',', engine='python')
        val_loss = hist['val_loss']
        loss = hist['loss']
        yPredicted = modelLoaded.predict(xTest).flatten()
        diff = yPredicted - yT.flatten()
        plot = sn.kdeplot(data=diff, label=labels[i], linewidth =0.8, color=colours[i], ax=ax)
    plt.legend()
    plt.title('Distribution of errors')
    plt.xlabel('Difference between predicted and true PV [cm]')
    plt.savefig(f"{nameData}_Hist_loss_{name}.png", dpi=1200)
    print('KDE plot made')

    plt.clf()
    fig, ax = plt.subplots()
    ax.minorticks_on()
    ax.grid(which='major', color='#CCCCCC', linewidth=0.8)
    ax.grid(which='minor', color='#DDDDDD', linestyle='--', linewidth=0.6)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for i in range(len(models)):
        # if i == 2:
        #     print('\n\n\n\n')
        #     xTest = xTest.reshape(xTest.shape[0], xTest.shape[1], xTest.shape[2], 1)
        if models[i][-2:] == 'h5':
            if i == 0:
                modelLoaded = loadWeightsSimple(models[i], xTest)
            elif i == 1:
                modelLoaded = loadWeightsGRU(models[i], xTest)
            else:
                break
        else:
            modelLoaded = loadModel(models[i])

        print()
        print(models[i])
        print(xTest.shape)

        hist = pd.read_csv(train[i], sep=',', engine='python')
        val_loss = hist['val_loss']
        loss = hist['loss']

        print(np.sort(val_loss)[:5])
        yPredicted = modelLoaded.predict(xTest).flatten()
        diff = yPredicted - yT.flatten()
        diff = diff[(diff>-1) & (diff<1)]
        print(max(diff), min(diff))
        print(np.std(diff), np.mean(diff))
        plot = sn.kdeplot(data=diff, label=labels[i], linewidth =0.8, color=colours[i], ax=ax)
    plt.legend()
    plt.title('Distribution of errors')
    plt.xlabel('Difference between predicted and true PV [cm]')
    plt.savefig(f"{nameData}_Hist_loss_shortened_{name}.png", dpi=1200)
    print('KDE plot made')


def loadModel(name,loss=None):
    loadedModel = tf.keras.models.load_model(name)
    loadedModel.summary()
    return loadedModel


def loadWeights(name, x, lr=0.001):
    form = x.shape[1:]
    print()
    print(form)
    print(name)
    model, typeM = rnn(form, op=keras.optimizers.Adam(learning_rate=lr), lossFunc=keras.losses.BinaryCrossentropy(), maskNo=MASK_NO)
    print()
    model.load_weights(name)
    model.summary()
    return model

def loadWeightsGRU(name, x):
    form = x.shape[1:]
    print()
    print(form)
    print(name)
    model = keras.models.Sequential([
         keras.layers.Masking(mask_value=MASK_NO, input_shape=form),
         keras.layers.GRU(20, return_sequences=True, use_cudnn=False),
         keras.layers.GRU(20, use_cudnn=False),
         keras.layers.Dense(1)
     ])
    model.compile(optimizer=keras.optimizers.Adam(), loss=keras.losses.MeanAbsoluteError())
    print()
    model.load_weights(name)
    model.summary()
    return model

def loadWeightsSimple(name, x):
    form = x.shape[1:]
    print()
    print(form)
    print(name)
    model = keras.models.Sequential([
         keras.layers.Masking(mask_value=MASK_NO, input_shape=form),
         keras.layers.SimpleRNN(20, return_sequences=True),
         keras.layers.SimpleRNN(20),
         keras.layers.Dense(1)
     ])
    model.compile(optimizer=keras.optimizers.Adam(), loss=keras.losses.MeanAbsoluteError())
    print()
    model.load_weights(name)
    model.summary()
    return model


def trainLoadedModel(name, train, xTrain, yTrain, xValid, yValid):
    hist = pd.read_csv(train, sep=',', engine='python')
    epochs = len(hist['loss'])
    bestEpoch = np.argmin(hist['val_loss']) + 1
    lrContinue = hist['lr'][np.argmin(hist['val_loss'])]
    print(bestEpoch)
    print(epochs)

    if name[-2:] == 'h5':
        modelLoaded = loadWeights(name, xTrain, lrContinue)
        weights = name
        model = name[:-11]+'.keras'
    else:
        weights = name[:-6] + '.weights.h5'
        model = name
        modelLoaded = loadModel(model)
        print(weights)
    
    print(model)
    

    time.sleep(3)
    checkpointCallback = keras.callbacks.ModelCheckpoint(filepath=weights, monitor="val_loss", save_weights_only=True, save_best_only=True, verbose=1)
    lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, cooldown = 1, min_lr=0.000001, verbose=1)
    csvLogger = keras.callbacks.CSVLogger(train, separator=',', append=True)
    stopTraining = haltCallback()
    earlyStop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=500)

    epochNo = 1000 - bestEpoch
    print('\n'+str(epochNo)+'\n')

    history = modelLoaded.fit(xTrain, yTrain, epochs=epochNo, batch_size=BATCH_SIZE,\
                        validation_data=(xValid, yValid),\
                        callbacks=[lr, checkpointCallback, csvLogger, earlyStop])
    
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


def binSizeComp(xT, yT, labels):
    plt.clf()
    fig, ax = plt.subplots()
    colours = ['green', 'red', 'blue']
    model = loadModel('TTbar_Raw_model_3inputs_rnn_adam_mean_absolute_error_overlap_bins_size05_1723540545.keras')
    yPredicted = model.predict(xT[0]).flatten()
    diff = abs(yPredicted - yT[0].flatten())
    print(max(diff), min(diff))
    print(np.std(diff), np.mean(diff))

    sortedDiff = np.sort(diff[diff<2])
    percent = (np.arange(0,len(sortedDiff),1)*100)/len(diff)
    tolPercent = (np.arange(0,len(diff),1)*100)/len(diff)
    per = 90
    tol = 0.15
    tolIndex = np.where(sortedDiff <= tol)
    perIndex = np.where(tolPercent <= per)
    
    print('Percentage where difference is <=', tol, ":", percent[tolIndex[0][-1]])
    print('Value of', per, 'th percentile:', np.sort(diff)[perIndex[0][-1]])

    percentile = np.zeros(len(sortedDiff)) + per
    tolerance = np.zeros(len(diff)) + tol
    plt.plot(sortedDiff, percent, label=labels[0], color=colours[0], linewidth=0.8)
    plt.scatter(tol, percent[tolIndex[0][-1]], color='c', label=str(tol)+' tolerance: '+str(round(percent[tolIndex[0][-1]], 3)), s=10)
    if np.sort(diff)[perIndex[0][-1]] < 2:
        plt.scatter(np.sort(diff)[perIndex[0][-1]], per, color='orange', label=str(per)+'th percentile: '+str(round(np.sort(diff)[perIndex[0][-1]], 3)), s=10)
    print()

    model = loadModel('TTbar_Raw_model_3inputs_rnn_adam_mean_absolute_error_overlap_bins_size1_1723539163.keras')
    yPredicted = model.predict(xT[1]).flatten()
    diff = abs(yPredicted - yT[1].flatten())
    print(max(diff), min(diff))
    print(np.std(diff), np.mean(diff))

    sortedDiff = np.sort(diff[diff<2])
    percent = (np.arange(0,len(sortedDiff),1)*100)/len(diff)
    tolPercent = (np.arange(0,len(diff),1)*100)/len(diff)
    per = 90
    tol = 0.15
    tolIndex = np.where(sortedDiff <= tol)
    perIndex = np.where(tolPercent <= per)
    
    print('Percentage where difference is <=', tol, ":", percent[tolIndex[0][-1]])
    print('Value of', per, 'th percentile:', np.sort(diff)[perIndex[0][-1]])

    percentile = np.zeros(len(sortedDiff)) + per
    tolerance = np.zeros(len(diff)) + tol
    plt.plot(sortedDiff, percent, label=labels[1], color=colours[1], linewidth=0.8)
    plt.scatter(tol, percent[tolIndex[0][-1]], color='c', label=str(tol)+' tolerance: '+str(round(percent[tolIndex[0][-1]], 3)), s=10)
    if np.sort(diff)[perIndex[0][-1]] < 2:
        plt.scatter(np.sort(diff)[perIndex[0][-1]], per, color='orange', label=str(per)+'th percentile: '+str(round(np.sort(diff)[perIndex[0][-1]], 3)), s=10)
    print()

    model = loadModel('TTbar_Raw_model_3inputs_rnn_adam_mean_absolute_error_overlap_bins_size2_pv_1723539044.keras')
    yPredicted = model.predict(xT[2]).flatten()
    diff = abs(yPredicted - yT[2].flatten())
    print(max(diff), min(diff))
    print(np.std(diff), np.mean(diff))

    sortedDiff = np.sort(diff[diff<2])
    percent = (np.arange(0,len(sortedDiff),1)*100)/len(diff)
    tolPercent = (np.arange(0,len(diff),1)*100)/len(diff)
    per = 90
    tol = 0.15
    tolIndex = np.where(sortedDiff <= tol)
    perIndex = np.where(tolPercent <= per)
    
    print('Percentage where difference is <=', tol, ":", percent[tolIndex[0][-1]])
    print('Value of', per, 'th percentile:', np.sort(diff)[perIndex[0][-1]])

    percentile = np.zeros(len(sortedDiff)) + per
    tolerance = np.zeros(len(diff)) + tol
    plt.plot(sortedDiff, percent, label=labels[2], color=colours[2], linewidth=0.8)
    plt.scatter(tol, percent[tolIndex[0][-1]], color='c', label=str(tol)+' tolerance: '+str(round(percent[tolIndex[0][-1]], 3)), s=10)
    if np.sort(diff)[perIndex[0][-1]] < 2:
        plt.scatter(np.sort(diff)[perIndex[0][-1]], per, color='orange', label=str(per)+'th percentile: '+str(round(np.sort(diff)[perIndex[0][-1]], 3)), s=10)
    print()
    
    ax.minorticks_on()
    ax.grid(which='major', color='#CCCCCC', linewidth=0.8)
    ax.grid(which='minor', color='#DDDDDD', linestyle='--', linewidth=0.6)
    plt.xlabel('Difference between predicted and true value [cm]')
    plt.ylabel('Percentage')
    plt.title("Percentage of values vs Difference")
    plt.legend()
    plt.plot(sortedDiff, percentile, color='orange', linestyle=':', label=str(per)+"th percentile")
    plt.plot(tolerance, tolPercent, color='c', linestyle=':', label=str(tol)+" tolerance")
    plt.savefig(f'{nameData}_comparison_of_bin_sizes_after_probabability_applied.png', dpi=1200)


def binSizeCompGivenProb(xT, yT, labels):
    plt.clf()

    colours = ['green', 'red', 'blue']
    model = loadModel('')
    xTestFocus, yTestFocus = findPVGivenProb(zRaw, modelName='', xT=xT[0], yT=yT[0])
    yPredicted = model.predict(xTestFocus).flatten()
    diff = abs(yPredicted - yTestFocus.flatten())
    print(max(diff), min(diff))
    print(np.std(diff), np.mean(diff))

    sortedDiff = np.sort(diff[diff<2])
    percent = (np.arange(0,len(sortedDiff),1)*100)/len(diff)
    tolPercent = (np.arange(0,len(diff),1)*100)/len(diff)
    per = 90
    tol = 0.15
    tolIndex = np.where(sortedDiff <= tol)
    perIndex = np.where(tolPercent <= per)
    
    print('Percentage where difference is <=', tol, ":", percent[tolIndex[0][-1]])
    print('Value of', per, 'th percentile:', np.sort(diff)[perIndex[0][-1]])

    percentile = np.zeros(len(sortedDiff)) + per
    tolerance = np.zeros(len(diff)) + tol
    plt.plot(sortedDiff, percent, label=labels[0], color=colours[0], linewidth=0.8)
    plt.scatter(tol, percent[tolIndex[0][-1]], color='c', label=str(tol)+' tolerance: '+str(round(percent[tolIndex[0][-1]], 3)), s=10)
    if np.sort(diff)[perIndex[0][-1]] < 2:
        plt.scatter(np.sort(diff)[perIndex[0][-1]], per, color='orange', label=str(per)+'th percentile: '+str(round(np.sort(diff)[perIndex[0][-1]], 3)), s=10)
    print()


    model = loadModel('')
    xTestFocus, yTestFocus = findPVGivenProb(zRaw, modelName='', xT=xT[1], yT=yT[1])
    yPredicted = model.predict(xTestFocus).flatten()
    diff = abs(yPredicted - yTestFocus.flatten())
    print(max(diff), min(diff))
    print(np.std(diff), np.mean(diff))

    sortedDiff = np.sort(diff[diff<2])
    percent = (np.arange(0,len(sortedDiff),1)*100)/len(diff)
    tolPercent = (np.arange(0,len(diff),1)*100)/len(diff)
    per = 90
    tol = 0.15
    tolIndex = np.where(sortedDiff <= tol)
    perIndex = np.where(tolPercent <= per)
    
    print('Percentage where difference is <=', tol, ":", percent[tolIndex[0][-1]])
    print('Value of', per, 'th percentile:', np.sort(diff)[perIndex[0][-1]])

    percentile = np.zeros(len(sortedDiff)) + per
    tolerance = np.zeros(len(diff)) + tol
    plt.plot(sortedDiff, percent, label=labels[1], color=colours[1], linewidth=0.8)
    plt.scatter(tol, percent[tolIndex[0][-1]], color='c', label=str(tol)+' tolerance: '+str(round(percent[tolIndex[0][-1]], 3)), s=10)
    if np.sort(diff)[perIndex[0][-1]] < 2:
        plt.scatter(np.sort(diff)[perIndex[0][-1]], per, color='orange', label=str(per)+'th percentile: '+str(round(np.sort(diff)[perIndex[0][-1]], 3)), s=10)
    print()


    model = loadModel('')
    xTestFocus, yTestFocus = findPVGivenProb(zRaw, modelName='', xT=xT[2], yT=yT[2])
    yPredicted = model.predict(xTestFocus).flatten()
    diff = abs(yPredicted - yTestFocus.flatten())
    print(max(diff), min(diff))
    print(np.std(diff), np.mean(diff))

    sortedDiff = np.sort(diff[diff<2])
    percent = (np.arange(0,len(sortedDiff),1)*100)/len(diff)
    tolPercent = (np.arange(0,len(diff),1)*100)/len(diff)
    per = 90
    tol = 0.15
    tolIndex = np.where(sortedDiff <= tol)
    perIndex = np.where(tolPercent <= per)
    
    print('Percentage where difference is <=', tol, ":", percent[tolIndex[0][-1]])
    print('Value of', per, 'th percentile:', np.sort(diff)[perIndex[0][-1]])

    percentile = np.zeros(len(sortedDiff)) + per
    tolerance = np.zeros(len(diff)) + tol
    plt.plot(sortedDiff, percent, label=labels[2], color=colours[2], linewidth=0.8)
    plt.scatter(tol, percent[tolIndex[0][-1]], color='c', label=str(tol)+' tolerance: '+str(round(percent[tolIndex[0][-1]], 3)), s=10)
    if np.sort(diff)[perIndex[0][-1]] < 2:
        plt.scatter(np.sort(diff)[perIndex[0][-1]], per, color='orange', label=str(per)+'th percentile: '+str(round(np.sort(diff)[perIndex[0][-1]], 3)), s=10)
    print()

 
    plt.plot(sortedDiff, percentile, color='orange', linestyle=':', label=str(per)+"th percentile")
    plt.plot(tolerance, tolPercent, color='c', linestyle=':', label=str(tol)+" tolerance")
    plt.savefig(f'{nameData}_comparison_of_bin_sizes_after_probabability_applied.png', dpi=1200)


# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------- MAIN -----------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------

MASK_NO = -9999.99
BATCH_SIZE = 30
EPOCHS = 50
CLOCK = int(time.time())

# loading numpy arrays of data
nameData = 'TTbar'
# rawD = np.load('TTbarRaw5.npz')
# binD = np.load('TTbarBin4.npz')
# rawBinD = np.load('TTbar_Raw_0.5_bin_size_overlap_0.npz')
# rawBinD = np.load('TTbar_Raw_0.5_bin_size_overlap_0.25.npz')
# rawBinD = np.load('TTbar_Raw_0.5_bin_size_overlap_0.25_single_pv.npz')
# rawBinD = np.load('TTbar_Raw_1_bin_size.npz')
rawBinD = np.load('TTbar_Raw_1.0_bin_size_overlap_0.npz')
# rawBinD = np.load('TTbar_Raw_1.0_bin_size_overlap_0.5.npz')
# rawBinD = np.load('TTbar_Raw_1.0_bin_size_overlap_0.5_single_pv.npz')
# rawBinD = np.load('TTbar_Raw_2_bin_size.npz')
# rawBinD = np.load('TTbar_Raw_2.0_bin_size_overlap_0.npz')
# rawBinD = np.load('TTbar_Raw_2_bin_size_overlap_1.0.npz')
# rawBinD = np.load('TTbar_Raw_2.0_bin_size_overlap_1.0_single_pv.npz')

# nameData = 'WJets'
# rawD = np.load('WJetsToLNu.npz')
# binD = np.load('WJetsToLNu_Bin.npz')

# nameData = 'QCD'
# rawD = np.load('QCD_Pt-15To3000.npz')
# binD = np.load('QCD_Pt-15To3000_Bin.npz')

# nameData = 'Merged'
# rawD = np.load('Merged_deacys_Raw.npz')
# binD = np.load('Merged_decays_Bin.npz')
# # raw binned 
# rawBinD = np.load('Merged_Raw_1.0_bin_size_overlap_0.npz')

print(nameData)

# zRaw, ptRaw, etaRaw, pvRaw = rawD['z'], rawD['pt'], rawD['eta'], rawD['pv']
# trackLength = rawD['tl']
zRaw, ptRaw, etaRaw, pvRaw, probability = rawBinD['z'], rawBinD['pt'], rawBinD['eta'], rawBinD['pv'], rawBinD['prob']
# pvPred = rawBinD['pv_pred']
# ptBin, trackBin = binD['ptB'], binD['tB']
# print(zRaw.shape, ptRaw.shape, etaRaw.shape, pvRaw.shape)
# print(np.argwhere(probability == 1))
# print(len(np.argwhere(probability == 1)))
# print(pvRaw.shape)
# print(np.argwhere(pvRaw!=pvRaw[np.isnan(pvRaw)]))
# print(len(np.argwhere(pvRaw[~np.isnan(pvRaw)])))

# indexProb = np.argwhere(probability == 1)
# indexNan = np.argwhere(np.isnan(pvRaw))
# pvRaw = np.argwhere(pvRaw, indexNan, 0)


# plt.hist(trackLength, bins=100, color='red')
# plt.plot()
# plt.savefig("TTbarTrackDistribution.png")

# print()
# xTrain, yTrain, xValid, yValid, xTest, yTest = binModelSplit(pt=ptBin, pv=pvRaw.flatten(), track=trackBin)
# xTrain = xTrain.reshape(xTrain.shape[0], xTrain.shape[1], xTrain.shape[2], 1)
# xValid = xValid.reshape(xValid.shape[0], xValid.shape[1], xValid.shape[2], 1)
# xTest = xTest.reshape(xTest.shape[0], xTest.shape[1], xTest.shape[2], 1)
# print(xTest.shape)
# model, history, name, lossFunc = binModel(xTrain, yTrain, xValid, yValid)
# testing(model, history, xTest, yTest, name, lossFunc)

print()
# print(zRaw[:3])
xTrain, yTrain, xValid, yValid, xTest, yTest = rawModelSplit(zRaw, ptRaw, etaRaw, pvRaw.flatten(), pvPr=None, prob=None)
# # print('\n\n\n\n\n')
# # print(yTest[:50])
# # print(xTest[:3])
# # print(xTrain.shape)
# model, history, name = rawModel(xTrain, yTrain, xValid, yValid)
# testing(model, history.history, xTest, yTest, name)


# prediting the pv given probability
# zRaw, ptRaw, etaRaw, pvRaw, probability = rawBinD['z'], rawBinD['pt'], rawBinD['eta'], rawBinD['pv'], rawBinD['prob']
# # pvPred = rawBinD['pv_pred']
# print(zRaw.shape, ptRaw.shape, etaRaw.shape, pvRaw.shape)
# xTrain, yTrain, xValid, yValid, xTest, yTest = rawModelSplit(zRaw, ptRaw, etaRaw, pvRaw.flatten(), pvPr=None, prob=probability)
# # # print(xTest.shape)
# probModel = 'TTbar_Raw_model_3inputs_rnn_adam_binary_crossentropy_bins_size1_1724058679.keras'
# xTestFocus, yTestFocus = findPVGivenProb(zRaw, probModel, xTest, yTest)
# regModel = 'TTbar_Raw_model_3inputs_rnn_adam_mean_absolute_error_bins_size1_1724059129.keras'
# train = 'TTbar_training_Raw_model_3inputs_rnn_adam_mean_absolute_error_bins_size1_1724059129.log'
# testing(model=loadModel(regModel), hist=pd.read_csv(train, sep=',', engine='python'), xT=xTestFocus, yT=yTestFocus, name=regModel[:-6]+'_focus_pred')

# probModel = 'TTbar_Raw_model_3inputs_rnn_adam_binary_crossentropy_bins_size1_fpga_1724061741.weights.h5'
# xTestFocus, yTestFocus = findPVGivenProb(zRaw, probModel, xTest, yTest)
# regModel = 'TTbar_Raw_model_3inputs_rnn_adam_mean_absolute_error_bins_size1_fpga_1724061293.keras'
# train = 'TTbar_training_Raw_model_3inputs_rnn_adam_mean_absolute_error_bins_size1_fpga_1724061293.log'
# testing(model=regModel, hist=pd.read_csv(train, sep=',', engine='python'), xT=xTestFocus, yT=yTestFocus, name=regModel[:-6]+'_focus_pred')


# Loaded model test and comparison to other models

# xTrain, yTrain, xValid, yValid, xTest, yTest = binModelSplit(ptBin, pvRaw.flatten(), track=trackBin)
# xTrain, yTrain, xValid, yValid, xTest, yTest = rawModelSplit(zRaw, ptRaw, etaRaw, pvRaw.flatten(), prob=None)
# xTrain = xTrain.reshape(xTrain.shape[0], xTrain.shape[1], xTrain.shape[2], 1)
# xValid = xValid.reshape(xValid.shape[0], xValid.shape[1], xValid.shape[2], 1)
# xTest = xTest.reshape(xTest.shape[0], xTest.shape[1], xTest.shape[2], 1)
# print(xTrain[0,0])
# print(xTrain.shape)

mod = 'TTbar_Raw_model_3inputs_rnn_adam_mean_absolute_error_bins_size1_fpga_1724061293.keras'
train = 'TTbar_training_Raw_model_3inputs_rnn_adam_mean_absolute_error_bins_size1_fpga_1724061293.log'
# trainLoadedModel(name, train, xTrain, yTrain, xValid, yValid)
testing(model=loadModel(mod), hist=pd.read_csv(train, sep=',', engine='python'), xT=xTest, yT=yTest, name=mod[:-6])


# xTrain = xTrain.reshape(xTrain.shape[0], xTrain.shape[2], xTrain.shape[1])
# xValid = xValid.reshape(xValid.shape[0], xValid.shape[2], xValid.shape[1])
# xTest = xTest.reshape(xTest.shape[0], xTest.shape[2], xTest.shape[1], 1)

# # Comparing various models
# modelsCompare = ['Merged_Bin_model_2inputs_conv_adam_huber_loss_1721923682.keras',\
#                 #  'Merged_Bin_model_2inputs_conv_adam_modified01_huber_loss_1722587835.keras',\
#                  'Merged_Bin_model_2inputs_conv_adam_huber_loss_1722256533.keras',\
#                  'Merged_Bin_model_2inputs_conv_adam_modified015_huber_loss_1722513936.keras']
# trainingCompare = ['Merged_training_Bin_model_2inputs_conv_adam_huber_loss_1721923682.log',\
#                 #    'Merged_training_Bin_model_2inputs_conv_adam_modified01_huber_loss_1722587835.log',\
#                    'Merged_training_Bin_model_2inputs_conv_adam_huber_loss_1722256533.log',\
#                    'training_Merged_Bin_model_2inputs_conv_adam_modified015_huber_loss_1722513936.log']

# endStart =[i for i, letter in enumerate(modelsCompare[0]) if letter == '_']
# print(modelsCompare[0][:endStart[2]])
# mod = loadModel(modelsCompare[0])
# config = mod.get_config()
# print(config["layers"][0]["config"])
# mod = loadModel(modelsCompare[1])
# config = mod.get_config()
# print(config["layers"][0]["config"])
# mod = loadModel(modelsCompare[2])
# config = mod.get_config()
# print(config["layers"][0]["config"])


# mod = loadModel('Bin_model_2inputs_wavenet_adam_huber_loss_1721316446.keras')
# config = mod.get_config()
# print(config["layers"][0]["config"])
# train = 'Merged_training_Bin_model_2inputs_conv_adamax_modified01_huber_loss_1722592046.log'
# hist = pd.read_csv(train, sep=',', engine='python')
# print(hist.columns)

# print(xTest.shape)
# comparison(modelsCompare, trainingCompare, xTest, yTest)


# comparing performance of different bin size
# labels = ['0.5cm bin size', '1cm bin size', '2cm bn size' ]

# rawBinD = np.load('TTbar_Raw_0.5_bin_size_overlap_0.npz')
# zRaw, ptRaw, etaRaw, pvRaw, probability = rawBinD['z'], rawBinD['pt'], rawBinD['eta'], rawBinD['pv'], rawBinD['prob']
# xTrainZero, yTrainZero, xValidZero, yValidZero, xTestZero, yTestZero = rawModelSplit(zRaw, ptRaw, etaRaw, pvRaw.flatten(), prob=None)

# rawBinD = np.load('TTbar_Raw_1_bin_size.npz')
# zRaw, ptRaw, etaRaw, pvRaw, probability = rawBinD['z'], rawBinD['pt'], rawBinD['eta'], rawBinD['pv'], rawBinD['prob']
# xTrainOne, yTrainOne, xValidOne, yValidOne, xTestOne, yTestOne = rawModelSplit(zRaw, ptRaw, etaRaw, pvRaw.flatten(), prob=None)

# rawBinD = np.load('TTbar_Raw_2_bin_size.npz')
# zRaw, ptRaw, etaRaw, pvRaw, probability = rawBinD['z'], rawBinD['pt'], rawBinD['eta'], rawBinD['pv'], rawBinD['prob']
# xTrainTwo, yTrainTwo, xValidTwo, yValidTwo, xTestTwo, yTestTwo = rawModelSplit(zRaw, ptRaw, etaRaw, pvRaw.flatten(), prob=None)

# xTestAll = [xTestZero, xTestOne, xTestTwo]
# yTestAll = [yTestZero, yTestOne, yTestTwo]

# binSizeComp(xT=xTestAll, yT=yTestAll, labels=labels)


# calc pv for each event
# # prevData = np.load('TTbar_Raw_1.0_bin_size_overlap_0.npz')
# prevData = np.load('TTbar_Raw_2.0_bin_size_overlap_0.npz')
# zRaw, ptRaw, etaRaw, pvRaw, probability = prevData['z'], prevData['pt'], prevData['eta'], prevData['pv'], prevData['prob']
# xTrain, yTrain, xValid, yValid, xTest, yTest = rawModelSplit(zRaw, ptRaw, etaRaw, pvRaw.flatten(), prob=probability)
# # model = loadModel('TTbar_Raw_model_3inputs_rnn_adam_mean_absolute_error_bins_size2_1723650181.keras')
# model = loadModel('TTbar_Raw_model_3inputs_rnn_adam_mean_absolute_error_bins_size2_1723650091.keras')
# print(xTest.shape, xValid.shape, xTrain.shape)
# rawBinAll = np.concatenate((xTest, xValid, xTrain), axis=0)
# print(rawBinAll.shape)
# pvPredicted = model.predict(rawBinAll).flatten()
# prevData = dict(prevData)
# prevData['pv_pred'] = pvPredicted
# # np.savez('TTbar_Raw_1.0_bin_size_overlap_0', **prevData)
# np.savez('TTbar_Raw_2.0_bin_size_overlap_0', **prevData)