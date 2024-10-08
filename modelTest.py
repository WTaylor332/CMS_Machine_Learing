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
print()
from model_types import convModel as cnn, rnn
from customFunction import welsch, learningRate, power_decay, piecewise_constant_fn

# callback used to stop training when the loss has got below a certain value in order to test models quicker
class haltCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if (logs.get('val_loss') < 0.01):
            print('\n\nValuation loss reach 0.01 so training stopped.\n\n')
            self.model.stop_training = True

# funciton to get train-test split for model using dataset of summed pt values in each bin of size 0.1cm
def binModelSplit(pt, pv, track=None):
    # scaling 
    columnPT = pt.reshape(pt.shape[0]*pt.shape[1], 1)
    scaler = StandardScaler().fit(columnPT)
    ptScale = scaler.transform(columnPT)
    pt = ptScale.reshape(pt.shape[0], pt.shape[1])

    if track is None:
        binDataAll = pt
    else: # scaling number of tracks in each bin
        columnT = track.reshape(track.shape[0]*track.shape[1], 1)
        scaler = StandardScaler().fit(columnT)
        tScale = scaler.transform(columnT)
        track = tScale.reshape(pt.shape[0], pt.shape[1])
        binDataAll = np.stack((track,pt), axis=1)

    # splitting data into test, validation and training data
    t = len(pt)//10
    v = len(pt)//5
    xTest, xValid, xTrain = binDataAll[:t], binDataAll[t:v], binDataAll[v:]
    yTest, yValid, yTrain = pv[:t], pv[t:v], pv[v:]

    # if conv model architecture is used uncomment this code to reshape data
    # xTrain = xTrain.reshape(xTrain.shape[0], xTrain.shape[1], xTrain.shape[2], 1)
    # xValid = xValid.reshape(xValid.shape[0], xValid.shape[1], xValid.shape[2], 1)
    # xTest = xTest.reshape(xTest.shape[0], xTest.shape[1], xTest.shape[2], 1)

    return xTrain, yTrain, xValid, yValid, xTest, yTest


def binModel(xTrain, yTrain, xValid, yValid):

    if len(xTrain.shape) > 3: # input shape used for conv models 
        form = (xTrain.shape[1], xTrain.shape[2], 1)
    else: # input shape used for all other models
        form = (xTrain.shape[1], xTrain.shape[2])
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
    csvLogger = keras.callbacks.CSVLogger(f"{nameData}_training_{modelName[start[0]+1:]}.log", separator=',', append=False) # logs loss changes with epochs 
    stopTraining = haltCallback()

    history = model.fit(xTrain, yTrain, epochs=EPOCHS, batch_size=BATCH_SIZE,\
                        validation_data=(xValid, yValid),\
                        callbacks=[lr, checkpointCallback, csvLogger, stopTraining])

    checkpointFilename = os.path.join(modelDirectory, weights)
    check = os.path.isdir(modelDirectory)
    if not check:
        os.makedirs(modelDirectory)
        print("Created directory:" , modelDirectory)

    # saves full model as a keras file
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


def pvToProbRNN(form, op, lossFunc, maskNo):

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

# reshapes data into shape (No. events x Bins, No. tracks)
def reshapeRawBin(z, pt, eta,):
    zData = z.reshape(z.shape[0]*z.shape[1], z.shape[2])
    ptData = pt.reshape(z.shape[0]*z.shape[1], z.shape[2])
    etaData = eta.reshape(z.shape[0]*z.shape[1], z.shape[2])

    return zData, ptData, etaData

# funciton to get train-test split for model using dataset of z, pt, and eta values
def rawModelSplit(z, pt, eta, pv, pvPredicted=None, binProbability=None, probToPV=False):
    if len(z.shape) > 2: # reshapes data if dataset is in binned regions
        z, pt, eta = reshapeRawBin(z, pt, eta)

    # scaling z
    columnZ = z.reshape(z.shape[0]*z.shape[1], 1)
    scaler = StandardScaler().fit(columnZ)
    columnZ = scaler.transform(columnZ)
    z = columnZ.reshape(pt.shape[0], pt.shape[1])
        
    if binProbability is None and len(z.shape) > 2: # if training a regression model and only want to train against true pv values this is used to delete bins without a pv
        indexNan = np.argwhere(np.isnan(pv))
        print(indexNan.shape)
        z = np.delete(z, indexNan, 0)
        pt = np.delete(pt, indexNan, 0)
        eta = np.delete(eta, indexNan, 0)
        pv = np.delete(pv, indexNan, 0)
        print(z.shape, pt.shape, eta.shape, pv.shape)

    if pvPredicted is not None: # used if we want to send predicted pv values for every bin into the model for training
        pvReshaped = np.zeros((z.shape[0], z.shape[1]))
        pvReshaped[pvReshaped==0] = np.nan
        for i in range(z.shape[0]):
            numNans = np.count_nonzero(~np.isnan(z[i]))
            pvReshaped[i, :numNans] = pvPredicted[i]
        pvReshaped = np.nan_to_num(pvReshaped, nan=MASK_NO)

    # puts right padded mask number for rnn to ignore
    z = np.nan_to_num(z, nan=MASK_NO) 
    pt = np.nan_to_num(pt, nan=MASK_NO)
    eta = np.nan_to_num(eta, nan=MASK_NO)
    pv = np.nan_to_num(pv, nan=MASK_NO)

    if pvPredicted is not None:
        rawDataAll = np.stack((z,pt,eta, pvReshaped), axis=1)
    else:
        rawDataAll = np.stack((z,pt,eta), axis=1)

    # splitting data into test, validation and training data
    t = len(pv)//10
    v = len(pv)//5

    # padded data split
    rawDataAll = rawDataAll.swapaxes(1,2) # to get in form (No.events, tracks, 3) or (No.events x Bins, tracks, 3) for binned data
    xTest, xValid, xTrain = rawDataAll[:t], rawDataAll[t:v], rawDataAll[v:]
    print(rawDataAll.shape)

    # desired values
    if binProbability is None:
        yTest, yValid, yTrain = pv[:t], pv[t:v], pv[v:]
    elif binProbability is not None and probToPV == False:
        yTest, yValid, yTrain = binProbability[:t], binProbability[t:v], binProbability[v:]
    elif probToPV == True: # dataset used when reconstructing the primary vertex based on the predicted bin
        yTestReg, yValidReg, yTrainReg = pv[:t], pv[t:v], pv[v:] # regression test data
        yTestClass, yValidClass, yTrainClass = binProbability[:t], binProbability[t:v], binProbability[v:] # probability test data
        yTrain = [yTrainReg, yTrainClass]
        yValid = [yValidReg, yValidClass]
        yTest = [yTestReg, yTestClass]
        
    return xTrain, yTrain, xValid, yValid, xTest, yTest


def rawModel(xTrain, yTrain, xValid, yValid): 
    num = xTrain.shape[2]
    form = (xTrain.shape[1], xTrain.shape[2])

    # creating model
    op = keras.optimizers.Adam(learning_rate=0.001)
    lossFunc = keras.losses.BinaryCrossentropy() # use when training a classification model
    # lossFunc = keras.losses.MeanAbsoluteError() # use when training a regression model

    # get model architecture
    model, typeM = rnn(form, op, lossFunc, MASK_NO)
    model.summary()
    
    # saving the model and best weights
    weights = "{d}_Raw_model_{n}inputs_{m}_{o}_{l}_bins_size1_fpga_{t}.weights.h5".format(n=num, m=typeM, o='adam', l=lossFunc.name, d=nameData, t=CLOCK)
    modelDirectory = "models"
    modelName = weights[:-11]
    start =[i for i, letter in enumerate(modelName) if letter == '_']

    # callbacks
    checkpointCallback = keras.callbacks.ModelCheckpoint(filepath=weights, monitor="val_loss",\
                                                         save_weights_only=True, save_best_only=True, verbose=1) # saves the best weights when val loss improves
    lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=20,\
                                            cooldown = 1, min_lr=0.000001, verbose=1) # changes learning rate if val loss hasn't improved in number of epochs specified by patience
    csvLogger = keras.callbacks.CSVLogger(f"{nameData}_training_{modelName[start[0]+1:]}.log", separator=',', append=False)

    history = model.fit(xTrain, yTrain, epochs=EPOCHS, batch_size=BATCH_SIZE,\
                        validation_data=(xValid, yValid),\
                        callbacks=[checkpointCallback, lr, csvLogger]) # runs training

    checkpointFilename = os.path.join(modelDirectory, weights) 
    check = os.path.isdir(modelDirectory)
    if not check:
        os.makedirs(modelDirectory)
        print("Created directory:" , modelDirectory)

    # saves full model
    modelFilename = os.path.join(modelDirectory, modelName)
    model.save(modelName+'.keras')

    return model, history, modelName


def testingRegression(model, hist, xT, yT, name):

    yPredicted = model.predict(xT).flatten() # predicts output using test input data
    diff = abs(yPredicted - yT.flatten())

    print(max(diff), min(diff))
    print(np.std(diff), np.mean(diff))
    start =[i for i, letter in enumerate(name) if letter == '_'] # used for naming files

    path = '/mercury/data3/bgz16927/tmp/CMS_Machine_Learing/Plots/'

    # plot of training and validation loss as the epoch increases
    print()
    loss = hist['loss']
    val_loss = hist['val_loss']
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
    plt.savefig(path+f"{name[:start[0]]}_Train_valid_loss_{name[start[0]+1:]}.png", dpi=1200)
    print('min val loss:', min(val_loss)) # prints min val loss and the epoch at which the min val loss is reached
    print('At epoch number:',np.argmin(val_loss)+1)
    print('min loss:', min(loss))
    print('At epoch number:',np.argmin(loss)+1)

    # plotting % of predictions vs difference
    plt.clf()
    per = 90 # 90th percentile
    tol = 0.15 # cm difference used as a gauge
    sortedDiff = np.sort(diff)
    shortenedDiff = sortedDiff[sortedDiff<2]
    percent = (np.arange(0,len(shortenedDiff),1)*100)/len(diff)
    percentile = np.zeros(len(shortenedDiff)) + per
    tolerance = np.zeros(len(diff)) + tol
    tolPercent = (np.arange(0,len(diff),1)*100)/len(diff)
    tolIndex = np.where(shortenedDiff <= tol)
    perIndex = np.where(tolPercent <= per)
    print('Percentage where difference is <=', tol, ":", percent[tolIndex[0][-1]])
    print('Value of', per, 'th percentile:', sortedDiff[perIndex[0][-1]])
    label = 'Integration of Gaussian distribution'

    # plots % of predictions for errors up to 2
    fig, ax = plt.subplots()
    plt.plot(shortenedDiff, percent, color="green", label=label, linewidth=0.7)
    plt.plot(shortenedDiff, percentile, color='blue', linestyle=':', label=str(per)+"th percentile")
    plt.plot(tolerance, tolPercent, color='red', linestyle=':', label=str(tol)+" tolerance")
    plt.scatter(tol, percent[tolIndex[0][-1]], color='red', label=str(tol)+' tolerance: '+str(round(percent[tolIndex[0][-1]], 3)))
    if sortedDiff[perIndex[0][-1]] < 2:
        plt.scatter(sortedDiff[perIndex[0][-1]], per, color='blue', label=str(per)+' percentile: '+str(round(sortedDiff[perIndex[0][-1]], 3)))
    ax.minorticks_on()
    ax.grid(which='major', color='#CCCCCC', linewidth=0.8)
    ax.grid(which='minor', color='#DDDDDD', linestyle='--', linewidth=0.6)
    ax.set_xlim(0,2)
    plt.xlabel('Difference between predicted and true value')
    plt.ylabel('Percentage')
    plt.title(f"{nameData} Percentage of values vs Difference")
    plt.legend()
    plt.savefig(path+f"Integration of Gaussian plots/{name[:start[0]]}_Percentage_vs_loss_up_to_2_{name[start[0]+1:]}.png", dpi=1200)
    
    # plots % of predictions for errors up to 10
    plt.clf()
    fig, ax = plt.subplots()
    shortenedDiff = sortedDiff[sortedDiff<10]
    percent = (np.arange(0,len(shortenedDiff),1)*100)/len(diff)
    percentile = np.zeros(len(shortenedDiff)) + per
    plt.plot(shortenedDiff, percent, color="green", label=label, linewidth=0.7)
    plt.plot(shortenedDiff, percentile, color='blue', linestyle=':', label=str(per)+"th percentile")
    plt.plot(tolerance, tolPercent, color='red', linestyle=':', label=str(tol)+" tolerance")
    plt.scatter(tol, percent[tolIndex[0][-1]], color='red', label=str(tol)+' tolerance: '+str(round(percent[tolIndex[0][-1]], 3)))
    if sortedDiff[perIndex[0][-1]] < 2:
        plt.scatter(sortedDiff[perIndex[0][-1]], per, color='blue', label=str(per)+' percentile: '+str(round(sortedDiff[perIndex[0][-1]], 3)))
    ax.minorticks_on()
    ax.grid(which='major', color='#CCCCCC', linewidth=0.8)
    ax.grid(which='minor', color='#DDDDDD', linestyle='--', linewidth=0.6)
    ax.set_xlim(0,10)
    plt.xlabel('Difference between predicted and true value')
    plt.ylabel('Percentage')
    plt.title(f"{nameData} Percentage of values vs Difference")
    plt.legend()
    plt.savefig(path+f"Integration of Gaussian plots/{name[:start[0]]}_Percentage_vs_loss_up_to_10_{name[start[0]+1:]}.png", dpi=1200)
    
    # plots % of predictions for errors up to max difference
    plt.clf()
    percent = (np.arange(0,len(sortedDiff),1)*100)/len(diff)
    percentile = np.zeros(len(sortedDiff)) + per
    fig, ax = plt.subplots()
    plt.plot(sortedDiff, percent, color="green", label=label, linewidth=0.7)
    plt.plot(sortedDiff, percentile, color='blue', linestyle=':', label=str(per)+"th percentile")
    plt.plot(tolerance, tolPercent, color='red', linestyle=':', label=str(tol)+" tolerance")
    plt.scatter(tol, percent[tolIndex[0][-1]], color='red', label=str(tol)+' tolerance: '+str(round(percent[tolIndex[0][-1]], 3)))
    if sortedDiff[perIndex[0][-1]] < 2:
        plt.scatter(sortedDiff[perIndex[0][-1]], per, color='blue', label=str(per)+' percentile: '+str(round(sortedDiff[perIndex[0][-1]], 3)))
    ax.minorticks_on()
    ax.grid(which='major', color='#CCCCCC', linewidth=0.8)
    ax.grid(which='minor', color='#DDDDDD', linestyle='--', linewidth=0.6)
    plt.xlabel('Difference between predicted and true value')
    plt.ylabel('Percentage')
    plt.title(f"{nameData} Percentage of values vs Difference")
    plt.legend()
    plt.savefig(path+f"Integration of Gaussian plots/{name[:start[0]]}_Percentage_vs_loss_up_to_maxdiff_{name[start[0]+1:]}.png", dpi=1200)
    print('Integration plot made')

    # plotting histogram of difference
    # plots difference against log of count to see separation at low counts
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
    sn.histplot(data=diffDataset, x='error', bins=300, ax=ax2, kde=True)
    ax2.set_title(f'{nameData} Distribution of errors')
    ax2.set_xlabel('Difference between predicted and true PV [cm]')
    plt.savefig(f"{nameData}_Hist_loss_log_{name[start[0]+1:]}.png", dpi=1200)
    
    # plots ifference against count
    plt.clf()
    fig, ax1 = plt.subplots()
    ax1.minorticks_on()
    ax1.grid(which='major', color='#CCCCCC', linewidth=0.8)
    ax1.grid(which='minor', color='#DDDDDD', linestyle='--', linewidth=0.6)
    ax1.set_ylabel('Count')
    ax1.set_xlim(-20,20)
    sn.histplot(data=diffDataset, x='error', bins=300, kde=True, ax=ax1)
    ax1.set_title('Distribution of errors')
    ax1.set_xlabel(f'{nameData} Difference between predicted and true PV [cm]')
    plt.savefig(path+f"Gaussian distribution plots/{nameData}_Hist_loss_{name[start[0]+1:]}.png", dpi=1200)

    # plots differnce where values are between 2 and -2 against count
    shortenedDiff = diff[(diff<2) & (diff>-2)]
    shortDiffDataset = pd.DataFrame(dict(error=shortenedDiff))
    plt.clf()
    fig, ax1 = plt.subplots()
    ax1.minorticks_on()
    ax1.grid(which='major', color='#CCCCCC', linewidth=0.8)
    ax1.grid(which='minor', color='#DDDDDD', linestyle='--', linewidth=0.6)
    ax1.set_ylabel('Count')
    sn.histplot(data=shortDiffDataset, x='error', bins=300, kde=True, ax=ax1)
    ax1.set_title(f'{nameData} Distribution of errors')
    ax1.set_xlabel('Difference between predicted and true PV [cm]')
    ax1.set_xlim(-2,2)
    plt.savefig(path+f"Gaussian distribution plots/{nameData}_Hist_loss_shortened_{name[start[0]+1:]}.png", dpi=1200)
    print('Hist plot made')

    # plot of scattered train and validation data
    plt.clf()
    fig, ax = plt.subplots()
    line = np.array([-20, 20])
    ax.axis('equal')
    ax.scatter(yT.flatten(), yPredicted.flatten(), marker='^', color='r', edgecolor='k')
    ax.plot(line, line, color='black')
    ax.plot(line, line+max(line)*0.2,'--', c='orange')
    ax.plot(line, line-max(line)*0.2, '--', c='orange')
    ax.plot(line, line+max(line)*0.1, '--', c='pink')
    ax.plot(line, line-max(line)*0.1, '--', c='pink')
    ax.set_title('Validation Set')
    ax.set_xlabel('True values')
    ax.set_ylabel('Predicted values')
    ax.set_ylim(-20,20)
    ax.set_xlim(-20,20)
    ax.minorticks_on()
    ax.grid(which='both', alpha=0.7, c='#DDDDDD')
    plt.savefig(path+f'Scatter plot true vs predicted PV/{name[:start[0]]}_True_vs_predicted_scatter_{name[start[0]+1:]}.png', dpi=1000)
    print('Scatter plot made')

# function to see performances of models ability to predict the correct bin that contains the pv
def testingProbability(model, hist, xT, yT, name):
    print()
    print(name)
    yPredicted = model.predict(xT).flatten()
    start =[i for i, letter in enumerate(name) if letter == '_']

    print()
    loss = hist['loss']
    val_loss = hist['val_loss']
    print('min val loss:', min(val_loss))
    print('At epoch number:',np.argmin(val_loss)+1)
    print('min loss:', min(loss))
    print('At epoch number:',np.argmin(loss)+1)

    # % values that predicted the correct bin
    yPredicted = yPredicted.reshape(xT.shape[0]//zRaw.shape[1], zRaw.shape[1]) # reshapes data into (No. events, Bins)
    indexPred = np.argmax(yPredicted, axis=1).flatten()[:-1]
    indexTest = np.argwhere(yT.flatten() == 1).flatten()
    print(len(indexTest))
    indexTest = indexTest%zRaw.shape[1]
    count = 0
    if len(indexTest) < len(indexPred):
        length = len(indexTest)
    else:
        length = len(indexPred)
    for i in tqdm(range(length)):
        if indexPred[i] == indexTest[i]:
            count += 1
    print('Percentage of correct predicted bin: ', round(count*100/len(indexTest), 5))
    print(len(indexTest), len(indexPred))

    # plots confusion matrix 
    path = '/mercury/data3/bgz16927/tmp/CMS_Machine_Learing/Plots/CM_Plots/'
    plt.clf()
    plt.figure(figsize=(30,20))
    plt.rcParams.update({'font.size': 40})
    yClassPredLabels = np.zeros(xT.shape[0])
    yClassPredLabels[(zRaw.shape[1]) * np.arange(indexPred.shape[0]) + indexPred] = 1
    cm = tf.math.confusion_matrix(labels=yT, predictions=yClassPredLabels)
    cmNormalized = cm / tf.reduce_sum(cm, axis=0, keepdims=True)
    countCorrectBin = ['{0:0.0f}'.format(value) for value in tf.reshape(cm, [-1])]
    percentageCorrectBin = ['{0:.2%}'.format(value) for value in tf.reshape(cmNormalized, [-1])]
    print(countCorrectBin[:5])
    print(percentageCorrectBin[:5])
    labels = [f'{v1}\n{v2}' for v1, v2 in zip(countCorrectBin, percentageCorrectBin)]
    labels = np.asarray(labels).reshape(2,2)
    sn.heatmap(cmNormalized, annot=labels, fmt='', xticklabels=['0', '1'], yticklabels=['0', '1'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    if nameData != name[:start[0]]:
        plt.savefig(path+f"{nameData}_cm_probability_{name}.png", dpi=1000)
    else:
        plt.savefig(path+f'{nameData}_cm_probability_{name[start[0]+1:]}.png')
    print('cm plot made')

# function used to compare the performance of various models
def comparison(models, train, xTest, yT):
    print()
    endStart =[i for i, letter in enumerate(models[0]) if letter == '_']
    name = "{start}_comparison_of_rnn_models_{t}".format(start=models[0][endStart[0]+1:endStart[7]], t=CLOCK)
    plt.clf()
    fig, ax = plt.subplots()
    ax.minorticks_on()
    ax.grid(which='major', color='#CCCCCC', linewidth=0.8)
    ax.grid(which='minor', color='#DDDDDD', linestyle='--', linewidth=0.6)
    colours = ['green', 'red', 'blue', 'purple', 'goldenrod']
    # labels = ['MAE', 'Huber delta=1','MSE', 'Huber delta=0.1']
    labels = ['Simple RNN', 'GRU']
    # labels = ['Mixed model', 'TTbar model']
    # labels = ['MLP', 'RNN', 'CNN + MLP']
    # labels = ['None', '32', '512', '2048', str(len(xTrain))]
    for i in range(0, len(models)): # loops through all models and plots there performance using the integration of the gaussian distribution plot

        modelLoaded = loadModel(models[i])

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

    # plots gaussian distribution of models to compare performance
    plt.clf()
    fig, ax = plt.subplots()
    ax.minorticks_on()
    ax.grid(which='major', color='#CCCCCC', linewidth=0.8)
    ax.grid(which='minor', color='#DDDDDD', linestyle='--', linewidth=0.6)
    ax.set_yscale('log')
    for i in range(len(models)):

        modelLoaded = loadModel(models[i])
        
        hist = pd.read_csv(train[i], sep=',', engine='python')
        val_loss = hist['val_loss']
        loss = hist['loss']
        yPredicted = modelLoaded.predict(xTest).flatten()
        diff = yPredicted - yT.flatten()
        sn.kdeplot(data=diff, label=labels[i], linewidth =0.8, color=colours[i], ax=ax)
    plt.legend()
    plt.title('Distribution of errors')
    plt.xlabel('Difference between predicted and true PV [cm]')
    plt.savefig(f"{nameData}_Hist_loss_{name}.png", dpi=1200)
    print('KDE plot made')

    # plots gaussian distribution plot only looking at errors between -1 and 1
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
        modelLoaded = loadModel(models[i])

        hist = pd.read_csv(train[i], sep=',', engine='python')
        val_loss = hist['val_loss']
        loss = hist['loss']

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

# function to load a keras model
def loadModel(name):
    loadedModel = tf.keras.models.load_model(name)
    loadedModel.summary()
    return loadedModel

# function to put weights onto a model - need to know model architecture first
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

# fucntion to continue training a model that has been trained a bit before
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


# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------- MAIN -----------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------

MASK_NO = -9999.99
BATCH_SIZE = 8192
EPOCHS = 500
CLOCK = int(time.time())

# loading numpy arrays of data
nameData = 'TTbar'
# rawD = np.load('TTbarRaw5.npz')
# binD = np.load('TTbarBin4.npz')
# rawBinD = np.load('TTbar_Raw_0.5_bin_size_overlap_0.npz')
# rawBinD = np.load('TTbar_Raw_0.5_bin_size_overlap_0.25.npz')
# rawBinD = np.load('TTbar_Raw_0.5_bin_size_overlap_0.25_single_pv.npz')
# rawBinD = np.load('TTbar_Raw_1_bin_size.npz')
# rawBinD = np.load('TTbar_Raw_1.0_bin_size_overlap_0.npz')
# rawBinD = np.load('TTbar_Raw_1.0_bin_size_overlap_0.5.npz')
# rawBinD = np.load('TTbar_Raw_1.0_bin_size_overlap_0.5_single_pv.npz')
# rawBinD = np.load('TTbar_Raw_2_bin_size.npz')
rawBinD = np.load('TTbar_Raw_2.0_bin_size_overlap_0.npz')
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

# used to binned data where we looked at summed pt values and summed number of tracks in each bin 
# xTrain, yTrain, xValid, yValid, xTest, yTest = binModelSplit(pt=ptBin, pv=pvRaw.flatten(), track=trackBin)
# model, history, name, lossFunc = binModel(xTrain, yTrain, xValid, yValid)
# testingRegression(model, history, xTest, yTest, name, lossFunc)

print()
xTrain, yTrain, xValid, yValid, xTest, yTest = rawModelSplit(zRaw, ptRaw, etaRaw, pvRaw.flatten(), pvPredicted=None, binProbability=probability)
# model, history, name = rawModel(xTrain, yTrain, xValid, yValid)
# testingRegression(model, history.history, xTest, yTest, name) # choose to test regression or classification model
# testingProbability(model, history.history, xTest, yTest, name)

# # prediting the pv given probability
# zRaw, ptRaw, etaRaw, pvRaw, probability = rawBinD['z'], rawBinD['pt'], rawBinD['eta'], rawBinD['pv'], rawBinD['prob']
# pvPred = rawBinD['pv_pred']
# print(zRaw.shape, ptRaw.shape, etaRaw.shape, pvRaw.shape)
# xTrain, yTrain, xValid, yValid, xTest, yTest = rawModelSplit(zRaw, ptRaw, etaRaw, pvRaw.flatten(), pvPr=None, binProbability=probability, probToPV=True)
# probModel = 'TTbar_Raw_model_3inputs_rnn_adam_binary_crossentropy_bins_size1_fpga_1724061741.keras'
# xTestFocus, yTestFocus = findPVGivenProb(zRaw, probModel, xTest, yTest)
# regModel = 'TTbar_Raw_model_3inputs_rnn_adam_mean_absolute_error_bins_size1_fpga_1724061293.keras'
# train = 'TTbar_training_Raw_model_3inputs_rnn_adam_mean_absolute_error_bins_size1_fpga_1724061293.log'
# testingRegression(model=loadModel(regModel), hist=pd.read_csv(train, sep=',', engine='python'), xT=xTestFocus, yT=yTestFocus, name=regModel[:-6]+'_focus_pred')


# Loaded model for more training and testing
# xTrain, yTrain, xValid, yValid, xTest, yTest = binModelSplit(ptBin, pvRaw.flatten(), track=trackBin)
# xTrain, yTrain, xValid, yValid, xTest, yTest = rawModelSplit(zRaw, ptRaw, etaRaw, pvRaw.flatten(), pvPredicted=None, binProbability=None)

# mod = 'TTbar_Raw_model_3inputs_rnn_adam_mean_absolute_error_bins_size1_fpga_1724061293.keras'
# train = 'TTbar_training_Raw_model_3inputs_rnn_adam_mean_absolute_error_bins_size1_fpga_1724061293.log'
# trainLoadedModel(name, train, xTrain, yTrain, xValid, yValid)
# testingRegression(model=loadModel(mod), hist=pd.read_csv(train, sep=',', engine='python'), xT=xTest, yT=yTest, name=mod[:-6])

mod = 'TTbar_Raw_model_3inputs_rnn_adam_binary_crossentropy_bins_size2_fpga_1724164275.keras'
train = 'TTbar_training_Raw_model_3inputs_rnn_adam_binary_crossentropy_bins_size2_fpga_1724164275.log'
# trainLoadedModel(name, train, xTrain, yTrain, xValid, yValid)
testingProbability(model=loadModel(mod), hist=pd.read_csv(train, sep=',', engine='python'), xT=xTest, yT=yTest, name=mod[:-6])

# # Comparing various models
# modelsCompare = ['Merged_Bin_model_2inputs_conv_adam_huber_loss_1721923682.keras',\
#                 #  'Merged_Bin_model_2inputs_conv_adam_modified01_huber_loss_1722587835.keras',\
#                  'Merged_Bin_model_2inputs_conv_adam_huber_loss_1722256533.keras',\
#                  'Merged_Bin_model_2inputs_conv_adam_modified015_huber_loss_1722513936.keras']
# trainingCompare = ['Merged_training_Bin_model_2inputs_conv_adam_huber_loss_1721923682.log',\
#                 #    'Merged_training_Bin_model_2inputs_conv_adam_modified01_huber_loss_1722587835.log',\
#                    'Merged_training_Bin_model_2inputs_conv_adam_huber_loss_1722256533.log',\
#                    'training_Merged_Bin_model_2inputs_conv_adam_modified015_huber_loss_1722513936.log']
# comparison(modelsCompare, trainingCompare, xTest, yTest)


# calc pv for each bin in each event - used to feed into model to see if it would boost probability acccuracy
# prevData = np.load('TTbar_Raw_1.0_bin_size_overlap_0.npz')
# zRaw, ptRaw, etaRaw, pvRaw, probability = prevData['z'], prevData['pt'], prevData['eta'], prevData['pv'], prevData['prob']
# xTrain, yTrain, xValid, yValid, xTest, yTest = rawModelSplit(zRaw, ptRaw, etaRaw, pvRaw.flatten(), pvPredicted=None, binProbability=probability)
# model = loadModel('TTbar_Raw_model_3inputs_rnn_adam_mean_absolute_error_bins_size1_fpga_1724061293.keras')
# print(xTest.shape, xValid.shape, xTrain.shape)
# rawBinAll = np.concatenate((xTest, xValid, xTrain), axis=0)
# print(rawBinAll.shape)
# pvPredicted = model.predict(rawBinAll).flatten()
# prevData = dict(prevData)
# prevData['pv_pred'] = pvPredicted
# np.savez('TTbar_Raw_1.0_bin_size_overlap_0', **prevData)
