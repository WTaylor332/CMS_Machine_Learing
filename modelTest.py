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
from model_types import convModel as cnn, pureCNN as pcnn, rnn, wavenet, multiLayerPerceptron as mlp
from customFunction import welsch, learningRate, power_decay, piecewise_constant_fn, OneCycleLr

# print("Num GPUs availabel: ", len(tf.config.list_physical_devices('GPU')))

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
    epochNo = 500
    bSize = 256

    op = keras.optimizers.Adadm()
    lossFunc = keras.losses.Huber(delta=0.1, name='modified01_huber_loss')
    # lossFunc = keras.losses.Huber()
    # lossFunc = keras.losses.MeanAbsoluteError()
    # lossFunc = welsch
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


def findPVGivenProb(z, modelName, xT, yT):
    model = loadModel(modelName)

    testPredProb = model.predict(xT).flatten()
    print('test predict done')
    print(testPredProb.shape)
    indexPred = np.argmax(testPredProb.reshape(xT.shape[0]//z.shape[1], z.shape[1]), axis=1) # change to take highest prob in each event as the bin with the pv in it
    print()
    print(indexPred.shape)
    print(yT.shape)
    print()
    oneDIndex = z.shape[1] * np.arange(indexPred.shape[0]) + indexPred
    xTestFocus = xT[oneDIndex]
    yTestFocus = yT[oneDIndex]

    print(xTestFocus.shape, yTestFocus.shape)

    return xTestFocus, yTestFocus


def pvToProbRNN(form , op, lossFunc, maskNo):

    # modelLoad = loadModel('TTbar_Raw_model_3inputs_rnn_adam_mean_absolute_error_overlap_bins_size1_1723539163.keras') # not overlap
    # modelLoad = loadModel('TTbar_Raw_model_3inputs_rnn_adam_mean_absolute_error_overlap_bins_size2_pv_1723539044.keras') # not overlap just bin size of 2
    # modelLoad = loadModel('TTbar_Raw_model_3inputs_rnn_adam_mean_absolute_error_overlap_bins_size2_pv_1723538801.keras')
    modelLoad = loadModel('TTbar_Raw_model_3inputs_rnn_adam_mean_absolute_error_overlap_bins_size05_1723540545.keras') # not overlap

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


def rawModelSplit(z, pt, eta, pv, prob=None):
    if len(z.shape) > 2:
        z, pt, eta = reshapeRawBin(z, pt, eta)
        print(z.shape, pt.shape, eta.shape, pv.shape)

        if prob is None:
            indexNan = np.argwhere(np.isnan(pv))
            z = np.delete(z, indexNan, 0)
            pt = np.delete(pt, indexNan, 0)
            eta = np.delete(eta, indexNan, 0)
            pv = np.delete(pv, indexNan, 0)
            pv = pv[~np.isnan(pv)]
            print(z.shape, pt.shape, eta.shape, pv.shape)


    # scaling z
    columnZ = z.reshape(z.shape[0]*z.shape[1], 1)
    scaler = StandardScaler().fit(columnZ)
    columnZ = scaler.transform(columnZ)
    z = columnZ.reshape(pt.shape[0], pt.shape[1])

    z = np.nan_to_num(z, nan=MASK_NO)
    pt = np.nan_to_num(pt, nan=MASK_NO)
    eta = np.nan_to_num(eta, nan=MASK_NO)

    # z = z[:,:150]
    # pt = pt[:,:150]
    # eta = eta[:,:150]

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
        

    return xTrain, yTrain, xValid, yValid, xTest, yTest


def rawModel(xTrain, yTrain, xValid, yValid): 
    num = xTrain.shape[2]
    form = (xTrain.shape[1], xTrain.shape[2])

    # creating model
    op = keras.optimizers.Adam(learning_rate=0.001)
    # lossFunc = keras.losses.Huber(delta=0.1, name='modified01_huber_loss')
    # lossFunc = keras.losses.Huber()
    lossFunc = keras.losses.BinaryCrossentropy() #from_logits=True)
    # lossFunc = welsch
    # lossFunc = keras.losses.MeanAbsoluteError()

    model, typeM = rnn(form, op, lossFunc, MASK_NO)
    # model, typeM = pvToProbRNN(form, op, lossFunc, MASK_NO)
    model.summary()
    
    # saving the model and best weights
    weights = "{d}_Raw_model_{n}inputs_{m}_{o}_{l}_bins_size2_{t}.weights.h5".format(n=num, m=typeM, o='adam', l=lossFunc.name, d=nameData, t=CLOCK)
    modelDirectory = "models"
    modelName = weights[:-11]
    start =[i for i, letter in enumerate(modelName) if letter == '_']
    print(modelName)
    print()
    # callbacks
    checkpointCallback = keras.callbacks.ModelCheckpoint(filepath=weights, monitor="val_loss", save_weights_only=True, save_best_only=True, verbose=1)
    lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=30, cooldown = 1, min_lr=0.000001, verbose=1)
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
    diff = abs(yPredicted.flatten() - yT.flatten())
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

    # plotting histogram of difference
    # plt.clf()
    # b = 100
    # sn.displot(diff, hist=True, kde=True, bins=b, color='blue')
    # plt.title('Error of Predicted values historgram')
    # plt.xlabel('Error')
    # plt.savefig(f"{name[:start[0]]}_Hist_loss_{name[start[0]+1:]}.png")

    # # plotting % of predictions vs difference
    # plt.clf()
    # per = 90
    # tol = 0.15
    # shortenedDiff = diff[diff<2]
    # percent = (np.arange(0,len(shortenedDiff),1)*100)/len(diff)
    # percentile = np.zeros(len(shortenedDiff)) + per
    # tolerance = np.zeros(len(diff)) + tol
    # tolPercent = (np.arange(0,len(diff),1)*100)/len(diff)
    # sortedDiff = np.sort(shortenedDiff)
    # tolIndex = np.where(sortedDiff <= tol)
    # perIndex = np.where(tolPercent <= per)
    # print('Percentage where difference is <=', tol, ":", percent[tolIndex[0][-1]])
    # print('Value of', per, 'th percentile:', np.sort(diff)[perIndex[0][-1]])

    # fig, ax = plt.subplots()
    # plt.plot(sortedDiff, percent, color="green", label=name[start[3]+1:start[-1]], linewidth=0.7)
    # plt.plot(sortedDiff, percentile, color='blue', linestyle=':', label=str(per)+"th percentile")
    # plt.plot(tolerance, tolPercent, color='red', linestyle=':', label=str(tol)+" tolerance")
    # plt.scatter(tol, percent[tolIndex[0][-1]], color='red', label=str(tol)+' tolerance: '+str(round(percent[tolIndex[0][-1]], 3)))
    # if np.sort(diff)[perIndex[0][-1]] < 2:
    #     plt.scatter(np.sort(diff)[perIndex[0][-1]], per, color='blue', label=str(per)+' percentile: '+str(round(np.sort(diff)[perIndex[0][-1]], 3)))
    # ax.minorticks_on()
    # ax.grid(which='major', color='#CCCCCC', linewidth=0.8)
    # ax.grid(which='minor', color='#DDDDDD', linestyle='--', linewidth=0.6)
    # plt.xlabel('Difference between predicted and true value')
    # plt.ylabel('Percentage')
    # plt.title("Percentage of values vs Difference")
    # plt.legend()
    # plt.savefig(f"{name[:start[0]]}_Percentage_vs_loss_{name[start[0]+1:]}.png", dpi=1200)

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
    # ax[0].set_ylim(-15,15)
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
    # ax[1].set_ylim(-15,15)
    # ax[1].minorticks_on()
    # ax[1].grid(which='both', alpha=0.7, c='#DDDDDD')
    # plt.savefig(f'{name[:start[0]]}_True_vs_predicted_scatter_{name[start[0]+1:]}.png', dpi=1000)
    # print('scatter plot made')

    # # plot of scattered train and validation data
    # print()
    # plt.clf()
    # fig, ax = plt.subplots(1, 2, figsize=(12,6), sharey=True)
    # ax[0].axis('equal')
    # extent = np.array([[min(yTrain), max(yTrain)], [min(yPredTrain), max(yPredTrain)]])
    # heatmap = ax[0].hist2d(yTrain, yPredTrain, bins=20, cmap='hot_r', range=extent)
    # fig.colorbar(heatmap[3], ax=ax[0])
    # line = np.array([-15, 15])
    # ax[0].plot(line, line, color='black')
    # ax[0].plot(line, line+max(line)*0.2, '--', c='orange')
    # ax[0].plot(line, line-max(line)*0.2, '--', c='orange')
    # ax[0].plot(line, line+max(line)*0.1, '--', c='pink')
    # ax[0].plot(line, line-max(line)*0.1, '--', c='pink')
    # ax[0].set_title('Test Set')
    # ax[0].set_xlabel('True values')
    # ax[0].set_ylabel('Predicted values')
    # ax[0].set_ylim(-15,15)
    # ax[0].grid(which='both', alpha=0.7, c='#DDDDDD')

    # ax[1].axis('equal')
    # extent = np.array([[min(yT), max(yT)], [min(yPredicted), max(yPredicted)]])
    # heatmap = ax[1].hist2d(yT, yPredicted, bins=20, cmap='hot_r', range=extent)
    # fig.colorbar(heatmap[3], ax=ax[1])
    # ax[1].plot([-15,15], [-15,15], color='black')
    # ax[1].plot(line, line+max(line)*0.2,'--', c='orange')
    # ax[1].plot(line, line-max(line)*0.2, '--', c='orange')
    # ax[1].plot(line, line+max(line)*0.1, '--', c='pink')
    # ax[1].plot(line, line-max(line)*0.1, '--', c='pink')
    # ax[1].set_title('Validation Set')
    # ax[1].set_xlabel('True values')
    # ax[1].set_ylabel('Predicted values')
    # ax[1].set_ylim(-15,15)
    # ax[1].grid(which='both', alpha=0.7, c='#DDDDDD')
    # plt.savefig(f'{name[:start[0]]}_True_vs_predicted_map_{name[start[0]+1:]}.png')
    # print('map plot made')

    # # plotting learning rate against epochs
    # print()
    # lr = hist.history['lr']
    # plt.clf()
    # plt.plot(epochs, lr, color='b', linewidth=0.7)
    # plt.grid(which='major', color='#DDDDDD', linewidth=0.8)
    # plt.grid(which='minor', color='#EEEEEE', linestyle=':', linewidth=0.6)
    # plt.xlabel('Epoch number')
    # plt.ylabel('Learning Rate')
    # plt.title('Learning Rate against epochs')
    # plt.savefig(f"{name[:start[0]]}_Learning_rate_{name[start[0]+1:]}.png")
    # print('learning rate plot made')

    # % values that predicted the correct bin
    yPredicted = yPredicted.reshape(xT.shape[0]//zRaw.shape[1], zRaw.shape[1])
    indexPred = np.argmax(yPredicted, axis=1).flatten()
    indexTest = np.argwhere(yT.flatten() == 1).flatten()
    count = 0
    print(indexTest.shape)
    print(indexTest[:5])
    print(indexPred.shape)
    print(indexPred[:5])
    print(np.round(yPredicted[:10]))
    print(yT[:10])
    print(yT.shape)
    print(yPredicted[:10])
    print(yPredicted.shape)
    if len(indexTest) < len(indexPred):
        length = len(indexTest)
    else:
        length = len(indexPred)
    for i in tqdm(range(length)):
        if indexPred[i] in indexTest:
            count += 1
    print()
    print('Percentage of correct predicted bin: ', round(count*100/len(indexTest), 5))

    # confunstion matrix
    print()
    plt.clf()
    plt.figure(figsize=(30,20))
    plt.rcParams.update({'font.size': 40})
    yClassPredLabels = np.round(yPredicted)
    print(yT.shape)
    print(yClassPredLabels.shape)
    cm = tf.math.confusion_matrix(labels=yT, predictions=yClassPredLabels)
    sn.heatmap(cm, annot=True, fmt='d')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    if nameData != name[:start[0]]:
        plt.savefig(f"{nameData}_cm_probability_{name}.png", dpi=1000)
    else:
        plt.savefig(f'{nameData}_cm_probability_{name[start[0]+1:]}.png')
    print('cm plot made')
    

def comparison(models, train, xTest, yT):
    print()
    endStart =[i for i, letter in enumerate(models[0]) if letter == '_']
    name = "{start}_comparison_of_rnns_{t}".format(start=models[0][endStart[0]+1:endStart[7]], t=CLOCK)
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
            modelLoaded = loadWeights(models[i], xTest)
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
            modelLoaded = loadWeights(models[i], xTest)
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
            modelLoaded = loadWeights(models[i], xTest)
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
    loadedModel = tf.keras.models.load_model(name, custom_objects=dict(loss=loss))
    loadedModel.summary()
    return loadedModel


def loadWeights(name, x, lr):
    form = x.shape[1:]
    print()
    print(form)
    print(name)
    model, typeM = rnn(form, op=keras.optimizers.Adam(learning_rate=lr), lossFunc=keras.losses.BinaryCrossentropy(), maskNo=MASK_NO)
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


def testLoadedModel(model, train, xT, yT):
    hist = pd.read_csv(train, sep=',', engine='python')
    print(hist.columns)
    start =[i for i, letter in enumerate(model) if letter == '_']
    print(model)
    if model[-2:] == 'h5':
        print(model)
        modelLoaded = loadWeights(model, xT, lr=hist['lr'][np.argmin(hist['val_loss'])])
        model = model[:-11]
    else:
        modelLoaded = loadModel(model)
        model = model[:6]
    
    print()
    print(model)

    if nameData != model[:start[0]]:
        name = nameData + '_' + model[:start[0]]
    else:
        name = model[:start[-1]] # + 'pv_given_probability'
    print(name)

    loss = hist['loss']
    val_loss = hist['val_loss']
    epochs = range(1, len(loss) + 1)
    print()
    print('min val loss:', min(val_loss))
    print('At epoch number:',np.argmin(val_loss)+1)
    print('min loss:', min(loss))
    print('At epoch number:',np.argmin(loss)+1)

    # plot of epochs against training and validation loss
    plt.clf()
    plt.plot(epochs, loss, color='blue', label='Training Loss', linewidth=0.7)
    plt.plot(epochs, val_loss, color='red', label='Validation Loss', linewidth=0.7)
    minX = np.argmin(val_loss) + 1
    minY = np.min(val_loss)
    plt.scatter(minX, minY, color='green', label='minimum', s=6)
    plt.xlabel('Epoch number')
    plt.ylabel('Loss') 
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig(f"{name}_Train_valid_loss_{model[start[0]+1:]}.png",dpi=1200)
    print('Train valid plot made')
    
    print()
    yPredicted = modelLoaded.predict(xT).flatten()
    print()
    diff = abs(yPredicted.flatten() - yT.flatten())
    print(max(diff), min(diff))
    print(np.std(diff), np.mean(diff))

    # plotting histogram of difference
    plt.clf()
    sn.kdeplot(data=diff[diff<2], linewidth =0.8)
    plt.title('Error of Predicted values historgram')
    plt.xlabel('Error')
    plt.savefig(f"{name}_Hist_loss_{model[start[0]+1:]}.png", dpi=1000)
    print('Hist plot made')

    # plotting % of predictions vs loss
    print()
    plt.clf()
    per = 90
    tol = 0.15
    sortedDiff = np.sort(diff[diff<2])
    percent = (np.arange(0,len(sortedDiff),1)*100)/len(diff)
    percentile = np.zeros(len(sortedDiff)) + per
    tolerance = np.zeros(len(diff)) + tol
    tolPercent = (np.arange(0,len(diff),1)*100)/len(diff)
    tolIndex = np.where(sortedDiff <= tol)
    perIndex = np.where(tolPercent <= per)
    print('Percentage where difference is <=', tol, ":", percent[tolIndex[0][-1]])
    print('Value of', per, 'th percentil:', np.sort(diff)[perIndex[0][-1]])
    fig, ax = plt.subplots()
    plt.plot(sortedDiff, percent, color="green", linewidth=0.7)
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
    plt.savefig(f"{name}_Percentage_vs_loss_{model[start[0]+1:]}.png", dpi=1200)
    print('Percentage vs difference plot made')

    # # plot of scattered train and validation data
    # print()
    # yPredTrain = modelLoaded.predict(xTrain).flatten()
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
    # ax[0].set_ylim(-15,15)
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
    # ax[1].set_ylim(-15,15)
    # ax[1].minorticks_on()
    # ax[1].grid(which='both', alpha=0.7, c='#DDDDDD')
    # plt.savefig(f'{name}_True_vs_predicted_scatter_{model[start[0]+1:]}.png', dpi=1000)
    # print('scatter plot made')

    # # plot of scattered train and validation data
    # print()
    # plt.clf()
    # fig, ax = plt.subplots(1, 2, figsize=(12,6), sharey=True)
    # ax[0].axis('equal')
    # extent = np.array([[min(yTrain), max(yTrain)], [min(yPredTrain), max(yPredTrain)]])
    # heatmap = ax[0].hist2d(yTrain, yPredTrain, bins=40, cmap='Wistia', range=extent)
    # fig.colorbar(heatmap[3], ax=ax[0])
    # line = np.array([-15, 15])
    # ax[0].plot(line, line, color='black')
    # ax[0].plot(line, line+max(line)*0.2, '--', c='orange')
    # ax[0].plot(line, line-max(line)*0.2, '--', c='orange')
    # ax[0].plot(line, line+max(line)*0.1, '--', c='pink')
    # ax[0].plot(line, line-max(line)*0.1, '--', c='pink')
    # ax[0].set_title('Test Set')
    # ax[0].set_xlabel('True values')
    # ax[0].set_ylabel('Predicted values')
    # ax[0].set_ylim(-15,15)
    # ax[0].grid(which='both', alpha=0.7, c='#DDDDDD')

    # ax[1].axis('equal')
    # extent = np.array([[min(yT), max(yT)], [min(yPredicted), max(yPredicted)]])
    # heatmap = ax[1].hist2d(yTrain, yPredTrain, bins=40, cmap='Wistia', range=extent)
    # fig.colorbar(heatmap[3], ax=ax[1])
    # ax[1].plot([-15,15], [-15,15], color='black')
    # ax[1].plot(line, line+max(line)*0.2,'--', c='orange')
    # ax[1].plot(line, line-max(line)*0.2, '--', c='orange')
    # ax[1].plot(line, line+max(line)*0.1, '--', c='pink')
    # ax[1].plot(line, line-max(line)*0.1, '--', c='pink')
    # ax[1].set_title('Validation Set')
    # ax[1].set_xlabel('True values')
    # ax[1].set_ylabel('Predicted values')
    # ax[1].set_ylim(-15,15)
    # ax[1].grid(which='both', alpha=0.7, c='#DDDDDD')
    # plt.savefig(f'{name}_True_vs_predicted_map_{model[start[0]+1:]}.png')
    # print('map plot made')

    # # plotting learning rate against epochs
    # print()
    # if 'lr' in hist.columns[0]:
    #     lr = hist['lr']
    #     print('Final lr:', lr[-1])
    #     plt.clf()
    #     plt.plot(epochs, lr, color='b', linewidth=0.7)
    #     plt.grid(which='major', color='#DDDDDD', linewidth=0.8)
    #     plt.grid(which='minor', color='#EEEEEE', linestyle=':', linewidth=0.6)
    #     plt.xlabel('Epoch number')
    #     plt.ylabel('Learning Rate')
    #     plt.title('Learning Rate against epochs')
    #     plt.savefig(f"{name}_Learning_rate_{model[start[0]+1:]}.png")
    #     print('learning rate plot made')
    # else:
    #     print('No learning rate stored for each epoch')

    # # % values that predicted the correct bin
    # yPredicted = yPredicted.reshape(xT.shape[0]//zRaw.shape[1], zRaw.shape[1])
    # indexPred = np.argmax(yPredicted, axis=1).flatten()
    # indexTest = np.argwhere(yT.flatten() == 1).flatten()
    # indexTest = indexTest//zRaw.shape[1] + indexTest%zRaw.shape[1]
    # count = 0
    # print(indexTest.shape)
    # print(indexTest[:5])
    # print(indexPred.shape)
    # print(indexPred[:5])
    # print(np.max(yPredicted[:10]))
    # print(yT[:10])
    # print(yT.shape)
    # print(yPredicted[:10])
    # print(yPredicted.shape)
    # if len(indexTest) < len(indexPred):
    #     length = len(indexTest)
    # else:
    #     length = len(indexPred)
    # for i in tqdm(range(length)):
    #     if indexPred[i] in indexTest:
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
    #     plt.savefig(f"{nameData}_cm_probability_{model}.png", dpi=1000)
    # else:
    #     plt.savefig(f'{nameData}_cm_probability_{model[start[0]+1:]}.png')
    # print('cm plot made')


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
BATCH_SIZE = 4096
EPOCHS = 500
CLOCK = int(time.time())

# name = 'Merged_Raw_model_3inputs_rnn_adam_modified01_huber_loss_1722601651.keras'
# x = loadModel(name)

# loading numpy arrays of data
nameData = 'TTbar'
# rawD = np.load('TTbarRaw5.npz')
# binD = np.load('TTbarBin4.npz')
# rawBinD = np.load('TTbar_Raw_0.5_bin_size_overlap_0.npz')
rawBinD = np.load('TTbar_Raw_1_bin_size.npz')
# rawBinD = np.load('TTbar_Raw_2_bin_size.npz')
# rawBinD = np.load('TTbar_Raw_2_bin_size_overlap_1.0.npz')
# rawBinD = np.load('TTbar_Raw_2_bin_size_overlap_1.npz')
# rawBinD = np.load('TTbar_Raw_2_bin_size_overlap_pv_far_from_boundary_1.npz')

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
# ptBin, trackBin = binD['ptB'], binD['tB']
print(zRaw.shape, ptRaw.shape, etaRaw.shape, pvRaw.shape)
# print(np.argwhere(probability == 1))
# print(len(np.argwhere(probability == 1)))
# print(pvRaw.shape)
# print(np.argwhere(pvRaw!=pvRaw[np.isnan(pvRaw)]))
# print(len(np.argwhere(pvRaw[~np.isnan(pvRaw)])))


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
# xTrain, yTrain, xValid, yValid, xTest, yTest = rawModelSplit(zRaw, ptRaw, etaRaw, pvRaw.flatten(), prob=probability)
# print(xTrain.shape)
# model, history, name = rawModel(xTrain, yTrain, xValid, yValid)
# testing(model, history, xTest, yTest, name)


# prediting the pv given probability
# rawBinD = np.load('TTbar_Raw_1_bin_size.npz')
# zRaw, ptRaw, etaRaw, pvRaw, probability = rawBinD['z'], rawBinD['pt'], rawBinD['eta'], rawBinD['pv'], rawBinD['prob']
# print(zRaw.shape, ptRaw.shape, etaRaw.shape, pvRaw.shape)
# xTrain, yTrain, xValid, yValid, xTest, yTest = rawModelSplit(zRaw, ptRaw, etaRaw, pvRaw.flatten(), prob=probability)
# print(xTest.shape)
# probModel = 'TTbar_Raw_model_3inputs_pv_to_prob_rnn_adam_binary_crossentropy_bins_size1_1723546241.keras'
# xTestFocus, yTestFocus = findPVGivenProb(zRaw, probModel, xTest, yTest)
# regModel = 'TTbar_Raw_model_3inputs_rnn_adam_mean_absolute_error_overlap_bins_size1_1723539163.keras'
# train = 'TTbar_training_Raw_model_3inputs_rnn_adam_mean_absolute_error_overlap_bins_size1_1723539163.log'
# testLoadedModel(model=regModel, train=train, xT=xTestFocus, yT=yTestFocus)


# rawBinD = np.load('TTbar_Raw_2_bin_size.npz')
# zRaw, ptRaw, etaRaw, pvRaw, probability = rawBinD['z'], rawBinD['pt'], rawBinD['eta'], rawBinD['pv'], rawBinD['prob']
# print(zRaw.shape, ptRaw.shape, etaRaw.shape, pvRaw.shape)
# xTrain, yTrain, xValid, yValid, xTest, yTest = rawModelSplit(zRaw, ptRaw, etaRaw, pvRaw.flatten(), prob=probability)
# print(xTest.shape)
# probModel = 'TTbar_Raw_model_3inputs_rnn_adam_binary_crossentropy_1723130617.keras'
# xTestFocus, yTestFocus = findPVGivenProb(zRaw, probModel, xTest, yTest)
# regModel = 'TTbar_Raw_model_3inputs_rnn_adam_mean_absolute_error_overlap_bins_size2_pv_1723539044.keras'
# train = 'TTbar_training_Raw_model_3inputs_rnn_adam_mean_absolute_error_overlap_bins_size2_pv_1723539044.log'
# testLoadedModel(model=regModel, train=train, xT=xTestFocus, yT=yTestFocus)

# rawBinD = np.load('TTbar_Raw_0.5_bin_size_overlap_0.npz')
# zRaw, ptRaw, etaRaw, pvRaw, probability = rawBinD['z'], rawBinD['pt'], rawBinD['eta'], rawBinD['pv'], rawBinD['prob']
# print(zRaw.shape, ptRaw.shape, etaRaw.shape, pvRaw.shape)
# xTrain, yTrain, xValid, yValid, xTest, yTest = rawModelSplit(zRaw, ptRaw, etaRaw, pvRaw.flatten(), prob=probability)
# print(xTest.shape)
# probModel = 'TTbar_Raw_model_3inputs_pv_to_prob_rnn_adam_binary_crossentropy_bins_size1_1723548447.keras'
# xTestFocus, yTestFocus = findPVGivenProb(zRaw, probModel, xTest, yTest)
# regModel = 'TTbar_Raw_model_3inputs_rnn_adam_mean_absolute_error_overlap_bins_size05_1723540545.keras'
# train = 'TTbar_training_Raw_model_3inputs_rnn_adam_mean_absolute_error_overlap_bins_size05_1723540545.log'
# testLoadedModel(model=regModel, train=train, xT=xTestFocus, yT=yTestFocus)


# Loaded model test and comparison to other models

# xTrain, yTrain, xValid, yValid, xTest, yTest = binModelSplit(ptBin, pvRaw.flatten(), track=trackBin)
xTrain, yTrain, xValid, yValid, xTest, yTest = rawModelSplit(zRaw, ptRaw, etaRaw, pvRaw.flatten(), prob=None)
# xTrain = xTrain.reshape(xTrain.shape[0], xTrain.shape[1], xTrain.shape[2], 1)
# xValid = xValid.reshape(xValid.shape[0], xValid.shape[1], xValid.shape[2], 1)
# xTest = xTest.reshape(xTest.shape[0], xTest.shape[1], xTest.shape[2], 1)
# print(xTrain[0,0])
# print(xTrain.shape)

# name = 'Merged_Raw_model_3inputs_rnn_adam_modified01_huber_loss_1722601651.keras'
# train = 'Merged_training_Raw_model_3inputs_rnn_adam_modified01_huber_loss_1722601651.log'
# trainLoadedModel(name, train, xTrain, yTrain, xValid, yValid)
# testLoadedModel(name, train, xTest, yTest)

# name = 'TTbar_Raw_model_3inputs_rnn_adam_mean_absolute_error_overlap_bins_pv_1723453162.keras'
# train = 'TTbar_training_Raw_model_3inputs_rnn_adam_mean_absolute_error_overlap_bins_pv_1723453162.log'
# trainLoadedModel(name, train, xTrain, yTrain, xValid, yValid)
# testLoadedModel(name, train, xTest, yTest)

# name = 'TTbar_Raw_model_3inputs_rnn_adam_binary_crossentropy_overlap_bins_size2_pv_1723470769.keras'
# train = 'TTbar_training_Raw_model_3inputs_rnn_adam_binary_crossentropy_overlap_bins_size2_pv_1723470769.log'
# # trainLoadedModel(name, train, xTrain, yTrain, xValid, yValid)
# testLoadedModel(name, train, xTest, yTest)

# name = 'TTbar_Raw_model_3inputs_rnn_adam_binary_crossentropy_overlap_bins_size2_pv_1723466613.weights.h5'
# train = 'TTbar_training_Raw_model_3inputs_rnn_adam_binary_crossentropy_overlap_bins_size2_pv_1723466613.log'
# trainLoadedModel(name, train, xTrain, yTrain, xValid, yValid)
# testLoadedModel(name, train, xTest, yTest)

# name = ''
# train = ''
# trainLoadedModel(name, train, xTrain, yTrain, xValid, yValid)
# testLoadedModel(name, train, xTest, yTest)

# xTrain = xTrain.reshape(xTrain.shape[0], xTrain.shape[2], xTrain.shape[1])
# xValid = xValid.reshape(xValid.shape[0], xValid.shape[2], xValid.shape[1])
# xTest = xTest.reshape(xTest.shape[0], xTest.shape[2], xTest.shape[1], 1)

# # Comparing various models
# modelsCompare = ['TTbar_Bin_model_2inputs_conv_adam_mean_absolute_error_1721663273.keras',\
#                  'TTbar_Bin_model_2inputs_conv_adam_huber_loss_1721663295.keras',\
#                  'TTbar_Bin_model_2inputs_conv_adam_mean_squared_error_1721663286.keras',\
#                  'TTbar_Bin_model_2inputs_conv_adam_modified01_huber_loss_1722332746.keras']
# trainingCompare = ['TTbar_training_Bin_model_2inputs_conv_adam_mean_absolute_error_1721663273.log',\
#                    'TTbar_training_Bin_model_2inputs_conv_adam_huber_loss_1721663295.log',\
#                    'TTbar_training_Bin_model_2inputs_conv_adam_mean_squared_error_1721663286.log',\
#                    'TTbar_training_Bin_model_2inputs_conv_adam_modified01_huber_loss_1722332746.log']
modelsCompare = ['TTbar_Raw_model_3inputs_rnn_adam_mean_absolute_error_overlap_bins_size1_1723539163.keras',\
                 'TTbar_Raw_model_3inputs_rnn_adam_mean_absolute_error_1723135768.keras']
trainingCompare = ['TTbar_training_Raw_model_3inputs_rnn_adam_mean_absolute_error_overlap_bins_size1_1723539163.log',\
                   'TTbar_training_Raw_model_3inputs_rnn_adam_mean_absolute_error_1723135768.log']

endStart =[i for i, letter in enumerate(modelsCompare[0]) if letter == '_']
print(modelsCompare[0][:endStart[2]])
mod = loadModel(modelsCompare[0])
config = mod.get_config()
print(config["layers"][0]["config"])
mod = loadModel(modelsCompare[1])
config = mod.get_config()
print(config["layers"][0]["config"])
# mod = loadModel(modelsCompare[2])
# config = mod.get_config()
# print(config["layers"][0]["config"])


# mod = loadModel('Bin_model_2inputs_wavenet_adam_huber_loss_1721316446.keras')
# config = mod.get_config()
# print(config["layers"][0]["config"])
# train = 'Merged_training_Bin_model_2inputs_conv_adamax_modified01_huber_loss_1722592046.log'
# hist = pd.read_csv(train, sep=',', engine='python')
# print(hist.columns)

print(xTest.shape)
comparison(modelsCompare, trainingCompare, xTest, yTest)


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