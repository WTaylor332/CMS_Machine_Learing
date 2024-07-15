import numpy as np 
import pandas as pd
import time
import os
import tqdm
print()
import tensorflow as tf 
from tensorflow import keras
print()
import matplotlib.pyplot as plt 
from sklearn.preprocessing import StandardScaler

class haltCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if (logs.get('val_loss') < 0.1):
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


def convModel(shape, op, lossFunc):
    if shape[1] < 2:
        #1D model
        model = keras.models.Sequential([
            # convolutional layer
            keras.layers.Conv1D(12, kernel_size=8, activation='relu', input_shape=(shape)),
            keras.layers.MaxPool1D(pool_size=4),

            keras.layers.Conv1D(12, kernel_size=8, activation='relu'),
            keras.layers.MaxPool1D(pool_size=4),

            keras.layers.Conv1D(12, kernel_size=8, activation='relu'),
            keras.layers.MaxPool1D(pool_size=2),


            # multi later perceptron
            keras.layers.Flatten(),
            keras.layers.Dense(15, activation="relu"),
            # keras.layers.Dropout(rate=0.3),
            keras.layers.Dense(5, activation="relu"),
            # keras.layers.Dropout(rate=0.3),
            keras.layers.Dense(1)
        ])

    else:
        # 2D model
        model = keras.models.Sequential([
            # convolutional layer
            keras.layers.Conv2D(12, kernel_size=(8,1), activation='relu', input_shape=(shape)),
            keras.layers.MaxPool2D(pool_size=(4,4)),

            keras.layers.Conv2D(12, kernel_size=(8,1), activation='relu'),
            keras.layers.MaxPool2D(pool_size=(4,4)),

            keras.layers.Conv2D(12, kernel_size=(8,1), activation='relu'),
            keras.layers.MaxPool2D(pool_size=(2,2)),


            # multi later perceptron
            keras.layers.Flatten(),
            keras.layers.Dense(15, activation="relu"),
            # keras.layers.Dropout(rate=0.3),
            keras.layers.Dense(5, activation="relu"),
            # keras.layers.Dropout(rate=0.3),
            keras.layers.Dense(1)
        ])
    model.compile(optimizer=op,
    loss=lossFunc)
    model.summary()
    return model


def binModel(xTrain, yTrain, xValid, yValid, xTest, yTest):
    print(xTrain.shape)
    print(len(xTrain.shape))
    if len(xTrain.shape) > 2:
        form = (xTrain.shape[1], 2, 1)
    else:
        form = (xTrain.shape[1], 1)

    # creating model
    # model = keras.models.Sequential([
    #     # multi later perceptron
    #     keras.Input(shape=form),
    #     keras.layers.Dense(100, activation="relu"),
    #     keras.layers.Dropout(rate=0.3),
    #     keras.layers.Dense(50, activation="relu"),
    #     keras.layers.Dropout(rate=0.3),
    #     keras.layers.Dense(1)
    # ])
    op = keras.optimizers.Adam(learning_rate=0.008)
    lossFunc = keras.losses.Huber()
    model = convModel(form, op, lossFunc)
    
    # saving the model and best weights
    weights = "Bin_model_conv_weights_{o}_{l}_{t}.weights.h5".format(o='adam', l=lossFunc.name, t=clock)
    modelDirectory = "models"
    modelName = "Bin_model_conv_{o}_{l}_{t}".format(o='adam', l=lossFunc.name, t=clock)
    
    # callbacks
    checkpointCallback = keras.callbacks.ModelCheckpoint(filepath=weights, monitor="val_loss", save_weights_only=True, save_best_only=True, verbose=1)
    lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, cooldown = 1, min_lr=0.000001, verbose=1)
    csvLogger = keras.callbacks.CSVLogger("training_{}.log".format(modelName), separator=',', append=False)
    stopTraining = haltCallback()
    earlyStop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)

    epochNo = 500
    print(modelName)
    history = model.fit(xTrain, yTrain, epochs=epochNo, validation_data=(xValid, yValid), callbacks=[lr, checkpointCallback, csvLogger, stopTraining, earlyStop])

    checkpointFilename = os.path.join(modelDirectory, weights)
    check = os.path.isdir(modelDirectory)
    if not check:
        os.makedirs(modelDirectory)
        print("Created directory:" , modelDirectory)

    # saves full model
    modelName = "Bin_model_conv_{o}_{l}_{t}.keras".format(o='adam', l=lossFunc.name, t=clock)
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
    form = (2, xTrain.shape[1])

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

    # plotting % of predictions vs loss
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


def comparison(models, xTest, yTest):
    # Percentage vs difference plot comparsion
    plt.clf()
    fig, ax = plt.subplots()
    ax.minorticks_on()
    ax.grid(which='major', color='#CCCCCC', linewidth=0.8)
    ax.grid(which='minor', color='#DDDDDD', linestyle='--', linewidth=0.6)
    yPredicted = np.zeros((len(models), len(yTest)))
    for i in range(2, len(models)):
        # if models[i][-2:] == 'h5':
        modelLoaded = loadWeights(models[i])
        # else:
        # modelLoaded = loadModel(models[i])
        yPredicted[i] = modelLoaded.predict(xTest).flatten()

        diff = abs(yPredicted[i].flatten() - yTest.flatten())
        sortedDiff = np.sort(diff[diff<2])
        percent = (np.arange(0,len(sortedDiff),1)*100)/len(sortedDiff)

        percentile = np.zeros(len(sortedDiff)) + 90
        tolerance = np.zeros(len(sortedDiff)) + 0.1

        plt.plot(sortedDiff, percent, label=models[i])

    plt.legend()
    plt.title("Percentage of values vs loss")
    # plt.plot(sortedDiff, percentile, color='blue')
    # plt.plot(tolerance, percent, color='red')
    name = "Bin_model_comparison_of_loss_functions_{t}".format(t=clock)
    plt.savefig("Percentage_vs_loss_{}.png".format(name), dpi=1200)


def loadModel(name):
    mod = name # + '/saved_model'
    loadedModel = tf.keras.models.load_model(mod)
    loadedModel.summary()
    return loadedModel


def loadWeights(name):
    if len(xTrain.shape) > 2:
        form = (xTrain.shape[1], 2, 1)
    else:
        form = (xTrain.shape[1], 1)
    print(form)
    model = convModel(form)
    model.load_weights(name)
    model.summary()
    return model


def testLoadedModel(model, xTest, yTest,name):
    modelLoaded = loadModel(model)
    hist = pd.read_csv('training.log', sep=',', engine='python')

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
binD = np.load('TTbarBin4.npz')
zRaw, ptRaw, pvRaw = rawD['z'], rawD['pt'], rawD['pv']
ptBin, trackBin = binD['ptB'], binD['tB']
trackLength, maxTrack = rawD['tl'], rawD['maxValue']

clock = int(time.time())

# plt.hist(trackLength, bins=100, color='red')
# plt.plot()
# plt.savefig("TTbarTrackDistribution.png")

print()
xTrain, yTrain, xValid, yValid, xTest, yTest = binModelSplit(pt=ptBin, pv=pvRaw.flatten()) #, track=trackBin)
model, history, name = binModel(xTrain, yTrain, xValid, yValid, xTest, yTest)
testing(model, history, xValid, yValid, xTest, yTest, name)

# print()
# xTrain, yTrain, xValid, yValid, xTest, yTest = rawModelSplit(zRaw, ptRaw, pvRaw.flatten())
# model, history, name = rawModel(xTrain, yTrain, xValid, yValid, xTest, yTest)
# testing(model, history, xValid, yValid, xTest, yTest, name)



# Loaded model test and comparison to other models

print()
#xTrain, yTrain, xValid, yValid, xTest, yTest = binModelSplit(ptBin, pvRaw.flatten())
# xTrain, yTrain, xValid, yValid, xTest, yTest = rawModelSplit(zRaw, ptRaw, pvRaw.flatten())

# model = "Bin_model_1720443577.h5"
# testLoadedModel(model, xTest, yTest)

#models = np.array(['Bin_model_conv_weights_1720686776.h5',\
#        'Bin_model_conv_weights_1720614498.h5'])


#training = np.array(['training_Bin_model_conv_adam_huber_loss_1720614426.log',\
#        'training_Bin_model_conv_adam_mean_squared_error_1720686776.log',\
#        'training_Bin_model_conv_adam_mean_absolute_error_1720614498.log'])


#for i in range(len(models)):
#    print(i)
#    print(xTrain.shape)
#    model = loadWeights(models[i])
#    hist = pd.read_csv(training[i], sep=',', engine='python')
#    val_loss = hist.history['val_loss']
#    print(val_loss)


# comparison(models, xTest, yTest)




