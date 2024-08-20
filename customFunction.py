import tensorflow as tf
from tensorflow import keras

# type of loss function 
def welsch(y_true: tf.Tensor, y_pred: tf.Tensor, c=1):
    loss =  1 - tf.math.exp(-0.5 * ((y_true - y_pred)/c)**2)
    return loss


# learning rate functions:

def learningRate(lr0=0.001, s=20, patience=3):
    def learningRateFunc(epoch, val_loss, learning_rate):
        if len(val_loss) > 3:
            if  val_loss[epoch] - val_loss[epoch-patience] > 0:
                return learning_rate[epoch-patience]
            else:
                return exponential_decay(lr0, s)
        else:
            return lr0
    return learningRateFunc
        

def power_decay(lr0=0.001, s=20):
    def power_decay_function(epoch):
        return lr0/(1 + (epoch / s))
    return power_decay_function

def exponential_decay(lr0, s):
    def exponential_decay_fn(epoch):
        return lr0 * 0.1**(epoch / s)
    return exponential_decay_fn

def piecewise_constant_fn(epoch):
    if epoch < 50:
        return 0.001
    elif epoch < 80:
        return 0.0005
    elif epoch < 120:
        return 0.00025
    elif epoch < 160:
        return 0.000125
    elif epoch < 200:
        return 0.0000625
    elif epoch < 250:
        return 0.00003625
    else:
        return 0.000001
