import numpy as np
import math as m
import tensorflow as tf

def welsch(y_true: tf.Tensor, y_pred: tf.Tensor, c=1.2):
    loss =  1 - tf.math.exp(-0.5 * ((y_true - y_pred)/c)**2)
    return loss



# learning rate function

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
    if epoch < 20:
        return 0.001
    elif epoch < 30:
        return 0.0005
    else:
        return 0.0000001