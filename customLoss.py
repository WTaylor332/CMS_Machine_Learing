import numpy as np
import math as m
import tensorflow as tf

def welsch(y_true: tf.Tensor, y_pred: tf.Tensor, c=0.1):
    loss =  1 - tf.math.exp(-0.5 * ((y_true - y_pred)/c)**2)
    return loss