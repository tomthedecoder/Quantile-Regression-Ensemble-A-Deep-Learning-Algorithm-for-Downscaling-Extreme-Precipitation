import tensorflow as tf 
#import tensorflow_probability as tfp 

from useful_functions import * 
from Modules import *


# tracks the progess of the model over epochs 
class EpochLogger(tf.keras.callbacks.Callback):
    def __init__(self):
        super(EpochLogger, self).__init__()
        
        self.statistics = {}

        
    def on_epoch_end(self, epoch, logs=None):
        for key, stat in logs.items():
            if epoch == 0:
                self.statistics[key] = [stat]
            else:
                self.statistics[key].append(stat)

            
    def get_statistics(self):
        return self.statistics
    

# The Bernoulli-Gamma loss 
class GammaLoss(tf.keras.losses.Loss):
    def __init__(
                    self,
                    epsilon=0.0001,                       # how does making smaller effect results  
                    y_thrs=0.5
                ):
        
        super(GammaLoss, self).__init__()

        self.epsilon = epsilon
        self.y_thrs = y_thrs
        
        
    # if training, then fit parameters to empirical distribution, 
    # otherwise return L2 norm between predictions and obs, given by distribution mean
    def call(self, y_true, y_pred):   
        new_shape = (tf.shape(y_pred)[0], tf.shape(y_pred)[1])
        p = tf.reshape(tf.gather(y_pred, indices=[0], axis=-1), new_shape)
        alpha =  tf.reshape(tf.gather(y_pred, indices=[1], axis=-1), new_shape)
        beta = tf.reshape(tf.gather(y_pred, indices=[2], axis=-1), new_shape)
        
        alpha = tf.exp(alpha)
        beta = tf.exp(beta)
        
        loss = -tf.reduce_mean(self.lBG_distribution(y_true, p, alpha, beta, self.y_thrs, self.epsilon))

        return loss 
    
    # train on the log of the distribution 
    @staticmethod
    def lBG_distribution(y, p, alpha, beta, y_thrs=0.5, eps=0.01):
        p_true = tf.where(y > y_thrs, 1.0, 0.0)
        branch1 = tf.math.log(1 - p + eps)
        branch2 = tf.math.log(p + eps) + (alpha - 1) * tf.math.log(y + eps) - alpha * tf.math.log(beta + eps) - tf.math.lgamma(alpha) - y / (beta + eps)
        
        return (1 - p_true) * branch1 + p_true * branch2   