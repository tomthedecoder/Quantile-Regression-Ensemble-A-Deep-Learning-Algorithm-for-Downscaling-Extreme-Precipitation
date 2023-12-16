import xarray as xr 
import numpy as np 
import tensorflow as tf 
import os 

from useful_functions import * 
from Modules import * 


# discretise the dataset, taking y sample to bin index  
def bin_y_var(y, Q_ranges):
    bins = []
    
    # get bin limits 
    for ql, qh in Q_ranges:
        yq = data_between(y, qh) 
        bins.append(np.sum(yq))

    # get quantile ranges 
    quantiles = []
    for y_sample in y:
        ys_sum = np.sum(y_sample)
        bin_index = len(bins) - 1
        for n, bin_thrs in enumerate(bins):
            if ys_sum <= bin_thrs:
                bin_index = n
                break

        new_y = np.zeros(len(Q_ranges))
        new_y[bin_index] = 1

        quantiles.append(new_y)        
    
    return quantiles


def weight(y1, y2):
    return np.abs(y1 - y2)

# implements EMD loss for multiclass classification
class OmegaLoss(tf.keras.losses.Loss):
    def __init__(self, num_classes, p=1, weight_func=weight, loss_weights=tf.constant([0.0, 0.0, 1.0])):
        super(OmegaLoss, self).__init__()

        self.num_classes = num_classes
        self.p = p
        self.weight_func = weight_func
        self.loss_weights = loss_weights
        self.weight_matrix = self.gen_matrix()
    
    def gen_matrix(self):
        matrix = np.array([[self.weight_func(n, m) for n in range(self.num_classes)] for m in range(self.num_classes)])
        matrix = tf.constant(matrix, dtype=tf.float32)
        return tf.transpose(matrix)

    def call(self, y_true, y_pred):
        y_pred = tf.squeeze(y_pred)
        y_true = tf.squeeze(y_true)
                
        col_indices = tf.argmax(y_true, axis=1)
        weight_cols = tf.pow(tf.gather(self.weight_matrix, col_indices), self.p)
        
        emd = tf.reduce_mean(weight_cols * y_pred, axis=1)
        
        return emd
    

# The weight network omega_theta 
class Omega(tf.keras.Model):
    def __init__(
                    self,
                    output_dim,
                    kernels=[6, 6, 6],
                    channels=[64, 128, 256],
                ):
        
        super(Omega, self).__init__()
        
        self.output_dim = output_dim
        self.kernels = kernels
        self.channels = channels
        self.post_process = False
        
        self._omega = self.omega()

    def call(self, x, training=False):
        x = self._omega(x)
        
        if self.post_process and not training and False:
            x = post_process(x)
            
        return x 
    
    def omega(self):
        omega = tf.keras.Sequential()
        
        for ks, nch in zip(self.kernels, self.channels):
            omega.add(tf.keras.layers.Conv2D(nch, ks, activation='relu'))
            omega.add(ChannelAttentionModule(nch, 1, True, True))
            omega.add(tf.keras.layers.Conv2D(nch, ks, activation='relu'))
            #omega.add(tf.keras.layers.MaxPool2D(2))
        
        omega.add(tf.keras.layers.Flatten())
        omega.add(tf.keras.layers.Dense(100, activation='relu'))
        omega.add(tf.keras.layers.Dropout(0.2))
        omega.add(tf.keras.layers.Dense(100, activation='relu'))
        omega.add(tf.keras.layers.Dropout(0.2))
        omega.add(tf.keras.layers.Dense(self.output_dim, activation='softmax'))
        
        return omega
    

# quantile boost ensemble g 
class ynetwork:
    def __init__(
                    self,
                    CNN_dict,
                    weight_model=None,
                    fixed_weights=None
                ):

        self.CNN_dict = CNN_dict
        
        if weight_model is not None:
            self.weights = weight_model
        elif fixed_weights is not None:
            self.weights = lambda x: fixed_weights 
        
    def __call__(self, x):
        pred = None
        for CNN_list in self.CNN_dict.values():
            mem = tf.expand_dims(evaluate_models(CNN_list, x, take_mean=True), axis=1)
            if pred is None:
                pred = mem
            else:
                pred = tf.concat((pred, mem), axis=1)
        
        weights = tf.expand_dims(self.weights(x), axis=-1)
        weights = tf.cast(weights, dtype=tf.float32)
        x = tf.reduce_sum(weights * pred, axis=1)
                
        return x