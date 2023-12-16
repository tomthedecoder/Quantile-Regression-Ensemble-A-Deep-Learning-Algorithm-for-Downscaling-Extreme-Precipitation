from data_handling import open_data 
from useful_functions import data_between 
from quantiles import ynetwork 
import tensorflow as tf 
import numpy as np 

class ConstantBaseline:
    def __init__(self, cnst_val):
        self.prediction = cnst_val
    
    def __call__(self, x=None):
        return self.prediction 

def empirical_mean_baseline(y_train, Q_intervals, num_repeats):
    emp_base = {}
    # segment y_train by intensity, create f_xi predictions, but fix them to empirical mean of training set 
    print(Q_intervals)
    for n, (q1, q2) in enumerate(Q_intervals):
        qseg = data_between(y_train, q1, q_high=q2)
        emp_mean = tf.cast(tf.reduce_mean(qseg.squeeze(), axis=0), dtype=tf.float32)
        emp_mean = tf.expand_dims(emp_mean, axis=0)
        emp_base[f'cnst_{n+1}'] = [ConstantBaseline(emp_mean) for k in range(num_repeats)]
    
    return emp_base  