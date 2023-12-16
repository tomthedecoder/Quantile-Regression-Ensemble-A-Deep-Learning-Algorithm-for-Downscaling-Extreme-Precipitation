import tensorflow as tf
import numpy as np


class EmbedModule(tf.keras.Model):
    def __init__(self, 
                         avg_pool=True, 
                         max_pool=False, 
                         n=1,
                         m=1
                 ):
        
        super(EmbedModule, self).__init__()

        self.avg_pool = avg_pool
        self.max_pool = max_pool
        self.n = n
        self.m = m 
        
        reduce_resolution = lambda nc, ks: tf.keras.Sequential([
                                                                    tf.keras.layers.Conv2D(nc, ks, activation='relu'),
                                                                    tf.keras.layers.MaxPool2D(3),
                                                                    #tf.keras.layers.Conv2D(nc, ks, activation='relu'),
                                                                    tf.keras.layers.Conv2D(1, 3, activation='relu')
                                                               ])
        
        self.module = tf.keras.Sequential([reduce_resolution(64, 7)])
        self.interpolate = tf.keras.layers.Lambda(lambda im: tf.image.resize(im, (self.n, self.m)))        

    def call(self, x):
        x = self.module(x)
        x = self.interpolate(x)
        
        return x
    

# if interpolation is used 
class InterpModule:
    def __init__(self, n=1, m=1):
        self.n = n
        self.m = m
        self.interpolate = lambda im: tf.image.resize(im, (self.n, self.m))
    
    def __call__(self, x):
        x = self.interpolate(x)
        
        return x 

    
# a single down sampling component of the network 
class DownScaleModule(tf.keras.Model):
    def __init__(self, 
                         is_baseline,
                         avg_pool, 
                         max_pool, 
                         ch_attn, 
                         sp_attn,
                         num_channels,
                         kernel_sizes,
                         reduction_rate=1,
                         aux_data=None,
                         aux_mode='embed',
                         n=1,
                         m=1
                 ):
        
        super(DownScaleModule, self).__init__()        
        
        self.num_channels = num_channels
        self.kernel_sizes = kernel_sizes
        self.reduction_rate = reduction_rate
        self.avg_pool = avg_pool
        self.max_pool = max_pool
        self.sp_attn = sp_attn
        self.ch_attn = ch_attn
        self.aux_data = aux_data
        
        self.input_layer = tf.keras.layers.InputLayer(input_shape=(36, 41, 5))
        self.ds_module = tf.keras.Sequential([self.bmodule(nc, ks) 
                                               if is_baseline 
                         else self.module(nc, ks) for nc, ks in zip(self.num_channels, self.kernel_sizes)])
        
        if self.aux_data is not None:
            self.aux_module = EmbedModule(
                                            self.avg_pool, 
                                            self.max_pool, 
                                            n,
                                            m
                                       ) if aux_mode == 'embed' else InterpModule(n, m)
        else:
            self.aux_module = None 

        
    def module(self, nc, ks):
        module = tf.keras.Sequential()
        module.add(tf.keras.layers.Conv2D(nc, ks, activation='relu'))
        module.add(ChannelAttentionModule(nc, self.reduction_rate, self.avg_pool, self.max_pool))
        module.add(tf.keras.layers.Conv2D(nc, ks, activation='relu'))

        return module 
    
    
    def bmodule(self, nc, ks):
        base_ds = tf.keras.Sequential()
    
        base_ds.add(tf.keras.layers.BatchNormalization())
        base_ds.add(tf.keras.layers.Conv2D(nc, ks, activation='relu'))
        base_ds.add(tf.keras.layers.MaxPool2D(2))
        
        return base_ds
                   
        
    def call(self, x):
        x = self.input_layer(x)
        # make sure the batch sizes are the same
        if self.aux_data is not None:
            n = x.shape[1]
            m = x.shape[2]
            z = self.aux_module(self.aux_data)
            z = tf.broadcast_to(z, [tf.shape(x)[0] if tf.shape(x)[0] is not None else 1, n, m, 1])
            x = tf.concat((x, z), axis=-1)
        
        x = self.ds_module(x)
                
        return x
        
        
# channel attention from zhang 2017. Used in convolutional component of the network 
class ChannelAttentionModule(tf.keras.Model):
    def __init__(self,
                        num_channels,
                        reduction_rate,
                        avg_pool=False,
                        max_pool=False
                 ):
        
        super(ChannelAttentionModule, self).__init__()
    
        # keepdims is a misunderstood key word argument
        self.avg_pool = tf.keras.layers.GlobalAveragePooling2D(data_format='channels_last') if avg_pool else None 
        self.max_pool = tf.keras.layers.GlobalMaxPool2D(data_format='channels_last') if max_pool else None
        self.pools = [pool for pool in (self.avg_pool, self.max_pool) if pool is not None]

        self.mlp = tf.keras.Sequential([
                                            tf.keras.layers.Dense(num_channels//reduction_rate, activation='relu'),
                                            tf.keras.layers.Dense(num_channels)
                                        ])

    def call(self, x):
        output = None

        for pool in self.pools:
            pool_out = tf.expand_dims(tf.expand_dims(pool(x), axis=1), axis=1)
            if output is None:
                output = self.mlp(pool_out)
            else:
                output = tf.add(output, self.mlp(pool_out))

        s = tf.math.sigmoid(output)

        return s * x


# Spatial attention module, concatenates a choice of global pooling functions
class SpatialAttentionModule(tf.keras.Model):
    def __init__(self,  avg_pool=True,
                        max_pool=False):

        super(SpatialAttentionModule, self).__init__()
        self.avg_pool = tf.keras.layers.Lambda(lambda im: tf.reduce_mean(im, axis=-1, keepdims=True)) if avg_pool else None
        self.max_pool = tf.keras.layers.Lambda(lambda im: tf.reduce_max(im, axis=-1, keepdims=True)) if max_pool else None

        self.pools = [pool for pool in (self.avg_pool, self.max_pool) if pool is not None]
        self.conv = tf.keras.layers.Conv2D(1, 7, padding='same', activation='sigmoid')

    def call(self, x, **kwargs):
        output = None
        for pool in self.pools:
            if output is None:
                output = pool(x)
            else:
                output = tf.concat([output, pool(x)], axis=-1)
                
        m = self.conv(output)
        
        return m * x

    
# combines spatial and channel attention if nessacary 
class AttentionModule(tf.keras.Model):
    def __init__(self, 
                         avg_pool, 
                         max_pool, 
                         ch_attn, 
                         sp_attn, 
                         num_channel, 
                         reduction_rate
                ):
        
        super(AttentionModule, self).__init__()
        
        self.avg_pool = avg_pool
        self.max_pool = max_pool
        self.ch_attn = ch_attn
        self.sp_attn = sp_attn
        self.num_channels = num_channel
        self.reduction_rate = reduction_rate
        
        self.module = self.attention_module()
        
    def attention_module(self):
        attn_module = tf.keras.Sequential([])

        if self.ch_attn:
            attn_module.add(ChannelAttentionModule(self.num_channels, self.reduction_rate, self.avg_pool, self.max_pool))

        if self.sp_attn:
            attn_module.add(SpatialAttentionModule(self.avg_pool, self.max_pool))

        return attn_module
    
    def call(self, x):
        x = self.module(x)
        
        return x
       
        
        
        
        
        
        
        
        
        
        
        
        