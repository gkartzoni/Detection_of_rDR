#!/usr/bin/python
# -*- coding: utf-8 -*-

from keras.layers.core import Dropout
#from keras.applications.inception_v3 import InceptionV3
from keras.models import Model
from keras.layers import Input


import os
from __init__ import get_submodules_from_kwargs
import keras_applications

from keras_applications import imagenet_utils
from keras_applications.imagenet_utils import decode_predictions
#from keras_applications.imagenet_utils import _obtain_input_shape
import tensorflow as tf
from keras import backend as K  
from keras.layers import multiply
import numpy as np
from keras.utils.data_utils import get_file
from keras.models import Model
from keras.layers.merge import Multiply, multiply, dot
from keras import utils 
from keras.layers import Concatenate,concatenate,Dense,Lambda,Dot, AveragePooling2D,MaxPooling2D, BatchNormalization, Conv2D, GlobalMaxPooling2D,Input,Activation,merge, Dense, GlobalAveragePooling2D, Dropout

backend = None
layer = None
models = None
keras_utils = None
name = None
def lambda_fn(x):
     #bool_x = x[1].astype(np.float)
     # var = tf.Variable(x[1],dtype=tf.float32,validate_shape=False,name = 'pl1')

      bool_x = x[1]*x[0]
    #  bool_x = x[1]>=1
      #bool_x = tf.cast(x[1], tf.bool)
      return bool_x
def reshape_fun(x,size):
   # import tensorflow as tf
    #return tf.reshape(x,size)
    return Lambda(lambda x: tf.reshape(x,size))(x)

def resize_fun(x,size):
    from keras.backend import tf as ktf
    return Lambda(lambda image: ktf.image.resize_images(x, size))(x)
    
    

# def reshape_fun(x,size):
    # import tensorflow as tf
    # return tf.reshape(x,size)


# def resize_fun(x,size):
    # from keras.backend import tf as ktf
    # return Lambda(lambda image: ktf.image.resize_images(x, size))(x)
    
def conv2d_bn(x,
              filters,
              num_row,
              num_col,
              padding='same',
              strides=(1, 1),
            ):

    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
    else:
        bn_name = None
        conv_name = None
    # if backend.image_data_format() == 'channels_first':
        # bn_axis = 1
    # else:
        # bn_axis = 3
    bn_axis = 3
    x = Conv2D(
        filters, (num_row, num_col),
        strides=strides,
        padding=padding,
        use_bias=False,
        name=conv_name)(x)
    x = BatchNormalization(axis=bn_axis, scale=False, name=bn_name)(x)
    x = Activation('relu', name=name)(x)
    return x
    
def con32(x2):
    x2 = Concatenate(axis = 3)([x2, x2, x2, x2, x2, x2, x2, x2])
    x2 = Concatenate(axis = 3)([x2, x2, x2, x2])  
    return x2

def con96(x2):
    x2 = Concatenate(axis = 3)([x2, x2, x2, x2, x2, x2, x2, x2])
    x2 = Concatenate(axis = 3)([x2, x2, x2, x2]) 
    x2 = Concatenate(axis = 3)([x2, x2, x2])    
    return x2
    
def con144(x2):
    x2 = Concatenate(axis = 3)([x2, x2, x2, x2, x2, x2, x2, x2])
    x2 = Concatenate(axis = 3)([x2, x2])  
    x2 = Concatenate(axis = 3)([x2, x2, x2, x2, x2, x2, x2, x2,x2])

    return x2

def con288(x2):
    x2 = Concatenate(axis = 3)([x2, x2, x2, x2, x2, x2, x2, x2])
    x2 = Concatenate(axis = 3)([x2, x2])  
    x2 = Concatenate(axis = 3)([x2, x2, x2, x2, x2, x2, x2, x2,x2])
    x2 = Concatenate(axis = 3)([x2, x2])  

    return x2    
def con64(x2):
    x2 = Concatenate(axis = 3)([x2, x2, x2, x2, x2, x2, x2, x2])
    x2 = Concatenate(axis = 3)([x2, x2, x2, x2])  
    x2 = Concatenate(axis = 3)([x2, x2])    
    return x2


def con80(x2):
    x2 = Concatenate(axis = 3)([x2, x2, x2, x2, x2, x2, x2, x2])
    x2 = Concatenate(axis = 3)([x2, x2, x2, x2,x2])  
    x2 = Concatenate(axis = 3)([x2, x2])  

    return x2

def con192(x2):
    x2 = Concatenate(axis = 3)([x2, x2, x2, x2, x2, x2, x2, x2])
    x2 = Concatenate(axis = 3)([x2, x2, x2, x2])  
    x2 = Concatenate(axis = 3)([x2, x2, x2, x2,x2, x2])  
    return x2    

def con48(x2):
    x2 = Concatenate(axis = 3)([x2, x2, x2, x2, x2, x2, x2, x2])
    x2 = Concatenate(axis = 3)([x2, x2, x2, x2, x2, x2])
    return x2  

def con96(x2):
    x2 = Concatenate(axis = 3)([x2, x2, x2, x2, x2, x2, x2, x2])
    x2 = Concatenate(axis = 3)([x2, x2, x2, x2,x2, x2]) 
    x2 = Concatenate(axis = 3)([x2, x2])
    return x2  

def con384(x2):
    x2 = Concatenate(axis = 3)([x2, x2, x2, x2, x2, x2, x2, x2])
    x2 = Concatenate(axis = 3)([x2, x2, x2, x2])  
    x2 = Concatenate(axis = 3)([x2, x2, x2, x2,x2, x2]) 
    x2 = Concatenate(axis = 3)([x2,x2]) 
    return x2     

def con768(x2):
    x2 = Concatenate(axis = 3)([x2, x2, x2, x2, x2, x2, x2, x2])
    x2 = Concatenate(axis = 3)([x2, x2, x2, x2])  
    x2 = Concatenate(axis = 3)([x2, x2, x2, x2,x2, x2]) 
    x2 = Concatenate(axis = 3)([x2,x2]) 
    x2 = Concatenate(axis = 3)([x2,x2]) 

    return x2   
    
def con128(x2):
    x2 = Concatenate(axis = 3)([x2, x2, x2, x2, x2, x2, x2, x2])
    x2 = Concatenate(axis = 3)([x2, x2, x2, x2])  
    x2 = Concatenate(axis = 3)([x2, x2]) 
    x2 = Concatenate(axis = 3)([x2, x2])     
    return x2    

def con256(x2):
    x2 = Concatenate(axis = 3)([x2, x2, x2, x2, x2, x2, x2, x2])
    x2 = Concatenate(axis = 3)([x2, x2, x2, x2])  
    x2 = Concatenate(axis = 3)([x2, x2]) 
    x2 = Concatenate(axis = 3)([x2, x2])     
    x2 = Concatenate(axis = 3)([x2, x2])     

    return x2  

def con2048(x2):
    x2 = Concatenate(axis = 3)([x2, x2, x2, x2, x2, x2, x2, x2])
    x2 = Concatenate(axis = 3)([x2, x2, x2, x2])  
    x2 = Concatenate(axis = 3)([x2, x2]) 
    x2 = Concatenate(axis = 3)([x2, x2])     
    x2 = Concatenate(axis = 3)([x2, x2])  
    x2 = Concatenate(axis = 3)([x2, x2, x2, x2, x2, x2, x2, x2])
    

    return x2    
def con160(x2):
    x2 = Concatenate(axis = 3)([x2, x2, x2, x2, x2, x2, x2, x2])
    x2 = Concatenate(axis = 3)([x2, x2, x2, x2,x2])  
    x2 = Concatenate(axis = 3)([x2, x2])  
    x2 = Concatenate(axis = 3)([x2, x2])      
    return x2       

def con1280(x2):
    x2 = Concatenate(axis = 3)([x2, x2, x2, x2, x2, x2, x2, x2])
    x2 = Concatenate(axis = 3)([x2, x2, x2, x2,x2])  
    x2 = Concatenate(axis = 3)([x2, x2])  
    x2 = Concatenate(axis = 3)([x2, x2])   
    x2 = Concatenate(axis = 3)([x2, x2, x2, x2, x2, x2, x2, x2])
    
    return x2       
    
def con320(x2):
    x2 = Concatenate(axis = 3)([x2, x2, x2, x2, x2, x2, x2, x2])
    x2 = Concatenate(axis = 3)([x2, x2, x2, x2,x2])  
    x2 = Concatenate(axis = 3)([x2, x2])  
    x2 = Concatenate(axis = 3)([x2, x2])      
    x2 = Concatenate(axis = 3)([x2, x2])      
    return x2       

def con448(x2):
    x2 = Concatenate(axis = 3)([x2, x2, x2, x2, x2, x2, x2, x2])
    x2 = Concatenate(axis = 3)([x2, x2, x2, x2])  
    x2 = Concatenate(axis = 3)([x2, x2])  
    x2 = Concatenate(axis = 3)([x2, x2, x2, x2, x2, x2, x2])
    
    return x2  
def con288(x2):
    x2 = Concatenate(axis = 3)([x2, x2, x2, x2, x2, x2, x2, x2])
    x2 = Concatenate(axis = 3)([x2, x2])  
    x2 = Concatenate(axis = 3)([x2, x2, x2, x2, x2, x2, x2, x2,x2])
    x2 = Concatenate(axis = 3)([x2, x2])  

    return x2  
    
def InceptionV3changed(include_top=True,
                input_tensor=None,
                input_shape=None,
                pooling=None,):
                
  
    channel_axis = 3
    img_input = Input( shape=(None,None,4),name = 'input')

    x2_ = Lambda(lambda x2_: x2_[:,:,:,3],)(img_input)
    x2_ = reshape_fun(x2_,[-1,299,299,1])

    x = Lambda(lambda x: x[:,:,:,:3],name = 'image')(img_input)
    x = conv2d_bn(x, 32, 3, 3, strides=(2, 2), padding='valid')

    x2 = resize_fun(x2_,[149,149],)
    x2 = con32(x2)
    x = Lambda(lambda_fn, name = 'mask')([x,x2])
  #  x = multiply([x, x2],name = 'mask')    
  #  x = Lambda(lambda x: x * x2)(layer)
#    x = Lambda(x*x2)([x, x2])



#l = keras.layers.Lambda(lambda_fn, output_shape=[(10,), (20,)])
   # x = Lambda(lambda y: y[0] * y[1])([x, x2])
    
    
    x = conv2d_bn(x, 32, 3, 3, padding='valid')

    x2 = resize_fun(x2_,[147,147])
    x2 = con32(x2)
    x = multiply([x, x2])    

    x = conv2d_bn(x, 64, 3, 3)

    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x2 = resize_fun(x2_,[73,73])
    x2 = con64(x2)
    x = multiply([x, x2]) 

    x = conv2d_bn(x, 80, 1, 1, padding='valid')
    x2 = resize_fun(x2_,[73,73])
    x2 = con80(x2)
    x = multiply([x, x2]) 

    x = conv2d_bn(x, 192, 3, 3, padding='valid')
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)  

    x2 = resize_fun(x2_,[35,35])
    x2 = con192(x2)
    x = multiply([x, x2])     

    # mixed 0: 35 x 35 x 256
    branch1x1 = conv2d_bn(x, 64, 1, 1)
    branch5x5 = conv2d_bn(x, 48, 1, 1)

    x2 = resize_fun(x2_,[35,35])
    x2 = con48(x2)
    branch5x5 = multiply([branch5x5, x2]) 

    branch5x5 = conv2d_bn(branch5x5, 64, 5, 5)

    branch3x3dbl = conv2d_bn(x, 64, 1, 1)

    x2 = resize_fun(x2_,[35,35])
    x2 = con64(x2)
    branch3x3dbl = multiply([branch3x3dbl, x2])

    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)

    x2 = resize_fun(x2_,[35,35])
    x2 = con96(x2)
    branch3x3dbl = multiply([branch3x3dbl, x2]) 

    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)

    branch_pool = AveragePooling2D((3, 3),
                                      strides=(1, 1),
                                      padding='same')(x)
                                      
    x2 = resize_fun(x2_,[35,35])
    x2 = con192(x2)
    branch_pool = multiply([branch_pool, x2])

    branch_pool = conv2d_bn(branch_pool, 32, 1, 1)
    x = Concatenate(axis=channel_axis)(
    [branch1x1, branch5x5, branch3x3dbl, branch_pool],)

    x2 = resize_fun(x2_,[35,35])
    x2 = con256(x2)
    x = multiply([x, x2]) 

    # mixed 1: 35 x 35 x 288
    branch1x1 = conv2d_bn(x, 64, 1, 1)

    branch5x5 = conv2d_bn(x, 48, 1, 1)

    x2 = resize_fun(x2_,[35,35])
    x2 = con48(x2)
    branch5x5 = multiply([branch5x5, x2]) 

    branch5x5 = conv2d_bn(branch5x5, 64, 5, 5)

    branch3x3dbl = conv2d_bn(x, 64, 1, 1)

    x2 = resize_fun(x2_,[35,35])
    x2 = con64(x2)
    branch3x3dbl = multiply([branch3x3dbl, x2]) 
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)

    x2 = resize_fun(x2_,[35,35])
    x2 = con96(x2)
    branch3x3dbl = multiply([branch3x3dbl, x2]) 

    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)

    branch_pool = AveragePooling2D((3, 3),
                                      strides=(1, 1),
                                      padding='same')(x)
    x2 = resize_fun(x2_,[35,35])
    x2 = con256(x2)
    branch_pool = multiply([branch_pool, x2])   

    branch_pool = conv2d_bn(branch_pool, 64, 1, 1)
    x = Concatenate(axis=channel_axis)(
    [branch1x1, branch5x5, branch3x3dbl, branch_pool],)



    x2 = resize_fun(x2_,[35,35])
    x2 = con288(x2)
    x = multiply([x, x2],name = '35')  

    # mixed 2: 35 x 35 x 288
    branch1x1 = conv2d_bn(x, 64, 1, 1)

    branch5x5 = conv2d_bn(x, 48, 1, 1)

    x2 = resize_fun(x2_,[35,35])
    x2 = con48(x2)
    branch5x5 = multiply([branch5x5, x2])
    branch5x5 = conv2d_bn(branch5x5, 64, 5, 5)


    branch3x3dbl = conv2d_bn(x, 64, 1, 1)

    x2 = resize_fun(x2_,[35,35])
    x2 = con64(x2)
    branch3x3dbl = multiply([branch3x3dbl, x2])

    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)

    x2 = resize_fun(x2_,[35,35])
    x2 = con96(x2)
    branch3x3dbl = multiply([branch3x3dbl, x2])

    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)

    branch_pool = AveragePooling2D((3, 3),
                                      strides=(1, 1),
                                      padding='same')(x)
                                      
    x2 = resize_fun(x2_,[35,35])
    x2 = con288(x2)
    branch_pool = multiply([branch_pool, x2])                                       

    branch_pool = conv2d_bn(branch_pool, 64, 1, 1)
    x = Concatenate(axis=channel_axis)(
    [branch1x1, branch5x5, branch3x3dbl, branch_pool],)


    x2 = resize_fun(x2_,[35,35])
    x2 = con288(x2)
    x = multiply([x, x2])  

    # mixed 3: 17 x 17 x 768
    branch3x3 = conv2d_bn(x, 384, 3, 3, strides=(2, 2), padding='valid')

    branch3x3dbl = conv2d_bn(x, 64, 1, 1)

    x2 = resize_fun(x2_,[35,35])
    x2 = con64(x2)
    branch3x3dbl = multiply([branch3x3dbl, x2]) 

    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)

    x2 = resize_fun(x2_,[35,35])
    x2 = con96(x2)
    branch3x3dbl = multiply([branch3x3dbl, x2])

    branch3x3dbl = conv2d_bn(
    branch3x3dbl, 96, 3, 3, strides=(2, 2), padding='valid')

    branch_pool = MaxPooling2D((3, 3), strides=(2, 2))(x)
    x = Concatenate(axis=channel_axis)(
    [branch3x3, branch3x3dbl, branch_pool],)

    x2 = resize_fun(x2_,[17,17])
    x2 = con768(x2)
    x = multiply([x, x2])     
    # mixed 4: 17 x 17 x 768
    branch1x1 = conv2d_bn(x, 192, 1, 1)

    branch7x7 = conv2d_bn(x, 128, 1, 1)

    x2 = resize_fun(x2_,[17,17])
    x2 = con128(x2)
    branch7x7 = multiply([branch7x7, x2])

    branch7x7 = conv2d_bn(branch7x7, 128, 1, 7)


    x2 = resize_fun(x2_,[17,17])
    x2 = con128(x2)
    branch7x7 = multiply([branch7x7, x2])

    branch7x7 = conv2d_bn(branch7x7, 192, 7, 1)

    branch7x7dbl = conv2d_bn(x, 128, 1, 1)


    x2 = resize_fun(x2_,[17,17])
    x2 = con128(x2)
    branch7x7dbl = multiply([branch7x7dbl, x2])

    branch7x7dbl = conv2d_bn(branch7x7dbl, 128, 7, 1)

    x2 = resize_fun(x2_,[17,17])
    x2 = con128(x2)
    branch7x7dbl = multiply([branch7x7dbl, x2])

    branch7x7dbl = conv2d_bn(branch7x7dbl, 128, 1, 7)
    x2 = resize_fun(x2_,[17,17])
    x2 = con128(x2)
    branch7x7dbl = multiply([branch7x7dbl, x2])
    branch7x7dbl = conv2d_bn(branch7x7dbl, 128, 7, 1)

    x2 = resize_fun(x2_,[17,17])
    x2 = con128(x2)
    branch7x7dbl = multiply([branch7x7dbl, x2])
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)

    branch_pool = AveragePooling2D((3, 3),
                                      strides=(1, 1),
                                      padding='same')(x)
                                      
    x2 = resize_fun(x2_,[17,17])
    x2 = con768(x2)
    branch_pool = multiply([branch_pool, x2])    

    branch_pool = conv2d_bn(branch_pool, 192, 1, 1)
    x = Concatenate(axis=channel_axis)(
    [branch1x1, branch7x7, branch7x7dbl, branch_pool],
    )

    x2 = resize_fun(x2_,[17,17])
    x2 = con768(x2)
    x = multiply([x, x2]) 
    # mixed 5, 6: 17 x 17 x 768
    for i in range(2):
        branch1x1 = conv2d_bn(x, 192, 1, 1)

        branch7x7 = conv2d_bn(x, 160, 1, 1)
        
        x2 = resize_fun(x2_,[17,17])
        x2 = con160(x2)
        branch7x7 = multiply([branch7x7, x2])
        
        branch7x7 = conv2d_bn(branch7x7, 160, 1, 7)
        x2 = resize_fun(x2_,[17,17])
        x2 = con160(x2)
        branch7x7 = multiply([branch7x7, x2])
        
        branch7x7 = conv2d_bn(branch7x7, 192, 7, 1)

        branch7x7dbl = conv2d_bn(x, 160, 1, 1)
       
        x2 = resize_fun(x2_,[17,17])
        x2 = con160(x2)
        branch7x7dbl = multiply([branch7x7dbl, x2])
        
        branch7x7dbl = conv2d_bn(branch7x7dbl, 160, 7, 1)
        x2 = resize_fun(x2_,[17,17])
        x2 = con160(x2)
        branch7x7dbl = multiply([branch7x7dbl, x2])
        
        branch7x7dbl = conv2d_bn(branch7x7dbl, 160, 1, 7)
       
        x2 = resize_fun(x2_,[17,17])
        x2 = con160(x2)
        branch7x7dbl = multiply([branch7x7dbl, x2])
        
        branch7x7dbl = conv2d_bn(branch7x7dbl, 160, 7, 1)
        
        x2 = resize_fun(x2_,[17,17])
        x2 = con160(x2)
        branch7x7dbl = multiply([branch7x7dbl, x2])
        
        branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)

        branch_pool = AveragePooling2D(
            (3, 3), strides=(1, 1), padding='same')(x)
        
        x2 = resize_fun(x2_,[17,17])
        x2 = con768(x2)
        branch_pool = multiply([branch_pool, x2])
        branch_pool = conv2d_bn(branch_pool, 192, 1, 1)
        x = Concatenate(axis=channel_axis)(
            [branch1x1, branch7x7, branch7x7dbl, branch_pool],)
            
        x2 = resize_fun(x2_,[17,17])
        x2 = con768(x2)
        x = multiply([x, x2])

    x2 = resize_fun(x2_,[17,17])
    x2 = con768(x2)
    x = multiply([x, x2])  
    
    # mixed 7: 17 x 17 x 768
    branch1x1 = conv2d_bn(x, 192, 1, 1)

    branch7x7 = conv2d_bn(x, 192, 1, 1)

    x2 = resize_fun(x2_,[17,17])
    x2 = con192(x2)
    branch7x7 = multiply([branch7x7, x2]) 

    branch7x7 = conv2d_bn(branch7x7, 192, 1, 7)
    x2 = resize_fun(x2_,[17,17])
    x2 = con192(x2)
    branch7x7 = multiply([branch7x7, x2]) 

    branch7x7 = conv2d_bn(branch7x7, 192, 7, 1)

    branch7x7dbl = conv2d_bn(x, 192, 1, 1)

    x2 = resize_fun(x2_,[17,17])
    x2 = con192(x2)
    branch7x7 = multiply([branch7x7, x2]) 
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 7, 1)

    x2 = resize_fun(x2_,[17,17])
    x2 = con192(x2)
    branch7x7 = multiply([branch7x7, x2]) 

    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)

    x2 = resize_fun(x2_,[17,17])
    x2 = con192(x2)
    branch7x7 = multiply([branch7x7, x2]) 
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 7, 1)

    x2 = resize_fun(x2_,[17,17])
    x2 = con192(x2)
    branch7x7 = multiply([branch7x7, x2]) 
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)

    branch_pool = AveragePooling2D((3, 3),
                                      strides=(1, 1),
                                      padding='same')(x)
                                      
    x2 = resize_fun(x2_,[17,17])
    x2 = con768(x2)
    branch_pool = multiply([branch_pool, x2]) 

    branch_pool = conv2d_bn(branch_pool, 192, 1, 1)
    x = Concatenate(axis=channel_axis)(
    [branch1x1, branch7x7, branch7x7dbl, branch_pool],
    )

    x2 = resize_fun(x2_,[17,17])
    x2 = con768(x2)
    x = multiply([x, x2]) 
#################################################
    # mixed 8: 8 x 8 x 1280
    branch3x3 = conv2d_bn(x, 192, 1, 1)
    branch3x3 = conv2d_bn(branch3x3, 320, 3, 3,
                      strides=(2, 2), padding='valid')

    branch7x7x3 = conv2d_bn(x, 192, 1, 1)
    branch7x7x3 = conv2d_bn(branch7x7x3, 192, 1, 7)
    branch7x7x3 = conv2d_bn(branch7x7x3, 192, 7, 1)
    branch7x7x3 = conv2d_bn(
    branch7x7x3, 192, 3, 3, strides=(2, 2), padding='valid')

    branch_pool = MaxPooling2D((3, 3), strides=(2, 2),name = 'last_1')(x)
    x = Concatenate(axis=channel_axis)(
    [branch3x3, branch7x7x3, branch_pool],
    )
    
    # mixed 9: 8 x 8 x 2048
    for i in range(2):
        branch1x1 = conv2d_bn(x, 320, 1, 1) 
        
        branch3x3 = conv2d_bn(x, 384, 1, 1)

        
        branch3x3_1 = conv2d_bn(branch3x3, 384, 1, 3)

        
        
        branch3x3_2 = conv2d_bn(branch3x3, 384, 3, 1)
        
        
        branch3x3 = Concatenate(axis=channel_axis)(
            [branch3x3_1, branch3x3_2],
        )

        branch3x3dbl = conv2d_bn(x, 448, 1, 1)
        
        
        branch3x3dbl = conv2d_bn(branch3x3dbl, 384, 3, 3)
        branch3x3dbl_1 = conv2d_bn(branch3x3dbl, 384, 1, 3)
       
        branch3x3dbl_2 = conv2d_bn(branch3x3dbl, 384, 3, 1)
        branch3x3dbl = Concatenate(axis=channel_axis)(
            [branch3x3dbl_1, branch3x3dbl_2])

        branch_pool = AveragePooling2D(
            (3, 3), strides=(1, 1), padding='same')(x)
        

        
        branch_pool = conv2d_bn(branch_pool, 192, 1, 1)
        x = Concatenate(axis=channel_axis)(
            [branch1x1, branch3x3, branch3x3dbl, branch_pool],
        )
        

    
        
    model = Model(img_input, x)
    return model




# Model inceptionV3  
def inceptionV3(img_rows,img_cols,img_channels,num_classes):   
     IN_SHAPE = (img_rows,img_cols,img_channels)
     #input_tensor = Input(shape=IN_SHAPE)
     base_model = InceptionV3changed(include_top=False,input_shape=IN_SHAPE)

     #base_model = InceptionV3changed(weights= 'imagenet', include_top=False, input_tensor1=img, input_tensor2 = roi_img)
     #input_shape=IN_SHAPE)
     x = base_model.output
     x = GlobalAveragePooling2D()(x)
     predictions = Dense(num_classes, activation='sigmoid')(x)
     model = Model(inputs=base_model.input, outputs=predictions)
     return model
    



     
     