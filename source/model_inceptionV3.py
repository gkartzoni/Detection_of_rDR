#!/usr/bin/python
# -*- coding: utf-8 -*-

from keras.layers.core import Dropout
from keras.applications.inception_v3 import InceptionV3
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, BatchNormalization, Dropout, GlobalMaxPooling2D
from keras.layers import Input
#keras.applications.inception_resnet_v2.InceptionResNetV2(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)

# Model inceptionV3  
def inceptionV3(img_rows,img_cols,img_channels,num_classes):     
     IN_SHAPE = (img_rows,img_cols,img_channels)
     input_tensor = Input(shape=IN_SHAPE)    
     base_model = InceptionV3(weights= 'imagenet', include_top=False,input_shape=IN_SHAPE)
     #base_model = InceptionV3(weights= 'imagenet', include_top=False,input_shape=IN_SHAPE)

     # base_model = InceptionResNetV2(weights= 'imagenet', include_top=False,input_shape=IN_SHAPE)
     x = base_model.output
     x = GlobalAveragePooling2D(name='gap')(x)
    # x = Dropout(0.5)(x)
   #  x = Dense(1024, activation='relu')(x)
     predictions = Dense(num_classes, activation='sigmoid')(x)
     model = Model(inputs=base_model.input, outputs=predictions)
     return model
    



     
     