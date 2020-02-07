# -*- coding: utf-8 -*-


import os
#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";

# The GPU id to use, usually either "0" or "1";
os.environ["CUDA_VISIBLE_DEVICES"]="0";  
import tensorflow as tf
from tensorflow import keras

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
keras.backend.set_session(sess)

from keras.applications.inception_v3 import preprocess_input
from model_inceptionV3 import inceptionV3

import numpy as np
import model_functions as mf
from keras.utils import multi_gpu_model
import json
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD, Adam,RMSprop
from keras.callbacks import EarlyStopping, ModelCheckpoint
import math as m
from shutil import copyfile #copyfile(src, dst)
import matplotlib
matplotlib.use('Agg')
from skimage.exposure import equalize_hist
import cv2
from matplotlib import pyplot as plt

from keras.utils import plot_model
import random
from random import shuffle


experiment_name = '26'
###################Set experiment#####################
################Before Starting########################
#train/retrain parameter
#optimizer
#lr decay
#experiment_name
# pathVal pathTrain
#nb_epoch
# nb_train_samples, nb_validation_samples prosoxi analoga me ta samples na to alalzo
#class_weight energopoihmenh
# allazo to sosto arxeio
# kano save to arxeio
# data Img generator - >  epilogi metatropon
# processing function kai sto train kai sto valid prepei na einai idia
# num_classes-> sto inceptionV3 exo mia eksodo

img_rows,img_cols,img_channels = 299 ,299,3
#nb_train_samples, nb_validation_samples, =58461, 3704
#nb_train_samples, nb_validation_samples, =190878, 15904
#nb_train_samples, nb_validation_samples, =72420, 3704
#nb_train_samples, nb_validation_samples, =63626, 6152
nb_train_samples, nb_validation_samples, =63128, 6152

#nb_train_samples, nb_validation_samples, =24618, 15904
nb_epoch, batch_size = 40, 8
num_classes = 1
lr, decay = 0.5 * 1e-3,0
 #4e-5

pathTrain = '/home/gkartzoni/thesis/images/3channel_train'
pathVal = '/home/gkartzoni/thesis/images/3channel_validation'

from sklearn.metrics import roc_auc_score
import tensorflow as tf
from sklearn.metrics import roc_auc_score
        

n = 1
mode = 'train'
if mode == 'train':
    experiment_file = '/home/gkartzoni/thesis/experiments/'+ experiment_name
    os.mkdir(experiment_file)
    experiment_file = '/home/gkartzoni/thesis/experiments/'+ experiment_name+'/TrainingProcess'
    try:  
        os.mkdir(experiment_file)
    except OSError:  
        print ("Creation of the directory %s failed" % experiment_file)
    else:  
        print ("Successfully created the directory %s " % experiment_file)
    json_file = experiment_file + '/log.json'
    model = inceptionV3(img_rows,img_cols,img_channels,num_classes)

elif mode == 'retrain':
    if n == 1:
        experiment_file = '/home/gkartzoni/thesis/experiments/'+ experiment_name+'/ReTrainingProcess'+str(n)
        try:  
            os.mkdir(experiment_file)
        except OSError:  
            print ("Creation of the directory %s failed" % experiment_file)
        else:  
            print ("Successfully created the directory %s " % experiment_file)
        path = '/home/gkartzoni/thesis/experiments/'+ experiment_name+'/TrainingProcess/model.h5'
        print('Load this model path: ')
        print(path)    
        model = mf.model_load(path)

    else:
        
        experiment_file = '/home/gkartzoni/thesis/experiments/'+ experiment_name+'/ReTrainingProcess'+str(n)
        try:  
            os.mkdir(experiment_file)
        except OSError:  
            print ("Creation of the directory %s failed" % experiment_file)
        else:  
            print ("Successfully created the directory %s " % experiment_file)
        path = '/home/gkartzoni/thesis/experiments/'+ experiment_name+'/ReTrainingProcess'+str(n-1) +'/model.h5' 
        print('Load this model path: ')
        print(path)        
        model = mf.model_load(path)

json_file = experiment_file + '/log.json'    
src = '/home/gkartzoni/thesis/source/runInceptionV3.py'
dst = experiment_file + '/runInceptionV3.py'
copyfile(src, dst)  
path_modelsave = experiment_file + '/model.h5'
      

def preprocessing_fun(img):       
        rotate_im = random.randint(1,2)         
        if rotate_im == 1:
            angle = random.randint(0,360)
            rotation_matrix = cv2.getRotationMatrix2D((img_rows/2, img_cols/2), angle, 1)
            img = cv2.warpAffine(img, rotation_matrix, (img_rows, img_cols))
        img  =  img.astype(np.uint8)
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        lab_planes = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))
        lab_planes[0] = clahe.apply(lab_planes[0])
        lab = cv2.merge(lab_planes)
        img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        img  =  img.astype(np.float32)
        return img
         
def preprocessing_fun_val(img):      
         
        img  =  img.astype(np.uint8)
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        lab_planes = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))
        lab_planes[0] = clahe.apply(lab_planes[0])
        lab = cv2.merge(lab_planes)
        img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        img  =  img.astype(np.float32)
        return img

         
train_gen  = ImageDataGenerator(
    #rotation_range=360,
   # fill_mode = "constant", # Use this for fill with 1-> black when rotate
   # cval = 0,
    vertical_flip = True,
    horizontal_flip = True,
    preprocessing_function = preprocessing_fun,
    rescale=1./255,

)
val_gen = ImageDataGenerator(

   # featurewise_center=True,
   # featurewise_std_normalization=True,
    preprocessing_function = preprocessing_fun_val,
    rescale=1./255,

    
)

train_generator = train_gen.flow_from_directory(
        pathTrain,
        target_size=(img_rows, img_cols),
        batch_size=batch_size,
        shuffle=True,
        class_mode='binary'
    )

validation_generator = val_gen.flow_from_directory(
        pathVal,
        target_size=(img_rows, img_cols),
        batch_size=batch_size,
        class_mode='binary'
     )


# define the name of the directory to be created
image_path = experiment_file + '/images/'
model_architecture_path = experiment_file + '/model_architecture.png'
try:  
    os.mkdir(image_path)
except OSError:  
    print ("Creation of the directory %s failed" % image_path)
else:  
    print ("Successfully created the directory %s " % image_path)
mf.save_generator_images(train_generator, batch_size,image_path)

model.compile(
    optimizer = SGD(lr=lr,decay=decay),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

callbacks = [ModelCheckpoint(filepath = path_modelsave, monitor='val_loss', save_best_only=True)]

#class_weight = {0:1, 1:4.194}
class_weight = {0:1, 1:4.16906}

history = model.fit_generator(
          train_generator,
          steps_per_epoch = m.ceil(nb_train_samples // batch_size),
          nb_epoch=nb_epoch,
          callbacks=callbacks,
          validation_data=validation_generator,
          validation_steps=m.ceil(nb_validation_samples // batch_size),
          verbose=1,
          class_weight=class_weight,

          )

          
with open(json_file, 'w') as f:
    json.dump(history.history, f)
    
# summarize history for loss
plt.plot(history.history['loss'],color='b')
plt.plot(history.history['val_loss'],color='r')
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
plt.savefig(experiment_file+'/loss.png', bbox_inches='tight')
plt.clf()

# summarize history for accuracy
plt.plot(history.history['acc'],color='b')
plt.plot(history.history['val_acc'],color='r')
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
plt.savefig(experiment_file+'/acc.png', bbox_inches='tight')
plt.clf()
