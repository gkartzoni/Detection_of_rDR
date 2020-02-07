
# -*- coding: utf-8 -*-
"""
Created on Aug 2019

@author: Ioanna
"""

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";
import glob
# The GPU id to use, usually either "0" or "1";
os.environ["CUDA_VISIBLE_DEVICES"]="1";  
import tensorflow as tf
from tensorflow import keras

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
keras.backend.set_session(sess)
from sklearn.utils import shuffle
from keras.models import Model
import numpy as np
import cv2
from keras.applications.inception_v3 import preprocess_input
import model_functions as mf
from keras import activations
from vis.visualization import visualize_cam
from vis.visualization import visualize_saliency,overlay
from vis.utils import utils

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

img_rows,img_cols,img_channels = 299 ,299,3
model = mf.load_model(r'/home/gkartzoni/thesis/experiments/expiriment_old/54/TrainingProcess/model.h5')
model_ = model
print("Remove Activation from Last Layer")
# Swap softmax with linear
model.layers[312].activation = activations.linear
print("Done. Now Applying changes to the model ...")
model = utils.apply_modifications(model)
layer_idx = 312

# File path
file_extension  = '*.jpeg'
image_path0 = '/home/gkartzoni/thesis/images/test/0'
image_path1 = '/home/gkartzoni/thesis/images/test/1'

#file_extension  = '*.jpg' #For messidor 2 use '*.tif' jpeg
#image_path0 = '/home/gkartzoni/thesis/images/2classesMessidor2_RDR/class0'
#image_path1 =  '/home/gkartzoni/thesis/images/2classesMessidor2_RDR/class1'

path2save0 = '/home/gkartzoni/thesis/images/visualize_camΚ/0/'
path2save1 = '/home/gkartzoni/thesis/images/visualize_camΚ/1/'

# data_path = os.path.join(image_path0,file_extension)
# files = glob.glob(data_path)
num_of_img = 500


i = 1
data_path = os.path.join(image_path1,file_extension)
files = glob.glob(data_path)

for image_file in files:
    normalizedImg = np.zeros([img_rows, img_cols, 3])
    if i == num_of_img:
        break
    img = cv2.imread(image_file)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB )   
    img_original = img 
    normalizedImg = cv2.normalize(img,  normalizedImg, 0, 255, cv2.NORM_MINMAX)
    img = normalizedImg
    img  =  img.astype(np.float32)    
    img = preprocess_input(img)
    X = np.asarray(img)
    X = np.reshape(X,[-1,299,299,3])
    y_pred = (model_.predict(X))
    y_pred = np.asscalar(y_pred)    
    y_pred = (round(y_pred,2))
    heatmap = visualize_cam(model, layer_idx, filter_indices=0, seed_input=img)
    
    name = os.path.basename(image_file)
    filename, file_extension = os.path.splitext(name)
    plt.subplots_adjust(left=0, bottom=0, right=2, top=5, wspace=-1, hspace=0)

    plt.figure(figsize=(15,15))
    plt.subplot(1,2,1)
    plt.axis('off') 

    plt.imshow(img_original)
    plt.subplot(1,2,2)
    plt.axis('off') 

   # plt.imshow(overlay(img_original, img_original))
    plt.imshow(overlay(img_original, heatmap))
    #plt.savefig(path2save1 + str(y_pred)+'_'+ str(i)+'.png')
    plt.savefig(path2save1+'_'+str(y_pred) +'_'+ filename +'.png')   
  #  plt.savefig(path2save1+'_'+ filename +'.png')   
    i = i + 1
    
    
    
    
    
