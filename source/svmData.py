
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 02:26:43 2019

@author: Ioanna
"""
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";
import glob
# The GPU id to use, usually either "0" or "1";
os.environ["CUDA_VISIBLE_DEVICES"]="0";  
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import minmax_scale
from sklearn.metrics import classification_report,precision_recall_curve, confusion_matrix, auc,roc_curve

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
keras.backend.set_session(sess)
from sklearn.utils import shuffle
from keras.models import Model
from model_inceptionV3 import inceptionV3
from sklearn.svm import SVC
import numpy as np
nb_epoch, batch_size = 20, 32
num_classes = 1
img_rows,img_cols,img_channels = 299 ,299,3
from keras.utils import to_categorical

import cv2
import matplotlib.pyplot as plt
from sklearn import preprocessing
from PIL import Image
from matplotlib import cm
import random
import matplotlib.image as mpimg 
import matplotlib.pyplot as plt 
from sklearn.utils import shuffle
import pickle
from sklearn.svm import libsvm



from keras.applications.inception_v3 import preprocess_input
img_rows,img_cols,img_channels = 299 ,299,3

file_extension  = '*.jpg'


path0 = '/home/gkartzoni/thesis/images/svm/0'
path1 = '/home/gkartzoni/thesis/images/svm/0'

# PATHS
#image_path_train0 = '/home/gkartzoni/thesis/images/train/0'
#image_path_train1 = '/home/gkartzoni/thesis/images/train/1'

#image_path_validation0 = '/home/gkartzoni/thesis/images/validation/0'
#image_path_validation0 = '/home/gkartzoni/thesis/images/validation/1'

#image_path_train0 = '/home/gkartzoni/thesis/images/test/0'
#image_path_train1 = '/home/gkartzoni/thesis/images/test/1'
image_path_train0 = '/home/gkartzoni/thesis/images/2classesMessidor2_RDR/class0'
image_path_train1 =  '/home/gkartzoni/thesis/images/2classesMessidor2_RDR/class1'

# #image_path = r'G:\images\imageTrain\originaltrain\4'
model = inceptionV3(img_rows,img_cols,img_channels,num_classes)
model.load_weights(r'/home/gkartzoni/thesis/experiments/55/TrainingProcess/model.h5')
model_feat = Model(inputs=model.input,outputs=model.get_layer('gap').output)


# FIND DENSE LAYER VECTORS
path = image_path_train0 
data_path = os.path.join(path,file_extension)
files = glob.glob(data_path)
X0 = []
Y0 = []
class_ = 0
feat_train_X = []
for image_file in files:
    Y0.append(class_)
    
    normalizedImg = np.zeros([img_rows, img_cols, 3])
    img = cv2.imread(image_file)
    img = cv2.resize(img, dsize=(img_rows, img_cols), interpolation = cv2.INTER_CUBIC)  
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB )   #img = cv2.imread(image_file)
    normalizedImg = cv2.normalize(img,  normalizedImg, 0, 255, cv2.NORM_MINMAX)
    img = normalizedImg
    img  =  img.astype(np.float32)    
    img = preprocess_input(img)
    
    img = np.reshape(img,[1,img_rows,img_cols,3])
    feat_train_X = model_feat.predict(np.asarray(img))    
    X0.append(feat_train_X)  #  if i<2:    
#
X0 = np.asarray(X0)
Y0 = np.asarray(Y0)

dataset_size = len(X0)
X0 = X0.reshape(dataset_size,-1)
##############
path = image_path_train1
data_path = os.path.join(path,file_extension)
files = glob.glob(data_path)

X1 = []
Y1 = []
class_ =1
feat_train_X = []
for image_file in files:
    Y1.append(class_)
    
    normalizedImg = np.zeros([img_rows, img_cols, 3])
    img = cv2.imread(image_file)
    img = cv2.resize(img, dsize=(img_rows, img_cols), interpolation = cv2.INTER_CUBIC)  
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB )   #img = cv2.imread(image_file)
    normalizedImg = cv2.normalize(img,  normalizedImg, 0, 255, cv2.NORM_MINMAX)
    img = normalizedImg
    img  =  img.astype(np.float32)    
    img = preprocess_input(img)
    
    img = np.reshape(img,[1,img_rows,img_cols,3])
    feat_train_X = model_feat.predict(np.asarray(img))    
    X1.append(feat_train_X)  #  if i<2:    
#
X1 = np.asarray(X1)
Y1 = np.asarray(Y1)

dataset_size = len(X1)
X1 = X1.reshape(dataset_size,-1)

X0 = np.concatenate((X0, X1), axis=0)
Y0 = np.concatenate((Y0, Y1), axis=0)

X1 = []
Y1 = []

p = np.random.permutation(len(X0))
X0 = X0[p] 
Y0 = Y0[p]
from sklearn.preprocessing import minmax_scale

np.save('X_M.npy', X0)
np.save('Y_M.npy', Y0)


