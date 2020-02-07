
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 02:26:43 2019

@author: Ioanna
"""
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";
import glob
# The GPU id to use, usually either "0" or "1";
os.environ["CUDA_VISIBLE_DEVICES"]="1";  
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

X0 = np.load('X.npy')
Y0 = np.load('Y.npy')

X1 = np.load('X_val.npy')
Y1 = np.load('Y_val.npy')

X0 = np.concatenate((X0, X1), axis=0)
Y0 = np.concatenate((Y0, Y1), axis=0)

X0 = minmax_scale(X0, feature_range=(-1, 1), axis=0)

class_weight = 'balanced'
svm = SVC(C = 0.01,kernel='rbf',probability = True,class_weight = class_weight, cache_size=2000, verbose = True)  

svm_model = svm.fit(X0,Y0)
filename = 'RBF_model.sav'
pickle.dump(svm_model, open(filename, 'wb'))




  
