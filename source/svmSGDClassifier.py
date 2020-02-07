
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
from sklearn.preprocessing import minmax_scale

from keras.applications.inception_v3 import preprocess_input
img_rows,img_cols,img_channels = 299 ,299,3


from sklearn.linear_model import SGDClassifier
from keras.models import load_model

class_weight = {0:1, 1:4.16906}
X0 = np.load('X.npy')
Y0 = np.load('Y.npy')

X1 = np.load('X_val.npy')
Y2 = np.load('Y_val.npy')

X0 = np.concatenate((X0, X1), axis=0)
Y0 = np.concatenate((Y0, Y1), axis=0)


max_it = 1000000

from sklearn.kernel_approximation import RBFSampler

s = SGDClassifier(loss='log',loss = hinge)
X0_ = minmax_scale(X0, feature_range=(-1, 1), axis=1)       
SGDmodel = s.fit(X0, Y0)

filename = 'model_linear.sav'
pickle.dump(SGDmodel, open(filename, 'wb'))   
       # linear
# s = SGDClassifier(alpha=0.0001, average=False, class_weight=class_weight,
       # early_stopping=False, epsilon=0.1, eta0=0.0, fit_intercept=True,
       # l1_ratio=0.15, learning_rate='optimal', loss='log', max_iter=10000,
       # n_iter_no_change=1000, n_jobs=None, penalty='l2', power_t=0.5,
       # random_state=None, shuffle=True, tol=1e-3, validation_fraction=0.1,
       # verbose=1, warm_start=True)
       
# SGDmodel = s.fit(X0, Y0)
#filename = 'modelSVM_SGD_log.sav'
#pickle.dump(SGDmodel, open(filename, 'wb'))
