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
import model_functions as mf

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
import json

import cv2

normalizedImg = np.zeros((img_rows, img_cols,3))
import matplotlib.pyplot as plt
from sklearn import preprocessing
from PIL import Image
from matplotlib import cm
import random
import matplotlib.image as mpimg 
import matplotlib.pyplot as plt 
from sklearn.utils import shuffle
import pickle
from sklearn.linear_model import SGDClassifier
from sklearn.kernel_approximation import RBFSampler
from sklearn.preprocessing import minmax_scale
from sklearn.metrics import roc_curve

from keras.applications.inception_v3 import preprocess_input
img_rows,img_cols,img_channels = 299 ,299,3


#X0 = X0[:,1:101]
#Y0 = Y0
from sklearn.metrics import classification_report,precision_recall_curve, confusion_matrix, auc,roc_curve

from sklearn.svm import libsvm
from sklearn import svm

#model = pickle.load(open('model_RBF.sav', 'rb'))
model = pickle.load(open('RBF_model.sav', 'rb'))

dataset = 2
if dataset == 1:
    evaluation_folder = '/home/gkartzoni/thesis/experiments/Eval_SVM_K_RBF/'
   # evaluation_folder = '/home/gkartzoni/thesis/experiments/Eval_SVM_K_RBF/'
    X0 = np.load('X_K.npy')
    Y0 = np.load('Y_K.npy')

elif dataset == 2:
    evaluation_folder = '/home/gkartzoni/thesis/experiments/Eval_SVM_M_RBF/'
   # evaluation_folder = '/home/gkartzoni/thesis/experiments/Eval_SVM_M_RBF/'
    X0 = np.load('X_M.npy')
    Y0 = np.load('Y_M.npy')
    
X0 = minmax_scale(X0, feature_range=(-1, 1), axis=0)
#result = svm.libsvm.predict(X0,*loaded_model)
X = X0
Y = Y0
metrics = []
# y_pred = (model.predict_proba(X))
# y_pred = y_pred.astype('float16')
#y_pred_ = (model.predict(X))
y_pred_ = (model.predict_proba(X))
#y_pred_ = svm.libsvm.predict(X0,*loaded_model)

#c = confusion_matrix(Y,y_pred_)
#sensitivity  = float(c[1, 1]) / (float(c[1, 1]) + float(c[1, 0]))    
#specificity = float(c[0, 0]) / (float(c[0, 1]) + float(c[0, 0]))
# print(sensitivity)
# print(specificity)
#   print( np.argmax(y_pred, axis=1))
#   print(y_pred)

#fpr, tpr, threshold = roc_curve(Y, np.argmax(y_pred_, axis=1))
fpr, tpr, threshold = roc_curve(Y, y_pred_[:,1])

roc_auc = auc(fpr, tpr)

class NumpyEncoder(json.JSONEncoder):
   def default(self, obj):
       if isinstance(obj, np.ndarray):
           return obj.tolist()
       return json.JSONEncoder.default(self, obj)

json_metrics = json.dumps({     
   'roc_auc': roc_auc,
   'fpr':fpr,
   'tpr':tpr,
   #'sensitivity':sensitivity,
   #'specificity':specificity,
   #'y_pred':y_pred_[1:100,:],
   'threshold':threshold,}, cls=NumpyEncoder,) 
   #'y_pred':y_pred[:]}, cls=NumpyEncoder)         
# json_metrics = mf.model_evaluation_1classOutputSVM(loaded_model,X0,Y0,num_classes,evaluation_folder)



# # #AUC
json_metrics = json.loads(json_metrics)
print(json_metrics['roc_auc'])
#print(json_metrics['sensitivity'])
#print(json_metrics['specificity'])

plt.plot(json_metrics['fpr'],json_metrics['tpr'], 'b', label = 'AUC = %0.2f' %json_metrics['roc_auc'])
plt.title('Receiver Operating Characteristic')
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
plt.savefig(evaluation_folder+'/AUC.png', bbox_inches='tight')
plt.clf()


with open(evaluation_folder+'/metrics.json', 'w') as json_file:  
     json.dump(json_metrics, json_file)