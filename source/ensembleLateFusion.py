
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

from keras import backend as K
from keras import layers
from keras.layers import Input
from keras.models import Model

import json
import cv2
import numpy as np
import keras
from skimage.exposure import equalize_hist
from keras.applications.vgg16 import preprocess_input
import model_functions as mf
import glob
from shutil import copyfile #copyfile(src, dst)
from keras.applications.inception_v3 import preprocess_input
from keras.optimizers import SGD
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report,precision_recall_curve, confusion_matrix, auc,roc_curve


experiment_name = 'Last'


lr, decay = 1e-3, 4e-5

from keras.models import load_model
models=[]
ensemble_path = '/home/gkartzoni/thesis/ensemble/EF9' 


path, dirs, files = next(os.walk(ensemble_path))
file_count = len(files)

for i in range(0,file_count):
    modelTemp=load_model(ensemble_path +  '/' + str(i) + '.h5') 
    modelTemp.name = 'model'+ str(i)
    models.append(modelTemp)
    

img_rows,img_cols,img_channels = 299 ,299,3

       
evaluation_folder = '/home/gkartzoni/thesis/experiments/ensemble'+experiment_name
try:  
    os.mkdir(evaluation_folder)
except OSError:  
    print ("Creation of the directory %s failed" % evaluation_folder)
else:  
    print ("Successfully created the directory %s " % evaluation_folder)


def endsemble(models,select,folder2save,weighted,experiment_name):
    img_rows,img_cols,img_channels = 299 ,299,3  
    if select == 0:
      image_path0 =  '/home/gkartzoni/thesis/images/test/0'
      image_path1 =   '/home/gkartzoni/thesis/images/test/1'
      file_extension  = '*.jpeg' #For messidor 2 use '*.tif' jpeg
    elif select == 1:
       image_path0 = '/home/gkartzoni/thesis/images/2classesMessidor2_RDR/class0'
       image_path1 =  '/home/gkartzoni/thesis/images/2classesMessidor2_RDR/class1'
       file_extension  = '*.jpg' #For messidor 2 use '*.tif' jpeg

    evaluation_folder = '/home/gkartzoni/thesis/experiments/'+ folder2save+experiment_name
    try:  
        os.mkdir(evaluation_folder)
    except OSError:  
        print ("Creation of the directory %s failed" % evaluation_folder)
    else:  
        print ("Successfully created the directory %s " % evaluation_folder)
       
    num_classes = 1
    X = []
    Y = []



    data_path = os.path.join(image_path0,file_extension)
    files = glob.glob(data_path)
    normalizedImg = np.zeros((img_rows, img_cols,3))

    i = 1;
    for image_file in files:
        normalizedImg = np.zeros([img_rows, img_cols, 3])
        img = cv2.imread(image_file)
        img = cv2.resize(img, dsize=(img_rows, img_cols), interpolation = cv2.INTER_CUBIC)  
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB )   #img = cv2.imread(image_file)
        normalizedImg = cv2.normalize(img,  normalizedImg, 0, 255, cv2.NORM_MINMAX)
        img = normalizedImg
        img  =  img.astype(np.float32)    
        img = preprocess_input(img)
        X.append(img)  #  if i<2:    
        Y.append(0)




    data_path = os.path.join(image_path1,file_extension)
    files = glob.glob(data_path)
     # #for i in range(1,len(image_name)+1):
    i = 1;
    for image_file in files:
        normalizedImg = np.zeros([img_rows, img_cols, 3])
        img = cv2.imread(image_file)
        img = cv2.resize(img, dsize=(img_rows, img_cols), interpolation = cv2.INTER_CUBIC)  
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB )   #img = cv2.imread(image_file)
        normalizedImg = cv2.normalize(img,  normalizedImg, 0, 255, cv2.NORM_MINMAX)
        img = normalizedImg
        img  =  img.astype(np.float32)    
        img = preprocess_input(img)
        X.append(img)  #  if i<2:    
        Y.append(1)


    # for i in range(0,2):
        # matplotlib.image.imsave(evaluation_folder+'/'+str(i)+'.png', X[i])   
     

    X = np.asarray(X)
    Y = np.asarray(Y)

    metrics = []
    l = len(models)
    l_x = len(X)

    y_pred = np.zeros((l_x,l))
    thresholds =   np.arange(0, 1, 0.002)
    #  threshold = numpy.arange([0, ]1, [0.01,])
    i = 0
    for model in models: 
        tmp = model.predict(X)
        y_pred[:,i] = np.reshape(tmp,[len(tmp)])
        i = i + 1

    del models 
    if weighted == 1:
    
        iter_thres = 0
        fpr = np.zeros_like(thresholds)  
        tpr = np.zeros_like(thresholds)  
        coef = [0.5,0.5,0.5,0.5,2,2,3.5,0.5]

        for threshold in thresholds:
                y_pred_ = np.zeros_like(y_pred[:,0])  
                for iter in range(0,l):
                    temp_val = ([1 if x >= threshold else 0 for x in y_pred[:, iter]]) * np.asarray(coef[iter])
                    y_pred_ = np.add(temp_val,y_pred_)
                y_pred_ = [1 if x >= (5) else 0 for x in y_pred_]
                c = confusion_matrix(Y,y_pred_)
                fpr[iter_thres] = 1 - (float(c[0, 0]) / (float(c[0, 1]) + float(c[0, 0])))
                tpr[iter_thres] =  float(c[1, 1]) / (float(c[1, 1]) + float(c[1, 0])) 
                iter_thres = iter_thres + 1 
        roc_auc = auc(fpr, tpr)
    elif weighted == 0: 
        iter_thres = 0
        fpr = np.zeros_like(thresholds)  
        tpr = np.zeros_like(thresholds) 
        for threshold in thresholds:
                y_pred_ = np.zeros_like(y_pred[:,0])  
                for iter in range(0,l):
                    y_pred_ = np.add([1 if x >= threshold else 0 for x in y_pred[:, iter]],y_pred_)
                y_pred_ = [1 if x >= 5 else 0 for x in y_pred_]
                c = confusion_matrix(Y,y_pred_)
                fpr[iter_thres] = 1 - (float(c[0, 0]) / (float(c[0, 1]) + float(c[0, 0])))
                tpr[iter_thres] =  float(c[1, 1]) / (float(c[1, 1]) + float(c[1, 0])) 
                iter_thres = iter_thres + 1 
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
       'threshold':thresholds,}, cls=NumpyEncoder,)    
    return json_metrics    

################################################################
folder2save = 'Kaggle_LF_notWeighted'
json_metrics =  endsemble(models,0,folder2save,0,experiment_name)
evaluation_folder = '/home/gkartzoni/thesis/experiments/'+ folder2save+experiment_name

#AUC
json_metrics = json.loads(json_metrics)
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

src = '/home/gkartzoni/thesis/source/ensembleLateFusion.py'
dst = evaluation_folder + '/ensembleLateFusion.py'
copyfile(src, dst)	

with open(evaluation_folder+'/metrics.json', 'w') as json_file:  
     json.dump(json_metrics, json_file)
#######################################################################
folder2save = 'Messidor_LF_notWeighted'
json_metrics = endsemble(models,1,folder2save,0,experiment_name)
evaluation_folder = '/home/gkartzoni/thesis/experiments/'+ folder2save+experiment_name

#AUC
json_metrics = json.loads(json_metrics)
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

src = '/home/gkartzoni/thesis/source/ensembleLateFusion.py'
dst = evaluation_folder + '/ensembleLateFusion.py'
copyfile(src, dst)	

with open(evaluation_folder+'/metrics.json', 'w') as json_file:  
     json.dump(json_metrics, json_file)
     
# #############################################
# folder2save = 'Kaggle_LF_Weighted'
# json_metrics = endsemble(models,0,folder2save,1,experiment_name)
# evaluation_folder = '/home/gkartzoni/thesis/experiments/'+ folder2save+experiment_name

# #AUC
# json_metrics = json.loads(json_metrics)
# plt.plot(json_metrics['fpr'],json_metrics['tpr'], 'b', label = 'AUC = %0.2f' %json_metrics['roc_auc'])
# plt.title('Receiver Operating Characteristic')
# plt.legend(loc = 'lower right')
# plt.plot([0, 1], [0, 1],'r--')
# plt.xlim([0, 1])
# plt.ylim([0, 1])
# plt.ylabel('True Positive Rate')
# plt.xlabel('False Positive Rate')
# plt.show()
# plt.savefig(evaluation_folder+'/AUC.png', bbox_inches='tight')
# plt.clf()

# src = '/home/gkartzoni/thesis/source/ensembleLateFusion.py'
# dst = evaluation_folder + '/ensembleLateFusion.py'
# copyfile(src, dst)

# #################################################
# folder2save = 'Messidor_LF_Weighted'
# json_metrics =  endsemble(models,1,folder2save,1,experiment_name) 
# evaluation_folder = '/home/gkartzoni/thesis/experiments/'+ folder2save+experiment_name
   
# #AUC
# json_metrics = json.loads(json_metrics)
# plt.plot(json_metrics['fpr'],json_metrics['tpr'], 'b', label = 'AUC = %0.2f' %json_metrics['roc_auc'])
# plt.title('Receiver Operating Characteristic')
# plt.legend(loc = 'lower right')
# plt.plot([0, 1], [0, 1],'r--')
# plt.xlim([0, 1])
# plt.ylim([0, 1])
# plt.ylabel('True Positive Rate')
# plt.xlabel('False Positive Rate')
# plt.show()
# plt.savefig(evaluation_folder+'/AUC.png', bbox_inches='tight')
# plt.clf()

# src = '/home/gkartzoni/thesis/source/ensembleLateFusion.py'
# dst = evaluation_folder + '/ensembleLateFusion.py'
# copyfile(src, dst)




