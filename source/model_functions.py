#!/usr/bin/python
# -*- coding: utf-8 -*-
from operator import add

import operator
from keras.models import load_model
from sklearn.metrics import classification_report,precision_recall_curve, confusion_matrix, auc,roc_curve
import keras
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from PIL import Image
import json
from sklearn.metrics import roc_curve
import tensorflow as tf

def model_save(path,model):
    model.save(path) 

def model_load(path):
    model = load_model(path)
    return model

def model_print_weights(model,nb_layer):
    weights = model.layers[nb_layer].get_weights() # list of numpy array
    print(weights)
  
def save_generator_images(generator, batchsize,path):
    x,y = generator.next()
    #print(y)
    for i in range(0,batchsize):
        image = x[i]
        matplotlib.image.imsave(path+str(i)+'.png', image)
#       print(image.min())
#        print(image.max())
def save_generator_imagesGray(generator, batchsize,path,image_size):
    x,y = generator.next()
    #x = np.array(x, dtype=np.float64) 
    #print(y)
    for i in range(0,batchsize):
        image = x[i]
        image =  np.reshape(image, (image_size, image_size))
        matplotlib.image.imsave(path+str(i)+'.png', image, cmap='gray')

    
def model_save_history(path,history):
    with open(path, 'w') as f:
        json.dump(history.history, f)
    

# def model_evaluation(model,X,Y,num_classes,evaluation_folder):  
 
# # metrics_file = open(evaluation_folder+"/metrics.txt","w")
# # metrics_file.write("This is a test\n") 
 


def model_evaluation(model,X,Y,num_classes,evaluation_folder):  
     
    Yhot = keras.utils.to_categorical(Y,num_classes)
    metrics = []
    score = model.evaluate(X,Yhot)
    y_pred = (model.predict(X))
    
    
    c = confusion_matrix(Y, np.argmax(y_pred, axis=1))
    
    sensitivity  = float(c[1, 1]) / (float(c[1, 1]) + float(c[1, 0]))    
    specificity = float(c[0, 0]) / (float(c[0, 1]) + float(c[0, 0]))
    print(sensitivity)
    print(specificity)

    fpr, tpr, threshold = roc_curve(Y, np.argmax(y_pred, axis=1))
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
       'threshold':threshold,
       'sensitivity':sensitivity,
       'specificity':specificity,
       'Test score':score[0],
       'Test accuracy':score[1],
       'confusion_matrix':c,
       'y_pred':y_pred[1][:]}, cls=NumpyEncoder
    )
       
       
   # json_metrics = json.dumps({'a': a, 'aa': [2, (2, 3, 4), a], 'bb': [2]}, cls=NumpyEncoder)
    return json_metrics
    
    
    
# def model_evaluation_1classOutput(model,X,Y,num_classes,evaluation_folder):  
    # metrics = []
    # y_pred = (model.predict(X))
    # #print(y_pred)
    # rounded = [round(x[0]) for x in y_pred]
    # y_pred1 = np.array(rounded,dtype='int64')
    # c = confusion_matrix(Y,y_pred1)
    # score = model.evaluate(X,y_pred1)

    # sensitivity  = float(c[1, 1]) / (float(c[1, 1]) + float(c[1, 0]))    
    # specificity = float(c[0, 0]) / (float(c[0, 1]) + float(c[0, 0]))
    # print(sensitivity)
    # print(specificity)
    
    # fpr, tpr, threshold = roc_curve(Y, y_pred)
    # roc_auc = auc(fpr, tpr)
   
    # max_indexSpec, max_valueSpec = max(enumerate(fpr), key=operator.itemgetter(1))
    # max_valueSpec
    # max_indexSens, max_valueSens = max(enumerate(tpr), key=operator.itemgetter(1))
   

  
    # class NumpyEncoder(json.JSONEncoder):
       # def default(self, obj):
           # if isinstance(obj, np.ndarray):
               # return obj.tolist()
           # return json.JSONEncoder.default(self, obj)

    # json_metrics = json.dumps({     
       # 'roc_auc': roc_auc,
       # 'fpr':fpr,
       # 'tpr':tpr,
       # 'threshold':threshold,
       # 'sensitivity':sensitivity,
       # 'specificity':specificity,
       # 'Test score':score[0],
       # 'Test accuracy':score[1],
       # 'confusion_matrix':c,
       # 'y_pred':y_pred[1][:]}, cls=NumpyEncoder,
    # )         
   # # json_metrics = json.dumps({'a': a, 'aa': [2, (2, 3, 4), a], 'bb': [2]}, cls=NumpyEncoder)
    # return json_metrics
    
    
def model_evaluation_1classOutput(model,X,Y,num_classes,evaluation_folder):  
    metrics = []

    
    y_pred = (model.predict(X))

  #  y_pred = list( map(add, y_pred, (model.predict(X)) ) )
    fpr, tpr, threshold = roc_curve(Y, y_pred)
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
       'threshold':threshold,}, cls=NumpyEncoder,)     
    return json_metrics
    
def model_evaluation_1classOutputGenerator(model,test_generator,totalTest,batch_size):  
    metrics = []  
    # y_pred = (model.predict(X))
    # fpr, tpr, threshold = roc_curve(Y, y_pred)
    # roc_auc = auc(fpr, tpr)   
  
    test_generator.reset()
    y_pred = model.predict_generator(test_generator,steps=(totalTest // batch_size) + 1)
    fpr, tpr, threshold = roc_curve(test_generator.classes, y_pred)
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
       'threshold':threshold,}, cls=NumpyEncoder,)     
    return json_metrics
    
def model_evaluation_1classOutputSVM(model,X,Y,num_classes,evaluation_folder):  
    metrics = []
   # y_pred = (model.predict_proba(X))
  # y_pred = y_pred.astype('float16')
    y_pred_ = (model.predict(X))
    c = confusion_matrix(Y,y_pred_)
    sensitivity  = float(c[1, 1]) / (float(c[1, 1]) + float(c[1, 0]))    
    specificity = float(c[0, 0]) / (float(c[0, 1]) + float(c[0, 0]))
   # print(sensitivity)
   # print(specificity)
 #   print( np.argmax(y_pred, axis=1))
 #   print(y_pred)

    fpr, tpr, threshold = roc_curve(Y, np.argmax(y_pred_, axis=1))
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
       'sensitivity':sensitivity,
       'specificity':specificity,
       'threshold':threshold,}, cls=NumpyEncoder,) 
       #'y_pred':y_pred[:]}, cls=NumpyEncoder)         
    return json_metrics
    
    

    
    
    
    
# def model_evaluation_1classOutputEnsembleEF(models,X,Y,num_classes,evaluation_folder):  
    # metrics = []
    # l = len(models)
    # l_x = len(X)
    
    # y_pred = np.zeros((l_x,1))

    # for model in models: 
        # y_pred = list( map(add, y_pred, (model.predict(X)) ) )
    # y_pred = [x / l for x in y_pred]
    # fpr, tpr, threshold = roc_curve(Y, y_pred)
    # roc_auc = auc(fpr, tpr)
   
  
    # class NumpyEncoder(json.JSONEncoder):
       # def default(self, obj):
           # if isinstance(obj, np.ndarray):
               # return obj.tolist()
           # return json.JSONEncoder.default(self, obj)

    # json_metrics = json.dumps({     
       # 'roc_auc': roc_auc,
       # 'fpr':fpr,
       # 'tpr':tpr,
       # 'threshold':threshold,}, cls=NumpyEncoder,)         
    # return json_metrics    
    
    
    
    
    

    
    
# def model_evaluation_1classOutputEnsembleLF(models,X,Y,num_classes,evaluation_folder):  
    # metrics = []
    # l = len(models)
    # l_x = len(X)
    
    # y_pred = np.zeros((l_x,l))
    # thresholds =   np.arange(0, 1, 0.002)
  # #  threshold = numpy.arange([0, ]1, [0.01,])
    # i = 0
    # for model in models: 
        # tmp = model.predict(X)
        # y_pred[:,i] = np.reshape(tmp,[len(tmp)])

        # i = i + 1
    # iter_thres = 0
    # fpr = np.zeros_like(thresholds)  
    # tpr = np.zeros_like(thresholds)  

    # for threshold in thresholds:
            # y_pred_ = np.zeros_like(y_pred[:,0])  
            # for iter in range(0,l):
                # y_pred_ = np.add([1 if x >= threshold else 0 for x in y_pred[:, iter]],y_pred_)
            # y_pred_ = [1 if x >= round(l/2) else 0 for x in y_pred_]
            # c = confusion_matrix(Y,y_pred_)
            # fpr[iter_thres] = 1 - (float(c[0, 0]) / (float(c[0, 1]) + float(c[0, 0])))
            # tpr[iter_thres] =  float(c[1, 1]) / (float(c[1, 1]) + float(c[1, 0])) 
            # iter_thres = iter_thres + 1 
    # roc_auc = auc(fpr, tpr)

    # class NumpyEncoder(json.JSONEncoder):
       # def default(self, obj):
           # if isinstance(obj, np.ndarray):
               # return obj.tolist()
           # return json.JSONEncoder.default(self, obj)

    # json_metrics = json.dumps({     
       # 'roc_auc': roc_auc,
       # 'fpr':fpr,
       # 'tpr':tpr,
       # 'threshold':threshold,}, cls=NumpyEncoder,)         
    # return json_metrics
    
    