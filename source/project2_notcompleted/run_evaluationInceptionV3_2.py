import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";
 
# # The GPU id to use, usually either "0" or "1";
os.environ["CUDA_VISIBLE_DEVICES"]="1"; 
from sklearn.metrics import roc_curve
from model_inceptionV3 import inceptionV3

from sklearn.metrics import classification_report,precision_recall_curve, confusion_matrix, auc,roc_curve

import tensorflow as tf
from tensorflow import keras
num_classes = 1
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
keras.backend.set_session(sess)
import json
import cv2
import numpy as np
import keras
from skimage.exposure import equalize_hist
import model_functions as mf
import glob
from shutil import copyfile #copyfile(src, dst)
from keras.applications.inception_v3 import preprocess_input
from imutils import paths
seed = 82

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

from keras.preprocessing.image import ImageDataGenerator

experiment_name = '24'

#idia metatropi eikonon
# evaluation_train / evaluation_retrain

img_rows,img_cols,img_channels = 299 ,299,4
select = 'evaluation_retrain'
select_dataSet = 2# select 1 for kaggle, 2 for messidor
n = 5
nb_epoch, batch_size = 8, 32


if select_dataSet == 1:
    totalTest = 8759

    pathTest = '/home/gkartzoni/thesis/images/test'
    image_path0 =  '/home/gkartzoni/thesis/images/4channel_test/0'
    image_path1 =   '/home/gkartzoni/thesis/images/4channel_test/1'
    file_extension  = '*.png' #For messidor 2 use '*.tif' jpeg
    if select == 'evaluation_train':
        evaluation_folder = '/home/gkartzoni/thesis/experiments/'+ experiment_name+'/EvaluationK'
        try:  
            os.mkdir(evaluation_folder)
        except OSError:  
            print ("Creation of the directory %s failed" % evaluation_folder)
        else:  
            print ("Successfully created the directory %s " % evaluation_folder)
        model_path = '/home/gkartzoni/thesis/experiments/'+ experiment_name+'/TrainingProcess/model.h5'
        model = inceptionV3(img_rows,img_cols,img_channels,num_classes)
        model.load_weights(model_path)  
    elif 'evaluation_retrain':
        
        evaluation_folder = '/home/gkartzoni/thesis/experiments/'+ experiment_name+'/EvaluationK'+str(n)
        try:  
            os.mkdir(evaluation_folder)
        except OSError:  
            print ("Creation of the directory %s failed" % evaluation_folder)
        else:  
            print ("Successfully created the directory %s " % evaluation_folder)
        model_path = '/home/gkartzoni/thesis/experiments/'+ experiment_name+'/ReTrainingProcess'+ str(n)+'/model.h5'
        model = inceptionV3(img_rows,img_cols,img_channels,num_classes)
        model.load_weights(model_path)  
        
elif select_dataSet == 2:
    totalTest = 1748

    image_path0 = '/home/gkartzoni/thesis/images/4channel_Messidor2/0'
    image_path1 =  '/home/gkartzoni/thesis/images/4channel_Messidor2/1'
    pathTest = '/home/gkartzoni/thesis/images/2classesMessidor2_RDR'
    file_extension  = '*.png' #For messidor 2 use '*.tif' jpeg
    if select == 'evaluation_train':
        evaluation_folder = '/home/gkartzoni/thesis/experiments/'+ experiment_name+'/EvaluationM'
        try:  
            os.mkdir(evaluation_folder)
        except OSError:  
            print ("Creation of the directory %s failed" % evaluation_folder)
        else:  
            print ("Successfully created the directory %s " % evaluation_folder)
        model_path = '/home/gkartzoni/thesis/experiments/'+ experiment_name+'/TrainingProcess/model.h5'
        model = inceptionV3(img_rows,img_cols,img_channels,num_classes)
        model.load_weights(model_path)  
    elif 'evaluation_retrain':
        evaluation_folder = '/home/gkartzoni/thesis/experiments/'+ experiment_name+'/EvaluationM'+str(n)
        try:  
            os.mkdir(evaluation_folder)
        except OSError:  
            print ("Creation of the directory %s failed" % evaluation_folder)
        else:  
            print ("Successfully created the directory %s " % evaluation_folder)
        model_path = '/home/gkartzoni/thesis/experiments/'+ experiment_name+'/ReTrainingProcess'+ str(n)+'/model.h5'
        model = inceptionV3(img_rows,img_cols,img_channels,num_classes)
        model.load_weights(model_path)          
num_classes = 1
X = []
Y = []

# def preprocessing_fun(img):
         # normalizedImg = np.zeros([img_rows, img_cols, 3])
         # img  =  img.astype(np.uint8)
        # # normalizedImg = cv2.normalize(img,  normalizedImg, 0, 255, cv2.NORM_MINMAX)
       # #  img = normalizedImg
         # img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
         # clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(8,8))
         # img[:,:,0] = clahe.apply(img[:,:,0]) 
         # img = cv2.cvtColor(img, cv2.COLOR_YUV2RGB)
         # img  =  img.astype(np.float32)
         # normalizedImg = cv2.normalize(img,  normalizedImg, 0, 1, cv2.NORM_MINMAX)
         # img = normalizedImg
         # #img = preprocess_input(img)

         # return img 
    
test_gen = ImageDataGenerator(
    rescale=1./255,

)
test_generator = test_gen.flow_from_directory(
        pathTest,
        target_size=(img_rows, img_cols),
        batch_size=batch_size,
        class_mode='binary',
        shuffle=False,
        color_mode='rgba',
        
     )

# test_generator.reset()
# predIdxs = model.predict_generator(test_generator,steps=(totalTest // batch_size) + 1)
# fpr, tpr, threshold = roc_curve(test_generator.classes, predIdxs)
# roc_auc = auc(fpr, tpr)
# #print(roc_auc)
# #json_metrics = mf.model_evaluation_1classOutput(model,X_test,Y_test,num_classes,evaluation_folder)
# #AUC
# #json_metrics = json.loads(json_metrics)
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

# src = '/home/gkartzoni/thesis/source/run_evaluationInceptionV3.py'
# dst = evaluation_folder + '/run_evaluationInceptionV3.py'
# copyfile(src, dst)	

json_metrics = mf.model_evaluation_1classOutputGenerator(model,test_generator,totalTest,batch_size)
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

src = '/home/gkartzoni/thesis/source/run_evaluationInceptionV3.py'
dst = evaluation_folder + '/run_evaluationInceptionV3.py'
copyfile(src, dst)	

with open(evaluation_folder+'/metrics.json', 'w') as json_file:  
     json.dump(json_metrics, json_file)


