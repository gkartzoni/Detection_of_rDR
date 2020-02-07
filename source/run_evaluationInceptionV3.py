import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";
 
# # The GPU id to use, usually either "0" or "1";
os.environ["CUDA_VISIBLE_DEVICES"]="0"; 


import tensorflow as tf
from tensorflow import keras

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

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt


experiment_name = '11'

#idia metatropi eikonon
# evaluation_train / evaluation_retrain

img_rows,img_cols,img_channels = 299 ,299,3
select = 'evaluation_retrain'
select_dataSet = 1# select 1 for kaggle, 2 for messidor
n = 2
model_name = 'model.h5'
if select_dataSet == 1:
    image_path0 =  '/home/gkartzoni/thesis/images/3_gray_test/0'
    image_path1 =   '/home/gkartzoni/thesis/images/3_gray_test/1'
    file_extension  = '*.png' #For messidor 2 use '*.tif' jpeg
    if select == 'evaluation_train':
        evaluation_folder = '/home/gkartzoni/thesis/experiments/'+ experiment_name+'/EvaluationK'
        try:  
            os.mkdir(evaluation_folder)
        except OSError:  
            print ("Creation of the directory %s failed" % evaluation_folder)
        else:  
            print ("Successfully created the directory %s " % evaluation_folder)
        model_path = '/home/gkartzoni/thesis/experiments/'+ experiment_name+'/TrainingProcess/' + model_name
        model =  mf.model_load(model_path)
    elif 'evaluation_retrain':
        
        evaluation_folder = '/home/gkartzoni/thesis/experiments/'+ experiment_name+'/EvaluationK'+str(n)
        try:  
            os.mkdir(evaluation_folder)
        except OSError:  
            print ("Creation of the directory %s failed" % evaluation_folder)
        else:  
            print ("Successfully created the directory %s " % evaluation_folder)
        model_path = '/home/gkartzoni/thesis/experiments/'+ experiment_name+'/ReTrainingProcess'+ str(n)+'/'+model_name
        model =  mf.model_load(model_path)

elif select_dataSet == 2:
    image_path0 = '/home/gkartzoni/thesis/images/3_gray_Messidor2/0'
    image_path1 =  '/home/gkartzoni/thesis/images/3_gray_Messidor2/1'
    file_extension  = '*.png' #For messidor 2 use '*.tif' jpeg
    if select == 'evaluation_train':
        evaluation_folder = '/home/gkartzoni/thesis/experiments/'+ experiment_name+'/EvaluationM'
        try:  
            os.mkdir(evaluation_folder)
        except OSError:  
            print ("Creation of the directory %s failed" % evaluation_folder)
        else:  
            print ("Successfully created the directory %s " % evaluation_folder)
        model_path = '/home/gkartzoni/thesis/experiments/'+ experiment_name+'/TrainingProcess/' + model_name
        model =  mf.model_load(model_path)
    elif 'evaluation_retrain':
        evaluation_folder = '/home/gkartzoni/thesis/experiments/'+ experiment_name+'/EvaluationM'+str(n)
        try:  
            os.mkdir(evaluation_folder)
        except OSError:  
            print ("Creation of the directory %s failed" % evaluation_folder)
        else:  
            print ("Successfully created the directory %s " % evaluation_folder)
        model_path = '/home/gkartzoni/thesis/experiments/'+ experiment_name+'/ReTrainingProcess'+ str(n)+'/'+ model_name
        model =  mf.model_load(model_path)

num_classes = 1
X = []
Y = []



data_path = os.path.join(image_path0,file_extension)
files = glob.glob(data_path)
normalizedImg = np.zeros((img_rows, img_cols,3))

i = 1;
for image_file in files:

    img = cv2.imread(image_file)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB )   #img = cv2.imread(image_file)

    img  =  img.astype(np.float32)
    img = preprocess_input(img)
    X.append(img)  #  if i<2:    
    Y.append(0)


data_path = os.path.join(image_path1,file_extension)
files = glob.glob(data_path)
 # #for i in range(1,len(image_name)+1):
i = 1;
for image_file in files:

    img = cv2.imread(image_file)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB )   #img = cv2.imread(image_file)

    img  =  img.astype(np.float32)
    img = preprocess_input(img)
    X.append(img)  #  if i<2:    
    Y.append(1)
    
#for i in range(0,7):
#    matplotlib.image.imsave(evaluation_folder+'/'+str(i)+'.png', X[i])   
 

X_test = np.asarray(X)
Y_test = np.asarray(Y)


json_metrics = mf.model_evaluation_1classOutput(model,X_test,Y_test,num_classes,evaluation_folder)
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




