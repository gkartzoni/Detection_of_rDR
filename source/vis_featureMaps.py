import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";
 
# # The GPU id to use, usually either "0" or "1";
os.environ["CUDA_VISIBLE_DEVICES"]="1"; 
from sklearn.metrics import roc_curve
from model_inceptionV3 import inceptionV3

from sklearn.metrics import classification_report,precision_recall_curve, confusion_matrix, auc,roc_curve
from PIL import Image
from keras.models import Model

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

experiment_name = '55'

#idia metatropi eikonon
# evaluation_train / evaluation_retrain

img_rows,img_cols,img_channels = 299 ,299,4
select = 'evaluation_train'
select_dataSet = 1# select 1 for kaggle, 2 for messidor
n = 7
nb_epoch, batch_size = 8, 32

if select_dataSet == 1:
    totalTest = 8759

    pathTest = '/home/gkartzoni/thesis/images/test'
    image_path0 =  '/home/gkartzoni/thesis/images/test/0'
    image_path1 =   '/home/gkartzoni/thesis/images/test/1'
    file_extension  = '*.jpeg' #For messidor 2 use '*.tif' jpeg
    if select == 'evaluation_train':
        evaluation_folder = '/home/gkartzoni/thesis/experiments/'+ experiment_name+'/EvaluationK'
        try:  
            os.mkdir(evaluation_folder)
        except OSError:  
            print ("Creation of the directory %s failed" % evaluation_folder)
        else:  
            print ("Successfully created the directory %s " % evaluation_folder)
        model_path = '/home/gkartzoni/thesis/experiments/'+ experiment_name+'/TrainingProcess/model.h5'
        model =  mf.model_load(model_path)
 
    elif 'evaluation_retrain':
        
        evaluation_folder = '/home/gkartzoni/thesis/experiments/'+ experiment_name+'/EvaluationK'+str(n)
        try:  
            os.mkdir(evaluation_folder)
        except OSError:  
            print ("Creation of the directory %s failed" % evaluation_folder)
        else:  
            print ("Successfully created the directory %s " % evaluation_folder)
        model_path = '/home/gkartzoni/thesis/experiments/'+ experiment_name+'/ReTrainingProcess'+ str(n)+'/model.h5'
        model =  mf.model_load(model_path)
 


i = 1
data_path = os.path.join(image_path1,file_extension)
files = glob.glob(data_path)
num_of_img = 10
file_extension  = '*.jpeg'
normalizedImg = np.zeros([img_rows, img_cols, 3])

img = Image.open(files[12])
img = np.array(img)
img  =  img.astype(np.float32)
angle = 60
rotation_matrix = cv2.getRotationMatrix2D((img_rows/2, img_cols/2), angle, 1)
img = cv2.warpAffine(img, rotation_matrix, (img_rows, img_cols)) 
img = img /255

img_original = img 
img = np.expand_dims(img, axis=0)
layer_dict = dict([(layer.name, layer) for layer in model.layers])
layer_name = 'activation_30'
#layer_name = 310
model = Model(inputs=model.inputs, outputs=layer_dict[layer_name].output)
model.summary()

# Apply the model to the image
feature_maps = model.predict(img)
i = i + 1
square = 8
index = 1
for _ in range(square):
    for _ in range(square):        
        ax = plt.subplot(square, square, index)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.imshow(feature_maps[0, :, :, index-1], cmap='viridis')
        index += 1
plt.savefig('feature_maps.png', bbox_inches='tight')
plt.clf()
plt.show()




