import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";
 
# # The GPU id to use, usually either "0" or "1";
os.environ["CUDA_VISIBLE_DEVICES"]="1"; 


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
from PIL import Image
from skimage.color import rgb2gray
from skimage.filters.rank import entropy

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt


file_extension = '*.jpg'


img_rows,img_cols,img_channels = 299 ,299,3
folder_name = '4channel_validation'

image_path0 =  '/home/gkartzoni/thesis/images/2classesMessidor2_RDR/class0'
image_path1 =  '/home/gkartzoni/thesis/images/2classesMessidor2_RDR/class1'

path2save0 = '/home/gkartzoni/thesis/images/3_gray_Messidor2/0/'
path2save1 = '/home/gkartzoni/thesis/images/3_gray_Messidor2/1/'

num_classes = 1
X = []
Y = []


data_path = os.path.join(image_path1,file_extension)
files = glob.glob(data_path)

normalizedImg = np.zeros((img_rows, img_cols,3))

img_rows,img_cols = 299,299

#files = files[45340:]

for image_file in files:
    img = cv2.imread(image_file)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB )   #img = cv2.imread(image_file)
    img_ = img.sum(axis=2)>15
    img *= img_[...,None]
    
    gray = rgb2gray(img)
    gray =np.uint8(cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX))
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)

    ent = entropy(gray, np.ones([9,9]) )
    ent =np.uint8(cv2.normalize(ent, None, 0, 255, cv2.NORM_MINMAX))

    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    v = hsv[:,:,2]
    
    gray = np.reshape(gray,[299,299,1])
    ent = np.reshape(ent,[299,299,1])
    v = np.reshape(v,[299,299,1])
    img = np.concatenate((gray, ent,v), axis=-1)

    name = os.path.basename(image_file)
    filename, file_extension = os.path.splitext(name)

    img = Image.fromarray(img)            
    img.save(path2save1 + filename + '.png')

# data_path = os.path.join(image_path1,file_extension)
# files = glob.glob(data_path)   

# for image_file in files:
    # img = cv2.imread(image_file)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB )   #img = cv2.imread(image_file)
    # img_ = img.sum(axis=2)>15
    # img *= img_[...,None]
    
    # gray = rgb2gray(img)
    # gray =np.uint8(cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX))
    # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    # gray = clahe.apply(gray)

    # ent = entropy(gray, np.ones([9,9]) )
    # ent =np.uint8(cv2.normalize(ent, None, 0, 255, cv2.NORM_MINMAX))

    # hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    # v = hsv[:,:,2]
    
    # gray = np.reshape(gray,[299,299,1])
    # ent = np.reshape(ent,[299,299,1])
    # v = np.reshape(v,[299,299,1])
    # img = np.concatenate((gray, ent,v), axis=-1)

    # name = os.path.basename(image_file)
    # filename, file_extension = os.path.splitext(name)
    
    # img = Image.fromarray(img)            
    # img.save(path2save1 + filename + '.png')


 