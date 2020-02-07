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
from PIL import Image

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt


file_extension = '*.jpg'


img_rows,img_cols,img_channels = 299 ,299,3
folder_name = '4channel_validation'

image_path0 =  '/home/gkartzoni/thesis/images/2classesMessidor2_RDR/class0'
path2save0 = '/home/gkartzoni/thesis/images/3channel_Messidor2/0/'

num_classes = 1
X = []
Y = []


data_path = os.path.join(image_path0,file_extension)
files = glob.glob(data_path)

normalizedImg = np.zeros((img_rows, img_cols,3))

img_rows,img_cols = 299,299
temp_img = np.zeros([img_rows, img_cols, 4])

rad = 146
center = (149,149)
circle_img_zeros = np.zeros([img_rows, img_cols])
circle_img = cv2.circle(circle_img_zeros, center, rad, (255), thickness=-1)

circle_img3 = np.zeros([img_rows, img_cols,3])
circle_img3[:,:,0] = circle_img
circle_img3[:,:,1] = circle_img
circle_img3[:,:,2] = circle_img
roi = np.zeros([img_rows, img_cols])
for image_file in files:
    img = cv2.imread(image_file)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB )   #img = cv2.imread(image_file)
    img_ = img.sum(axis=2)>15
    img *= img_[...,None]
    
    img = img.astype('float32')  
    normalizedImg = np.zeros([img_rows, img_cols, 3])
    normalizedImg = cv2.normalize(img,  normalizedImg, 0, 1, cv2.NORM_MINMAX)
    img = normalizedImg


    if ((img[0,150,:]== [0,0,0]).all() and (img[1,150,:]== [0,0,0]).all() and (img[2,150,:]== [0,0,0]).all()): 
    
            circle_img_zeros = np.zeros([img_rows, img_cols])

            circle_img = cv2.circle(circle_img_zeros, center, rad, (255), thickness=-1)
            circle_img3[:,:,0] = circle_img
            circle_img3[:,:,1] = circle_img
            circle_img3[:,:,2] = circle_img           
            img = circle_img3 * img

            

            
            img = cv2.line(img,(0,18),(298,18),(0,0,0),31)
            img = cv2.line(img,(0,280),(298,280),(0,0,0),31)
            
    else:
            circle_img_zeros = np.zeros([img_rows, img_cols])
            circle_img = cv2.circle(circle_img_zeros, center, rad, (255), thickness=-1)
            circle_img3[:,:,0] = circle_img
            circle_img3[:,:,1] = circle_img
            circle_img3[:,:,2] = circle_img
            img = circle_img3 * img
           

    
    img = img.astype('uint8')

    
    name = os.path.basename(image_file)
    filename, file_extension = os.path.splitext(name)
   
    img = Image.fromarray(img)            
    img.save(path2save0 + filename + '.png')




 