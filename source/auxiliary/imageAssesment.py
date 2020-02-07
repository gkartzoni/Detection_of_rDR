# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 13:01:19 2019

@author: Ioanna
"""

import cv2
import numpy as np
import pandas as pd
import keras
import os
import glob
import matplotlib.pyplot as plt
from model_vgg16 import vgg16

img_rows,img_cols,img_channels = 300 ,300,3
nb_train_samples, nb_validation_samples, =1000, 200
nb_epoch, batch_size = 100, 32 
num_classes = 2

image_path = r"C:\Users\Ioanna\Desktop\thesis\images\croppedImagesRGB\train\5_classes\class1"


X = []
data_path = os.path.join(image_path,'*.jpeg')
#data_path = os.path.join(image_path,'*.tif')
files = glob.glob(data_path)
iter1 = 0
iter2 = 0
#for i in range(1,len(image_name)+1):
for image_file in files:
   # print(image_file) 
    img = cv2.imread(image_file)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    fm = cv2.Laplacian(gray, cv2.CV_64F).var()
    text = "Not Blurry"
    iter1 = iter1+1
    if fm < 10:
        text = 'Blury'
        iter2 = iter2+1
        iter1 = iter1-1
    #cv2.putText(img, "{}: {:.2f}".format(text, fm), (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
    #cv2.imshow("Image", img)
    #key = cv2.waitKey(0)
print(iter1)
print(iter2)
    




















