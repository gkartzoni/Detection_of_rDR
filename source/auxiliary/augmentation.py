# -*- coding: utf-8 -*-
 
# -----------------------------------------------------------
# 2019 Ioanna Gkartzonika
# email gkartzoni@gmail.com
# -----------------------------------------------------------

"""
This is a stript for image augmentation.
Takes a file folder with images and does flip, flop and random rotation so from 1 image, 3 images are produced.
The new images are saved in [folder].
"""

from random import seed
from random import randint
import numpy as np
import os
import glob
import cv2
import matplotlib
image_path =  '/home/gkartzoni/thesis/images'
class_name = '/2classTrain/class1'
path = image_path + class_name
file_extension  = '*.jpeg' #For messidor 2 use '*.tif' jpeg

# Images' path 
data_path = os.path.join(path,file_extension)
# Vector with images full path 
files = glob.glob(data_path)

# Create a new folder to save augmented images
folder = path + '/AugmentedImagesTrain'
try:  
	os.mkdir(folder)
except OSError:  
	print ("Creation of the directory %s failed" % folder)
else:  
	print ("Successfully created the directory %s " % folder)


img_rows,img_cols,img_channels = 300 ,300,3
# Find image center to do the rotation.
center = (img_rows / 2, img_cols / 2)
scale = 1    
seed(1)

i = 1;
for image_file in files:
    # Random angle
    angle = randint(1, 20)
    # Produce rotation matrix
    M = cv2.getRotationMatrix2D(center, angle, scale)
    img = cv2.imread(image_file)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)   
    # Flips up/down the image
    img1 = np.flipud(img)
    # Flips left/right the image
    img2 = np.fliplr(img)
    # Rotates image
    img3 = cv2.warpAffine(img, M, (img_rows, img_cols))
    
    # Save images
    matplotlib.image.imsave(folder+'/'+str(i)+'flippedVer.jpeg', img1)   
    matplotlib.image.imsave(folder+'/'+str(i)+'flippedHor.jpeg', img2)   
    matplotlib.image.imsave(folder+'/'+str(i)+'rotated.jpeg', img3)   
    i = i + 1
