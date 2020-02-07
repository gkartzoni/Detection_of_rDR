# -*- coding: utf-8 -*-

# -----------------------------------------------------------
# 2019 Ioanna Gkartzonika
# email gkartzoni@gmail.com
# It is taken from https://www.pyimagesearch.com/2015/09/07/blur-detection-with-opencv/
# -----------------------------------------------------------


"""
This is a strict to 
move blurred images in other folder.
"""

import cv2
import glob
import os
import shutil


image_path = '/home/gkartzoni/thesis/images/train'
# Path to images in class 0
image_path0 = image_path + '/0'
# Path to images in class 1
image_path1 = image_path + '/1'

# Image extension
file_extension  = '*.jpeg' 
# Threshold for what would be consider blurred image 
threshold = 50

# Create folders to save blurred images. main folder: [experiment_file folder] and subfolder1 = [blurred_path0] and subfolder2 = [blurred_path1]. 
experiment_file =  '/home/gkartzoni/thesis/images/BlurredTrain'
try:  
    os.mkdir(experiment_file)
except OSError:  
    print ("Creation of the directory %s failed" % experiment_file)
else:  
    print ("Successfully created the directory %s " % experiment_file)
  
blurred_path0 =  experiment_file+ '/class0'  
try:  
    os.mkdir(blurred_path0)
except OSError:  
    print ("Creation of the directory %s failed" % blurred_path0)
else:  
    print ("Successfully created the directory %s " % blurred_path0)
      
blurred_path1 =  experiment_file+ '/class1'  
try:  
    os.mkdir(blurred_path1)
except OSError:  
    print ("Creation of the directory %s failed" % blurred_path1)
else:  
    print ("Successfully created the directory %s " % blurred_path1)

    
# Class 0    
data_path0 = os.path.join(image_path0,file_extension)
files0 = glob.glob(data_path0)

iter1= 0
iter2=0 
for image_file in files0:
    img = cv2.imread(image_file)
    # Convert image to gray
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Find laplacian
    fm = cv2.Laplacian(gray, cv2.CV_64F).var()
    # Compare with threshold 
    iter1 = iter1+1
    if fm < threshold:
        shutil.move(image_file, blurred_path0)
        iter2 = iter2+1
        iter1 = iter1-1
print(iter1)
print(iter2)  



# Class 1   
data_path1 = os.path.join(image_path1,file_extension)
files1 = glob.glob(data_path1)

iter1= 0
iter2=0 
for image_file in files1:
    img = cv2.imread(image_file)
    # Convert image to gray
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Find laplacian
    fm = cv2.Laplacian(gray, cv2.CV_64F).var()
    # Compare with threshold 
    iter1 = iter1+1
    if fm < threshold:
        shutil.move(image_file, blurred_path1)
        iter2 = iter2+1
        iter1 = iter1-1
print(iter1)
print(iter2)  
