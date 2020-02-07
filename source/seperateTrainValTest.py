# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 04:12:02 2019

@author: Ioanna
"""

"""
Seperate training and validation set
Take a percentage of images from every folder(5)
"""

import cv2
import glob
import os
import shutil
import numpy as np
import shutil, random, os

# Source folder
path1 = '/home/gkartzoni/thesis/images/all'
# New folder that the files will move
path2 = '/home/gkartzoni/thesis/images/validation'
file_extension  = '*.jpeg' 
#  Percentage of images that will be moved to path2
percentage = 0.2

for ii in range(0,5):
    pathSRC  = path1 + '/' + str(ii)
    pathDEST  = path2 +'/' + str(ii)
    l = (len([name for name in os.listdir(pathSRC) if os.path.isfile(os.path.join(pathSRC, name))]))
    nmbFiles =  int(l* percentage)
    filenames = random.sample(os.listdir(pathSRC),nmbFiles)
    for fname in filenames:
        srcpath = os.path.join(pathSRC, fname)
        shutil.move(srcpath, pathDEST)                    

    
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    