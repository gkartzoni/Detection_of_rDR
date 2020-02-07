# -*- coding: utf-8 -*-
 
# -----------------------------------------------------------
# (C) 2019 Ioanna Gkartzonika
# email gkartzoni@gmail.com
# -----------------------------------------------------------

"""
This is a stript to join files from destination folder(pathDEST) and source folder(pathSRC) to destination folder(pathDEST)
Move only the files that have [file_extension] extension.
"""

import glob
import os
import shutil


pathDEST = '/home/gkartzoni/thesis/images/train/2'
pathSRC = '/home/gkartzoni/thesis/images/train/4'
# File extension
file_extension  = '*.jpeg' #For messidor 2 use '*.tif' 
data_path = os.path.join(pathSRC,file_extension)
files = glob.glob(data_path)

# Move the [*.jpeg] files from source folder to destination folder
for image_file in files:
    srcpath = os.path.join(pathSRC, image_file)
    shutil.move(srcpath, pathDEST)                    




    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    