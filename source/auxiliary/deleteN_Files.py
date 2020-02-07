# -*- coding: utf-8 -*-

# -----------------------------------------------------------
# 2019 Ioanna Gkartzonika
# email gkartzoni@gmail.com
# -----------------------------------------------------------


"""
This is a strict to 
delete [nmbFiles] files in a folder.
"""
   
import os
import random

# Path to the file
pathSRC = '/home/gkartzoni/thesis/images/validation/0/'
# Number of files to delete
nmbFiles = 9219
# Find the random samples that will be deleted
filenames = random.sample(os.listdir(pathSRC),nmbFiles)
# Delete files
for fname in filenames:
    srcpath = os.path.join(pathSRC, fname)
    os.remove(srcpath)
    




    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    