# -*- coding: utf-8 -*-

# -----------------------------------------------------------
# 2019 Ioanna Gkartzonika
# email gkartzoni@gmail.com
# -----------------------------------------------------------


"""
This is a strict to delete files that have a specific string in their name.
In our case the string are "LR" and "UD"
"""    
    
import os
import glob
pathSRC = '/home/gkartzoni/thesis/images/train/0/'
file_extension  = '*.jpeg' #For messidor 2 use '*.tif' 
data_path = os.path.join(pathSRC,file_extension)
files = glob.glob(data_path)

for fname in files:
    # Delete the files that their name includes the string "LR" or "UD".
    #srcpath = os.path.join(pathSRC, fname)
    if (("LR" in fname) or (("UD" in fname)) ):
        os.remove(fname)
        

