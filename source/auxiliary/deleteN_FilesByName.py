# -*- coding: utf-8 -*-

# -----------------------------------------------------------
# 2019 Ioanna Gkartzonika
# email gkartzoni@gmail.com
# -----------------------------------------------------------


"""
This is a strict to delete [nmbFiles] files that their name starts from a specific string.
In our case the string is "UD"
"""  
    
import os
import random

# File name
pathSRC = '/home/gkartzoni/thesis/images/train_new/0/'
# Number of image that are going to deleted
nmbFiles = 50186/2
# Find the files that their name starts from "UD"
[filenames = random.sample([x for x in os.listdir(pathSRC) if file.startswith("UD")]],nmbFiles])

for fname in filenames:
    srcpath = os.path.join(pathSRC, fname)
    os.remove(srcpath)


   