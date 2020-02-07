import os
import glob
import cv2 as cv
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from random import shuffle

rows,columns = 299, 299
image_path =  '/home/gkartzoni/thesis/images'
class_name = '/trainAugmented/1'
path = image_path + class_name

file_extension  = '[!UD][!LR]*.jpeg' #For messidor 2 use '*.tif' jpeg
data_path = os.path.join(path,file_extension)
files = glob.glob(data_path)


BRIGHTNESS_MAX_DELTA = 0.15
SATURATION_LOWER = 0.8
SATURATION_UPPER = 1.2
HUE_MAX_DELTA = 0.05
CONTRAST_LOWER = 0.8
CONTRAST_UPPER = 1.2
augmentations = [
    {'fn': tf.image.random_brightness,
     'args': [BRIGHTNESS_MAX_DELTA]},
    {'fn': tf.image.random_saturation,
     'args': [SATURATION_LOWER, SATURATION_UPPER]},
    {'fn': tf.image.random_hue,
     'args': [HUE_MAX_DELTA]},
    {'fn': tf.image.random_contrast,
     'args': [CONTRAST_LOWER, CONTRAST_UPPER]}]
normalizedImg1 = np.zeros([rows, columns, 3])
normalizedImg2 = np.zeros([rows, columns, 3])

for image_file in files:

    shuffle(augmentations)
    img = cv.imread(image_file)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB )   #img = cv2.imread(image_file)
 
    original_image = img
    img= np.fliplr(original_image)
    tf.reset_default_graph()
    img_holder = tf.placeholder(dtype = tf.uint8, shape = [None, rows, columns, 3])
    with tf.Session() as sess:
          img = np.asarray(img).astype(np.uint8).reshape(-1,rows,columns,3) 
          img = sess.run(augmentations[0]['fn'](img, *augmentations[0]['args']), feed_dict = {img_holder: img})
    with tf.Session() as sess:
      img = np.asarray(img).astype(np.uint8).reshape(-1,rows,columns,3) 
      img = sess.run(augmentations[1]['fn'](img, *augmentations[1]['args']), feed_dict = {img_holder: img})    
    with tf.Session() as sess:
          img = np.asarray(img).astype(np.uint8).reshape(-1,rows,columns,3) 
          img = sess.run(augmentations[2]['fn'](img, *augmentations[2]['args']), feed_dict = {img_holder: img})
    with tf.Session() as sess:
       img = np.asarray(img).astype(np.uint8).reshape(-1,rows,columns,3) 
       img = sess.run(augmentations[3]['fn'](img, *augmentations[3]['args']), feed_dict = {img_holder: img})
    img = np.asarray(img).astype(np.uint8).reshape(rows,columns,3) 
    img[:,:,0] = img[:,:,0] - img[:,:,0].min()
    img[:,:,1] = img[:,:,1] - img[:,:,1].min()
    img[:,:,2] = img[:,:,2] - img[:,:,2].min()    
    normalizedImg1 = cv.normalize(img,  normalizedImg1, 0, 255, cv.NORM_MINMAX)
    

################################################################
    shuffle(augmentations)
    img= np.flipud(original_image)
    tf.reset_default_graph()
    img_holder = tf.placeholder(dtype = tf.uint8, shape = [None, rows, columns, 3])
    with tf.Session() as sess:
          img = np.asarray(img).astype(np.uint8).reshape(-1,rows,columns,3) 
          img = sess.run(augmentations[0]['fn'](img, *augmentations[0]['args']), feed_dict = {img_holder: img})
    with tf.Session() as sess:
      img = np.asarray(img).astype(np.uint8).reshape(-1,rows,columns,3) 
      img = sess.run(augmentations[1]['fn'](img, *augmentations[1]['args']), feed_dict = {img_holder: img})    
    with tf.Session() as sess:
          img = np.asarray(img).astype(np.uint8).reshape(-1,rows,columns,3) 
          img = sess.run(augmentations[2]['fn'](img, *augmentations[2]['args']), feed_dict = {img_holder: img})
    with tf.Session() as sess:
       img = np.asarray(img).astype(np.uint8).reshape(-1,rows,columns,3) 
       img = sess.run(augmentations[3]['fn'](img, *augmentations[3]['args']), feed_dict = {img_holder: img})
    img = np.asarray(img).astype(np.uint8).reshape(rows,columns,3) 
    img[:,:,0] = img[:,:,0] - img[:,:,0].min()
    img[:,:,1] = img[:,:,1] - img[:,:,1].min()
    img[:,:,2] = img[:,:,2] - img[:,:,2].min()    
    normalizedImg2 = cv.normalize(img,  normalizedImg2, 0, 255, cv.NORM_MINMAX)   
    
    name = (os.path.basename(image_file))
    # print(path_+'/'+ 'lr' + name)
    cv.imwrite(path+'/'+ 'LR' + name , cv.cvtColor(normalizedImg1, cv.COLOR_BGR2RGB))
    cv.imwrite(path+'/'+ 'UD' + name , cv.cvtColor(normalizedImg2, cv.COLOR_BGR2RGB))