import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";
 
# # The GPU id to use, usually either "0" or "1";
os.environ["CUDA_VISIBLE_DEVICES"]="1"; 


import tensorflow as tf
from tensorflow import keras

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
keras.backend.set_session(sess)
from keras import backend as K
from keras import layers
from keras.layers import Input
from keras.models import Model

import json
import cv2
import numpy as np
import keras
from skimage.exposure import equalize_hist
from keras.applications.vgg16 import preprocess_input
import model_functions as mf
import glob
from shutil import copyfile #copyfile(src, dst)
from keras.applications.inception_v3 import preprocess_input
from keras.optimizers import SGD
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt


experiment_name = '1'


lr, decay = 1e-3, 4e-5

from keras.models import load_model
models=[]
ensemble_path = '/home/gkartzoni/thesis/ensemble/2_exp' 
for i in range(1,6):
    modelTemp=load_model(ensemble_path +  '/' + str(i) + '.h5') 
    modelTemp.name = 'model'+ str(i)
    models.append(modelTemp)
    
def ensembleModels(models, model_input):
    # collect outputs of models in a list
    yModels=[model(model_input) for model in models] 
    # averaging outputs
    yAvg=layers.average(yModels) 
    # build model from same input and avg output
    modelEns = Model(inputs=model_input, outputs=yAvg,    name='ensemble')  
    return modelEns
    
model_input = Input(shape=models[0].input_shape[1:]) # c*h*w
model = ensembleModels(models, model_input)
#model.summary()
#idia metatropi eikonon
# evaluation_train / evaluation_retrain

img_rows,img_cols,img_channels = 299 ,299,3

       
evaluation_folder = '/home/gkartzoni/thesis/experiments/ensemble'+experiment_name
try:  
    os.mkdir(evaluation_folder)
except OSError:  
    print ("Creation of the directory %s failed" % evaluation_folder)
else:  
    print ("Successfully created the directory %s " % evaluation_folder)

model.compile(
    optimizer = SGD(lr=lr,decay=decay),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

select = 1# select 1 for kaggle, 2 for messidor
if select == 1:
  image_path0 =  '/home/gkartzoni/thesis/images/test/0'
  image_path1 =   '/home/gkartzoni/thesis/images/test/1'
  file_extension  = '*.jpeg' #For messidor 2 use '*.tif' jpeg
elif select == 2:
   image_path0 = '/home/gkartzoni/thesis/images/2classesMessidor_RDR/class0'
   image_path1 =  '/home/gkartzoni/thesis/images/2classesMessidor_RDR/class1'
   file_extension  = '*.tif' #For messidor 2 use '*.tif' jpeg


num_classes = 1
X = []
Y = []



data_path = os.path.join(image_path0,file_extension)
files = glob.glob(data_path)
normalizedImg = np.zeros((img_rows, img_cols,3))

i = 1;
for image_file in files:
    img = cv2.imread(image_file)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB )   #img = cv2.imread(image_file)
    img = cv2.resize(img, dsize=(img_rows, img_cols), interpolation=cv2.INTER_CUBIC)
    img = preprocess_input(img)
    X.append(img)  #  if i<2:    

  #  img = cv2.resize(img, dsize=(img_rows, img_cols), interpolation=cv2.INTER_CUBIC)
  #  normalizedImg = np.zeros((256, 256,3))
#    img = equalize_hist(img)
#    normalizedImg = cv2.normalize(img,  normalizedImg, 0, 1, cv2.NORM_MINMAX) 
  #  img = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB )   #img = cv2.imread(image_file)
   # if i<2:
   #     matplotlib.image.imsave(evaluation_folder+'/'+str(i)+'.PNG', normalizedImg)   
   # i = i +1
    Y.append(0)



data_path = os.path.join(image_path1,file_extension)
files = glob.glob(data_path)
 # #for i in range(1,len(image_name)+1):
i = 1;
for image_file in files:
    img = cv2.imread(image_file)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB )   #img = cv2.imread(image_file)
    img = cv2.resize(img, dsize=(img_rows, img_cols), interpolation=cv2.INTER_CUBIC)
    img = preprocess_input(img)
    X.append(img)  #  if i<2:
  #      matplotlib.image.imsave(evaluation_folder+'/'+str(i)+'.PNG', normalizedImg)   
  #  i = i +1
    Y.append(1)


# for i in range(0,2):
    # matplotlib.image.imsave(evaluation_folder+'/'+str(i)+'.png', X[i])   
 

X_test = np.asarray(X)
Y_test = np.asarray(Y)



#for i in range(500,550):
#    matplotlib.image.imsave(evaluation_folder+'/'+str(i)+'.png', X_test[i])   
json_metrics = mf.model_evaluation_1classOutput(model,X_test,Y_test,num_classes,evaluation_folder)
#AUC
json_metrics = json.loads(json_metrics)
plt.plot(json_metrics['fpr'],json_metrics['tpr'], 'b', label = 'AUC = %0.2f' %json_metrics['roc_auc'])
plt.title('Receiver Operating Characteristic')
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
plt.savefig(evaluation_folder+'/AUC.png', bbox_inches='tight')
plt.clf()

src = '/home/gkartzoni/thesis/source/ensemble.py'
dst = evaluation_folder + '/ensemble.py'
copyfile(src, dst)	

with open(evaluation_folder+'/metrics.json', 'w') as json_file:  
     json.dump(json_metrics, json_file)




