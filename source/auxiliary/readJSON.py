# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 14:01:05 2019

@author: Ioanna
"""
import numpy as np

import json
from pprint import pprint
#path = '/home/gkartzoni/thesis/experiments/ensembles/ensemble5model/ensembleM/metrics.json'
#path = '/home/gkartzoni/thesis/experiments/54/EvaluationK/metrics.json'
#path = '/home/gkartzoni/thesis/experiments/25/EvaluationM2/metrics.json'
path = '/home/gkartzoni/thesis/experiments/Messidor_EF_notWeightedLast/metrics.json'

with open(path) as data_file:    
    data = json.load(data_file)
l = len(data['fpr'])
dat = np.zeros((l,2))
#pprint(data['fpr'])
dat[:,0] = data['tpr']
dat[:,1] = data['fpr']
tr = data['threshold']

i = 0;

c= np.zeros((l,1))
c_maxSens = np.zeros((l,1))
c_maxSpec = np.zeros((l,1))
max_val = 0.9

for i in range(0,l-1):
    c[i] =  (dat[i,0]) + (1-dat[i,1])
    c_maxSens[i] = 0
    c_maxSpec[i] = 0
    
    if (dat[i,0]) > max_val:
        c_maxSens[i] = c[i]
    if (1-dat[i,1]) > max_val:
        c_maxSpec[i] = c[i]
        
selected_op = np.argmax(c)
selected_opMaxSens = np.argmax(c_maxSens)
selected_opMaxSpec = np.argmax(c_maxSpec)
avrOP = [dat[selected_op,0],(1-dat[selected_op,1]),tr[selected_op] ]
maxSpecOP = [dat[selected_opMaxSens,0],1-dat[selected_opMaxSens,1],tr[selected_opMaxSens]]
maxSensOP = [dat[selected_opMaxSpec,0],(1-dat[selected_opMaxSpec,1]),tr[selected_opMaxSpec]]

print(avrOP)
print(maxSensOP)
print(maxSpecOP)


    #cost[i] =  coef_x * dat[i,0] + coef_y * (1-dat[i,1])
   # print(c[i])
#print(cost.index(min(cost)))
#print( np.argmax(c))
#print(1-dat[np.argmax(c),0])
#print(dat[np.argmax(c),1])

#dat[::-1,1] = data['tpr']
#pprint(dat)

#dat = data['fpr']
#tpr =  data['tpr']