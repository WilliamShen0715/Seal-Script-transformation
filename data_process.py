# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 02:32:32 2022

@author: USER
"""
from pHash import aHash,cmpHash
import os
import random
import cv2
import numpy as np
from tensorflow.keras.models import load_model

def group_sampling(Num_Classes,TraningDataPath,duplicate_times,group_num,group_sample_amount,test_data):
    ngruop = []
    num_of_element = int(Num_Classes/group_num)
    hash1= aHash(test_data)
    for i in range(group_num):
        minhash=100
        for j in range(num_of_element):
            place = os.listdir(TraningDataPath+'/'+str(i))[j]
            # print(place)
            realplace = os.listdir(TraningDataPath+'/'+str(i)+'/'+place)[random.randint(1,group_sample_amount)]
            # print(realplace)
            # print(TraningDataPath+'/'+str(i)+'/'+place+'/'+realplace)
            img2=cv2.imread(TraningDataPath+'/'+str(i)+'/'+place+'/'+realplace)
            hash2= aHash(img2)
            n=cmpHash(hash1,hash2)
            if n < minhash:
                minhash = n
        ngruop.append(minhash)
    print("ngruop:",ngruop)
    return np.argmin(ngruop)

def module_predict(Num_Classes,TraningDataPath,test_data,duplicate_times,group_num,group_sample_amount):
    Predicts=[]
    for i in range(len(test_data)):
        group = group_sampling(Num_Classes,TraningDataPath,duplicate_times,group_num,group_sample_amount,test_data[i])
        CNN = load_model( 'CNN_Model_'+str(group)+'.h5' )
        #print(CNN.predict( test_data ))
        Predicts.append(os.listdir(TraningDataPath+'/'+str(group))[int(np.argmax(CNN.predict( test_data )[i]))])
    return Predicts