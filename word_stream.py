# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 02:57:57 2022

@author: USER
"""
from data_process import module_predict
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from img_process import Plot_Predict

RawDataPath = 'C:/Users/USER/Downloads/dataset'
TraningDataPath = 'train'
TestingDataPath = 'test'

Num_Classes = 

Image_Size = ( 50, 50 )

Epochs = 

Batch_Size = 

group_num = 

group_sample_amount = 

duplicate_times = 


Test_Data_Genetor = ImageDataGenerator( rescale=1./255 )
Test_Generator = Test_Data_Genetor.flow_from_directory( TestingDataPath,
                                                        target_size = Image_Size,
                                                        shuffle = False,
                                                        class_mode = 'categorical' )
#%% sample randomly from group to decide
# Loading pretrained model

test_data, test_label = Test_Generator.next()

Predicts = module_predict(Num_Classes,TraningDataPath,test_data,duplicate_times,group_num,group_sample_amount)


#%% plot prediction
import json
with open("number_dictionary.json", newline="",encoding="utf-8") as jsonfile:
    index = json.load(jsonfile)
Plot_Predict( index, test_data, test_label, Predicts )