# -*- coding: utf-8 -*-
"""
Created on Mon Dec 13 21:11:54 2021

@author: William0715
"""

from platform import python_version
import os
import numpy as np

from matplotlib import pyplot as plt
import cv2

plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei'] 
plt.rcParams['axes.unicode_minus'] = False


import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout  
from tensorflow.keras.optimizers import Adam

print( 'Python Version: ', python_version() )
print( 'TensorFlow Version: ', tf.__version__ )
print( 'Keras Version: ', tf.keras.__version__ ) 
#%% dataset path

RawDataPath = 'C:/Users/USER/Downloads/dataset'
TraningDataPath = 'train'
TestingDataPath = 'test'

os.chdir( RawDataPath )
print( 'Current working directory:', os.getcwd() ) 

#%%
Num_Classes = 

Image_Size = ( 50, 50 )

Epochs = 

Batch_Size = 

group_num = 

group_sample_amount = 

duplicate_times = 
#%% p-hash for img clustering
#avghash
similarity_matrix=[]
from pHash import aHash, cmpHash

for i in range(0,Num_Classes):
    img1=cv2.imread('./test/'+str(i%Num_Classes)+'.jpg')
    hash1= aHash(img1)
    print ('similarity pointer '+str(i)+':')
    sim=np.zeros((1,Num_Classes))
    for j in range(0,Num_Classes-1):
        img2=cv2.imread('./test/'+str((i+j+1)%Num_Classes)+'.jpg')
        hash2= aHash(img2)
        n=cmpHash(hash1,hash2)
        sim[0][(i+j+1)%Num_Classes] = n
    print(sim)
    similarity_matrix.append(sim[0])
#%% sample and find group
num_of_element = int(Num_Classes/group_num)
lookup_table = []
lookup_buffer = []
for i in range(Num_Classes):
    lookup_table.append(0)
    lookup_buffer.append(0)
for i in range(group_num):
    itr=num_of_element
    while itr!=0:
        if lookup_buffer[np.argmin(similarity_matrix[i])]==0:
            print(np.argmin(similarity_matrix[i]),':')
            lookup_table[np.argmin(similarity_matrix[i])]=i
            lookup_buffer[i]=1
            lookup_buffer[np.argmin(similarity_matrix[i])]=1
            similarity_matrix[i][np.argmin(similarity_matrix[i])]=100
            itr=itr-1
            print(similarity_matrix[i])
        else:
            similarity_matrix[i][np.argmin(similarity_matrix[i])]=100

#%% generate training data to aug
for i in range(Num_Classes):
    os.mkdir(TestingDataPath+'/'+str(i))
    os.replace(TestingDataPath+'/'+str(i)+'.jpg', TestingDataPath+'/'+str(i)+'/'+str(i)+'.jpg')
#%% data augmentation
Train_Data_Genetor = ImageDataGenerator( rescale = 1./255, validation_split = 0.2,
                                width_shift_range = 0.05,   
                                height_shift_range = 0.05,
                                zoom_range = 0.1,  
                                horizontal_flip = False )
Train_Generator = Train_Data_Genetor.flow_from_directory( TestingDataPath ,
                                                        target_size = Image_Size,
                                                        batch_size = Batch_Size,
                                                        class_mode = 'categorical',
                                                        shuffle = True,
                                                        save_to_dir='aug',
                                                        save_format='jpg',
                                                        subset = 'training')
for i in range(duplicate_times):
    Train_Generator.next()
#%% send aug to train
for i in range(group_num):
    os.mkdir(TraningDataPath+'/'+str(i))
    for j in range(len(lookup_table)):
        if lookup_table[j]==i:
            os.mkdir(TraningDataPath+'/'+str(i)+'/'+str(j))
#%%
for i in range(group_num):
    rd=0
    for j in range(len(lookup_table)):
        if lookup_table[j]==i:
            for filename in os.listdir(RawDataPath+'/aug'):
                if int(filename[1]) < num_of_element+10:
                    rd+=1
                    os.replace(RawDataPath+'/aug/'+filename, RawDataPath+'/'+TraningDataPath+'/'+str(lookup_table[int(filename[1])])+'/'+filename[1]+'/'+filename[1]+'_'+str(rd)+'.jpg')

#%% grouping validation data and testing data
Val_Data_Genetor = ImageDataGenerator( rescale=1./255, validation_split = 0.2 )
Val_Generator=[]
Train_Generator_use=[]
for i in range(group_num):
    Train_Generator_use.append(Train_Data_Genetor.flow_from_directory( TraningDataPath+'/'+str(i) ,
                                                            target_size = Image_Size,
                                                            batch_size = Batch_Size,
                                                            class_mode = 'categorical',
                                                            shuffle = True,
                                                            save_to_dir='aug',
                                                            save_format='jpg',
                                                            subset = 'training'))
    Val_Generator.append(Train_Data_Genetor.flow_from_directory( TraningDataPath+'/'+str(i),
                                                        target_size = Image_Size,
                                                        batch_size = Batch_Size,
                                                        class_mode = 'categorical',
                                                        shuffle = True, 
                                                        subset = 'validation' ))
#%%
CNN = Sequential( name = 'CNN_Model' )
CNN.add( Conv2D( 5, kernel_size = (2,2), padding = 'same', 
                 input_shape = (Image_Size[0],Image_Size[1],3), name = 'Convolution' ) )
CNN.add( MaxPooling2D( pool_size = (2,2), name = 'Pooling' ) )
CNN.add( Flatten( name = 'Flatten' ) )
CNN.add( Dropout( 0.5, name = 'Dropout_1' ) )
CNN.add( Dense( 512, activation = 'relu', name = 'Dense' ) )
CNN.add( Dropout( 0.5, name = 'Dropout_2' ) )
CNN.add( Dense( num_of_element, activation = 'softmax', name = 'Softmax' ) )
CNN.summary()

CNN.compile( optimizer = Adam(),
             loss = 'categorical_crossentropy', 
             metrics = ['accuracy'] )
#%%
History = []
for i in range(group_num):
    my_History = CNN.fit( Train_Generator_use[i],
                       steps_per_epoch = Train_Generator_use[i].samples//Batch_Size,
                       validation_data = Val_Generator[i],
                       validation_steps = Val_Generator[i].samples//Batch_Size,
                       epochs = Epochs )
    CNN.save( 'CNN_Model_'+str(i)+'.h5' )
    History.append(my_History)
#%%
Train_Accuracy =np.zeros((1,len(History[0].history['accuracy'])))
# Val_Accuracy=np.zeros((1,len(History[0].history['val_accuracy'])))
# Train_Loss=np.zeros((1,len(History[0].history['loss'])))
# Val_Loss=np.zeros((1,len(History[0].history['val_loss'])))
for  i in range(group_num):
    
    Train_Accuracy = np.array(Train_Accuracy) + np.array(History[i].history['accuracy'])
    # Val_Accuracy = np.array(Val_Accuracy) + np.array(History[i].history['val_accuracy'])
    # Train_Loss = np.array(Train_Loss) + np.array(History[i].history['loss'])
    # Val_Loss = np.array(Val_Loss) + np.array(History[i].history['val_loss'])
epochs_range = range(Epochs)
Train_Accuracy = [i/group_num for i in Train_Accuracy]
# Val_Accuracy /= [i/group_num for i in Val_Accuracy]
# Train_Loss /= [i/group_num for i in Train_Loss]
# Val_Loss /= [i/group_num for i in Val_Loss]

plt.figure( figsize=(14,4) )
# plt.subplot( 1,2,1 )
plt.plot( range( len(Train_Accuracy[0]) ), Train_Accuracy[0], label='Train' ) 
# plt.plot( range( len(Val_Accuracy[0]) ), Val_Accuracy[0], label='Validation' ) 
plt.legend( loc='lower right' )
plt.title( 'Accuracy' )

# plt.subplot( 1,2,2 )
# plt.plot( range( len(Train_Loss[0]) ), Train_Loss[0], label='Train' )
# plt.plot( range( len(Val_Loss[0]) ), Val_Loss[0], label='Validation' )
# plt.legend( loc='upper right' )
# plt.title( 'Loss')

plt.show()