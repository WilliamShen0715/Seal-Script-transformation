# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 02:08:33 2022

@author: USER
"""

import tensorflow as tf
import cv2
import numpy as np
from matplotlib import pyplot as plt

def Loading_Image( image_path ):
    img = cv2.imread(image_path,-1)
    img = tf.constant( np.array(img) )  
    return img

def Show( image, title=None ) :
    if len( image.shape )>3 :
        image = tf.squeeze( image, axis=0 )   

    plt.imshow( image )
    if title:
        plt.title( title )

def Plot_Genetor( imgs, labels=[], grid=(3,10), size=(16,4) ):
    n = len( imgs )
    plt.gcf().set_size_inches(size) 
    for i in range(n):           
        ax = plt.subplot( grid[0], grid[1], i+1 )   
        ax.imshow( imgs[i] )            
        if len(labels):
            ax.set_title( f'Label={labels[i]}' )   
        ax.set_xticks([]); ax.set_yticks([]) 
    plt.show()

def Plot_Predict( index,imgs, labels=[], predicts=[], grid=(3,10), size=(15,10) ):                
    n = len( imgs )             
    plt.gcf().set_size_inches(size) 
    for i in range(n):           
        ax = plt.subplot( grid[0], grid[1], i+1 )   
        ax.imshow( imgs[i] )            
        if len(labels):
            ax.set_title( f'Predict={index[predicts[i]]} \nLabel={index[str(np.argmax(labels[i], axis=0))]}' )   
        ax.set_xticks([]); ax.set_yticks([]) 
    plt.show() 