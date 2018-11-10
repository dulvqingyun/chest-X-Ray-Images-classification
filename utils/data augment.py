#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  2 14:50:09 2018

@author: dulvqingyun
"""

# -*- coding: utf-8 -*-

 
 # import packages
import os


from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
 
datagen = ImageDataGenerator(
         rotation_range=30,
         width_shift_range=0.2,
         height_shift_range=0.2,
         shear_range=0.2,
         zoom_range=0.2,
         horizontal_flip=True,
         fill_mode='nearest')
 
class_path='./chest_xray/train/NORMAL/'


    
for filename in os.listdir(class_path):
    if filename.endswith('jpeg'):
        
        fullfilename = class_path + filename
        img = load_img(fullfilename) 
        x = img_to_array(img)  
        x = x.reshape((1,) + x.shape)  
        i=0
        for batch in datagen.flow(x,batch_size=1,
                     save_to_dir= './chest_xray/train/NORMAL',
                     save_prefix=os.path.splitext(filename)[0]+'pre',
                     save_format='jpeg'):
             i+=1
             if i>1:
                 break

 
