# -*- coding: utf-8 -*-
"""
Created on Wed Jun 16 09:54:39 2021

@author: Marc
"""

import numpy as np
import tensorflow as tf
import keras
import os
from keras import backend as K
import hyperspy.api as hs
import tkinter 
from tkinter import filedialog
import matplotlib.pyplot as plt

data1=np.loadtxt(r'E:\Arxius varis\PhD\2nd_year\Code\datasets\FFT_peak_intensity_profiles\vertical_profiles\Prof1.txt')
print(data1)
x1=data1[:,0]
y1=data1[:,1]

#center the x to 0 at the maximum (peak center)
max_intensity_coord=x1[np.argmax(y1)]
x1=x1-max_intensity_coord
#normalise intensity from 0 to 1
y1=(y1-np.min(y1))/(np.max(y1)-np.min(y1))

print(x1)
print(y1)
dirname=r'E:\Arxius varis\PhD\2nd_year\Code\datasets\FFT_peak_intensity_profiles\vertical_profiles'

x_tots=np.array([])
fft_calibs=np.array([])
y_tots=np.array([])

for exp_file in os.listdir(dirname):
    filename=dirname+'/'+exp_file
    data1=np.loadtxt(filename)

    x1=data1[:,0]
    y1=data1[:,1]
    fft_calib=np.abs(x1[1]-x1[0])
    fft_calib_array=np.ones(np.shape(x1)[0])*fft_calib
    
    #center the x to 0 at the maximum (peak center)
    max_intensity_coord=x1[np.argmax(y1)]
    x1=x1-max_intensity_coord
    #normalise intensity from 0 to 1
    y1=(y1-np.min(y1))/(np.max(y1)-np.min(y1))
    
    x_tots=np.concatenate((x_tots, x1))
    fft_calibs=np.concatenate((fft_calibs, fft_calib_array))
    y_tots=np.concatenate((y_tots, y1))
    

print(np.shape(x_tots))
print(np.shape(fft_calibs))
print(np.shape(y_tots))

#set the variables in the correct formating of rows and columns: rows are training example and column variables
x_tots=np.reshape(x_tots, (np.shape(x_tots)[0],1))
fft_calibs=np.reshape(fft_calibs, (np.shape(fft_calibs)[0],1))
training_set=np.hstack((x_tots,fft_calibs))
y_tots=np.reshape(y_tots, (np.shape(y_tots)[0],1))
print(training_set)
print(np.shape(training_set))
plt.scatter(x_tots, y_tots)
plt.show()
#create the machine learning model: shallow neural network
initializer = keras.initializers.glorot_normal()
opt = keras.optimizers.SGD(lr=0.01, momentum=0.9)

nn_model=keras.models.Sequential()
nn_model.add(keras.layers.Dense(10, input_dim=2, activation='sigmoid', kernel_initializer=initializer))
nn_model.add(keras.layers.BatchNormalization())
nn_model.add(keras.layers.Dense(20, activation='sigmoid', kernel_initializer=initializer))
nn_model.add(keras.layers.BatchNormalization())
nn_model.add(keras.layers.Dense(30, activation='sigmoid', kernel_initializer=initializer))
nn_model.add(keras.layers.BatchNormalization())
nn_model.add(keras.layers.Dense(20, activation='sigmoid', kernel_initializer=initializer))
nn_model.add(keras.layers.BatchNormalization())
nn_model.add(keras.layers.Dense(10, activation='sigmoid', kernel_initializer=initializer))
nn_model.add(keras.layers.BatchNormalization())
nn_model.add(keras.layers.Dense(1, activation='relu', kernel_initializer=initializer))

nn_model.compile(loss='mean_squared_error', optimizer=opt, metrics=['accuracy'])
nn_model.fit(training_set, y_tots, epochs=500, batch_size=32)

_, accuracy = nn_model.evaluate(training_set, y_tots)
print('Accuracy: %.2f' % (accuracy*100))


nn_model.save(r'E:\Arxius varis\PhD\2nd_year\Code\datasets\FFT_peak_intensity_profiles\trained_models\global_model.h5')

data1=np.loadtxt(r'E:\Arxius varis\PhD\2nd_year\Code\datasets\FFT_peak_intensity_profiles\vertical_profiles\Prof1.txt')

x1=data1[:,0]
y1=data1[:,1]

x1=np.arange(-4,4,0.01)
fft_calib=np.abs(x1[1]-x1[0])
fft_calib_array=np.ones(np.shape(x1)[0])*fft_calib

max_intensity_coord=x1[np.argmax(y1)]
#x1=x1-max_intensity_coord

x1=np.reshape(x1, (np.shape(x1)[0],1))
fft_calibs=np.reshape(fft_calib_array, (np.shape(fft_calib_array)[0],1))
prediction_inputs=np.hstack((x1,fft_calibs))

prediction_outs_ref = nn_model.predict(prediction_inputs)
plt.scatter(prediction_inputs[:,0], prediction_outs_ref)
plt.show()

list1=[0.25,0.5,1,2,3,4]
for val in list1:
    fft_calib_array=np.ones(np.shape(x1)[0])*fft_calib
    fft_calibs=np.reshape(fft_calib_array, (np.shape(fft_calib_array)[0],1))
    prediction_inputs=np.hstack((x1,fft_calibs))
    prediction_outs_vals= nn_model.predict(prediction_inputs)
    plt.scatter(prediction_inputs[:,0], prediction_outs_ref-prediction_outs_vals)
    print(np.unique(prediction_outs_ref-prediction_outs_vals))
    plt.show()

#up to here, single model with the consideration of all the pixel sizes, but theres too litle info of each to make
#the pixel size parameter count, so it is finally an average of all of them, which can also be used


#%%

import numpy as np
import tensorflow as tf
import keras
import os
from keras import backend as K
import hyperspy.api as hs
import tkinter 
from tkinter import filedialog
import matplotlib.pyplot as plt


#here we train a model ofr each of the piel sizes considered. If a pixel size not considered is needed,
#then just use the closer one. Remember that all of these models are too simple to be picky about them
#they do not need to be perfect as it is just an approximation of the fringes


#no need for two variables here just the main one of the reciproca postiion

dirname=r'E:\Arxius varis\PhD\2nd_year\Code\datasets\FFT_peak_intensity_profiles\vertical_profiles'

x_tots=np.array([])
fft_calibs=np.array([])
y_tots=np.array([])

x_tots1=np.array([])
y_tots1=np.array([])

x_tots2=np.array([])
y_tots2=np.array([])

x_tots3=np.array([])
y_tots3=np.array([])

x_tots4=np.array([])
y_tots4=np.array([])

x_tots5=np.array([])
y_tots5=np.array([])
    

for exp_file in os.listdir(dirname):
    
    number_of_file=int(exp_file[exp_file.find('f')+1:exp_file.find('.')])
    filename=dirname+'/'+exp_file
    data1=np.loadtxt(filename)
    
    x1=data1[:,0]
    y1=data1[:,1]
    
    #center the x to 0 at the maximum (peak center)
    max_intensity_coord=x1[np.argmax(y1)]
    x1=x1-max_intensity_coord
    #normalise intensity from 0 to 1
    y1=(y1-np.min(y1))/(np.max(y1)-np.min(y1))
    
    
    if number_of_file>=1 and number_of_file<=5:
        #pixel size 4
        fft_pixel_size= 0.026294 #1/nm
        
        x_tots4=np.concatenate((x_tots4, x1))
        y_tots4=np.concatenate((y_tots4, y1))
        
    elif number_of_file>=6 and number_of_file<=8:
        #pixel size 1
        fft_pixel_size= 0.00929  #1/nm   
        
        x_tots1=np.concatenate((x_tots1, x1))
        y_tots1=np.concatenate((y_tots1, y1))
     
    elif number_of_file>=9 and number_of_file<=13:
        #pixel size 3
        fft_pixel_size= 0.01854  #1/nm    
        
        x_tots3=np.concatenate((x_tots3, x1))
        y_tots3=np.concatenate((y_tots3, y1))
    
    elif number_of_file>=14 and number_of_file<=20:
        #pixel size 5
        fft_pixel_size= 0.037185  #1/nm 
        
        x_tots5=np.concatenate((x_tots5, x1))
        y_tots5=np.concatenate((y_tots5, y1))
        
    else:
        # which is number_of_file>=21 and number_of_file<=23
        #pixel size 2
        fft_pixel_size= 0.013147  #1/nm   
        
        x_tots2=np.concatenate((x_tots2, x1))
        y_tots2=np.concatenate((y_tots2, y1))
        
        
        
x_tots1=np.reshape(x_tots1, (np.shape(x_tots1)[0],1))
y_tots1=np.reshape(y_tots1, (np.shape(y_tots1)[0],1))

x_tots2=np.reshape(x_tots2, (np.shape(x_tots2)[0],1))
y_tots2=np.reshape(y_tots2, (np.shape(y_tots2)[0],1))

x_tots3=np.reshape(x_tots3, (np.shape(x_tots3)[0],1))
y_tots3=np.reshape(y_tots3, (np.shape(y_tots3)[0],1))

x_tots4=np.reshape(x_tots4, (np.shape(x_tots4)[0],1))
y_tots4=np.reshape(y_tots4, (np.shape(y_tots4)[0],1))

x_tots5=np.reshape(x_tots5, (np.shape(x_tots5)[0],1))
y_tots5=np.reshape(y_tots5, (np.shape(y_tots5)[0],1))


#train the models

x_tots_list=[x_tots1, x_tots2,x_tots3,x_tots4,x_tots5]
y_tots_list=[y_tots1,y_tots2,y_tots3,y_tots4,y_tots5]

initializer = keras.initializers.glorot_normal()
opt = keras.optimizers.SGD(lr=0.01, momentum=0.9)
for index,(training_set, y_tots) in enumerate(zip(x_tots_list, y_tots_list)):
    #create the machine learning model: shallow neural network
    plt.scatter(training_set, y_tots)
    
    nn_model_i=keras.models.Sequential()
    nn_model_i.add(keras.layers.BatchNormalization())
    nn_model_i.add(keras.layers.Dense(10, input_dim=1, activation='sigmoid', kernel_initializer=initializer))
    nn_model_i.add(keras.layers.BatchNormalization())
    nn_model_i.add(keras.layers.Dense(20, activation='sigmoid', kernel_initializer=initializer))
    nn_model_i.add(keras.layers.BatchNormalization())
    nn_model_i.add(keras.layers.Dense(30, activation='sigmoid', kernel_initializer=initializer))
    nn_model_i.add(keras.layers.BatchNormalization())
    nn_model_i.add(keras.layers.Dense(20, activation='sigmoid', kernel_initializer=initializer))
    nn_model_i.add(keras.layers.BatchNormalization())
    nn_model_i.add(keras.layers.Dense(10, activation='sigmoid', kernel_initializer=initializer))
    nn_model_i.add(keras.layers.BatchNormalization())
    nn_model_i.add(keras.layers.Dense(1, activation='relu', kernel_initializer=initializer))
    
    nn_model_i.compile(loss='mean_squared_error', optimizer=opt, metrics=['accuracy'])
    nn_model_i.fit(training_set, y_tots, epochs=400, batch_size=16, verbose=0 )
    
    _, accuracy = nn_model_i.evaluate(training_set, y_tots, verbose=0 )
    print('Accuracy: %.2f' % (accuracy*100))
    savepath=r'E:\Arxius varis\PhD\2nd_year\Code\datasets\FFT_peak_intensity_profiles\trained_models'+'/local_pixsize_model'+str(index+1)+'.h5'
    nn_model_i.save(savepath)
    #check resulting model
    x1=np.arange(-3,3,0.01)
    fft_calib=np.abs(x1[1]-x1[0])
    fft_calib_array=np.ones(np.shape(x1)[0])*fft_calib
    
    max_intensity_coord=x1[np.argmax(y1)]
    #x1=x1-max_intensity_coord
    
    x1=np.reshape(x1, (np.shape(x1)[0],1))
    
    prediction_outs_ref = nn_model_i.predict(x1)
    plt.scatter(x1, prediction_outs_ref)
    plt.show()
  
    
  
#%%

#work with the created models and lead them for generating the fringes

import numpy as np
import tensorflow as tf
from tensorflow.keras import initializers
import keras
import os
from keras import backend as K
import hyperspy.api as hs
import tkinter 
from tkinter import filedialog
import matplotlib.pyplot as plt
 



#!!! IMPORTANT: When applying this to the FFTs be sure to apply this to half of the image,
#and then just create the other half of the image as the centrosymmetric representation of that half

#!!! hyperparameters
#calibate_image
fft_calib_pixsize=0.01  #1/nm
#define lenght of line
line_length= 1 #1/nm


def compute_gaussian_value(pixel_x, pixel_y, x_coord, y_coord, radiix, radiiy):
    #let us fix the stdev as radii/4 for sake of a more visible plot
    desvest_x=radiix/4
    desvest_y=radiiy/4
    
    gaussian_val=1*np.exp(-(((pixel_x-x_coord)**2/(2*desvest_x**2))+((pixel_y-y_coord)**2/(2*desvest_y**2))))
    
    return gaussian_val


def gaussian_circle_generator(image):
    
    pixels_x=np.shape(image)[1]
    pixels_y=np.shape(image)[0]
    
    x_coord=int(np.floor(pixels_x/2))
    y_coord=int(np.floor(pixels_y/2))
    radiix=np.floor(x_coord/2)
    radiiy=np.floor(y_coord/2)
    
    
    for pixel_y in range(int(pixels_y)):
        for pixel_x in range(int(pixels_x)):
            if ((pixel_x-x_coord)/radiix)**2+((pixel_y-y_coord)/radiiy)**2<1:
                
                gaussian_val=compute_gaussian_value(pixel_x, pixel_y, x_coord, y_coord, radiix, radiiy)
                image[pixel_y,pixel_x]=gaussian_val
                #set conditions for gaussian distribution
    
    return image


image_circle=np.zeros((20,20))

image_circle=gaussian_circle_generator(image_circle)

plt.imshow(image_circle)



#choose the model according to the pixel size, can be done also with the general and this could be skipped
#global model
# loadpath=r'E:\Arxius varis\PhD\2nd_year\Code\datasets\FFT_peak_intensity_profiles\trained_models\global_model.h5'
# model_i=keras.models.load_model(loadpath)

fft_reference_pixel_sizes=np.array([0.00929,0.013147,0.01854,0.026294,0.037185])
#index referencing the fft pixel size closer to the wanted one
index_reference=np.argmin(np.abs(fft_reference_pixel_sizes-fft_calib_pixsize))+1
fft_calib_pixsize=fft_reference_pixel_sizes[index_reference-1]

loadpath=r'E:\Arxius varis\PhD\2nd_year\Code\datasets\FFT_peak_intensity_profiles\trained_models'+'/local_pixsize_model'+str(index_reference)+'.h5'
model_i=keras.models.load_model(loadpath)

#create fake long image
long_image_trial=np.zeros((320,20))

long_image_trial[int(np.floor(np.shape(long_image_trial)[0]/2))-int(np.floor(np.shape(image_circle)[0]/2)):int(np.floor(np.shape(long_image_trial)[0]/2))-int(np.floor(np.shape(image_circle)[0]/2))+int(np.shape(image_circle)[0]),:]=image_circle
plt.imshow(long_image_trial)
long_image_trial_gaussian_ref=np.copy(long_image_trial)



total_pixels=int(np.floor(line_length/fft_calib_pixsize))  #per each size 

#create the size of the wanted reciprocal size
x1=np.arange(-line_length,line_length,line_length/total_pixels)
line_profile_values_ref = model_i.predict(x1)

#normalise the values given the maximum value per each column or row dependeing on horizontal or vertical line definition

for column_index in range(np.shape(long_image_trial)[1]):
    column=long_image_trial[:, column_index]
    #given that all the values are between 0 and 1, we can directly normalise from this
    
    line_profile_values=np.copy(line_profile_values_ref)
    max_val_column=np.max(column)
    index_max_val_column=np.argmax(column)
    
    #normalise intensity from 0 to the max value in each column
    norm_line_profile_vals=max_val_column*(line_profile_values-np.min(line_profile_values))/(np.max(line_profile_values)-np.min(line_profile_values))
    norm_line_profile_vals=np.reshape(norm_line_profile_vals, (np.shape(norm_line_profile_vals)[0],))
    #substitute the column to the array
    #center the substitution where the maximum for eahc column was found
    
    if max_val_column==0:
        continue
    else:
        if total_pixels<=int(np.shape(long_image_trial)[0]/2):
            long_image_trial[index_max_val_column-int(np.floor(np.shape(norm_line_profile_vals)[0]/2)):index_max_val_column-int(np.floor(np.shape(norm_line_profile_vals)[0]/2))+np.shape(norm_line_profile_vals)[0], column_index]=norm_line_profile_vals
        else:
            long_image_trial[:, column_index]=norm_line_profile_vals[:np.shape(long_image_trial)[0]]


plt.imshow(long_image_trial)

#do the same in the horizontal direction

square_image_trial=np.zeros((320,320))
square_image_gaussian_ref=np.copy(square_image_trial)

square_image_gaussian_ref[:,int(np.floor(np.shape(square_image_gaussian_ref)[0]/2))-int(np.floor(np.shape(image_circle)[1]/2)):int(np.floor(np.shape(square_image_gaussian_ref)[0]/2))-int(np.floor(np.shape(image_circle)[1]/2))+int(np.shape(image_circle)[1])]=long_image_trial_gaussian_ref
plt.imshow(square_image_gaussian_ref)
plt.show()

square_image_trial[:,int(np.floor(np.shape(square_image_gaussian_ref)[0]/2))-int(np.floor(np.shape(image_circle)[1]/2)):int(np.floor(np.shape(square_image_gaussian_ref)[0]/2))-int(np.floor(np.shape(image_circle)[1]/2))+int(np.shape(image_circle)[1])]=long_image_trial
plt.imshow(square_image_trial)
plt.show()

'''
#do the process but referencing the square image with the gaussian circle and substituting in the 
#square but with vertical line image
for row_index in range(np.shape(square_image_gaussian_ref)[0]):
    row=square_image_gaussian_ref[row_index,:]
    #given that all the values are between 0 and 1, we can directly normalise from this
    
    line_profile_values=np.copy(line_profile_values_ref)
    
    max_val_row=np.max(row)
    index_max_val_row=np.argmax(row)
    
    #normalise intensity from 0 to the max value in each column
    norm_line_profile_vals=max_val_row*(line_profile_values-np.min(line_profile_values))/(np.max(line_profile_values)-np.min(line_profile_values))
    norm_line_profile_vals=np.reshape(norm_line_profile_vals, (np.shape(norm_line_profile_vals)[0],))
    #substitute the column to the array
    
    if max_val_row==0:
        continue
    else:
        if total_pixels<=int(np.shape(long_image_trial)[0]/2):
            square_image_trial[row_index,index_max_val_row-int(np.floor(np.shape(norm_line_profile_vals)[0]/2)):index_max_val_row-int(np.floor(np.shape(norm_line_profile_vals)[0]/2))+np.shape(norm_line_profile_vals)[0]]=norm_line_profile_vals
        else:
            square_image_trial[row_index,:]=norm_line_profile_vals[:np.shape(square_image_trial)[1]]
        
plt.figure(figsize = (100,50))   
plt.imshow(square_image_trial)
plt.show()

'''
#with the previous loop some pixels remain very light and produce this kind of division
#between the horizontal and vertical lines. to avoid this, just threhsold the intensity up to which
#the horizontal line will be created


for row_index in range(np.shape(square_image_gaussian_ref)[0]):
    row=square_image_gaussian_ref[row_index,:]
    #given that all the values are between 0 and 1, we can directly normalise from this
    
    line_profile_values=np.copy(line_profile_values_ref)
    
    max_val_row=np.max(row)
    index_max_val_row=np.argmax(row)
    if max_val_row>0.5:
        #normalise intensity from 0 to the max value in each column
        norm_line_profile_vals=max_val_row*(line_profile_values-np.min(line_profile_values))/(np.max(line_profile_values)-np.min(line_profile_values))
        norm_line_profile_vals=np.reshape(norm_line_profile_vals, (np.shape(norm_line_profile_vals)[0],))
        #substitute the column to the array
        
        if max_val_row==0:
            continue
        else:
            if total_pixels<=int(np.shape(long_image_trial)[0]/2):
                square_image_trial[row_index,index_max_val_row-int(np.floor(np.shape(norm_line_profile_vals)[0]/2)):index_max_val_row-int(np.floor(np.shape(norm_line_profile_vals)[0]/2))+np.shape(norm_line_profile_vals)[0]]=norm_line_profile_vals
            else:
                square_image_trial[row_index,:]=norm_line_profile_vals[:np.shape(square_image_trial)[1]]
        
plt.figure(figsize = (100,50))   
plt.imshow(square_image_trial)
plt.show()



