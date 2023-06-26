# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 17:46:10 2023

@author: Marc
"""


import numpy as np
import os
import imutils
import cv2
import hyperspy.api as hs
import matplotlib.pyplot as plt
import sys
import tkinter
from tkinter import filedialog
import skimage.measure
from scipy.signal import argrelextrema
import stemtool
from sympy.utilities.iterables import multiset_permutations, permute_signs
import torch
import torch.nn

sys.path.append(r'E:\Arxius varis\PhD\3rd_year\Code\Functions')

# General functions

# Peak finding functions
sys.path.append(r'E:\Arxius varis\PhD\4rth_year\Code\Functions\Peak_detector_Final')
sys.path.append(r'E:\Arxius varis\PhD\4rth_year\Code\Functions\Ivans_Files_2\IVAN\Segmentation_model')

import PF_FFT_processing as FFT_Procs
import PF_Peaks_detector as Peaks_detector



def Peak_Detection_Wrapped_CNN(
        FFT_image_array_No_Log, FFT_image_array, total_pixels_crop, 
        crop_FOV, visualisation = False):
    '''
    Function wrapping the peak detection, final function: 2D scan of the FFT, the without the logarithm of
    the FFT
    Choose whether to visualise the overlapping of the found peak positions with the FFT with 
    the logairhtm

    Returns
    -------
    list_refined_cryst_spots.

    '''

    # new_FFT = np.zeros((np.shape(FFT_image_array)[0]+1, np.shape(FFT_image_array)[0]+1))
    # new_FFT[:np.shape(FFT_image_array)[0], :np.shape(FFT_image_array)[0]] = FFT_image_array
    
    # FFT_image_array = new_FFT
    
    # print('New FFT')
    # print(FFT_image_array, np.shape(FFT_image_array))
    
    # Load CNN model
    device = torch.device("cuda")
    
    CNN_model = torch.load(r'E:\Arxius varis\PhD\4rth_year\Code\Functions\Ivans_Files\DL2.pt')
    
    #CNN prediction
    CNN_prediction, _ = CNN_model.predict(FFT_image_array)
    del CNN_model
    
    
    print(CNN_prediction)
    print(np.shape(CNN_prediction))
    CNN_predishow = np.copy(CNN_prediction)
    CNN_predishow.shape = CNN_prediction.shape[1:]
    
    plt.imshow(CNN_predishow)
    plt.title('cnn pred')
    plt.show()
    
    #Final prediction output
    # CNN hyperparams
    CNN_side = 21 #It has to be a odd number
    CNN_treshold= 0.2 #It has to be less than 1
    
    # Get the binary matrix with the peaks found 
    
    # peaks_in_matrix, prediction_No_background = Peaks_detector.peaks_recognition_old(
    #     CNN_side, CNN_treshold, CNN_prediction, np.shape(FFT_image_array)[0])
    
    # peaks_in_matrix, prediction_No_background = Peaks_detector.peaks_recognition(
    #     CNN_side, CNN_treshold, CNN_prediction, FFT_image_array, np.shape(FFT_image_array)[0])
    
    peaks_in_matrix, prediction_No_background = Peaks_detector.peaks_recognition(
        CNN_side, CNN_treshold, CNN_prediction, FFT_image_array, total_pixels_crop)
    
    print('peaks in matrix len')
    print(np.shape(peaks_in_matrix))
    print(np.unique(peaks_in_matrix))
    
    # Extract the pixel coordinates
    pixels_of_peaks = Peaks_detector.peaks_image_to_coords(peaks_in_matrix)
    
    print('pixels of peaks')
    print(np.shape(pixels_of_peaks))
    
    # FFT_resized = cv2.resize(prediction[0], dsize=(size, size), interpolation=cv2.INTER_LINEAR)
    added_image, fft_color, peaks_color = Peaks_detector.show_peaks_detected(
        FFT_image_array, peaks_in_matrix, ratio = 0.8, plot = visualisation)
    plt.title('CNN:'+str(len(pixels_of_peaks))+' peaks found, pixel size '+str(crop_FOV/total_pixels_crop)+' FOV '+str(crop_FOV))
    plt.show()
    
    # First check if the !st approx peak finding finds 1,2 or more pixels
    # as this would tell if we have an amorphous material (1st central spot),
    # a non identifiable material (central and 1 additional), or some crystal identifiable
   
    return pixels_of_peaks, added_image, CNN_predishow



image_filepath  = r'E:\Arxius varis\PhD\2nd_year\Code\trial_images\dm3_atomic_resolution\GeQW2.dm3'


hyperspy_2Dsignal=hs.load(image_filepath)

x_calibration=hyperspy_2Dsignal.axes_manager['x'].scale
total_pixels=hyperspy_2Dsignal.axes_manager['x'].size
fov = x_calibration*total_pixels

image_arraynp=np.asarray(hyperspy_2Dsignal)

window_size = 2048
# window_size = 1234
# window_size = 123

_, windowFFT, FFT_image_array = FFT_Procs.experimental_fft_with_log(
    image_filepath, window_size, [0,0])

FFT_image_array_No_Log = FFT_Procs.experimental_fft(
    image_filepath, window_size, [0,0])

print('shapes before')
print(np.shape(FFT_image_array_No_Log))
print(np.shape(FFT_image_array))

FFT_image_array = cv2.resize(
    FFT_image_array, 
    dsize=(window_size, window_size), 
    interpolation=cv2.INTER_LINEAR)

plt.imshow(windowFFT)
plt.show()

plt.imshow(FFT_image_array)
plt.show()

print('shapes after')
print(np.shape(FFT_image_array_No_Log))
print(np.shape(FFT_image_array))




pixels_of_peaks, peaks_color, CNN_prediction = Peak_Detection_Wrapped_CNN(
    FFT_image_array_No_Log, FFT_image_array, 
    window_size, fov, visualisation = True)
print(len(pixels_of_peaks))



lista= np.arange(0, len(pixels_of_peaks))
for el in lista:

    peak_1 = pixels_of_peaks[el]
    
    FFT_cropped = FFT_image_array[peak_1[0]-20:peak_1[0]+20, peak_1[1]-20:peak_1[1]+20]
    
    FFT_CNN = CNN_prediction[peak_1[0]-20:peak_1[0]+20, peak_1[1]-20:peak_1[1]+20]
    
    plt.imshow(FFT_cropped)
    plt.show()
        
    plt.imshow(FFT_CNN)
    plt.show()
    
    FFT_cropped_peaks_col = peaks_color[peak_1[0]-20:peak_1[0]+20, peak_1[1]-20:peak_1[1]+20]
    
    plt.imshow(FFT_cropped_peaks_col)
    plt.show()
    
    
    
    
    
    