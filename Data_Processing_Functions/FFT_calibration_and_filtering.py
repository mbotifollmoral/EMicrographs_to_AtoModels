# -*- coding: utf-8 -*-
"""
Created on Fri Jan 15 18:42:46 2021

@author: Marc
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import scipy 
import skimage
import skimage.measure as measure
import sklearn.cluster
import cv2
import pycroscopy
import torch
import torch.utils.data
import numba
import atomai as aoi
import stemtool
import pyUSID as usid
import pyNSID
import sidpy as sid
import h5py
import pyfftw
import time
import hyperspy.api as hs



    
    
def Find_percentage_of_thresholded(FFT_image, threshold):
    FFT_image[FFT_image<=threshold]=0
    percentage_of_tresholded=np.count_nonzero(FFT_image.ravel())/len(FFT_image.ravel())
    return percentage_of_tresholded

def Threshold_given_percentage(FFT_image, percentage):
    y_pixs,x_pixs=np.shape(FFT_image)
    n_int_pixs=int(round(percentage*y_pixs*x_pixs))
    FFT_ravel=np.sort(np.ravel(FFT_image))[::-1]
    threshold=FFT_ravel[n_int_pixs]
    return threshold

def mpfit_Distance(FFT_image,FOV):
    mpfit_model=[[-2.87175127e-11],
                     [ 8.11320079e-09],
                     [-8.18658056e-07],
                     [ 3.33222163e-05],
                     [-2.02745223e-04],
                     [-2.26140649e-02],
                     [ 5.95346985e-01],
                     [-7.69005862e-01]]
    # without the averages
    mpfit_model_c=[[-3.46636981e-11],
                   [ 1.00423053e-08],
                   [-1.06223267e-06],
                   [ 4.84860471e-05],
                   [-6.82330526e-04],
                   [-1.58450088e-02],
                   [ 5.79540436e-01],
                   [-1.10510783e+00]]
    #set the working limits of the model
    if FOV >=30:
        mpfit_dist=np.array([40])
    else:
        
        fov_vals=np.array([FOV**7,FOV**6,FOV**5,FOV**4,FOV**3,FOV**2,FOV**1,1])
        mpfit_dist=np.e**np.dot(fov_vals,mpfit_model)
        mpfit_dist=np.e**np.dot(fov_vals,mpfit_model_c)
     
    #Adjustments depending on the sizze of the image
    if np.shape(FFT_image)[0]==2048:
        mpfit_dist=mpfit_dist*1.30
    elif np.shape(FFT_image)[0]<256:
        mpfit_dist=mpfit_dist*1.55     
    elif np.shape(FFT_image)[0]==256:
        mpfit_dist=mpfit_dist*1.55     
    elif np.shape(FFT_image)[0]==1024:
        mpfit_dist=mpfit_dist*1.05
    elif np.shape(FFT_image)[0]==512:
        mpfit_dist=mpfit_dist*1.15
    else:
        mpfit_dist=mpfit_dist*1.15
        
    return mpfit_dist[0]

def FFT_threshold(FOV):
    FFT_thresh_model=[[-1.01291174e-11],
                          [ 2.88297492e-09],
                          [-3.01778444e-07],
                          [ 1.44327587e-05],
                          [-3.23378617e-04],
                          [ 3.61163733e-03],
                          [-3.72515413e-02],
                          [-1.96361805e-01]]
    # without the averages
    FFT_thresh_model_c=[[ 1.54099057e-12],
                        [-6.56354380e-10],
                        [ 1.05878669e-07],
                        [-8.09680716e-06],
                        [ 2.96148198e-04],
                        [-4.30807411e-03],
                        [ 1.81389577e-03],
                        [-2.45698182e-01]]
    #set the working limits of the model
    if FOV >=80:
        FFT_thresh=np.array([0.6])
    else:
                
        fov_vals=np.array([FOV**7,FOV**6,FOV**5,FOV**4,FOV**3,FOV**2,FOV**1,1])
        FFT_thresh=np.e**np.dot(fov_vals,FFT_thresh_model)
        FFT_thresh=np.e**np.dot(fov_vals,FFT_thresh_model_c)
      
    
    return FFT_thresh[0]

def FFT_percentage(FFT_image,FOV):
    FFT_perc_model=[[-3.00411834e-11],
                        [ 1.17313244e-08],
                        [-1.81232383e-06],
                        [ 1.40635117e-04],
                        [-5.76020214e-03],
                        [ 1.20704617e-01],
                        [-1.20113823e+00],
                        [-2.14024711e+00]]
    # without the averages
    FFT_perc_model_c=[[ 1.38602821e-11],
                      [-2.46874956e-09],
                      [-1.63526870e-08],
                      [ 2.67725990e-05],
                      [-1.91230990e-03],
                      [ 5.28789844e-02],
                      [-6.40863899e-01],
                      [-3.71037505e+00]]
    #set the working limits of the model
    if FOV >=110:
        FFT_perc=np.array([0.00025])  #In case it is too much for higher FOVs, just delete this and keep the FFT_perc_model for all ranges
    # elif FOV <3:
    #     FFT_perc=np.array([0.01])
    else:
        
        
        fov_vals=np.array([FOV**7,FOV**6,FOV**5,FOV**4,FOV**3,FOV**2,FOV**1,1])
        FFT_perc=np.e**np.dot(fov_vals,FFT_perc_model)
        FFT_perc=np.e**np.dot(fov_vals,FFT_perc_model_c)        
        
        if FOV <4.5:
            FFT_perc=FFT_perc*(10**(np.log(128/np.shape(FFT_image)[0])/np.log(4)))
        elif FOV >=4.5 and FOV <=20 :
            FFT_perc=FFT_perc*(10**(np.log(512/np.shape(FFT_image)[0])/np.log(4)))
        else:
            FFT_perc=FFT_perc*(10**(np.log(2048/np.shape(FFT_image)[0])/np.log(4)))
    
    #Adjustments depending on the sizze of the image
    if np.shape(FFT_image)[0]<256:
        FFT_perc=FFT_perc*0.25
    elif np.shape(FFT_image)[0]==256:
        FFT_perc=FFT_perc*0.45  
    elif np.shape(FFT_image)[0]==512:
        FFT_perc=FFT_perc*0.55
    elif np.shape(FFT_image)[0]==1024:
        FFT_perc=FFT_perc*0.80    
    else:
        FFT_perc=FFT_perc*0.80
        
    return FFT_perc[0]

def FFT_hyperparams(FFT_image,FOV):
    #Return, in order, the mpfit dist, the threshold, and the percentage
    
    mpfit_dist=mpfit_Distance(FFT_image,FOV)
    FFT_thresh=FFT_threshold(FOV)
    FFT_perc=FFT_percentage(FFT_image,FOV)
    
    return mpfit_dist,FFT_thresh,FFT_perc



def FFT_calibration(hyperspy_2D_signal):
    fft_shifted = hyperspy_2D_signal.fft(shift=True)
    
    FFT_calibration=fft_shifted.axes_manager['x'].scale
    FFT_pixels=fft_shifted.axes_manager['x'].size
    FFT_units=fft_shifted.axes_manager['x'].units
    
    return FFT_calibration,FFT_pixels,FFT_units
    

def Spot_coord_To_d_spacing_indiv(coords, FFT_calibration, FFT_pixels):
    
    (y_max, x_max)=coords

    FFT_distance_point_x=np.abs(x_max-int(FFT_pixels/2))*FFT_calibration
    FFT_distance_point_y=np.abs(y_max-int(FFT_pixels/2))*FFT_calibration
    
    FFT_distance_total=np.sqrt(FFT_distance_point_x**2+FFT_distance_point_y**2)
    
    
    d_spacing_spot=1/FFT_distance_total
    
    return d_spacing_spot



def Spot_coord_To_d_spacing_vect(coord_vects, FFT_calibration, FFT_pixels):
    y_vects=coord_vects[:,0]    
    x_vects=coord_vects[:,1] 

    FFT_distance_point_x=np.abs(x_vects-int(FFT_pixels/2))*FFT_calibration
    FFT_distance_point_y=np.abs(y_vects-int(FFT_pixels/2))*FFT_calibration
    
    FFT_distance_total=np.sqrt(FFT_distance_point_x**2+FFT_distance_point_y**2)
    
    
    d_spacing_spot=1/FFT_distance_total
    
    return d_spacing_spot


def Spot_coord_To_Angles_to_X_indiv(coords,FFT_pixels):
    
    (y_max, x_max)=coords
    
    cont_dist=x_max-int(FFT_pixels/2)
    opp_dist=int(FFT_pixels/2)-y_max
    
    angles_to_X=np.arctan2(opp_dist,cont_dist)*180/np.pi
    
    return angles_to_X


def Spot_coord_To_Angles_to_X_vect(coord_vects,FFT_pixels):
    y_vects=coord_vects[:,0]    
    x_vects=coord_vects[:,1] 
    
    cont_dist=x_vects-int(FFT_pixels/2)
    opp_dist=int(FFT_pixels/2)-y_vects
    
    angles_to_X=np.arctan2(opp_dist,cont_dist)*180/np.pi
    
    return angles_to_X


#Hyperparameters
gauss_blur_filter_size=(5,5)  #size of smoothing filter, go to line to change sigma
downscaling_factor=1 #for trials, n factor of downsampling size of image
FFT_thresholding=0.5  #value above which the pixels are kept
st_distance=30 #distance parameter in the Stem tool method
FFT_thresholdingG=0.6 #value above which the pixels are kept, in the gaussian filtered FFT
window_size=128  #window size of the sliding windows


#dm3 loading, and calibration extraction
imagedm3=hs.load(r'E:\Arxius varis\PhD\2nd_year\Code\trial_images\dm3_atomic_resolution\GeQW2.dm3')
meta1=imagedm3.metadata
meta2=imagedm3.original_metadata.export('parameters')


imagedm3.plot()
plt.show()

x_calibration=imagedm3.axes_manager['x'].scale
y_calibration=imagedm3.axes_manager['y'].scale

x_pixels=imagedm3.axes_manager['x'].size
y_pixels=imagedm3.axes_manager['y'].size

x_units=imagedm3.axes_manager['x'].units
y_units=imagedm3.axes_manager['y'].units


#FFT calibration

FFT_calibration,FFT_pixels,FFT_units=FFT_calibration(imagedm3)


imagearray=np.asarray(imagedm3)
image=imagearray

plt.imshow(image, cmap=plt.cm.gray, vmin=image.min(), vmax=image.max())
plt.show()
#First standarisation of the image for filtering/blurring it with gaussian filter
image_st=(image-np.min(image))/np.max(image-np.min(image))
plt.hist(image_st.ravel(),256,[np.min(np.array([image_st])),np.max(np.array([image_st]))])
plt.show()


#Application of Gaussian filter for denoising


denoised_image=cv2.GaussianBlur(image_st, gauss_blur_filter_size, 1)


plt.imshow(denoised_image, cmap=plt.cm.gray, vmin=denoised_image.min(), vmax=denoised_image.max())
plt.show()
#Second standarisation of the image after filtering/blurring it with gaussian filter

image_st=(denoised_image-np.min(denoised_image))/np.max(denoised_image-np.min(denoised_image))

#Print histogram

plt.hist(image_st.ravel(),256,[np.min(np.array([image_st])),np.max(np.array([image_st]))])
plt.show()

#For sake of evaluation, better work with an image with less pixels, as only the consecutive pixel evaluation would take
#approximately 6 hours to run for a big region of 250.000 pixels in total.

#Then downsample the image and upsample it posteriorly 
#We select a max pooling method to keep track of the brighter elements and this way keep a higher contrast


ds_image=measure.block_reduce(image_st, block_size=tuple(np.int32(downscaling_factor*np.ones(len(np.shape(image_st))))), func=np.mean, cval=0)

#and standarise it again to ensure 0-1 values

ds_image_st=(ds_image-np.min(ds_image))/np.max(ds_image-np.min(ds_image))


print('FFT computing and refinement')

# take the fft of the image
fft_image_w_background = np.fft.fftshift(np.log(np.fft.fft2(ds_image_st)))
fft_abs_image_background = np.abs(fft_image_w_background)

print('Original FFT')
plt.hist(fft_abs_image_background.ravel(),256,[np.min(np.array([fft_abs_image_background])),np.max(np.array([fft_abs_image_background]))])
plt.show()
plt.imshow(fft_abs_image_background)
plt.show()
# apply the filter
fft_abs_image_background2=np.copy(fft_abs_image_background)
fft_abs_image_backgroundc=np.copy(fft_abs_image_background)


fft_abs_image_backgroundc=(fft_abs_image_backgroundc-np.min(fft_abs_image_backgroundc))/np.max(fft_abs_image_backgroundc-np.min(fft_abs_image_backgroundc))


print('Original FFT')
fft_abs_image_background2=(fft_abs_image_background2-np.min(fft_abs_image_background2))/np.max(fft_abs_image_background2-np.min(fft_abs_image_background2))
plt.hist(fft_abs_image_background2.ravel(),256,[np.min(np.array([fft_abs_image_background2])),np.max(np.array([fft_abs_image_background2]))])
plt.show()

plt.imshow(fft_abs_image_background2)
plt.show()
fft_abs_image_background2=cv2.GaussianBlur(fft_abs_image_background2, gauss_blur_filter_size, 1)
fft_abs_image_background2=(fft_abs_image_background2-np.min(fft_abs_image_background2))/np.max(fft_abs_image_background2-np.min(fft_abs_image_background2))

print('Gaussian Filtered FFT')
plt.imshow(fft_abs_image_background2)
plt.show()

plt.hist(fft_abs_image_background2.ravel(),256,[np.min(np.array([fft_abs_image_background2])),np.max(np.array([fft_abs_image_background2]))])
plt.show()


#trial with original FFT
#fft_abs_image_background2=fft_abs_image_backgroundc

#Automatic hyperparameter finding
fov=np.shape(fft_abs_image_background2)[0]*y_calibration
print('fov',fov)
st_distance,_,FFT_perc=FFT_hyperparams(fft_abs_image_background2,fov)
print('mpfit',st_distance,'perc',FFT_perc )
FFT_thresholdingG=Threshold_given_percentage(fft_abs_image_background2, FFT_perc)
print('fft_threhs',FFT_thresholdingG)


print('STEM Tool based method (2D Gaussians)')

center_difractogram=(int(FFT_pixels/2), int(FFT_pixels/2))

print('ST gaussian FFT')
twodfit_blur=stemtool.afit.peaks_vis(fft_abs_image_background2, dist=st_distance, thresh=FFT_thresholdingG, imsize=(15, 15))

d_distances=Spot_coord_To_d_spacing_vect(twodfit_blur, FFT_calibration, FFT_pixels)

angles_to_x=Spot_coord_To_Angles_to_X_vect(twodfit_blur,FFT_pixels)

print(d_distances)

print(angles_to_x)

