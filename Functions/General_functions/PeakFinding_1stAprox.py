# -*- coding: utf-8 -*-
"""
Created on Thu Feb 17 17:29:38 2022

@author: Marc
"""

import numpy as np



#Functions for the identificaiton of the spots from FFT, old 2D gaussian based for trials

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

