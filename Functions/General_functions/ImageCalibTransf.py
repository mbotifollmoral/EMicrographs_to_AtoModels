# -*- coding: utf-8 -*-
"""
Created on Thu Feb 17 18:01:53 2022

@author: Marc
"""


import numpy as np
import matplotlib.pyplot as plt
import cv2
import hyperspy.api as hs
import time
import skimage.measure
from matplotlib.patches import Rectangle
import sys
import os



def FFT_calibration(hyperspy_2D_signal):
    fft_shifted = hyperspy_2D_signal.fft(shift=True)
    FFT_calibration=fft_shifted.axes_manager['x'].scale
    # FFT_pixels=fft_shifted.axes_manager['x'].size
    FFT_units=fft_shifted.axes_manager['x'].units
    
    return FFT_calibration, FFT_units
    
def FFT_calibration_Correction(hyperspy_2D_signal, real_calibration_factor):
    '''
    distances in real space--> d_real=d_measured*real_calibration_factor
    '''
    x_calibration_original=hyperspy_2D_signal.axes_manager['x'].scale
    y_calibration_original=hyperspy_2D_signal.axes_manager['y'].scale
    hyperspy_2D_signal.axes_manager['x'].scale=x_calibration_original*real_calibration_factor
    hyperspy_2D_signal.axes_manager['y'].scale=y_calibration_original*real_calibration_factor
    
    fft_shifted = hyperspy_2D_signal.fft(shift=True)
    FFT_calibration=fft_shifted.axes_manager['x'].scale
    # FFT_pixels=fft_shifted.axes_manager['x'].size
    FFT_units=fft_shifted.axes_manager['x'].units
    
    return FFT_calibration,FFT_units
    

def Load_dm3(image_path):
    # Load a dm3 image with hyperspy and get calibration and numpy array from it
    image_hs_signal=hs.load(image_path)
    image_array=np.asarray(image_hs_signal)

    # meta1=imagedm3.metadata
    # meta2=imagedm3.original_metadata.export('parameters')
    
    im_calibration=image_hs_signal.axes_manager['x'].scale
    total_pixels_image=image_hs_signal.axes_manager['x'].size
    im_FOV=im_calibration*total_pixels_image
    im_units=image_hs_signal.axes_manager['x'].units
    
    return image_hs_signal, image_array, im_calibration, total_pixels_image, im_FOV, im_units
 
def Standarise_Image(image_array):
    #set intensity values between 0 and 1    
    image_st=(image_array-np.min(image_array))/np.max(image_array-np.min(image_array))
    return image_st

def Crop_Image_Randomly(image_hs_signal, crop_size):
    '''
    Crop image in a random position of the image and generate a hyperspy signal of the cropped 
    image as wanted, and the corresponding numpy array of the cropped region

    Parameters
    ----------
    image_hs_signal : hyperspy 2d signal of the image
    crop_size : 2**n window size

    Returns
    -------
    crop_image_hs_signal : 
    crop_image_array : 

    '''
    np.random.seed(int(time.time()))
    total_pixels_original=image_hs_signal.axes_manager['x'].size

    if crop_size>=total_pixels_original:
        init_x=0
        init_y=0
    else:
        init_y=np.random.choice(np.arange(0,total_pixels_original-crop_size,1)) 
        init_x=np.random.choice(np.arange(0,total_pixels_original-crop_size,1)) 
    
    crop_image_hs_signal=image_hs_signal.isig[init_y:init_y+crop_size,init_x:init_x+crop_size]
    crop_image_array=np.asarray(crop_image_hs_signal)
    total_pixels_image=crop_size
    im_FOV=image_hs_signal.axes_manager['x'].scale*total_pixels_image
    
    return crop_image_hs_signal, crop_image_array, total_pixels_image, im_FOV


def Crop_Image_PixelCoords(image_signal, pixel_coords, pixel_size):
    '''
    Crop an image in a  position of the image and generate a hyperspy signal of the cropped 
    image as wanted, and the corresponding numpy array of the cropped region

    Parameters
    ----------
    images_signal : it can be either a numpy array or a hyperspy signal
    crop_size : 2**n window size

    Returns
    -------
    crop_image_hs_signal : 
    crop_image_array : 

    '''
    (pix_x_start, pix_y_start, pix_x_end, pix_y_end) = pixel_coords
    # if it is a numpy array
    if type(image_signal) == np.ndarray:
        crop_image_array = image_signal[pix_y_start:pix_y_end+1,pix_x_start:pix_x_end+1]
        total_pixels_image = pix_y_end - pix_y_start
        im_FOV = pixel_size*total_pixels_image
        # generate the hs signal from the array
        crop_image_hs_signal = hs.signals.Signal2D(crop_image_array)
        crop_image_hs_signal.axes_manager[0].name = 'x'
        crop_image_hs_signal.axes_manager[1].name = 'y'
        crop_image_hs_signal.axes_manager['x'].scale = pixel_size
        crop_image_hs_signal.axes_manager['y'].scale = pixel_size
        crop_image_hs_signal.axes_manager['x'].units = 'nm'
        crop_image_hs_signal.axes_manager['y'].units = 'nm'
        crop_image_hs_signal.axes_manager['x'].size = total_pixels_image
        crop_image_hs_signal.axes_manager['y'].size = total_pixels_image

    else:
        # if it is a hs signal
        crop_image_hs_signal = image_signal.isig[pix_y_start:pix_y_end+1,pix_x_start:pix_x_end+1]
        crop_image_array = np.asarray(crop_image_hs_signal)
        total_pixels_image = pix_y_end - pix_y_start
        im_FOV = image_signal.axes_manager['x'].scale*total_pixels_image
    
    return crop_image_hs_signal, crop_image_array, total_pixels_image, im_FOV



    
def Image_Calibration_Correction(im_calibration,im_FOV, real_calibration_factor):
    # when d_real=d_measured*real_calibration_factor, modify the pixel size and fov accordingly
    im_calibration=im_calibration*real_calibration_factor
    im_FOV=im_FOV*real_calibration_factor
    return im_calibration, im_FOV


def Downscale_Image_Factor(image_array, downscaling_factor, im_calibration, total_pixels_image):
    # downscale the image a factor of downscaling_factor and return the new image array, 
    # real space calibration (the FFT calib does not change with downscaling), 
    #the new total number of pixels
    
    # the downscaling factor must be 1 or a power of 2
    image_array_down=skimage.measure.block_reduce(image_array, block_size=tuple(np.int32(downscaling_factor*np.ones(len(np.shape(image_array))))), func=np.mean, cval=0)
    im_calibration=im_calibration*downscaling_factor
    total_pixels_image=int(total_pixels_image/downscaling_factor)
    
    return image_array_down, im_calibration, total_pixels_image



def Downscale_Image_FinalSize(image_array, final_image_size, im_calibration, total_pixels_image):
    # downscale the image until a given image size final_image_size and return the new image array, 
    # new real space calibration (the FFT calib does not change with downscaling), 
    # and the new total number of pixels
    
    # the final_image_size must be a power of 2
    downscaling_factor=int(total_pixels_image/final_image_size)
    image_array_down=skimage.measure.block_reduce(image_array, block_size=tuple(np.int32(downscaling_factor*np.ones(len(np.shape(image_array))))), func=np.mean, cval=0)
    im_calibration=im_calibration*downscaling_factor
    total_pixels_image=int(total_pixels_image/downscaling_factor)
    
    return image_array_down, im_calibration, total_pixels_image



def Compute_FFT_ImageArray(image_array):
    '''
    Outputs the FFT of an image array, and outputs the absolute value of the log (complex FFT) for doing
    the image processing and peak identification: FFT_image_array , and complex FFT, FFT_image_complex,
    for doing reciprocal space filtering and consequent iFFT

    Parameters
    ----------
    image_array : 

    Returns
    -------
    FFT_image_array : 
    FFT_image_complex :

    '''
    FFT_image_complex = np.log(np.fft.fftshift(np.fft.fft2(image_array)))
    FFT_image_array = np.abs(FFT_image_complex)
    return  FFT_image_array, FFT_image_complex


def Compute_FFT_ImageArray_NO_Log(image_array):
    '''
    Outputs the FFT of an image array, and outputs the absolute value of the log (complex FFT) for doing
    the image processing and peak identification: FFT_image_array , and complex FFT, FFT_image_complex,
    for doing reciprocal space filtering and consequent iFFT

    Parameters
    ----------
    image_array : 

    Returns
    -------
    FFT_image_array : 
    FFT_image_complex :

    '''
    FFT_image_complex = np.fft.fftshift(np.fft.fft2(image_array))
    FFT_image_array = np.abs(FFT_image_complex)
    return  FFT_image_array, FFT_image_complex



def Right_Half_FFT_Only(
        refined_distances, refined_angles_to_x, refined_pixels):
    # Only keep distances, angles and pixels of the right half of the FFT
    # Accounted in the compute g vectors function in GPA already

    #only account for the spots of the right half of the FFT
    refined_distances=np.asarray(refined_distances)
    refined_angles_to_x=np.asarray(refined_angles_to_x)
    refined_pixels=np.asarray(refined_pixels)
    
    # both 90 an d-90 angles included so it goes from 90 to -90 -> [-90,90]
    newdists=refined_distances[refined_angles_to_x>=-90]
    newpixels=refined_pixels[refined_angles_to_x>=-90]
    newangls=refined_angles_to_x[refined_angles_to_x>=-90]
    
    refined_distances=newdists[newangls<=90]
    refined_pixels=newpixels[newangls<=90]
    refined_angles_to_x=newangls[newangls<=90]
            
    return refined_distances, refined_angles_to_x, refined_pixels


def FFT_from_crop(
        image_hs_signal, image_array):
    
    # wrap the FFT obtaining and calibration based on the crop in one line
    # inputs are the hs and np signals of the CROP
    
    # Calibrate the FFT, which does change the calibration with cropping
    FFT_calibration_,FFT_units= FFT_calibration(image_hs_signal)
    
    # Denoise the image if wanted, might help to denoise but typically the peak finding works worse
    #image_array=FiltersNoise.Adaptative_Gaussian_Blur(image_array, im_calibration)
    
    # Compute the FFT 
    FFT_image_array, FFT_image_complex= Compute_FFT_ImageArray(image_array)
    
    # Filter the FFT in case we see it is convenient
    FFT_image_array= FFT_Gaussian_Convolution_Filter(FFT_image_array)
    
    # Standarise the FFT
    FFT_image_array= Standarise_Image(FFT_image_array)
    # plt.imshow(FFT_image_array, cmap=plt.cm.gray, vmin=FFT_image_array.min(), vmax=FFT_image_array.max())
    # plt.show()
        
    return FFT_image_array, FFT_calibration_, FFT_units


def FFT_No_Log_from_crop(
        image_hs_signal, image_array):
    
    # wrap the FFT obtaining and calibration based on the crop in one line
    # inputs are the hs and np signals of the CROP
    
    # Calibrate the FFT, which does change the calibration with cropping
    FFT_calibration_,FFT_units= FFT_calibration(image_hs_signal)
    
    # Denoise the image if wanted, might help to denoise but typically the peak finding works worse
    #image_array=FiltersNoise.Adaptative_Gaussian_Blur(image_array, im_calibration)
    
    # Compute the FFT 
    FFT_image_array_No_Log, FFT_image_complex= Compute_FFT_ImageArray_NO_Log(image_array)
    
    # We see that filtering is not convenient for the analysis
    #FFT_image_array=ImCalTrans.FFT_Gaussian_Convolution_Filter(FFT_image_array)
    
    # Standarise the FFT
    FFT_image_array_No_Log = Standarise_Image(FFT_image_array_No_Log)
    # plt.imshow(FFT_image_array, cmap=plt.cm.gray, vmin=FFT_image_array.min(), vmax=FFT_image_array.max())
    # plt.show()
        
    return FFT_image_array_No_Log, FFT_calibration_, FFT_units


def Compute_iFFT_ImageArray(FFT_image_complex):
    '''
    It computes the iFFT of a complex FFT. it does not work with the absolute value FFT used for processing,
    that is why the Compute_FFT_ImageArray function outputs the complex FFT as well
    Outputs a real space image with real numbers (does absolute funciton to the iFFT)

    Parameters
    ----------
    FFT_image_complex : 

    Returns
    -------
    iFFT_image_array : 

    '''
    iFFT_image_array=np.fft.ifft2(np.fft.ifftshift(np.exp(FFT_image_complex)))
    iFFT_image_array=np.abs(iFFT_image_array)
    return iFFT_image_array

def FFT_Gaussian_Convolution_Filter(FFT_image_array):
    '''
    Gaussian convolution to denoise the FFT, it can be beneficious and the fixed settings seem to work well
    in all the cases. it is not mandatory and the process is fine without it although it seems to help

    Parameters
    ----------
    FFT_image_array : 

    Returns
    -------
    FFT_image_array :

    '''
    gauss_blur_filter_size=(5,5)
    desvest_pixels=1
    FFT_image_array=cv2.GaussianBlur(FFT_image_array, gauss_blur_filter_size, desvest_pixels)
    return FFT_image_array


def nm_to_Angstroms(refined_distances):
    '''
    Changes the an array set from nm to Angstroms --> A = 1 nm *(10 A/ 1 nm)
    The function is created to keep a better track of where the unit changes are done
    After the Ensure_Center_Diff function, which works with the distances in nm (keep only distances <1.5nm),
    then this function needs to be applied to refined_distances array to be prepeared for the 
    Prepare_exp_distances_angles_pixels function which demands the distances in angstroms to be 
    compared with the min_d paramter in angstroms (keept in angstroms as it is the way defined in
    the phase identification algorithm with the .dll)
    The other functions are also thought to work in A as it is more crystallographic

    Parameters
    ----------
    refined_distances : 

    Returns
    -------
    refined_distances : 

    '''
    refined_distances=refined_distances*10    
    return refined_distances

def Angstroms_to_nm(distances):
    '''
    Changes the an array set from Angstroms to nm --> nm = 1 A *(1 nm/ 10 A)
    The function is created to keep a better track of where the unit changes are done

    Parameters
    ----------
    refined_distances : 

    Returns
    -------
    refined_distances : 

    '''
    distances=distances/10   
    return distances



def Plot_Image_with_GPA_Reference(image_array, scaled_reference_coords):
    # Plot the image and the square where the reference is computed 
    fig,ax = plt.subplots(1)
    ax.imshow(image_array,cmap=plt.cm.gray, vmin=image_array.min(), vmax=image_array.max())
    reference_position = Rectangle((scaled_reference_coords[2],scaled_reference_coords[0]), scaled_reference_coords[3]-scaled_reference_coords[2], scaled_reference_coords[1]-scaled_reference_coords[0], angle=0.0, facecolor='none',edgecolor="blue", linewidth=1, alpha=1)
    ax.add_patch(reference_position)
    plt.show()


def Build_hs_2DSignal(image_array, pixel_size):
    '''
    Generate a hyprespy 2D signal from calibrated image crop/array

    Parameters
    ----------
    image_array : 
    pixel_size : 

    Returns
    -------
    image_hs_signal : 
    image_array : 
    total_pixels_image : 
    crop_FOV : 
    '''
    
    # generate the hs signal from the array
    image_hs_signal = hs.signals.Signal2D(image_array)
    image_hs_signal.axes_manager[0].name = 'x'
    image_hs_signal.axes_manager[1].name = 'y'
    image_hs_signal.axes_manager['x'].scale = pixel_size
    image_hs_signal.axes_manager['y'].scale = pixel_size
    image_hs_signal.axes_manager['x'].units = 'nm'
    image_hs_signal.axes_manager['y'].units = 'nm'
    total_pixels_image = np.shape(image_array)[0]
    image_hs_signal.axes_manager['x'].size = total_pixels_image
    image_hs_signal.axes_manager['y'].size = total_pixels_image
    image_FOV = pixel_size * total_pixels_image
 
    return image_hs_signal, image_array, total_pixels_image, image_FOV






