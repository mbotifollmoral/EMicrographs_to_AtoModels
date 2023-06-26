# -*- coding: utf-8 -*-
"""
Created on Mon Jan 31 12:04:49 2022

@author: Marc
"""

'''
This code is for extracting the information from the images and tailor it to the 
automatic GPA script: position of the spots, choosing the spots, defining the interface 
to refine the rotation of the axes, defining the reference region
'''

import os
import numpy as np
import matplotlib.pyplot as plt
import stemtool
import time
from PIL import Image
import cv2

import tkinter
from tkinter import filedialog

import sklearn.cluster
import sklearn.mixture

import hyperspy.api as hs
import skimage.measure
from matplotlib.patches import Rectangle

from numpy import fft
import ctypes
from matplotlib.patches import Circle




plt.rcParams["figure.figsize"] = (12,12)

def FT(img):
    return fft.ifftshift(fft.fft2(fft.fftshift(img)))

def IFT(img):
    return fft.fftshift(fft.ifft2(fft.ifftshift(img)))

lib = ctypes.CDLL(r"GPA_dll\GPA.dll")

Handle = ctypes.POINTER(ctypes.c_char)
c_float_array = np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags='C_CONTIGUOUS')
c_int_array = np.ctypeslib.ndpointer(dtype=int, ndim=1, flags='C_CONTIGUOUS')

lib.createGPA.argtypes = None
lib.createGPA.restype = Handle

lib.deleteGPA.argtypes = [Handle]
lib.deleteGPA.restype = None

lib.load_img.argtypes = [Handle, c_float_array, c_int_array, ctypes.c_float]
lib.load_img.restypes = None

lib.mark_spot1.argtypes = [Handle,c_int_array, ctypes.c_int, c_float_array]
lib.mark_spot1.restypes = None

lib.mark_spot2.argtypes = [Handle,c_int_array, ctypes.c_int ,c_float_array]
lib.mark_spot2.restypes = None

lib.select_ref_area.argtypes = [Handle,c_int_array, c_int_array]
lib.select_ref_area.restypes = None

lib.calc_GPA.argtypes = [Handle]
lib.calc_GPA.restypes = None

lib.apply_rotation.argtypes = [Handle, ctypes.c_float] 
lib.apply_rotation.restypes = None

lib.get.argtypes = [Handle,c_float_array,c_float_array,c_float_array,c_float_array,c_float_array,c_float_array]
lib.get.restypes = None

class GPA:    
    def __init__(self):
        self.instance = lib.createGPA()
        self.ref_load = False
        self.Sp1_marked = False
        self.Sp2_marked = False
        self.get_ready = False

    def __del__(self):
        lib.deleteGPA(self.instance)   
        
    def load_image(self,img, calibration):
        #image = np.empty(self.Npix2, dtype=np.double)
        self.dim = img.shape
        size = np.asarray(self.dim,dtype = int)
        image = np.asarray(img.flatten(), dtype = np.float32)
        lib.load_img(self.instance, image, size, calibration*size[0])
        self.ref_load = True
    
    def mark_spot1(self, coordinates, win_size):
        if(self.ref_load != True):
            print("Load an image first")
            return np.empty((2,2), dtype=np.float32)
        amp = np.empty(self.dim[0]*self.dim[1], dtype=np.float32)
        coo = np.asarray(coordinates,dtype = int)
        lib.mark_spot1(self.instance,coo,win_size,np.asarray(amp, dtype=np.float32))
        self.Sp1_marked = True
        return amp.reshape(self.dim[0],self.dim[1])
    
    def mark_spot2(self, coordinates, win_size):
        if(self.ref_load != True):
            print("Load an image first")
            return np.empty((2,2), dtype=np.float32)
        amp = np.empty(self.dim[0]*self.dim[1], dtype=np.float32)
        coo = np.asarray(coordinates,dtype = int)
        lib.mark_spot2(self.instance,coo,win_size,np.asarray(amp, dtype=np.float32))
        self.Sp2_marked = True
        return amp.reshape(self.dim[0],self.dim[1])
    
    def select_ref_area(self, TL, BR):
        if(self.ref_load != True):
            print("Load an image first")
        tl = np.asarray(TL,dtype = int)    
        br = np.asarray(BR,dtype = int)
        lib.select_ref_area(self.instance,tl,br)
        
    def calc_GPA(self):
        if(self.ref_load and self.Sp1_marked and self.Sp2_marked == True):
            lib.calc_GPA(self.instance)
            self.get_ready = True
        else:
            print("Mark two diffraction spots first!")
            
    def apply_rotation(self, angle):
        lib.apply_rotation(self.instance, angle)
        
    def get(self):
        if(self.get_ready != True):
            x = np.empty((2,2), dtype=np.float32)
            print("run a calculation first!!")
            return x,x,x,x,x,x
        dxx = np.empty(self.dim[0]*self.dim[1], dtype=np.float32)
        dyy = np.empty(self.dim[0]*self.dim[1], dtype=np.float32)
        dxy = np.empty(self.dim[0]*self.dim[1], dtype=np.float32)
        dyx = np.empty(self.dim[0]*self.dim[1], dtype=np.float32)
        rot = np.empty(self.dim[0]*self.dim[1], dtype=np.float32)
        shear = np.empty(self.dim[0]*self.dim[1], dtype=np.float32)
        lib.get(self.instance, dxx, dyy, dxy, dyx, rot, shear)
        dxx = dxx.reshape(self.dim[0],self.dim[1])
        dyy = dyy.reshape(self.dim[0],self.dim[1])
        dxy = dxy.reshape(self.dim[0],self.dim[1])
        dyx = dyx.reshape(self.dim[0],self.dim[1])
        rot = rot.reshape(self.dim[0],self.dim[1])
        shear = shear.reshape(self.dim[0],self.dim[1])
        return dxx, dyy, dxy, dyx, rot, shear
 
    
#Adjust reference to the calculated images and phases function??    

    
def GPA_full(image_array,calibration,spot1, spot2, mask_size, reference, rotation_angle,display=True):
    '''
    Full GPA process, to call the full wrapped dll at once with all the necessary inputs.
    spot1 and spot2 must be tuples or lists with two elements, first the y coord (row) and then 
    x coord (col), in pixels, referring to the IMAGES reference system and not cartesian
    Mask size must be in pixels as well
    reference, coords in pixels, of the top-left corner and bottom-right corner of the rectangle/square
    referring to the reference, in format: row_i (yi) , col_i(xi), row_f (yf) , col_f (xf)
    Display =True to show the images of the process

    Parameters
    ----------
    image_array : 
    calibration : 
    spot1 : 
    spot2 : 
    mask_size : 
    reference:
    rotation_angle : angle to the x axis to the first g vector or spot, in degrees

    Returns
    -------
    dxx, dyy, dxy, dyx, rot, shear

    '''
    #image system
    # row1,col1=spot1
    # row2,col2=spot2
    
    #cartesian system
    Sp1Y,Sp1X = spot1
    Sp2Y,Sp2X = spot2
    
    row_i , row_f, col_i , col_f = reference
    
    Analysis = GPA()
    
    Analysis.load_image(image_array,calibration)
    
    #---
    # amp1 = Analysis.mark_spot1((Sp1X,Sp1Y),mask_size)
    
    # amp2 = Analysis.mark_spot2((Sp2X,Sp2Y),mask_size)
    
    # Analysis.calc_GPA()
    
    # Analysis.apply_rotation(rotation_angle)
        
    # exx, eyy, exy, eyx, rot, shear = Analysis.get()
    # #---
    
    
    # reference, watch out about the order of the parameters as the function demands top left and 
    # bottom right but in col (x), and row (y) order, and no need to add 1 as these are coordinates
    # not axes to be cropped (in the plotting we do crop)
    Analysis.select_ref_area((col_i,row_i),(col_f,row_f))
    
    amp1 = Analysis.mark_spot1((Sp1X,Sp1Y),mask_size)
    
    amp2 = Analysis.mark_spot2((Sp2X,Sp2Y),mask_size)
    
    Analysis.calc_GPA()
    
    Analysis.apply_rotation(rotation_angle)
        
    exx, eyy, exy, eyx, rot, shear = Analysis.get()
    
    #Adjust reference to the calculated images and phases function?? 

    if display==True:
        #show the spots positions in the FFT and the strain components
        
        #FFT and chosen spot positions
        fft_img = np.log(abs(FT(image_array)))
        fig,ax = plt.subplots(1,constrained_layout=True)
        fig.set_size_inches(12, 12)
        fig.suptitle('FFT and chosen g vectors',fontsize=18)
        ax.imshow(fft_img,interpolation='nearest', cmap='gray')
        ax.scatter(Sp1X, Sp1Y, c="red", marker="x")
        circle = Circle((Sp1X, Sp1Y), mask_size, facecolor='none',edgecolor="red", linewidth=1, alpha=1)
        ax.add_patch(circle)
        ax.scatter(Sp2X, Sp2Y, c="blue", marker="x")
        circle = Circle((Sp2X, Sp2Y), mask_size, facecolor='none',edgecolor="blue", linewidth=1, alpha=1)
        ax.add_patch(circle)
        
        #Calculated phase
        fig,ax=plt.subplots(1,2, constrained_layout=True)
        fig.set_size_inches(12, 6)
        fig.suptitle('Spot intensity maps',fontsize=18)
        ax[0].imshow(amp1,interpolation='nearest', cmap='gray')
        ax[0].set_title('Spot 1 (g1)')
        ax[1].imshow(amp2,interpolation='nearest', cmap='gray')
        ax[1].set_title('Spot 2 (g2)')
        
        #Calculated components
        c = 2
        clims_fix=(-0.05,0.05)
        clims_fix_rot=(-5,5)
        cm = 'jet'
        colorbarshrink=1
        fig,ax=plt.subplots(3,2, constrained_layout=True)
        fig.set_size_inches(16, 18)
        fig.suptitle('Calculated components', fontsize=18)
        M = exx[row_i:row_f+1,col_i:col_f+1].mean()
        S = exx[row_i:row_f+1,col_i:col_f+1].std()
        clims_ranged=(M-c*S,M+c*S)
        im=ax[0][0].imshow(exx,interpolation='nearest', cmap=cm, clim=clims_fix)
        ax[0][0].set_title('e_xx')
        fig.colorbar(im, ax=ax[0][0], shrink=colorbarshrink)
        M = eyy[row_i:row_f+1,col_i:col_f+1].mean()
        S = eyy[row_i:row_f+1,col_i:col_f+1].std()
        clims_ranged=(M-c*S,M+c*S)
        im=ax[1][1].imshow(eyy,interpolation='nearest', cmap=cm, clim=clims_fix)
        ax[1][1].set_title('e_yy')
        fig.colorbar(im, ax=ax[1][1], shrink=colorbarshrink)
        M = exy[row_i:row_f+1,col_i:col_f+1].mean()
        S = exy[row_i:row_f+1,col_i:col_f+1].std()
        clims_ranged=(M-c*S,M+c*S)
        im=ax[0][1].imshow(exy,interpolation='nearest', cmap=cm, clim=clims_fix)
        ax[0][1].set_title('e_xy')
        fig.colorbar(im, ax=ax[0][1], shrink=colorbarshrink)
        M = eyx[row_i:row_f+1,col_i:col_f+1].mean()
        S = eyx[row_i:row_f+1,col_i:col_f+1].std()
        clims_ranged=(M-c*S,M+c*S)
        im=ax[1][0].imshow(eyx,interpolation='nearest', cmap=cm, clim=clims_fix)
        ax[1][0].set_title('e_yx')
        fig.colorbar(im, ax=ax[1][0], shrink=colorbarshrink)
        M = rot[row_i:row_f+1,col_i:col_f+1].mean()
        S = rot[row_i:row_f+1,col_i:col_f+1].std()
        clims_ranged=(M-c*S,M+c*S)
        im=ax[2][0].imshow(rot,interpolation='nearest', cmap=cm, clim=clims_fix_rot)
        ax[2][0].set_title('Rotation') 
        fig.colorbar(im, ax=ax[2][0], shrink=colorbarshrink)
        M = shear[row_i:row_f+1,col_i:col_f+1].mean()
        S = shear[row_i:row_f+1,col_i:col_f+1].std()
        clims_ranged=(M-c*S,M+c*S)
        im=ax[2][1].imshow(shear,interpolation='nearest', cmap=cm, clim=clims_fix)
        ax[2][1].set_title('Shear')  
        fig.colorbar(im, ax=ax[2][1], shrink=colorbarshrink)
        plt.show(block=False)
        
    return exx, eyy, exy, eyx, rot, shear 
        




def FFT_calibration(hyperspy_2D_signal):
    fft_shifted = hyperspy_2D_signal.fft(shift=True)
    FFT_calibration=fft_shifted.axes_manager['x'].scale
    # FFT_pixels=fft_shifted.axes_manager['x'].size
    FFT_units=fft_shifted.axes_manager['x'].units
    
    return FFT_calibration,FFT_units
    
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
    plt.show(block=False)


def Compute_Smallest_Feature_Thickness(segmented_image):
    #To be done once the segmentation routine is strong enough
    smallest_feature_thikness=1
    return smallest_feature_thikness












def Spot_coord_To_d_spacing_vect(coord_vects, FFT_calibration, FFT_pixels):
    # output distances in nm
    y_vects=coord_vects[:,0]    
    x_vects=coord_vects[:,1] 

    FFT_distance_point_x=np.abs(x_vects-int(FFT_pixels/2))*FFT_calibration
    FFT_distance_point_y=np.abs(y_vects-int(FFT_pixels/2))*FFT_calibration
    
    FFT_distance_total=np.sqrt(FFT_distance_point_x**2+FFT_distance_point_y**2)
    
    FFT_distance_total[FFT_distance_total==0]=0.0001
    
    d_spacing_spot=1/FFT_distance_total
    
    return d_spacing_spot



def Spot_coord_To_Angles_to_X_vect(coord_vects,FFT_pixels):
    y_vects=coord_vects[:,0]    
    x_vects=coord_vects[:,1] 
    
    cont_dist=x_vects-int(FFT_pixels/2)
    opp_dist=int(FFT_pixels/2)-y_vects
    
    angles_to_X=np.arctan2(opp_dist,cont_dist)*180/np.pi
    
    return angles_to_X




def Ensure_Center_Diff(distances_array, angles_to_x_array, pixels_array):
    #input distances in nm
    #define hyperparameter, that should not be very modifyable, as it is the 
    #maximum interplanar distance to consider, in nm, let us say d_int > 1-1.5nm
    #as no plane should be  bigger than 1-1.5nm. Let us say 1.5 to include 1/2 indices 
    
    #do the same but delete the center of the pixels array
    angles_to_x_array_c=angles_to_x_array[distances_array<=1.5]
    pixels_array_c=pixels_array[distances_array<=1.5]    
    distances_array_c=distances_array[distances_array<=1.5]

    
    return distances_array_c, angles_to_x_array_c, pixels_array_c


def Prepare_exp_distances_angles_pixels(refined_distances_exp, refined_angles_exp, refined_pixels_exp, min_d):
    '''
    Needs the  refined distances, without the center of diffraction. It is not essential, but it is 
    preferred that it has already been extracted: As the process of extracting the center of the diffraction
    also takes everything that is bigger than 1.5nm out =15 A-> if d>15 A then it is removed (in case it 
    needed this distance can be increased if a very big unit cell is involved)
    IMPORTANT: The units must be in angstroms, as min_d is in Angrstoms
    Takes all the distances smaller than mind and deletes them.
    Then sort both axis from max to min
    '''
    
    more_refined_distances_exp=refined_distances_exp[refined_distances_exp>min_d]
    more_refined_angles_exp=refined_angles_exp[refined_distances_exp>min_d]
    more_refined_pixels_exp=refined_pixels_exp[refined_distances_exp>min_d]
    
    more_refined_angles_exp_f=[angle for _,angle in sorted(zip(more_refined_distances_exp,more_refined_angles_exp), key=lambda pair: pair[0])][::-1]
    more_refined_pixels_exp_f=[pixels for _,pixels in sorted(zip(more_refined_distances_exp,more_refined_pixels_exp), key=lambda pair: pair[0])][::-1]
    more_refined_distances_exp_f=sorted(more_refined_distances_exp)[::-1]
      
    more_refined_pixels_exp_f=np.int32(more_refined_pixels_exp_f)

    return more_refined_distances_exp_f, more_refined_angles_exp_f, more_refined_pixels_exp_f



#GPA specific functions

def Choose_g_vectors(refined_distances, refined_angles_to_x, refined_pixels, g2='perpendicular'):
    '''
    Define the vectors or pixels positions to be used for the GPA calculation
    
    g1 is computed based on the closest spot to x, as typically the interfaces will
    be almost paralel to x. If it was not the case in some scenarions, another 
    setting just having as input the interface orientation would be used to choose 
    the spot closer to this interface orientation
    
    Two possible setups for computing g2, based on g1 to ensure orthogonality
    
    if g2=='closest':
    # choose the largest distance (closest spot to the center) that is not g1, in
    # the right half of the FFT
        
    if g2=='perpendicular':
    # choose the most perpendicular spot to the choosen g1, also from the right half of FFT
        
    Parameters
    ----------
    refined_distances : distances, of physically meaningful spots, no central spot, 
                        in angstroms (although it does not matter here)
    refined_angles_to_x : angles to x axis, of physically meaningful spots, 
                        no central spot, in degrees, from 180 to -180
    refined_pixels : pixels positions, of physically meaningful spots, no central spot, y and x

    Returns
    -------
    The positions, in pixels, of the spots to be used as g vectors for the GPA calculation
    in an array of shape=(2,2), first array is g1 with y1,x1 and second array g2 with y2,x2
    
    We output as well a 2 elements list with the angles to x of the given spots g1, g2
    As the angle counts from x positive axis, the smallest angle is at the right of
    the FFT and we discard the spots on the left (in case we use just one half of the FFT later)
    
    The 0,0 of the pixels reference is top left corner 
    '''
    
    #if the vertical planes are just the closest to the horizontal plane, then:
    # as the distance array is sorted from largest distances to 
    
    #only account for the spots of the right half of the FFT
    refined_distances=np.asarray(refined_distances)
    refined_angles_to_x=np.asarray(refined_angles_to_x)
    refined_pixels=np.asarray(refined_pixels)
    
    # both 90 an d-90 angles included so it goes from 90 to -90 -> [-90,90]
    newdists=refined_distances[refined_angles_to_x>=-90]
    newangls=refined_angles_to_x[refined_angles_to_x>=-90]
    newpixels=refined_pixels[refined_angles_to_x>=-90]
    
    newdists=newdists[newangls<=90]
    newpixels=newpixels[newangls<=90]
    newangls=newangls[newangls<=90]
    
    g_vectors_pixels=[]
    angles_to_x_of_g=[]  
    
    if len(newdists)==0:
        print('! G vectors selection warning:')
        print('No spots to select: Either amorphous or not enough resolution')
        return g_vectors_pixels, angles_to_x_of_g
    
    if len(newdists)==1:
        print('G vectors selection warning:')
        print('Only 1 spot identified, GPA is not possible')
        return g_vectors_pixels, angles_to_x_of_g
    
    if len(newdists)==2:
        g_vectors_pixels=newpixels
        angles_to_x_of_g=newangls
        return g_vectors_pixels, angles_to_x_of_g

    g1_vertical_plane=newpixels[np.argmin(np.abs(newangls))]
    g_vectors_pixels.append(g1_vertical_plane)
    g1_angle=newangls[np.argmin(np.abs(newangls))]
    angles_to_x_of_g.append(g1_angle)
    
    #to find the second vector, two options are given:
    if g2=='closest':
        # choose the largest distance (closest spot to the center) that is not g1, in
        # the right half of the FFT

        newpixels=newpixels[newangls !=g1_angle]
        newangls=newangls[newangls !=g1_angle]
        # as they are sorted the first element is the largest distance
        g2_horizontal_plane=newpixels[0]
        g2_angle=newangls[0] 
                           
        g_vectors_pixels.append(g2_horizontal_plane)
        angles_to_x_of_g.append(g2_angle)
        
    if g2=='perpendicular':
        # choose the most perpendicular spot to the choosen g1

        newdists=newdists[newangls !=g1_angle]
        newpixels=newpixels[newangls !=g1_angle]
        newangls=newangls[newangls !=g1_angle]
        
        #the most perpendicular one to g1
        modenewangls=np.abs(np.abs(newangls-g1_angle)-90)
        
        newdists1=[dist for _,dist in sorted(zip(modenewangls,newdists), key=lambda pair: pair[0])]
        newpixels1=[pixels for _,pixels in sorted(zip(modenewangls,newpixels), key=lambda pair: pair[0])]
        newangls1=[angls for _,angls in sorted(zip(modenewangls,newangls), key=lambda pair: pair[0])]
          
        g2_horizontal_plane= newpixels1[0]
        g2_angle=newangls1[0]             
        # this chooses the angle closest to the perpendicular, but make sure it is
        # the smallest distance with that angle, as the pixelwise precision definition
        # of the central spot may make the angle vary slighly and make the final spot be 
        # the one farthest from the center
        
        for dist,angle,pixel in zip(newdists1,newangls1,newpixels1):

            if abs(angle)<abs(newangls1[0])+0.5 and abs(angle)>abs(newangls1[0])-0.5 and dist>=newdists1[0]:
            
                g2_horizontal_plane= pixel
                g2_angle=angle
            
        g_vectors_pixels.append(g2_horizontal_plane)
        angles_to_x_of_g.append(g2_angle)
        
    return g_vectors_pixels, angles_to_x_of_g


#Functions also important for the general algorithm of low to high mag final communication


# Provisional function for a quick segmentation to give the reference automatically
def Quick_Segmentation_for_Reference(image_array, k_clusters):
    #input the image ready for doing the segmentation already, if denoised or standarised or downscaled...
    # image_st=(image_array-np.min(image_array))/np.max(image_array-np.min(image_array))    
    
    # denoised_image=Adaptative_Gaussian_Blur(image_st, x_calibration)
    #denoised_image=cv2.GaussianBlur(image_st, gauss_blur_filter_size, 1)
    
    #Second standarisation of the image after filtering/blurring it with gaussian filter
    
    image_st=(image_array-np.min(image_array))/np.max(image_array-np.min(image_array))
    
    downscaling_factor=int(image_st.shape[0]/64)
    ds_image=skimage.measure.block_reduce(image_st, block_size=tuple(np.int32(downscaling_factor*np.ones(len(np.shape(image_st))))), func=np.mean, cval=0)
    
    values, labels, cost = best_km(ds_image, n_clusters =k_clusters)
    labels_ms_reshaped = np.choose(labels, values)
    labels_ms_reshaped.shape = ds_image.shape
  
    labels_ms_reshaped=Multiple_Consecutive_Clustered_regions(labels_ms_reshaped, criteria='Tolerance', tolerance=0.10)

    # plt.imshow(labels_ms_reshaped,cmap=plt.cm.gray)
    # plt.show()
    
    substrate_segment=np.copy(labels_ms_reshaped)
    last_row_sgmented=labels_ms_reshaped[-1,:]
    substrate_val_inter=np.unique(last_row_sgmented)
    counted=[]
    for val in substrate_val_inter:
        counts=len(last_row_sgmented[last_row_sgmented==substrate_val_inter])
        counted.append(counts)       
    substrate_val=substrate_val_inter[np.argmax(np.asarray(counted))]
    
    for ini, i in enumerate(substrate_segment):
        for inij, j in enumerate(i):
            if j!=substrate_val:
                substrate_segment[ini,inij]=0
            else:
                substrate_segment[ini,inij]=1
    
    conts_vertx_per_region=Contour_draw_All_regions(substrate_segment)

    final_extra_pixs=Denoise_All_Regions(substrate_segment,conts_vertx_per_region)
      
    for label in final_extra_pixs:
        for pixel in final_extra_pixs[label]:   
            substrate_segment[pixel[0],pixel[1]]=int(label[0])    
    
    size_down=(labels_ms_reshaped.shape[0]*downscaling_factor,labels_ms_reshaped.shape[0]*downscaling_factor)
    segmented_image= cv2.resize(substrate_segment, size_down, interpolation = cv2.INTER_AREA)
    # plt.imshow(segmented_image,cmap=plt.cm.gray)
    # plt.show()
     
    # this line tailors the process to the images used to test, but should be removed as soon as the 
    #independent region finding for each label is calculated  
          
    #segmented_image[0:int(np.floor(size_down[0]/2)),:]=0
    
    return segmented_image
          


def Segmented_Region_Conversion_into_Matrix(segmented_image, label_of_region):
    '''
    Turn a segmented region identified by the label=label_of_region into 
    the smallest possible rectangular matrix that contains all the pixels with a value
    equal to label_of_region
    The segmented image only needs that the specified region to consider has the value
    label_of_region and the rest of pixels can be whatever
    
    IMPORTANT: It extracts the coordinates of the cropped region to be stored for later
    steps to know where the region came from in the original image
    The tuple format is coordinates_of_region=(first_row,last_row,first_col,last_col)
    So to crop the region from the original image it can be done like the following:
    region=image[first_row:last_row+1,first_col:last_col+1]

    Parameters
    ----------
    segmented_image : matrix of the segmented image
    label_of_region : int

    Returns
    -------
    matrix_of_region 

    '''
    
    image_of_region=np.copy(segmented_image)
    
    image_of_region[image_of_region==label_of_region]=1
    image_of_region[image_of_region!=label_of_region]=0
    
    
    positions_1=np.where(image_of_region==1)
   
    first_row=min(positions_1[0])
    last_row=max(positions_1[0])
    

    # to discard the correct case in which the lowest index is 0, where second index must be 1
    first_col=min(positions_1[1])
    last_col=max(positions_1[1])
    
    matrix_of_region=image_of_region[first_row:last_row+1,first_col:last_col+1]
    #coordinates, with row i, row final, col i, col final
    coords_of_region=(first_row,last_row,first_col,last_col)
    return matrix_of_region, coords_of_region



def Find_Biggest_Squares_per_Segmented_Region(matrix_of_region):
    
    #first turn the result matrix a copy of the original and put everything to 0
    #except the border pixels of the top and left edges
    
    matrix_of_squares=np.copy(matrix_of_region)
    
    matrix_of_squares[1:matrix_of_squares.shape[0],1:matrix_of_squares.shape[1]]=0
    
    for y in range(1,matrix_of_squares.shape[0]):
        for x in range(1,matrix_of_squares.shape[1]):
            if matrix_of_region[y,x]==1:
                matrix_of_squares[y,x]=min(matrix_of_squares[y-1,x],matrix_of_squares[y,x-1],matrix_of_squares[y-1,x-1])+1
                  
    return matrix_of_squares
    
 
    
def Find_Biggest_Square_Closest_to_Region_Center(matrix_of_region, matrix_of_squares):
    '''
    Returns the coordinates of the square, in format,
    coords_of_square= [first_row,last_row,first_col,last_col], 
    [closest_square_top_left_corner[0],closest_square_bottom_right_corner[0],closest_square_top_left_corner[1],closest_square_bottom_right_corner[1]]
    to crop the image add a 1 in the final coords of x and y

    Parameters
    ----------
    matrix_of_region : TYPE
        DESCRIPTION.
    matrix_of_squares : TYPE
        DESCRIPTION.

    Returns
    -------
    coords_of_square_to_crop : TYPE
        DESCRIPTION.

    '''
    
    #in regular regions, just the central position:
    # probably the weighted to the positions of the region would be better but if better just very slightly
    # and only in counted occasions so not really worth programming it
    
    central_coords=(matrix_of_region.shape[0]//2, matrix_of_region.shape[1]//2) 
    
    size_of_biggest_square=int(np.max(matrix_of_squares))
    
    pos_bottom_right_corner_bigst_sqrs=np.where(matrix_of_squares==size_of_biggest_square)
    
    pos_center_bigst_sqrs=np.array([pos_bottom_right_corner_bigst_sqrs[0],pos_bottom_right_corner_bigst_sqrs[1]])
    pos_center_bigst_sqrs[0]=pos_bottom_right_corner_bigst_sqrs[0]-size_of_biggest_square//2
    pos_center_bigst_sqrs[1]=pos_bottom_right_corner_bigst_sqrs[1]-size_of_biggest_square//2
    
    distance_to_center=np.sqrt((pos_center_bigst_sqrs[0]-central_coords[0])**2+(pos_center_bigst_sqrs[1]-central_coords[1])**2)
    
    closest_square_bottom_right_corner=(pos_bottom_right_corner_bigst_sqrs[0][np.argmin(distance_to_center)],pos_bottom_right_corner_bigst_sqrs[1][np.argmin(distance_to_center)])
    closest_square_top_left_corner=[closest_square_bottom_right_corner[0]-size_of_biggest_square,closest_square_bottom_right_corner[1]-size_of_biggest_square]
    
    coords_of_square=[closest_square_top_left_corner[0],closest_square_bottom_right_corner[0],closest_square_top_left_corner[1],closest_square_bottom_right_corner[1]]
   
    return coords_of_square




def Locate_Reference_in_Original_Image(coords_of_region, coords_of_reference):
    '''
    Converts the coordiantes of the square of the reference, in terms of the cropped segmented region
    into the corresponding coordinates of the image where the region is taken from, to be given to GPA for 
    the final reference (they can be slightly reduced to avoid the effects of the image border)
    
    The coordiantes must be in the index format, to crop a 1 must be added: [first_row,last_row,first_col,last_col] 

    output, as usual, of indices to add 1 to crop coords_of_square= [first_row,last_row,first_col,last_col]
    
    Parameters
    ----------
    coordinates_of_region : 
    coords_of_reference : 

    Returns
    -------
    scaled_reference_coordinates : 

    '''
      
    (row_reg_i,row_reg_f,col_reg_i,col_reg_f)=coords_of_region
    (row_ref_i,row_ref_f,col_ref_i,col_ref_f)=coords_of_reference
    
    square_size=col_ref_f-col_ref_i
    
    #coords initial and final, with (y (row), x (col)) format
    scal_ref_cord_i=(row_reg_i+row_ref_i,col_reg_i+col_ref_i)
    scal_ref_cord_f=(scal_ref_cord_i[0]+square_size,scal_ref_cord_i[1]+square_size)
   
    scaled_reference_coords=(scal_ref_cord_i[0],scal_ref_cord_f[0],scal_ref_cord_i[1],scal_ref_cord_f[1])

    return scaled_reference_coords
    

def Reduce_Reference_if_Border(image_st_reference, scaled_reference_coords, reduction=0.1):
    '''
    Reduce a % = reduction of the size of the square of the reference in case it contacts the edges of
    the image
    '''
    image_size=image_st_reference.shape[0]
    row_i,row_f,col_i,col_f=scaled_reference_coords
    square_size=row_f+1-row_i
    
    # if the square constacts the right, left or bottom or top (top for other cases than reference)
    if row_f+1==image_size or col_f+1==image_size or row_i==0 or col_i==0:
        
        pxls_delete_side=int(np.floor(square_size*reduction)/2)
        row_i,row_f,col_i,col_f=row_i+pxls_delete_side,row_f-pxls_delete_side,col_i+pxls_delete_side,col_f-pxls_delete_side
    
    reduced_reference_coords=row_i,row_f,col_i,col_f
    return reduced_reference_coords
    
#compute mask size

def Define_GPA_Mask_Size(image, FFT_calibration, smallest_feature=0):
    # FFT calibration in nm^-1/pixel
    # As GPA is statistical, it should be applied mostly to low magnification/high FOV images
    # we do not want too strong local variations within the images to get good averages to extract
    #and use them to translate to the final model: we do not require high resolution, then something
    # 9 - 6nm should be good enough for most cases.
    
    # make mask of a given number of pixels coinciding with some nm, to make it sample dependent adjunst the
    # resolution to the features observed in the image
    
    #the smallest_feature argument stands for the value computed, in nm ,of the smalles segmented feature in
    # the device we are analysing: if we have a QW between two layers, then the QW is smallest and is 20nm thick
    
    if smallest_feature !=0:
        # define hyperparameter feature resolution, of the variation within the smallest element identified:
        # if the smallest feature is a QW of 20nm, then the resolution we get is 20/2.5. If more resolution is
        # wanted, then increase this value but it will be much noisier for resolutions < 5nm
        feature_resolution=2.5
        resolution=smallest_feature/feature_resolution
        mask_size=int(np.floor(1/(resolution*FFT_calibration)))
        GPA_eff_resolution=1/(mask_size*FFT_calibration)
        # Put a security check in case the feature is too large and then the resolution is too low:
        if GPA_eff_resolution > 10:
            resolution=10
            mask_size=int(np.floor(1/(resolution*FFT_calibration)))
            GPA_eff_resolution=1/(mask_size*FFT_calibration)
    else:
        resolution=7  #nm
        mask_size=int(np.floor(1/(resolution*FFT_calibration)))
        GPA_eff_resolution=1/(mask_size*FFT_calibration)
        
    return mask_size, GPA_eff_resolution

#compute the angle based on the peaks identified or on the interface

def Compute_GPA_Rotation_X_Axis(list_of_ALL_spots_angles, substrate_segment_image,substrate_segment_contour):
    '''
    Computes the angle to be used for GPA
    It firstly copmutes if there is a certain spot identified to have an angle close to 0 or 90 and in the 
    latter case it is substracted by 90
    In case there is not, the angle of the interface is computed. It can be modified as if "angle"
    or "interface" are the different criterias possible to choose

    Parameters
    ----------
    list_of_ALL_spots_angles : all the angles from -180 to 180 of the identified spots
    substrate_segment_image : image with only the substrate as a segment, 1, and the rest 0 intenstiy
    substrate_segment_contour : contours computed for this substrate with intensity 1

    Returns
    -------
    angle_of_horizontal_planes : TYPE
        DESCRIPTION.

    '''
    # compute the angle between the theoretical x axis, perfectly horizontal, with the horizontal planes
    # actually observed in the image to correct it to GPA
    
    # First compute the angles by taking into account the spots that were found by the peak finding
    # Adjust the finding so it only computes angle if it is closer to 0, or to +-90 and then substract 90
       
    for angle in list_of_ALL_spots_angles:
    # as the angles are sorted the first will appear being the spots closest to the cener   
        if angle<=0+5 and angle>=0-5:
            angle_of_horizontal_planes=angle
            return angle_of_horizontal_planes
        if angle<=90+5 and angle>=90-5:
            angle_of_horizontal_planes=angle-90
            return angle_of_horizontal_planes
            
    # Otherwise, work with finding out the interface orientation
    #work with segmented profile where 0 intensity is not the substrate and 1 intensity is the substrate pixels
    # then only substrate_segment_contour[str(int(1))+'_contours'] contains contours
    coords_same_y=[]
    
    contour_vectors=substrate_segment_contour[str(int(1))+'_contours']
    
    for contour in contour_vectors:
        #format of this coords is (y,x)
        init_c=contour.init_coords
        final_c=contour.final_coords
        #contours coordinates are not pixel coordinates as each pixel has 4 coordinates, and the
        # last pixel in an image with last index=image.shape[0]-1, has the 1st coordinate=image.shape[0]-1
        # and the last coordiante= image.shape[0]
        
        #pixels that share the same y (height) and do not belong to the edges of the image
        if init_c[0] != substrate_segment_image.shape[0] and init_c[0] !=0 and init_c[0]==final_c[0]  :
            coords_same_y.append([init_c[0],init_c[1]])
            
    coords_same_y=np.asarray(coords_same_y)   
    #unify the coordinates with the same y
    y_values=np.unique(coords_same_y[:,0])
    
    coord_values_compressed=[]
    
    # way of computing the stacked segments in the interface with same y, done by substracting the
    # values that coincide with the x cooridnates and finding the 0s, much more efficient than loop

    for y_val in y_values:
        y_val_coords=coords_same_y[coords_same_y[:,0]==y_val]
        y_val_coords=np.asarray(sorted(y_val_coords, key=lambda x: x[1])[::-1])
        next_index_to_start=0
        while next_index_to_start<y_val_coords.shape[0]:       
            first_x_val=y_val_coords[next_index_to_start,1]
            y_coords_shape=y_val_coords.shape[0]
            x_vals_array=np.arange(first_x_val-(y_coords_shape-next_index_to_start)+1, first_x_val+next_index_to_start+1, 1)[::-1]
            x_vals_array.shape=y_coords_shape
            y_val_coords_to_eval=np.subtract(y_val_coords[:,1],x_vals_array)
            y_eval_0s=np.where(y_val_coords_to_eval==0)
            starting_y_val_sgmnt=y_val_coords[y_eval_0s[0][0]]
            ending_y_val_sgmnt=y_val_coords[y_eval_0s[0][-1]]
            coord_values_compressed.append(starting_y_val_sgmnt)  
            coord_values_compressed.append(ending_y_val_sgmnt)  
            
            next_index_to_start=y_eval_0s[0][-1]+1
     
    coord_values_compressed=np.asarray(coord_values_compressed)
    
    #plot x y graph
    #sort from lower x to higher x
    coord_values_compressed=np.asarray(sorted(coord_values_compressed, key=lambda x: x[1]))
    
    #define segments in the format (y , x_start, y_final)
    
    segment_definition=[]
    index=0
    while index<coord_values_compressed.shape[0]:
        segment=[coord_values_compressed[index,0],coord_values_compressed[index,1],coord_values_compressed[index+1,1]]
        segment_definition.append(segment)
        index=index+2
    
    segment_definition=np.asarray(segment_definition)
    #find largest segment
    largest_segment=segment_definition[np.argmax(np.subtract(segment_definition[:,2],segment_definition[:,1]))]
    central_position=np.floor((largest_segment[2]-largest_segment[1])/2)
    
    min_index_down=np.min(segment_definition[:,1])
    max_index_up=np.max(segment_definition[:,2])
    
    angles_array=[]

    down_index=central_position
    up_index=central_position+1
    
    # doing the increase of the segment lenght first on one side and then to the other keeping the same
    # central position is less sensitive to noise in case there is a bad segmentation at some point 
    while down_index>min_index_down:
        
        index_vals=[]
        for index in [down_index, up_index]:
            init_x_vals=segment_definition[:,1]
            init_x_vals=init_x_vals-index
            init_x_vals[init_x_vals>0]=2**10
            index_val=segment_definition[np.argmin(abs(init_x_vals)),0]
            index_vals.append(index_val)
            
        yi,yf=index_vals
       
        angle=np.arctan2((yf-yi),(up_index-down_index))*(180/np.pi)
        
        angles_array.append(angle)
        down_index=max(down_index-1, min_index_down)
        
    down_index=central_position
    up_index=central_position+1  
    
    while up_index<max_index_up:
         
        index_vals=[]
        for index in [down_index, up_index]:
            init_x_vals=segment_definition[:,1]
            init_x_vals=init_x_vals-index
            init_x_vals[init_x_vals>0]=2**10
            index_val=segment_definition[np.argmin(abs(init_x_vals)),0]
            index_vals.append(index_val)
            
        yi,yf=index_vals
       
        angle=np.arctan2((yf-yi),(up_index-down_index))*(180/np.pi)
        
        angles_array.append(angle)
        
        up_index=min(up_index+1,max_index_up)
    # opposite angle as the quadrants are vertically inverted because the y are smaller at the top of the image
    angles_array=-np.asarray(angles_array)    
    
    # delete the outliers that are caused by noise generating big angles that are not possible
    # for an interface, at least a "growth" one
    # we assume that difficult to have a real rotation of the interface of >2.5 degrees, so allow noise 
    # rotation of up to +-5º in case they are centered arround 2.5
    angles_array=angles_array[abs(angles_array)<5]
    
    # plt.hist(angles_array,30,[np.min(angles_array),np.max(angles_array)])
    # plt.show()
    
    angle_of_horizontal_planes=np.mean(angles_array)
    return angle_of_horizontal_planes



#!!! To be completed as soon as segmentation is robust

def Compute_Smallest_Feature_Thickness(segmented_image):
    #To be done once the segmentation routine is strong enough
    smallest_feature_thikness=1
    return smallest_feature_thikness


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



#K-means clustering functions
def km_clust(array, n_clusters):
    # Create a line array, the lazy way
    X = array.reshape((-1, 1))
    # Define the k-means clustering problem
    k_m = sklearn.cluster.KMeans(n_clusters=n_clusters, init='random',n_init=4,random_state=2**13)
    # Solve the k-means clustering problem
    k_m.fit(X)
    # Get the coordinates of the clusters centres as a 1D array
    values = k_m.cluster_centers_.squeeze()
    # Get the label of each point
    labels = k_m.labels_
    cost=k_m.inertia_
    return(values, labels, cost)

#Gets the best K-means clustering option from 100 iterations
def best_km(array, n_clusters):
    iterations=100
    kmeans_values=[]
    kmeans_labels=[]
    kmeans_cost=[]
    for index,i in enumerate(range(iterations)):
        values, labels, cost= km_clust(array, n_clusters = n_clusters)
        kmeans_values.append(values)
        kmeans_labels.append(labels)
        kmeans_cost.append(cost)
    kmeans_cost=np.array(kmeans_cost)
    best_cost=kmeans_cost.min()    
    best_values=kmeans_values[kmeans_cost.argmin()]
    best_labels=kmeans_labels[kmeans_cost.argmin()]
    return (best_values, best_labels,best_cost)

#Mean shift clustering algorithm
def Mean_shift(image):
    image_reshaped=np.reshape(image, (image.shape[0]*image.shape[1],1))
    # The following bandwidth can be automatically detected using
    #bandwidth = sklearn.cluster.estimate_bandwidth(image, quantile=0.2, n_samples=500)
    
    ms = sklearn.cluster.MeanShift( bin_seeding=True)
    ms.fit(image_reshaped)
    labels = ms.labels_
    cluster_centers = ms.cluster_centers_
    
    return labels, cluster_centers


#Consecutive pixels for a given label, giving an initial target pixel to start with
def Consecutive_pixels_particular_pixel(image_array_labels, target_coords):
    # Same as previous but giving a starting pixel to start from to do the consecutive value analysis


    class pixels_state:
        def __init__(self, coordinates, value, state):
            '''
            State can be 'Old', referring to pixels that were in the array before and therefore do not need to be considered 
            for checking again, or 'New', meaning that it has just been uploaded to the array and has to act as a point for getting new 
            conscutive pixels
            '''  
            self.coords=coordinates
            self.value=value
            self.state=state
        
    ver_pixs=image_array_labels.shape[0]
    hor_pixs=image_array_labels.shape[1]
    
    contiguous_cluster=[]
           
    target_pixel=image_array_labels[target_coords[0],target_coords[1]] 
    
    #process of converting into the class of pixels and appending them into the cluster's array
    
    
        
    #the list contains all the pixels that are contiguous to the first one, each element is a pixels_state class
    
    
    target_pixel_classed=pixels_state((target_coords[0],target_coords[1]), target_pixel, state='New')
    contiguous_cluster.append(target_pixel_classed) 
    
    
    #a list contains all the new pixels found in each iteration
    
    all_coordinates=[(target_coords[0],target_coords[1])]
    for pixel in contiguous_cluster:
        
        
        if pixel.state=='New':
          
            array_containing_all_new_pixels=[]
            (pixel_y,pixel_x)=pixel.coords
            
            if pixel_y==0 and pixel_x!=0 and pixel_x!=hor_pixs-1:
                pixel_N=None
                pixel_NW=None
                pixel_NE=None
                pixel_W=image_array_labels[pixel_y,pixel_x-1] 
                pixel_SW=image_array_labels[pixel_y+1,pixel_x-1] 
                pixel_S=image_array_labels[pixel_y+1,pixel_x] 
                pixel_SE=image_array_labels[pixel_y+1,pixel_x+1] 
                pixel_E=image_array_labels[pixel_y,pixel_x+1] 
            
            #condition for left upper pixel
            elif pixel_y==0 and pixel_x==0: 
                pixel_N=None
                pixel_NW=None
                pixel_NE=None
                pixel_W=None
                pixel_SW=None
                pixel_S=image_array_labels[pixel_y+1,pixel_x] 
                pixel_SE=image_array_labels[pixel_y+1,pixel_x+1] 
                pixel_E=image_array_labels[pixel_y,pixel_x+1] 
                
            #condition for right upper pixel
            elif pixel_y==0 and pixel_x==hor_pixs-1: 
                pixel_N=None
                pixel_NW=None
                pixel_NE=None
                pixel_E=None
                pixel_SE=None
                pixel_W=image_array_labels[pixel_y,pixel_x-1] 
                pixel_SW=image_array_labels[pixel_y+1,pixel_x-1] 
                pixel_S=image_array_labels[pixel_y+1,pixel_x] 
                
            #condition for bottom limit pixels, not edge ones
            elif pixel_y==ver_pixs-1 and pixel_x!=0 and pixel_x!=hor_pixs-1:
                pixel_S=None
                pixel_SW=None
                pixel_SE=None
                pixel_N=image_array_labels[pixel_y-1,pixel_x] 
                pixel_NW=image_array_labels[pixel_y-1,pixel_x-1] 
                pixel_W=image_array_labels[pixel_y,pixel_x-1] 
                pixel_E=image_array_labels[pixel_y,pixel_x+1] 
                pixel_NE=image_array_labels[pixel_y-1,pixel_x+1] 
                
            #condition for left bottom pixel 
            elif pixel_y==ver_pixs-1 and pixel_x==0: 
                pixel_S=None
                pixel_SW=None
                pixel_SE=None
                pixel_W=None
                pixel_NW=None
                pixel_N=image_array_labels[pixel_y-1,pixel_x] 
                pixel_E=image_array_labels[pixel_y,pixel_x+1] 
                pixel_NE=image_array_labels[pixel_y-1,pixel_x+1] 
            
            #condition for right bottom pixel
            elif pixel_y==ver_pixs-1 and pixel_x==hor_pixs-1: 
                pixel_S=None
                pixel_SW=None
                pixel_SE=None
                pixel_E=None
                pixel_NE=None
                pixel_N=image_array_labels[pixel_y-1,pixel_x] 
                pixel_NW=image_array_labels[pixel_y-1,pixel_x-1] 
                pixel_W=image_array_labels[pixel_y,pixel_x-1] 
                
            #condition for left limit pixels, not edge ones
            elif pixel_x==0 and pixel_y!=0 and pixel_y!=ver_pixs-1:
                pixel_W=None
                pixel_SW=None
                pixel_NW=None
                pixel_N=image_array_labels[pixel_y-1,pixel_x] 
                pixel_S=image_array_labels[pixel_y+1,pixel_x] 
                pixel_SE=image_array_labels[pixel_y+1,pixel_x+1] 
                pixel_E=image_array_labels[pixel_y,pixel_x+1] 
                pixel_NE=image_array_labels[pixel_y-1,pixel_x+1] 
                
            #condition for right limit pixels, not edge ones
            elif pixel_x==hor_pixs-1 and pixel_y!=0 and pixel_y!=ver_pixs-1:
                pixel_E=None
                pixel_NE=None
                pixel_SE=None
                pixel_N=image_array_labels[pixel_y-1,pixel_x] 
                pixel_NW=image_array_labels[pixel_y-1,pixel_x-1] 
                pixel_W=image_array_labels[pixel_y,pixel_x-1] 
                pixel_SW=image_array_labels[pixel_y+1,pixel_x-1] 
                pixel_S=image_array_labels[pixel_y+1,pixel_x] 
                
            else:
                #it is a pixel out from the edges and borders, typical one
                pixel_N=image_array_labels[pixel_y-1,pixel_x] 
                pixel_NW=image_array_labels[pixel_y-1,pixel_x-1] 
                pixel_W=image_array_labels[pixel_y,pixel_x-1] 
                pixel_SW=image_array_labels[pixel_y+1,pixel_x-1] 
                pixel_S=image_array_labels[pixel_y+1,pixel_x] 
                pixel_SE=image_array_labels[pixel_y+1,pixel_x+1] 
                pixel_E=image_array_labels[pixel_y,pixel_x+1] 
                pixel_NE=image_array_labels[pixel_y-1,pixel_x+1] 

            array_of_immediate_pixels=np.array([pixel_N,pixel_NW,pixel_W, pixel_SW, pixel_S, pixel_SE, pixel_E, pixel_NE])
            coords_of_imm_pixels=[(pixel_y-1,pixel_x),(pixel_y-1,pixel_x-1),(pixel_y,pixel_x-1),
                          (pixel_y+1,pixel_x-1),(pixel_y+1,pixel_x),(pixel_y+1,pixel_x+1),
                          (pixel_y,pixel_x+1),(pixel_y-1,pixel_x+1)]

            
            for pixel_i,coordinate in zip(array_of_immediate_pixels,coords_of_imm_pixels):

                if pixel_i==pixel.value:
                    already=0
                    for coordinate_old in all_coordinates:
                        
                        if coordinate[0]==coordinate_old[0] and coordinate[1]==coordinate_old[1]:
                            already=1
                    if already !=1:
                        all_coordinates.append(coordinate)
                        
                        pixel_classed=pixels_state(coordinate,pixel_i,state='New')
                    
                        array_containing_all_new_pixels.append(pixel_classed)        
            
          
                
            contiguous_cluster.extend(array_containing_all_new_pixels)
            
        pixel.state='Old'         
                    
                    
    return contiguous_cluster                    


#Find all the clusters present in an image labelled with n labels, and select them given a criteria
def Multiple_Consecutive_Clustered_regions(image_array_clusters, criteria, tolerance=0, n_clusters=0):
    '''
    For each big cluster, find the consecutive big regions, meaning that each big region should have an independent cluster.
    Watch out about reducing it to a cluster per value problem, as one value may have one or more associated clusters. The number
    of final regions to get should be what the scientist considers are distingishable regions (materials, materials domains, etc..)
    
    'Tolerance': Criteria can be either 'Tolerance', in which case all the clusters containing less than the specified number of pixels are removed.
    It should be specified as the a % of the total number of pixels (e.g tolerance=0.2 --> select the clusters that have more than a 
                                                                     20% of the total number of pixels)
    'Num_Clusters': On the other hand, criteria can be 'Num_Clusters', which means that only the specified number of clusters will be returned. These 
    are ordered from more to less pixels per cluster, and only the n_clusters with more pixels, independently on the pixel value, will 
    be returned.
    

    Returns
    -------
    Changed sligthly from previous version, now it outputs directly the discretised image, with
    integers of the pixels of interest, and the remaining unclustered pixels having a 0 intensity
    watch out as if the criteria is 'Tolerance', the output is a dictionary containing the clusters, in which each cluster is defined
    as dict [value of the cluster + _ + some index to avoid repetition]
    while if criteria is 'Num_Clusters', the output is a list with n_clusters clusters stored as ordered_clusters objects, and to extract 
    cluster itself and its pixels (stored as pixels_state objects) they should be taken as output[i].cluster, 
    where i goes from 0 to n_clusters

    '''
    
    class pixels_state:
        def __init__(self, coordinates, value, state):
            '''
            State can be 'Old', referring to pixels that were in the array before and therefore do not need to be considered 
            for checking again, or 'New', meaning that it has just been uploaded to the array and has to act as a point for getting new 
            conscutive pixels
            '''  
            self.coords=coordinates
            self.value=value
            self.state=state
    
    class order_clusters:
        def __init__(self, cluster_list):
            self.cluster=cluster_list
            self.lenght=len(cluster_list)
             
        
    image_array_clusters_copy=np.copy(image_array_clusters)
    
   
    
    ver_pixs=image_array_clusters.shape[0]
    hor_pixs=image_array_clusters.shape[1]
    
    radom_value_for_difference=2**14
    #whether it is tolerance or number of final regions
    
    
    # variables for tolerance criteria
    
    tolerance_pixels=int(round(tolerance*ver_pixs*hor_pixs))
    
    dictionary_of_clusters=dict()
    
    # variables for number of clusters criteria
    
    all_possible_clusters=[]

    
    for pixel_y in range(ver_pixs):
        
        for pixel_x in range(hor_pixs):
            
            
            if image_array_clusters_copy[pixel_y,pixel_x] != radom_value_for_difference:
                
                #then it means that this pixel does not belong to any cluster and can therefore form one, even if it is
                #just a single pixel cluster
                contiguous_cluster=Consecutive_pixels_particular_pixel(image_array_clusters_copy, (pixel_y,pixel_x))
                
                if criteria=='Tolerance':
                    
                    for pixel in contiguous_cluster:
                        
                        (pixel_change_y, pixel_change_x)=pixel.coords
                        image_array_clusters_copy[pixel_change_y, pixel_change_x]=radom_value_for_difference
                    
                    if len(contiguous_cluster) > tolerance_pixels:
                        
                        value_pixels_cluster=contiguous_cluster[0].value
                        
                        index_addition=len(dictionary_of_clusters)
                        
                        dictionary_of_clusters[str(value_pixels_cluster)+'_'+str(index_addition)]=contiguous_cluster
                        
                        #this way the dictionary is able to classify the clusters given its corresponding value, as
                        #the first character of the element is the value that the cluster has
                    
                    
                    output=dictionary_of_clusters    
                    
                elif criteria=='Num_Clusters':
                    
                    for pixel in contiguous_cluster:
                        
                        (pixel_change_y, pixel_change_x)=pixel.coords
                        image_array_clusters_copy[pixel_change_y, pixel_change_x]=radom_value_for_difference
                        
                    cluster_with_lenght_obj=order_clusters(contiguous_cluster)
                    
                    all_possible_clusters.append(cluster_with_lenght_obj)
                        
                    
                    
    if criteria=='Num_Clusters':
        
        #sort depending on the number of elements per cluster (descending order, first the larger clusters)
        #and then select only the first n_clusters, which will be the largest ones
        
        all_possible_clusters.sort(key=(lambda x: x.lenght), reverse=True)
        output=all_possible_clusters[:n_clusters]
        
    # Once the clusters are ordered and selected according to the criteria, convert the image into 
    # a discretised array with every contigous regions with a specific number 
    
    output_discretised_image=np.zeros(np.shape(image_array_clusters))
    
    if criteria=='Tolerance':
    # Way of dealing with the 'Tolerance' criteria to get output discretised with 0,1,2....
        value_image=1
        for cluster in output:
            cluster_i=output[str(cluster)]
            for pixel in cluster_i:
                (pixel_y,pixel_x)=pixel.coords
                output_discretised_image[pixel_y,pixel_x]=value_image
            value_image=value_image+1
    
    if criteria=='Num_Clusters':
    # Way of dealing with the 'Num_clusters' criteria to get output discretised with 0,1,2....
        value_image=1
        for cluster in output:
            cluster_i=cluster.cluster
            for pixel in cluster_i:               
                (pixel_y,pixel_x)=pixel.coords
                output_discretised_image[pixel_y,pixel_x]=value_image
        
            value_image=value_image+1
      
    return  output_discretised_image


#Function that draws the contours (vertex and vectors) of a region given an initial pixel from its contour
def Contour_drawing_initial_pixel(image_array_labels, initial_pixel):
    
    class contour_vector:
        def __init__(self,m_in,m_out,init_coords,final_coords):
            self.m_in=m_in
            self.m_out=m_out
            self.init_coords=init_coords
            self.final_coords=final_coords
        
    #first define the rules for defining a contour in each of the 4 (N,S,E,W)
    
    def check_North(image_array,mat,initial_pixel_coords):
        #mat indicates the value of the pixels that are being contained inside the drawn contour
        (pixel_y,pixel_x)=initial_pixel_coords
        
        m_in=image_array[pixel_y-1,pixel_x-1]
        
        #specify conditions if the pixel is checked on the edge
        
        if pixel_x == hor_pixs:
            m_out=None
        else:            
            m_out=image_array[pixel_y-1,pixel_x]
        
        if m_in == mat and m_out != mat:
            next_vertex_coords=(pixel_y-1,pixel_x)
        else:
            next_vertex_coords=None
            
        return next_vertex_coords, m_out
    
    def check_South(image_array,mat,initial_pixel_coords):
        (pixel_y,pixel_x)=initial_pixel_coords
        
        m_in=image_array[pixel_y,pixel_x]
        
        #specify conditions if the pixel is checked on the edge
        
        if pixel_x==0:
            m_out=None
        else:
            m_out=image_array[pixel_y,pixel_x-1]
        
        if m_in == mat and m_out != mat:
            next_vertex_coords=(pixel_y+1,pixel_x)
        else:
            next_vertex_coords=None
            
        return next_vertex_coords, m_out
        
    
    def check_East(image_array,mat,initial_pixel_coords):
        (pixel_y,pixel_x)=initial_pixel_coords
        
        m_in=image_array[pixel_y-1,pixel_x]
        
        #specify conditions if the pixel is checked on the edge
        
        if pixel_y==ver_pixs:
            m_out=None
        else:
            m_out=image_array[pixel_y,pixel_x]
        
        if m_in == mat and m_out != mat:
            next_vertex_coords=(pixel_y,pixel_x+1)
        else:
            next_vertex_coords=None
            
        return next_vertex_coords, m_out
    
    def check_West(image_array,mat,initial_pixel_coords):
        (pixel_y,pixel_x)=initial_pixel_coords
        
        m_in=image_array[pixel_y,pixel_x-1]
        
        #specify conditions if the pixel is checked on the edge
        
        if pixel_y==0:
            m_out=None
        else:
            m_out=image_array[pixel_y-1,pixel_x-1]
        
        if m_in == mat and m_out != mat:
            next_vertex_coords=(pixel_y,pixel_x-1)
        else:
            next_vertex_coords=None
            
        return next_vertex_coords, m_out
    
    
    
    ver_pixs=image_array_labels.shape[0]
    hor_pixs=image_array_labels.shape[1]
    
    contour_vectors_list=[]
    
    image_array_labels_copy=np.copy(image_array_labels)
    
    initial_pixel_coords=initial_pixel
            
    # as the initial pixel coordinate is found as the bottom left edge of the pixel targeted, which is the actual coordinate,
    # (defined as top left corner of the pixel) of the pixel given as the initial, then the material inside for what is a 
    #contour EAST is the pixel_y-1 , pixel_x
    
    target_pixel=image_array_labels[initial_pixel_coords[0]-1,initial_pixel_coords[1]] 
    
    
    #and then define the exceptions depending if the pixel is in a contour or not 
    
    #add the padding to the image to make it ver_pixs+1 * hor_pixs+1
    
    # m_out material will be the following for the bottom and right edges, as the padding will have this value
    #so all the contours in which m_out = extra_pixels_value will be edge pixels from either the bottom or right edge
    #although maybe it is better to just set all the non existing m out as None rather than a value...
    extra_pixels_value=2**16
    
    padded_image=cv2.copyMakeBorder(image_array_labels_copy,0,1,0,1,cv2.BORDER_CONSTANT,value=extra_pixels_value)
    
    #list containing all the coordinates from all the vertex that are being analysed
    vertex_array=[]
    
    vertex_array.append(initial_pixel_coords)
    
    #list containing the vertexs that led to two contours (special case)
    list_double_vertexs=[]
    
    for vertex in vertex_array:
        
        #check if vertex is contained in an edge or corner to tailor the different checkings
        
        if vertex[0]==0 and vertex[1]==0:
            #top left corner
            #only check south
            next_vertex_coords, m_out=check_South(padded_image,target_pixel,vertex)
            
            if next_vertex_coords == initial_pixel_coords:
                new_contour=contour_vector(target_pixel,m_out,vertex,next_vertex_coords)
                contour_vectors_list.append(new_contour)
                #break  
            else:
                vertex_array.append(next_vertex_coords)
            
                #generate new contour vector and append it to the contour vectors list
                new_contour=contour_vector(target_pixel,m_out,vertex,next_vertex_coords)
                contour_vectors_list.append(new_contour)
            
        elif vertex[0]==0 and vertex[1]==hor_pixs:
            #top right corner
            #only check west
            next_vertex_coords, m_out=check_West(padded_image,target_pixel,vertex)
            
            if next_vertex_coords == initial_pixel_coords:
                new_contour=contour_vector(target_pixel,m_out,vertex,next_vertex_coords)
                contour_vectors_list.append(new_contour)
                #break  
            else:
                vertex_array.append(next_vertex_coords)
            
                #generate new contour vector and append it to the contour vectors list
                new_contour=contour_vector(target_pixel,m_out,vertex,next_vertex_coords)
                contour_vectors_list.append(new_contour)
        
        elif vertex[0]==ver_pixs and vertex[1]==0:
            #bottom left corner
            #only check east
            next_vertex_coords, m_out=check_East(padded_image,target_pixel,vertex)
            
            if next_vertex_coords == initial_pixel_coords:
                new_contour=contour_vector(target_pixel,m_out,vertex,next_vertex_coords)
                contour_vectors_list.append(new_contour)
                #break  
            else:
                vertex_array.append(next_vertex_coords)
            
                #generate new contour vector and append it to the contour vectors list
                new_contour=contour_vector(target_pixel,m_out,vertex,next_vertex_coords)
                contour_vectors_list.append(new_contour)
        elif vertex[0]==ver_pixs and vertex[1]==hor_pixs:
            #bottom right corner
            #only check north
            next_vertex_coords, m_out=check_North(padded_image,target_pixel,vertex)
            
            if next_vertex_coords == initial_pixel_coords:
                new_contour=contour_vector(target_pixel,m_out,vertex,next_vertex_coords)
                contour_vectors_list.append(new_contour)
                #break  
            else:
                vertex_array.append(next_vertex_coords)
            
                #generate new contour vector and append it to the contour vectors list
                new_contour=contour_vector(target_pixel,m_out,vertex,next_vertex_coords)
                contour_vectors_list.append(new_contour)
            
        elif vertex[0]==0 and vertex[1] !=0 and vertex[1] != hor_pixs:
            #north edge, excluding both corners
            #only check west and south
            
            possible_vertexs=[]
            m_outs=[]
            
            next_vertex_coords, m_out=check_West(padded_image,target_pixel,vertex)
            possible_vertexs.append(next_vertex_coords)
            m_outs.append(m_out)
            
            next_vertex_coords, m_out=check_South(padded_image,target_pixel,vertex)
            possible_vertexs.append(next_vertex_coords)
            m_outs.append(m_out)
            
            for possible_vertex, m_out in zip(possible_vertexs,m_outs):
                if possible_vertex != None:
                    #this should also handle the issue of one vertex diverging into two different contours, as this would
                    #append all the new possible vertices and make them new starting points for considering new vertices
                    #!!! nevertheless, repetition should be checked because on of the contours is going to led to the same vertexm
                    #and therefore it would be repeated if it is not considered
                    if possible_vertex == initial_pixel_coords:
                        new_contour=contour_vector(target_pixel,m_out,vertex,possible_vertex)
                        contour_vectors_list.append(new_contour)
                        #break  
                    else:
                        
                        vertex_array.append(possible_vertex)
                    
                        #generate new contour vector and append it to the contour vectors list
                        new_contour=contour_vector(target_pixel,m_out,vertex,possible_vertex)
                        contour_vectors_list.append(new_contour)
            
        elif vertex[0]==ver_pixs and vertex[1] !=0 and vertex[1] != hor_pixs:
            #south edge, excluding both corners
            #only check north and east
        
            possible_vertexs=[]
            m_outs=[]
            
            next_vertex_coords, m_out=check_North(padded_image,target_pixel,vertex)
            possible_vertexs.append(next_vertex_coords)
            m_outs.append(m_out)  
            
            next_vertex_coords, m_out=check_East(padded_image,target_pixel,vertex)
            possible_vertexs.append(next_vertex_coords)
            m_outs.append(m_out)
            
            for possible_vertex, m_out in zip(possible_vertexs,m_outs):
                if possible_vertex != None:
                    #this should also handle the issue of one vertex diverging into two different contours, as this would
                    #append all the new possible vertices and make them new starting points for considering new vertices
                    #!!! nevertheless, repetition should be checked because on of the contours is going to led to the same vertexm
                    #and therefore it would be repeated if it is not considered
                    if possible_vertex == initial_pixel_coords:
                        new_contour=contour_vector(target_pixel,m_out,vertex,possible_vertex)
                        contour_vectors_list.append(new_contour)
                        #break  
                    else:
                        
                        vertex_array.append(possible_vertex)
                    
                        #generate new contour vector and append it to the contour vectors list
                        new_contour=contour_vector(target_pixel,m_out,vertex,possible_vertex)
                        contour_vectors_list.append(new_contour)
            
            
            
        elif vertex[1]==0 and vertex[0] !=0 and vertex[0] != ver_pixs:
            #left edge, excluding both corners
            #only check south and east
            
            possible_vertexs=[]
            m_outs=[]            
            
            next_vertex_coords, m_out=check_South(padded_image,target_pixel,vertex)
            possible_vertexs.append(next_vertex_coords)
            m_outs.append(m_out)  
            
            next_vertex_coords, m_out=check_East(padded_image,target_pixel,vertex)
            possible_vertexs.append(next_vertex_coords)
            m_outs.append(m_out)
            
            for possible_vertex, m_out in zip(possible_vertexs,m_outs):
                if possible_vertex != None:
                    #this should also handle the issue of one vertex diverging into two different contours, as this would
                    #append all the new possible vertices and make them new starting points for considering new vertices
                    #!!! nevertheless, repetition should be checked because on of the contours is going to led to the same vertexm
                    #and therefore it would be repeated if it is not considered
                    if possible_vertex == initial_pixel_coords:
                        new_contour=contour_vector(target_pixel,m_out,vertex,possible_vertex)
                        contour_vectors_list.append(new_contour)
                        #break  
                    else:
                        
                        vertex_array.append(possible_vertex)
                    
                        #generate new contour vector and append it to the contour vectors list
                        new_contour=contour_vector(target_pixel,m_out,vertex,possible_vertex)
                        contour_vectors_list.append(new_contour)
            
        elif vertex[1]==hor_pixs and vertex[0] !=0 and vertex[0] != ver_pixs:
            #right edge, excluding both corners
            #only check north and west
            
            possible_vertexs=[]
            m_outs=[] 
            
            next_vertex_coords, m_out=check_North(padded_image,target_pixel,vertex)
            possible_vertexs.append(next_vertex_coords)
            m_outs.append(m_out)
            
            next_vertex_coords, m_out=check_West(padded_image,target_pixel,vertex)
            possible_vertexs.append(next_vertex_coords)
            m_outs.append(m_out)
            
            for possible_vertex, m_out in zip(possible_vertexs,m_outs):
                if possible_vertex != None:
                    #this should also handle the issue of one vertex diverging into two different contours, as this would
                    #append all the new possible vertices and make them new starting points for considering new vertices
                    #!!! nevertheless, repetition should be checked because on of the contours is going to led to the same vertexm
                    #and therefore it would be repeated if it is not considered
                    if possible_vertex == initial_pixel_coords:
                        new_contour=contour_vector(target_pixel,m_out,vertex,possible_vertex)
                        contour_vectors_list.append(new_contour)
                        #break 
                    else:
                        
                        vertex_array.append(possible_vertex)
                    
                        #generate new contour vector and append it to the contour vectors list
                        new_contour=contour_vector(target_pixel,m_out,vertex,possible_vertex)
                        contour_vectors_list.append(new_contour)
        
        else:
            #the majority, all the pixels that are not in an edge nor a corner
            
            possible_vertexs=[]
            m_outs=[] 
            
            next_vertex_coords, m_out=check_North(padded_image,target_pixel,vertex)
            possible_vertexs.append(next_vertex_coords)
            m_outs.append(m_out)
            
            next_vertex_coords, m_out=check_West(padded_image,target_pixel,vertex)
            possible_vertexs.append(next_vertex_coords)
            m_outs.append(m_out)
            
            next_vertex_coords, m_out=check_South(padded_image,target_pixel,vertex)
            possible_vertexs.append(next_vertex_coords)
            m_outs.append(m_out)
            
            next_vertex_coords, m_out=check_East(padded_image,target_pixel,vertex)
            possible_vertexs.append(next_vertex_coords)
            m_outs.append(m_out)
            
            
            number_of_found_vertexs=0
            for possible_vertex, m_out in zip(possible_vertexs,m_outs):
                if possible_vertex != None:
                    #this should also handle the issue of one vertex diverging into two different contours, as this would
                    #append all the new possible vertices and make them new starting points for considering new vertices
                    #!!! nevertheless, repetition should be checked because on of the contours is going to led to the same vertexm
                    #and therefore it would be repeated if it is not considered
                    if possible_vertex == initial_pixel_coords:
                        new_contour=contour_vector(target_pixel,m_out,vertex,possible_vertex)
                        contour_vectors_list.append(new_contour)
                        #break 
                    else:
                    
                        if len(list_double_vertexs)==0:
                            
                            vertex_array.append(possible_vertex)
                        
                            #generate new contour vector and append it to the contour vectors list
                            new_contour=contour_vector(target_pixel,m_out,vertex,possible_vertex)
                            contour_vectors_list.append(new_contour)
                        else:
                            
                            repeated_element=[repeated_vertex for repeated_vertex in list_double_vertexs if repeated_vertex==possible_vertex]
                            
                            if len(repeated_element)==0:
                                vertex_array.append(possible_vertex)
                            
                                #generate new contour vector and append it to the contour vectors list
                                new_contour=contour_vector(target_pixel,m_out,vertex,possible_vertex)
                                contour_vectors_list.append(new_contour)
                            else:
                                
                                #there is some element that is repeated
                                # only generate new contour vector and append it to the contour vectors list
                                # but do not append the coordinate of the new vertex as it already exists
                                new_contour=contour_vector(target_pixel,m_out,vertex,possible_vertex)
                                contour_vectors_list.append(new_contour)
                       
                    number_of_found_vertexs=number_of_found_vertexs+1       
                    
            if number_of_found_vertexs==2:
                list_double_vertexs.append(vertex)
            
   
    
    return contour_vectors_list, vertex_array


#Function that given a label, finds the initial pixel of the contour to draw the contour itself

def Find_Initial_Pixel_for_contour(image_array_labels, label_of_region):
    '''
    This function finds the inital pixel for defining the contour of a region in an image in which each region is labeled 
    differently. Then, the input of the function is the image, in which each pixel is labeled with a characteristic number,
    and the number corresponding to the label or region which has to be contoured
    
    !!! This function is sensitive to noise! If the first pixel found is in an edge, and this edge has an isolated pixel of noise 
    from another label, then this lonely (or a few of them together, but definitely not a region) one will be defined as
    region and analysed to be contoured. So the input must be a noiseless region definition

    Parameters
    ----------
    image_array_labels : TYPE
        DESCRIPTION.
    label_of_region: integer
        integer represing the regin we want to extract the inital pixel from

    Returns
    -------
    initial_pixel : TYPE
        DESCRIPTION.

    '''
    
    ver_pixs=image_array_labels.shape[0]
    hor_pixs=image_array_labels.shape[1]
    
    image_array_labels_copy=np.copy(image_array_labels)
    
    extra_pixels_value=2**16
    
    padded_image=cv2.copyMakeBorder(image_array_labels_copy,0,1,0,1,cv2.BORDER_CONSTANT,value=extra_pixels_value)
    
    
    #do the analysis from south to north, to avoid noisy regions that are typically on the top area of the microgrpahs, such as Pt
    
    for pixel_x in range(hor_pixs+1):
        for pixel_y in reversed(range(ver_pixs+1)):
            
            target_pixel=padded_image[pixel_y,pixel_x]
            
            if target_pixel==label_of_region:
                initial_pixel=(pixel_y+1,pixel_x)
                break
            
        if target_pixel==label_of_region:
            initial_pixel=(pixel_y+1,pixel_x)
            break   
        
        
    #!!! just make sure that when defining the material inside in the contour drawing function it is given by the pixel just 
    #above the given as initial one, as the iniital one is the one corresponding to the bottom of it and will have a different 
    #value, and of course, corresponds to a south interface
        
    return initial_pixel

#Draws the contour after computing the initial pixel given a label
def Contour_draw_computing_initial_pixel(image_array_labels, label_of_region):
    
    initial_pixel=Find_Initial_Pixel_for_contour(image_array_labels, label_of_region)
    contour_vectors_list, vertex_array=Contour_drawing_initial_pixel(image_array_labels, initial_pixel)
    
    return contour_vectors_list, vertex_array

   
#from all the labels in all the regions, computes the contour relative to each label, and therefore, its region (not 0)
def Contour_draw_All_regions(image_array_labels):
    '''
    The next step is to loop thought the different regions, each with a different label, and get as many contours arrays and 
    vertexs arrays as different regions or labels we have in the image (ideally, all the regions that have an important and
    characteristic feature that differentiates them from the rest, such as material, position, shape, and so on)


    Parameters
    ----------
    image_array_labels : array

    Returns
    -------
    dictionary containing the vertex and contours in each region labeled distinctly
    the vertexs and contours of each region can be extracted the following way:
    dictionary_name['label_vertexs'] or dictionary_name['label_contours']

    '''
   
    conts_vertx_per_region=dict()
    label_values=np.unique(image_array_labels)
    #label 0 is associated to pixels that do not have a label
    label_values=np.sort(label_values)[1:]
    
    for label in label_values:
        
        contour_vectors_list, vertex_array=Contour_draw_computing_initial_pixel(image_array_labels, label)
        conts_vertx_per_region[str(int(label))+'_'+'vertexs']=vertex_array
        conts_vertx_per_region[str(int(label))+'_'+'contours']=contour_vectors_list

  
    return  conts_vertx_per_region

#Make pixels considered as noise by the initial clustering be assigned to the cluster in which they should be,
#making the pixels inside a given border be assigned to this cluster
def Denoise_region(image_array,conts_vertx_per_region,region_label):
    '''
    

    Parameters
    ----------
    image_array : image with the labels obtained after the consecutive algorithm
    conts_vertx_per_region : dict, output from the function Contour_draw_All_regions to the previous image
    conregion_label : TYPE
        DESCRIPTION.

    Returns
    -------
    denoised_region : TYPE
        DESCRIPTION.

    '''
    
    def Check_vector_North(pixel_coords,contour_vectors):
        '''
        Check if given the pixel coordinates, if the FIRST vector on its north is following the contour direction,
        from right to left (only the first, as it may found a good one afterwards to misleading results)

        Parameters
        ----------
        pixel_coords : the coordinates of the evaluated pixel
        contour_vectors : list containing contour_vector objects
        limits_to_check: tuple, (min_top_vertex,max_bottom_vertex,min_left_vertex,max_right_vertex)


        Returns
        -------
        boolean True or False whether the pixel is or not inside the region

        '''
        
        possible_contours=[]
        for contour_vector in contour_vectors:
            if contour_vector.init_coords[0]==contour_vector.final_coords[0]:
                if contour_vector.final_coords[0]<=pixel_coords[0]:
                    if (contour_vector.init_coords[1]==pixel_coords[1] and contour_vector.final_coords[1]==pixel_coords[1]+1) or (contour_vector.init_coords[1]==pixel_coords[1]+1 and contour_vector.final_coords[1]==pixel_coords[1]):
                        possible_contours.append(contour_vector)
                    
        #now sort the possible contours from the pixel to the north and only check the first one just above it
        #sort depending on the value of y, putting first the bigger coordinate of y, which means 
        #that first it is the coordinate directly on top of the target pixel
        # sort in descending order, first the larger y
        #and then select only the first contour
        
        if len(possible_contours)==0:
            to_north=False
        else:
            possible_contours.sort(key=(lambda x: x.init_coords[0]), reverse=True)
        
            #check if the direction of the vector is from right to left
            if possible_contours[0].init_coords[1]==pixel_coords[1]+1 and possible_contours[0].final_coords[1]==pixel_coords[1]:
                to_north=True
            else:
                to_north=False
        
        return to_north
    
    
    def Check_vector_South(pixel_coords,contour_vectors):
        '''
        Check if given the pixel coordinates, if the FIRST vector on its south is following the contour direction,
        from left to right (only the first, as it may found a good one afterwards to misleading results)

        Parameters
        ----------
        pixel_coords : the coordinates of the evaluated pixel
        contour_vectors : list containing contour_vector objects


        Returns
        -------
        boolean True or False whether the pixel is or not inside the region

        '''
        
        possible_contours=[]
        for contour_vector in contour_vectors:
            if contour_vector.init_coords[0]==contour_vector.final_coords[0]:
                if contour_vector.final_coords[0]>pixel_coords[0]:
                    if (contour_vector.init_coords[1]==pixel_coords[1] and contour_vector.final_coords[1]==pixel_coords[1]+1) or (contour_vector.init_coords[1]==pixel_coords[1]+1 and contour_vector.final_coords[1]==pixel_coords[1]):
                        possible_contours.append(contour_vector)
                    
        #now sort the possible contours from the pixel to the south and only check the first one just below it
        #sort depending on the value of y, putting first the bigger coordinate of y, which means 
        #that first it is the coordinate directly below the target pixel
        # sort in ascending order, first the smaller y
        #and then select only the first contour
        
        if len(possible_contours)==0:
            to_south=False
        else:
            possible_contours.sort(key=(lambda x: x.init_coords[0]), reverse=False)
        
            #check if the direction of the vector is from left to right
            if possible_contours[0].init_coords[1]==pixel_coords[1] and possible_contours[0].final_coords[1]==pixel_coords[1]+1:
                to_south=True
            else:
                to_south=False
        
        return to_south
    
    
    def Check_vector_East(pixel_coords, contour_vectors):
        '''
        Check if given the pixel coordinates, if the FIRST vector on its east is following the contour direction,
        from bottom to top (only the first, as it may found a good one afterwards to misleading results)

        Parameters
        ----------
        pixel_coords : the coordinates of the evaluated pixel
        contour_vectors : list containing contour_vector objects


        Returns
        -------
        boolean True or False whether the pixel is or not inside the region

        '''
        
        possible_contours=[]
        for contour_vector in contour_vectors:
            if contour_vector.init_coords[1]==contour_vector.final_coords[1]:
                if contour_vector.final_coords[1]>pixel_coords[1]:
                    if (contour_vector.init_coords[0]==pixel_coords[0] and contour_vector.final_coords[0]==pixel_coords[0]+1) or (contour_vector.init_coords[0]==pixel_coords[0]+1 and contour_vector.final_coords[0]==pixel_coords[0]):
                        possible_contours.append(contour_vector)
                    
        #now sort the possible contours from the pixel to the east and only check the first one just at its right
        #sort depending on the value of x, putting first the bigger coordinate of x, which means 
        #that first it is the coordinate directly at right of the target pixel
        # sort in ascending order, first the smaller x
        #and then select only the first contour
        
        if len(possible_contours)==0:
            to_east=False
        else:
            possible_contours.sort(key=(lambda x: x.init_coords[1]), reverse=False)
        
            #check if the direction of the vector is from right to left
            if possible_contours[0].init_coords[0]==pixel_coords[0]+1 and possible_contours[0].final_coords[0]==pixel_coords[0]:
                to_east=True
            else:
                to_east=False
        
        return to_east
    
    
    def Check_vector_West(pixel_coords, contour_vectors):
        '''
        Check if given the pixel coordinates, if the FIRST vector on its west is following the contour direction,
        from top to bottom (only the first, as it may found a good one afterwards to misleading results)

        Parameters
        ----------
        pixel_coords : the coordinates of the evaluated pixel
        contour_vectors : list containing contour_vector objects


        Returns
        -------
        boolean True or False whether the pixel is or not inside the region

        '''
        
        possible_contours=[]
        for contour_vector in contour_vectors:
            if contour_vector.init_coords[1]==contour_vector.final_coords[1]:
                if contour_vector.final_coords[1]<=pixel_coords[1]:
                    if (contour_vector.init_coords[0]==pixel_coords[0] and contour_vector.final_coords[0]==pixel_coords[0]+1) or (contour_vector.init_coords[0]==pixel_coords[0]+1 and contour_vector.final_coords[0]==pixel_coords[0]):
                        possible_contours.append(contour_vector)
                    
        #now sort the possible contours from the pixel to the west and only check the first one just at its left
        #sort depending on the value of x, putting first the bigger coordinate of x, which means 
        #that first it is the coordinate directly at left of the target pixel
        # sort in descending order, first the larger x
        #and then select only the first contour
        
        if len(possible_contours)==0:
            to_west=False
        else:
            possible_contours.sort(key=(lambda x: x.init_coords[1]), reverse=True)
        
            #check if the direction of the vector is from right to left
            if possible_contours[0].init_coords[0]==pixel_coords[0] and possible_contours[0].final_coords[0]==pixel_coords[0]+1:
                to_west=True
            else:
                to_west=False
        
        return to_west
    
    label=region_label
    contour_vectors=conts_vertx_per_region[str(int(label))+'_contours']
    
    #limit the number of pixels to analise and loop through to reduce computing time
    #get maximum index for top bottom right and left and evaluate only within this range
    
    #these values can already be bigger than the original image, so the vertex indices can go from 0 to image size 
    #not image size -1, they are organised as [(y0,x1), (y1,x1),...,(yf,xf)]
    vertexs_for_label=conts_vertx_per_region[str(int(label))+'_vertexs']

    min_top_vertex=np.min(vertexs_for_label, axis=0)[0]
    max_bottom_vertex=np.max(vertexs_for_label, axis=0)[0]
    min_left_vertex=np.min(vertexs_for_label, axis=0)[1]
    max_right_vertex=np.max(vertexs_for_label, axis=0)[1]
    
    #these values define the range of values that can be contained inside the region
    #the max index (both bottom and right) does not need to be included in the interval, as these only 
    #define pixel borders and not real ones (the extra ones will never be inside the region)
    
    extra_pixels=[]
    
    for pixel_y in range(min_top_vertex,max_bottom_vertex):
        for pixel_x in range(min_left_vertex,max_right_vertex):
            
            if image_array[pixel_y,pixel_x] ==0:
                #only evaluate if the pixel is different from the label, as are the ones we want to change
                
                to_north=Check_vector_North((pixel_y,pixel_x),contour_vectors)
                to_south=Check_vector_South((pixel_y,pixel_x),contour_vectors)
                to_east=Check_vector_East((pixel_y,pixel_x),contour_vectors)
                to_west=Check_vector_West((pixel_y,pixel_x),contour_vectors)
                
                if to_north==True and to_south==True and to_east==True and to_west==True:
                    extra_pixels.append((pixel_y,pixel_x))

    return extra_pixels

#Denoise all the regions, looping through the labels that represent actual regions (not 0)
def Denoise_All_Regions(image_array,conts_vertx_per_region):
    '''
    Apply the denoising for all the regions and all the labels that have an associated region

    Parameters
    ----------
    image_array : TYPE
        DESCRIPTION.
    conts_vertx_per_region : TYPE
        DESCRIPTION.

    Returns
    -------
    dict_extra_pixels : TYPE
        DESCRIPTION.

    '''

    label_values=np.unique(image_array)
    #label 0 is associated to pixels that do not have a label
    label_values=np.sort(label_values)[1:]

    
    dict_extra_pixels=dict()
    for label in label_values:

        dict_extra_pixels[str(int(label))+'_extra_pixels']=Denoise_region(image_array,conts_vertx_per_region,label)
    
    return dict_extra_pixels

def Gaussian_pre_clustering(image_st,number_of_gaussians,variance_threshold):
    '''
    finds the bayesian GMM, and selects these pixels that belong to a sharp enough gaussian curve, as these features are
    difficult to be captured by K-means. Then, it redraws the original image by keeping all the original pixels except
    these that belong to these curves, which are assigned a label from 2 to 2+n where n is the thresholded gaussians -1
    
    Parameters
    ----------
    image : TYPE
        DESCRIPTION.
    number_of_gaussians : number of gaussians that fill the histogram
        DESCRIPTION.
    variance_threshold: defines these curves that will be considered sharp enough. It is the variance below which 
                    the curve is kept. The variances of the curves can be tracked by the covariances output of BGMM

    Returns
    -------
    None.

    '''   
    def BGMM(image_,number_clusters):
    
        weight_concentration_prior=1/number_clusters
        
        image_reshaped=np.reshape(image_, (image_.shape[0]*image_.shape[1],1))
        gmm = sklearn.mixture.BayesianGaussianMixture(n_components=number_clusters, n_init=5)
        gmm.fit(image_reshaped)
        
        #predictions from gmm
        labels = gmm.predict(image_reshaped)
        
        labels.shape=image_.shape
        means=gmm.means_
        covariances=gmm.covariances_
        
        return labels, means, covariances
    
    labels, means, covs=BGMM(image_st,number_clusters=number_of_gaussians)

    image_to_refine=np.copy(image_st)

    # ensure the labels have values that range from 2 to any number, to avoid comparing values with 0 and 1
    copy_labels_gauss=np.copy(labels)+2
    
    label_final_val=2
    for index,cov in enumerate(covs):
        if cov < variance_threshold:    
            print('yes')
            target_label=index+2
            image_to_refine=label_final_val*(copy_labels_gauss==target_label)+image_to_refine*(copy_labels_gauss!=target_label)
            
            label_final_val=label_final_val+1
    
    return image_to_refine





def main():
    return print('Start program')

if __name__ == "__main__":
    main()


def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)


#Hyperparameters
downscaling_factor=1 #for trials, n factor of downsampling size of image
FFT_thresholding=0.5  #value above which the pixels are kept
st_distance=30 #distance parameter in the Stem tool method
FFT_thresholdingG=0.6 #value above which the pixels are kept, in the gaussian filtered FFT
window_size=2048  #window size of the sliding windows
tol=0.05 #tolerance: how different from theoretical values the previous values can be to get good output
min_d=0.5    #minimum interplanar distance computed in the diffraction
forbidden = True  #Include (True) or not (False) the forbidden reflections
segmentation_clusters=3 #hyperparameter for the quick segmentation to do the GPA (should be provisional)
np.random.seed(int(time.time()))


root = tkinter.Tk()
root.attributes("-topmost", True)
root.withdraw()
images_directory = filedialog.askdirectory(parent=root,initialdir="/",title='Apply GPA to every image in the selected folder')

for image_file in os.listdir(images_directory):
    # Image directory
    
    image_path=images_directory+'/'+str(image_file) 
    
    #Check if dm3 image and only do the process if it is dm3 
    if image_path[::-1][0:4][::-1]!='.dm3':
        # if the extension does not match dm3 then skip this iteration
        continue
    
    # Load image
    image_hs_signal, image_array, im_calibration, total_pixels_image, im_FOV, im_units=Load_dm3(image_path)
    
    #Input the number of clusters per each image
    #segmentation_clusters=int(input('Number of intensity clusters of image '+str(image_file)+':'))

    # Calibrate the FFT, which does not change the calibration with cropping
    FFT_calibration_number,FFT_units=FFT_calibration(image_hs_signal)
    
    # Crop the image if wanted
    image_hs_signal, image_array, total_pixels_image, im_FOV=Crop_Image_Randomly(image_hs_signal, window_size)
    
    # Recalibrate image and FFT in case it is necessary
    real_calibration_factor=1
    im_calibration,im_FOV=Image_Calibration_Correction(im_calibration,im_FOV, real_calibration_factor)
    FFT_calibration_number,FFT_units=FFT_calibration_Correction(image_hs_signal, real_calibration_factor)
    
    
    # Downscale the image if wanted
    downscaling_factor=1
    image_array, im_calibration, total_pixels_image=Downscale_Image_Factor(image_array, downscaling_factor, im_calibration, total_pixels_image)
    
    # print('Image calibration',  im_calibration, im_units)
    # print('FFT calib', FFT_calibration_number, FFT_units)
    
    # Standarise the image
    image_array=Standarise_Image(image_array)
    
    # Denoise the image if wanted, might help to denoise byt tpyically the peak finding works worse
    #image_array=FiltersNoise.Adaptative_Gaussian_Blur(image_array, im_calibration)
    
    # Wiener filter instead
    # image_array=scisignal.wiener(image_array, 3)
    
    # Standarise the image
    image_array=Standarise_Image(image_array)
    
    # plt.imshow(image_array, cmap=plt.cm.gray, vmin=image_array.min(), vmax=image_array.max())
    # plt.show()
    
    # Compute the FFT 
    FFT_image_array, FFT_image_complex=Compute_FFT_ImageArray(image_array)
    
    # Filter the FFT in case we see it is convenient
    
    FFT_image_array=FFT_Gaussian_Convolution_Filter(FFT_image_array)
    
    # Standarise the FFT
    FFT_image_array=Standarise_Image(FFT_image_array)
    # plt.imshow(FFT_image_array, cmap=plt.cm.gray, vmin=FFT_image_array.min(), vmax=FFT_image_array.max())
    # plt.show()
    
    #1st Approx peak finding hyperparameter finding 
    st_distance,_,FFT_perc=FFT_hyperparams(FFT_image_array,im_FOV)
    FFT_thresholdingG=Threshold_given_percentage(FFT_image_array, FFT_perc)
    
    #1st Approx peak finding hyperparameter fitting
    pixels_of_peaks=stemtool.afit.peaks_vis(FFT_image_array, dist=st_distance, thresh=FFT_thresholdingG, imsize=(15, 15))
    plt.show(block=False)
    
    
    # Extract distances, angles, and pixel positions
    d_distances=Spot_coord_To_d_spacing_vect(pixels_of_peaks, FFT_calibration_number, total_pixels_image)
    angles_to_x=Spot_coord_To_Angles_to_X_vect(pixels_of_peaks,total_pixels_image)
    refined_distances, refined_angles_to_x, refined_pixels=Ensure_Center_Diff(d_distances, angles_to_x, pixels_of_peaks)
    
    # Set the values of distances in angstroms as required by Prepare_exp_distances_angles_pixels and other funcs
    refined_distances=nm_to_Angstroms(refined_distances)
    
    # Refine distances, angles, and pixel positions
    refined_distances, refined_angles_to_x, refined_pixels=Prepare_exp_distances_angles_pixels(refined_distances, refined_angles_to_x,refined_pixels, min_d)
    
    #Compute th e g vectors
    g_vectors_pixels, angles_to_x_of_g=Choose_g_vectors(
        refined_distances, refined_angles_to_x, refined_pixels,g2='perpendicular')
    
    # Quick segmentation for finding the substrate as the reference region for gPA
    substrate_segment=Quick_Segmentation_for_Reference(image_array, segmentation_clusters)
    
    # Find contours for the interface angle calculation afterwards
    substrate_segment_contour=Contour_draw_All_regions(substrate_segment)
    
    # Find the reference region
    matrix_of_region, coordinates=Segmented_Region_Conversion_into_Matrix(substrate_segment, 1)    
    matrix_of_squares= Find_Biggest_Squares_per_Segmented_Region(matrix_of_region)
    coords_of_square=Find_Biggest_Square_Closest_to_Region_Center(matrix_of_region, matrix_of_squares)
    scaled_reference_coords=Locate_Reference_in_Original_Image(coordinates, coords_of_square)
    
    #Reduce the coordiantes in case they coincide with the border of the images
    scaled_reference_coords=Reduce_Reference_if_Border(image_array, scaled_reference_coords, reduction=0.1)
    
    # Plot the image with the square reference if wanted
    Plot_Image_with_GPA_Reference(image_array, scaled_reference_coords)
    
    # Compute the mask size
    mask_size, GPA_eff_resolution=Define_GPA_Mask_Size(image_array, FFT_calibration_number)
    
    # COmpute the correction angle between mathematical x axis and image horizontal planes
    rotation_angle=Compute_GPA_Rotation_X_Axis(refined_angles_to_x, substrate_segment,substrate_segment_contour)
    
    # Set up the rest of the GPA params
    
    spot1=g_vectors_pixels[0]
    spot2=g_vectors_pixels[1]
    Sp1X, Sp1Y=spot1[1],spot1[0]
    Sp2X, Sp2Y=spot2[1],spot2[0]
    
    # Compute GPA and store the images
    exx, eyy, exy, eyx, rot, shear=GPA_full(image_array, im_calibration,spot1, spot2, mask_size, scaled_reference_coords, rotation_angle,display=True)
    
    save_folder_directory=image_path.replace('.dm3','')
    createFolder(save_folder_directory)
    
    for result_image_name, result_image_array in zip(['exx','eyy','exy','eyx','rot','shear'],[exx,eyy,exy,eyx,rot,shear]):
        save_image_directory=save_folder_directory+'/'+result_image_name+'.tiff'        
        im = Image.fromarray(result_image_array, mode='F') # float32
        im.save(save_image_directory, "TIFF")
        