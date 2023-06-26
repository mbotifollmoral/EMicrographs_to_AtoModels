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

import tkinter
from tkinter import filedialog

# custom imported files
import sys
import GPA_automated_Wrapper as autoGPA

sys.path.append(r'E:\Arxius varis\PhD\3rd_year\Code\Functions')

import Segmentation_1stAprox as Segment
import PeakFinding_1stAprox as PeakFind
import GPA_specific as GPA_sp
import Filters_Noise as FiltersNoise
import Phase_Identificator as PhaseIdent
import ImageCalibTransf as ImCalTrans




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
segmentation_clusters=4 #hyperparameter for the quick segmentation to do the GPA (should be provisional)
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
    image_hs_signal, image_array, im_calibration, total_pixels_image, im_FOV, im_units=ImCalTrans.Load_dm3(image_path)
    
    #Input the number of clusters per each image
    #segmentation_clusters=int(input('Number of intensity clusters of image '+str(image_file)+':'))

    # Calibrate the FFT, which does not change the calibration with cropping
    FFT_calibration,FFT_units=ImCalTrans.FFT_calibration(image_hs_signal)
    
    # Crop the image if wanted
    image_hs_signal, image_array, total_pixels_image, im_FOV=ImCalTrans.Crop_Image_Randomly(image_hs_signal, window_size)
    
    # Recalibrate image and FFT in case it is necessary
    real_calibration_factor=1
    im_calibration,im_FOV=ImCalTrans.Image_Calibration_Correction(im_calibration,im_FOV, real_calibration_factor)
    FFT_calibration,FFT_units=ImCalTrans.FFT_calibration_Correction(image_hs_signal, real_calibration_factor)
    
    
    # Downscale the image if wanted
    downscaling_factor=1
    image_array, im_calibration, total_pixels_image=ImCalTrans.Downscale_Image_Factor(image_array, downscaling_factor, im_calibration, total_pixels_image)
    
    # print('Image calibration',  im_calibration, im_units)
    # print('FFT calib', FFT_calibration, FFT_units)
    
    # Standarise the image
    image_array=ImCalTrans.Standarise_Image(image_array)
    
    # Denoise the image if wanted, might help to denoise byt tpyically the peak finding works worse
    #image_array=FiltersNoise.Adaptative_Gaussian_Blur(image_array, im_calibration)
    
    # Wiener filter instead
    # image_array=scisignal.wiener(image_array, 3)
    
    # Standarise the image
    image_array=ImCalTrans.Standarise_Image(image_array)
    
    # plt.imshow(image_array, cmap=plt.cm.gray, vmin=image_array.min(), vmax=image_array.max())
    # plt.show()
    
    # Compute the FFT 
    FFT_image_array, FFT_image_complex=ImCalTrans.Compute_FFT_ImageArray(image_array)
    
    # Filter the FFT in case we see it is convenient
    
    FFT_image_array=ImCalTrans.FFT_Gaussian_Convolution_Filter(FFT_image_array)
    
    # Standarise the FFT
    FFT_image_array=ImCalTrans.Standarise_Image(FFT_image_array)
    # plt.imshow(FFT_image_array, cmap=plt.cm.gray, vmin=FFT_image_array.min(), vmax=FFT_image_array.max())
    # plt.show()
    
    #1st Approx peak finding hyperparameter finding 
    st_distance,_,FFT_perc=PeakFind.FFT_hyperparams(FFT_image_array,im_FOV)
    FFT_thresholdingG=PeakFind.Threshold_given_percentage(FFT_image_array, FFT_perc)
    
    #1st Approx peak finding hyperparameter fitting
    pixels_of_peaks=stemtool.afit.peaks_vis(FFT_image_array, dist=st_distance, thresh=FFT_thresholdingG, imsize=(15, 15))
    plt.show()
    
    
    # Extract distances, angles, and pixel positions
    d_distances=PhaseIdent.Spot_coord_To_d_spacing_vect(pixels_of_peaks, FFT_calibration, total_pixels_image)
    angles_to_x=PhaseIdent.Spot_coord_To_Angles_to_X_vect(pixels_of_peaks,total_pixels_image)
    refined_distances, refined_angles_to_x, refined_pixels=PhaseIdent.Ensure_Center_Diff(d_distances, angles_to_x, pixels_of_peaks)
    
    # Set the values of distances in angstroms as required by Prepare_exp_distances_angles_pixels and other funcs
    refined_distances=ImCalTrans.nm_to_Angstroms(refined_distances)
    
    # Refine distances, angles, and pixel positions
    refined_distances, refined_angles_to_x, refined_pixels=PhaseIdent.Prepare_exp_distances_angles_pixels(refined_distances, refined_angles_to_x,refined_pixels, min_d)
    
    #Compute th e g vectors
    
    g_vectors_pixels, angles_to_x_of_g=GPA_sp.Choose_g_vectors(refined_distances, refined_angles_to_x, refined_pixels,g2='closest')
    
    # Quick segmentation for finding the substrate as the reference region for gPA
    substrate_segment=GPA_sp.Quick_Segmentation_for_Reference(image_array, segmentation_clusters)
    
    # Find contours for the interface angle calculation afterwards
    substrate_segment_contour=Segment.Contour_draw_All_regions(substrate_segment)
    
    # Find the reference region
    matrix_of_region, coordinates=GPA_sp.Segmented_Region_Conversion_into_Matrix(substrate_segment, 1)    
    matrix_of_squares= GPA_sp.Find_Biggest_Squares_per_Segmented_Region(matrix_of_region)
    coords_of_square=GPA_sp.Find_Biggest_Square_Closest_to_Region_Center(matrix_of_region, matrix_of_squares)
    scaled_reference_coords=GPA_sp.Locate_Reference_in_Original_Image(coordinates, coords_of_square)
    
    #Reduce the coordiantes in case they coincide with the border of the images
    scaled_reference_coords=GPA_sp.Reduce_Reference_if_Border(image_array, scaled_reference_coords, reduction=0.1)
    
    # Plot the image with the square reference if wanted
    ImCalTrans.Plot_Image_with_GPA_Reference(image_array, scaled_reference_coords)
    
    # Compute the mask size
    mask_size, GPA_eff_resolution=GPA_sp.Define_GPA_Mask_Size(image_array, FFT_calibration)
    
    # COmpute the correction angle between mathematical x axis and image horizontal planes
    rotation_angle=GPA_sp.Compute_GPA_Rotation_X_Axis(refined_angles_to_x, substrate_segment,substrate_segment_contour)
    
    # Set up the rest of the GPA params
    
    spot1=g_vectors_pixels[0]
    spot2=g_vectors_pixels[1]
    Sp1X, Sp1Y=spot1[1],spot1[0]
    Sp2X, Sp2Y=spot2[1],spot2[0]
    
    # Compute GPA and store the images
    exx, eyy, exy, eyx, rot, shear=autoGPA.GPA_full(image_array, im_calibration,spot1, spot2, mask_size, scaled_reference_coords, rotation_angle,display=True)
    
    save_folder_directory=image_path.replace('.dm3','')
    createFolder(save_folder_directory)
    
    for result_image_name, result_image_array in zip(['exx','eyy','exy','eyx','rot','shear'],[exx,eyy,exy,eyx,rot,shear]):
        save_image_directory=save_folder_directory+'/'+result_image_name+'.tiff'        
        im = Image.fromarray(result_image_array, mode='F') # float32
        im.save(save_image_directory, "TIFF")
        