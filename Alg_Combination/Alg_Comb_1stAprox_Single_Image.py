# -*- coding: utf-8 -*-
"""
Created on Wed Jan 11 11:37:05 2023

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
from pathlib import Path

sys.path.append(r'E:\Arxius varis\PhD\3rd_year\Code\Functions')

# General functions
import HighToLowTM 
import Segmentation_1stAprox as Segment
import Filters_Noise as FiltersNoise
import Phase_Identificator as PhaseIdent
import ImageCalibTransf as ImCalTrans
import PeakFinding_1stAprox as PeakFind
import GPA_specific as GPA_sp

# Peak finding functions
sys.path.append(r'E:\Arxius varis\PhD\4rth_year\Code\Functions\Peak_detector_Final')
sys.path.append(r'E:\Arxius varis\PhD\4rth_year\Code\Functions\Ivans_Files_2\IVAN\Segmentation_model')
sys.path.append(r'E:\Arxius varis\PhD\4rth_year\Code\Functions\General_functions')
sys.path.append(r'E:\Arxius varis\PhD\4rth_year\Code\Atomistic_model_builder')

import Atomistic_Model_Builder as AtomBuild
import PF_FFT_processing as FFT_Procs
import PF_Peaks_detector as Peaks_detector
import PF_ImageTreatment_and_interfaces as PF_II
import SG_Segmentation_algorithms as SegmAlgs
import Segmentation_Wrapped as SegmWrap
import FFT_indexer


#%%

# High to Low functions of the combined algorithm 1st approx

def TooLow_Magnification_Cutoff(images_in_dataset_by_pixel_size,pixel_sizes, tooLow_FOV):
    '''
    Cutoff these images that are too low magnificaiton, for instance, displaying the whole lamella or even 
    a larger area, as they do not provide any meaningful information and just make the process harder 
    Just delete them from the main arrays and return them with the same format
    
    Returns
    -------
    None.

    '''
    
    for list_im in images_in_dataset_by_pixel_size:
        for image_in_dataset in list_im:
            if image_in_dataset.FOV > tooLow_FOV: # in nm
                list_im.remove(image_in_dataset)
                
    images_in_dataset_by_pixel_size_cut=images_in_dataset_by_pixel_size.copy()
    pixel_sizes_cut=np.copy(pixel_sizes)
       
    for index, (list_im, pixel_size ) in enumerate(zip(images_in_dataset_by_pixel_size_cut, pixel_sizes_cut)):
        if len(list_im)==0:
            images_in_dataset_by_pixel_size.remove(list_im)
            pixel_sizes[index]=0
    pixel_sizes=pixel_sizes[pixel_sizes!=0]
             
    return images_in_dataset_by_pixel_size,pixel_sizes

def Images_Position_PixelSizes_Arrays(images_in_dataset_by_pixel_size,relative_positions,pixel_sizes):
    '''
    Transition function to make the lists arrays of same size the three of them, images, relative positions, 
    and pixel sizes, to correlate each position with the image, relative position and pixel size
    images_in_dataset_by_pixel_size which is a list of lists with images of same pixel size is kept as a list
    with the same name (i.e. not modifyed)    

    Returns
    -------
    flat_images_in_dataset_by_pixel_size : TYPE
        DESCRIPTION.
    relative_positions : TYPE
        DESCRIPTION.
    flat_pixel_sizes : TYPE
        DESCRIPTION.

    '''
    # Get the images that pair in the position of the relative positions
    #Now the relative_positions are linked with its corresponding image (image_in_dataset object) in flat_images_in_dataset_by_pixel_size
    flat_images_in_dataset_by_pixel_size=[item for sublist in images_in_dataset_by_pixel_size for item in sublist]
 
    flat_pixel_sizes=[[pixesize]*len(images_pixsize) for pixesize,images_pixsize in zip(pixel_sizes,images_in_dataset_by_pixel_size)]
    flat_pixel_sizes=[item for sublist in flat_pixel_sizes for item in sublist]
    flat_pixel_sizes=np.asarray(flat_pixel_sizes)
    
    relative_positions=np.asarray(relative_positions)
    flat_images_in_dataset_by_pixel_size=np.asarray(flat_images_in_dataset_by_pixel_size)

    return flat_images_in_dataset_by_pixel_size, relative_positions, flat_pixel_sizes
    


#%%
'''
Functions for communicating the segmented regions as querys with the atomic resoltion images as
templates and find atomically resolved crops within the low mag segmented images
'''
  

def Find_Segmented_Query(
        image_in_dataset, list_of_best_query_template_pairs, images_to_segment):
    '''
    For a given image, image_in_dataset, select the query image that has been segmented in 
    the process before, from which we can correlate the segmented regions with the magnified version of it

    Parameters
    ----------
    image_in_dataset : TYPE
        DESCRIPTION.
    list_of_best_query_template_pairs : TYPE
        DESCRIPTION.
    images_to_segment

    Returns
    -------
    query_image : TYPE
        DESCRIPTION.

    '''
    for query_template_pair in list_of_best_query_template_pairs:
        # the problem image, image_in_dataset, is the template and find the related queries
        if query_template_pair.template == image_in_dataset:
            # define the direct query of the image problem (template)
            query_image = query_template_pair.query
            # check if the query image of the template of our interest has been segmented
            if image_in_dataset in images_to_segment:
                return image_in_dataset
            if query_image in images_to_segment:
                return query_image
            # if nothing else, repreat the process with a new template being the previous query
            else:
                return Find_Segmented_Query(query_image, list_of_best_query_template_pairs, images_to_segment)
            

def Relative_PixelCoords_from_Query_Template_pair(
        query_total_pixels, query_relative_coords, template_relative_coords):
    '''
    Get the coordinates, in pixels positions, of where the template is located within a query
    as this will be the pixels that are going to be cropped
    output them as (x_start, y_start, x_end, y_end)
    Inputs are the coordiantes in the nm scale

    Parameters
    ----------
    query_relative_coords : 
    template_relative_coords :

    Returns
    -------
    pixel_q_t_rel_coords. (pix_x_start, pix_y_start, pix_x_end, pix_y_end) pixels within the query image

    '''
    (q_x_start, q_y_start, q_x_end, q_y_end) = query_relative_coords
    (t_x_start, t_y_start, t_x_end, t_y_end) = template_relative_coords
    
    rel_x_start = (t_x_start - q_x_start)/(q_x_end - q_x_start)
    rel_y_start = (t_y_start - q_y_start)/(q_y_end - q_y_start)
    
    pix_x_start = int(round(rel_x_start*query_total_pixels))
    pix_y_start = int(round(rel_y_start*query_total_pixels))
    
    rel_x_end = rel_x_start + (t_x_end - t_x_start)/(q_x_end - q_x_start)
    rel_y_end = rel_y_start + (t_y_end - t_y_start)/(q_y_end - q_y_start)
    
    pix_x_end = int(round(rel_x_end*query_total_pixels))
    pix_y_end = int(round(rel_y_end*query_total_pixels))
    
    # ensure both sides are the same length, just in case
    x_side = pix_x_end - pix_x_start
    y_side = pix_y_end - pix_y_start   
    
    if x_side == y_side:
        pixel_q_t_rel_coords = (pix_x_start, pix_y_start, pix_x_end, pix_y_end)
        return pixel_q_t_rel_coords
    # If the segments are not the same, then adjust the segment to the smallest one
    else:
        if x_side > y_side:
            pix_x_end = pix_x_start + y_side
        else:
            # x_side > y_side
            pix_y_end = pix_y_start + x_side
    pixel_q_t_rel_coords = (pix_x_start, pix_y_start, pix_x_end, pix_y_end)
    return pixel_q_t_rel_coords
 
  
def Correlate_Segmented_Atomic_q_t_Images(
        image_in_dataset, query_image, flat_images_in_dataset_by_pixel_size, 
        relative_positions, images_to_segment, images_segmented):
    '''
    Function to correlate the atomic resolution image coming from image_in_datset with the corresponding
    segmented image query_image, and crop the region within the segmented query that corresponds to the
    area of the atomic reoslution one, so we have the corresponding segmented regions within the atomic 
    resolution image
    We upscale that crop to fit the pixels of the atomic reoslution one so then the coordinates of the
    found regions match 1 to 1 between the upscaled crop and the atomic res image

    Parameters
    ----------
    image_in_dataset : 
    query_image : 
    flat_images_in_dataset_by_pixel_size : 
    relative_positions : 
    images_to_segment : 
    images_segmented :
        
    Returns
    -------
    crop_image_array_ups : 
    image_relative_coords : we output the rel coordinates of the image in dataset (tempalte, atomic one)
                        as will be useful later
    '''
    
    # Get the relative coordinates for both the image and segmented query
    image_relative_coords = relative_positions[flat_images_in_dataset_by_pixel_size == image_in_dataset][0]
    query_relative_coords = relative_positions[flat_images_in_dataset_by_pixel_size == query_image][0]
    
    # Get the segmented query
    # query_segmented = images_segmented[np.where(images_to_segment == query_image)[0][0]]
    
    # Quick fix for np.where, which seems to be bugged in some occassions (the function 
    # does not recognise an element even if its 100% there, happens only whith few images datasets)
    for index_seg, image_to_segment in enumerate(images_to_segment):
        if image_to_segment == query_image:
            index_segm_array = index_seg

    query_segmented = images_segmented[index_segm_array]
    plt.imshow(query_segmented)
    plt.show()
    # Get the pixel coordinates of the query in which to locate the image (template)
    pixel_q_t_rel_coords = Relative_PixelCoords_from_Query_Template_pair(
        query_image.total_pixels, query_relative_coords, image_relative_coords)
    
    # Crop the segmented image according to the position defined by the template (get segmented region of template)
    crop_image_hs_signal, crop_image_array, total_pixels_image, im_FOV = ImCalTrans.Crop_Image_PixelCoords(
        query_segmented, pixel_q_t_rel_coords, query_image.x_calibration)
    
    # After croping, it is interesting, for the sake of comparing the images properly,to upscale the crop to 
    # the size of the template that is going to be compared to
    # Important to make sure that the upsaled image keeps having integers only as the pixels intensity
    target_size=(image_in_dataset.total_pixels, image_in_dataset.total_pixels)
    crop_image_array_ups = np.int16(cv2.resize(crop_image_array, target_size, interpolation = cv2.INTER_NEAREST))
    # !!! Here the effective pixel size of this image has changed, altough I do not think we need to use it 
    # directly from this image, anyway, it is the same as the template one!

    return crop_image_array_ups, image_relative_coords

   
def Get_Atomic_Crop_from_Segmented_Crop(
        crop_image_array_ups, label, image_in_dataset):
    '''
    Get the segmented upsampled image of the crop of the atomically resolved
    image and get a crop of the region corresponding to label

    Parameters
    ----------
    crop_image_array_ups : 
    label : 

    Returns
    -------
    image_crop_hs_signal : 
    image_crop_array : 
    total_pixels_crop : 
    crop_FOV : 
    scaled_reference_coords : 

    '''
    
    # Find the reference region
    matrix_of_region, coords_of_full_label = GPA_sp.Segmented_Region_Conversion_into_Matrix(
        crop_image_array_ups, label)    
    matrix_of_squares = GPA_sp.Find_Biggest_Squares_per_Segmented_Region(
        matrix_of_region)
    coords_of_square = GPA_sp.Find_Biggest_Square_Closest_to_Region_Center(
        matrix_of_region, matrix_of_squares)
    scaled_reference_coords=GPA_sp.Locate_Reference_in_Original_Image(
        coords_of_full_label, coords_of_square)
    scaled_reference_coords = GPA_sp.Reduce_Reference_if_Border(
        crop_image_array_ups, scaled_reference_coords, reduction=0.1)
    # to plot where the found region is located
    # ImCalTrans.Plot_Image_with_GPA_Reference(
    #     crop_image_array_ups, scaled_reference_coords)
    ImCalTrans.Plot_Image_with_GPA_Reference(
        image_in_dataset.image_arraynp_st, scaled_reference_coords)
    
    #Crop the image in dataset given the found coordinates of the square
    [first_row,last_row,first_col,last_col] = scaled_reference_coords
    image_crop_to_analysis = image_in_dataset.image_arraynp_st[first_row:last_row, first_col:last_col]
    
    # Normalise the crop as seems to slightly help with the peak identification
    image_crop_to_analysis=(image_crop_to_analysis-np.min(image_crop_to_analysis))/np.max(image_crop_to_analysis-np.min(image_crop_to_analysis))

    # Build the hs signal out of the crop to gather the main info on pixel size and FOV and image size
    image_crop_hs_signal, image_crop_array, total_pixels_crop, crop_FOV = ImCalTrans.Build_hs_2DSignal(
        image_crop_to_analysis, image_in_dataset.x_calibration)    
    
    return image_crop_hs_signal, image_crop_array, total_pixels_crop, crop_FOV, scaled_reference_coords



#%%
'''
Hyperparameter definition
'''
# High to Low hyperparams
real_calibration_factor=1  # change calibration of the image
N_ps_higher=3   # for the Find_best_Query_Template_pairs_Restrictive, number of pixel size differences between checked query-image pairs
# No need to use N_ps_higher if Find_best_Query_Template_pairs algorithm is used instead  (all pairs considered) 

# Segmentation hyperparams
# 0.06 nm/pixel size is the size below which good atomic resolution can be obtained, obove this also
# for pixel sizes like 0.074 and 0.11 also atomic resolution, but also good to be segmented (good for GPA)
pixel_size_segment_thresh = 0.06  # in nm/pixel
tooLow_FOV = 1500 #nm, is a FOV which is too


# Phase identification hypersparams
tol=0.05 #tolerance: how different from theoretical values the previous values can be to get good output
min_d=0.5    #minimum interplanar distance computed in the diffraction
forbidden = True  #Include (True) or not (False) the forbidden reflections


#%%
'''
Template matching and generation of the relative coordinates per image over the lowest mag one
'''


# Emerging windows to choose the path to folder with the images
# root = tkinter.Tk()
# root.attributes("-topmost", True)
# root.withdraw()
# dataset_system_path_name = filedialog.askdirectory(parent=root,initialdir="/",title='Folder with the images dataset')

dataset_system_path_name=r'E:\Arxius varis\PhD\2nd_year\Code\trial_images\full_device_STEM_datasets\QT543AlNb\\'
dataset_system_path_name=r'E:\Arxius varis\PhD\2nd_year\Code\trial_images\full_device_STEM_datasets\Qdev439_InAs120nm_Al full 25nm\NW1\\'
dataset_system_path_name=r'E:\Arxius varis\PhD\2nd_year\Code\trial_images\full_device_STEM_datasets\SQ20-250-2\\'

dataset_system_path_name = r'E:\Arxius varis\PhD\2nd_year\Code\trial_images\single_image_analysis\\'

# Browse the images in the folder and also calibrate
# !!! CALIBRATION CORRECTION DONE HERE --> NO NEED TO CHANGE THE CALIBRATION OF THE IMAGES AT ANY POINT
# !!! UNLESS IMAGES ARE BINNED




images_in_dataset_list, pixel_sizes = HighToLowTM.Browse_Dataset_Images_and_Recalibrate(
    dataset_system_path_name, real_calibration_factor)

# Sort the dataset with a list of lists in which each list has images with the same pixel size
images_in_dataset_by_pixel_size = HighToLowTM.Sort_Dataset_by_PixelSize(
    images_in_dataset_list, pixel_sizes)

# Delete images which are too low magnificaiton (lamella appearence)
images_in_dataset_by_pixel_size, pixel_sizes = TooLow_Magnification_Cutoff(
    images_in_dataset_by_pixel_size, pixel_sizes, tooLow_FOV)

# Ensure to pick only one lowest magnification image to start the template matching at
# Better to ensure there is only one when prepearing the dataset (manually just keep the lowest mag image of interest)
images_in_dataset_by_pixel_size = HighToLowTM.Make_Sure_1_Lowest_Mag_Image(
    images_in_dataset_by_pixel_size)

# Find the best query-template pairs, restrictive typically works better use the N_ps_higher hyperparameter
list_of_best_query_template_pairs = HighToLowTM.Find_best_Query_Template_pairs_Restrictive(
    images_in_dataset_by_pixel_size, N_ps_higher)
#list_of_best_query_template_pairs=HighToLowTM.Find_best_Query_Template_pairs(images_in_dataset_by_pixel_size)

# Get the relative positions for each image with respect the lowest mag one
relative_positions = HighToLowTM.Coordinate_system_Calculation(
    images_in_dataset_by_pixel_size, list_of_best_query_template_pairs)

# Print the images collage if wanted
# HighToLowTM.Plot_Images_Stack_Scaling(images_in_dataset_by_pixel_size,relative_positions)

# Convert the lists into flat arrays to correlate 1 to 1 images, relative positions and pixel sizes
flat_images_in_dataset_by_pixel_size, relative_positions, flat_pixel_sizes = Images_Position_PixelSizes_Arrays(
    images_in_dataset_by_pixel_size, relative_positions, pixel_sizes)


#%%
'''
Segment the images which do not have atomic resolution, by a certain calibration or FOV threshold
'''

# Select only the images to be segmented, low mag ones, to define the image regions
images_to_segment, relative_positions_to_segment, pixel_sizes_to_segment = SegmWrap.Images_to_Segment_Cutoff(
    pixel_size_segment_thresh, flat_images_in_dataset_by_pixel_size, relative_positions, flat_pixel_sizes)

# Perform the segmentation to the images that have been defined to be segmented (not too low mag but low
# enough to almost do not have atomic resolution)
# images_segmented, conts_vertxs_per_region_segmented = SegmWrap.Segment_Images(
#     images_to_segment, relative_positions_to_segment, pixel_sizes_to_segment)

images_segmented, conts_vertxs_per_region_segmented = SegmWrap.Segment_Images_ContourBased(
    images_to_segment, relative_positions_to_segment, pixel_sizes_to_segment)


#%%


'''
Functions for finding out the high mag image within the segmented map 
'''

# Initialise the possible unit cells before going into the image per image analysis as the step is common
# Get the possible unit cells into a Crystal object to hold a DP simulation
unit_cells_path = r'E:\Arxius varis\PhD\3rd_year\Code\unit_cells'


# Initialise the crystal objects for every possible unit cell in the path of possible cells
# crystal_objects_list, space_group_list = PhaseIdent.Init_Unit_Cells(unit_cells_path)
crystal_objects_list, space_group_list = PhaseIdent.Init_Unit_Cells_AnyFormat(unit_cells_path)

# Limit the cells for the trials

# crystal_objects_list = crystal_objects_list[0:3]
# space_group_list = space_group_list[0:3]

# list to store all the Analysed_Image objects with all the images, crops and ZA found
list_analysed_images = []

# From the list of images_in_dataset objects, relative positions and pixel sizes, ordered 1 to 1
# we can proceed with the atomic identification    



# set pixel size below which images are considered: for TITAN and 2k images
# setting it to a bit smaller ( < 0.075 or below may help as the preivous is on the edge of atomic res)
# for single image analysis we want them all to be analysed, so force a high one

max_pixel_size_atom_res = 10  # !!! HYPERPARAMETER

for image_in_dataset in flat_images_in_dataset_by_pixel_size:
    
    if image_in_dataset.x_calibration <= max_pixel_size_atom_res:
        
        # BEWARE if all the images within flat images are sent to analysis, the first image, lowest mag,
        # WILL NEVER be a template and the algorithm will fail
        # this is assessed by the following exception

        # indent the full process here

        # Initialyse the Analysed_Image object to store the realtive coords of the crops within the image in dataset
        # and the found ZA within the crop
        analysed_image_obj = PhaseIdent.Analysed_Image(image_in_dataset)
        
        # Get query image, which is the lowest mag image that was segmented and is query of the target image        
        query_image = flat_images_in_dataset_by_pixel_size[0]
            
        
        # Get upsampled crop of the segmented equivalent of the image in dataset from the segmented query to
        # find the squares per region to perform the phase identification analysis
        # Get the relative cooords of the image as will be useful for storing in the final Analysed_image object
        crop_image_array_ups, image_relative_coords = Correlate_Segmented_Atomic_q_t_Images(
            image_in_dataset, query_image, flat_images_in_dataset_by_pixel_size,  
            relative_positions, images_to_segment, images_segmented)
        
        
        # plot segmented crop if wanted
        plt.imshow(crop_image_array_ups)
        plt.show()
        
        
        
        # Work with all the labels within the segmented crop
        labels_in_crop = np.int16(np.unique(crop_image_array_ups))
        
        # !!! all labels except 0, as it is noise with 1st approx segmentation
        labels_in_crop = labels_in_crop[labels_in_crop != 0]
        for label in labels_in_crop:
            
            image_crop_hs_signal, image_crop_array, total_pixels_crop, crop_FOV, scaled_reference_coords = Get_Atomic_Crop_from_Segmented_Crop(
                crop_image_array_ups, label, image_in_dataset)
            
            # # After finding the square per label check whether the obtained 
            # # field of view and pixel size is enoguh to extract a reliable FFT
            minimum_FOV = PhaseIdent.Atomic_Resolution_WindowsAnalysis_Cutoff(
                image_in_dataset.x_calibration)
            
            # Only proceed with the phase identification if the minum FOV given the pixel size is met
            # as if it is not met, then no meaningful information would be extracted and the crop would be useless
            
            if crop_FOV >= minimum_FOV:
                # From atomic res image crop get the ZA and crystal spot pairs classified accordingly
                # to get the list_refined_cryst_spots
                
                list_refined_cryst_spots, refined_pixels, spots_int_reference = PhaseIdent.From_Image_to_ZA_CrystSpots_Wrapped(
                    image_crop_hs_signal, image_crop_array, total_pixels_crop, 
                    crop_FOV, crystal_objects_list, space_group_list, 
                    forbidden, min_d, tol)

                # After the analysis of the atomic resolution crop, store the info into the initialised Analysed_Image class
                analysed_image_obj.Add_Crop_Output(
                    image_crop_hs_signal, scaled_reference_coords, 
                    image_relative_coords, list_refined_cryst_spots,
                    refined_pixels, spots_int_reference)
                
            else:
                # the FOV is too small to have a reasonable phase identification, 
                # so just add the crop with no phases identified in it
                # This way Add_Crop_Output keeps track of all the labels that checked
                # and there is 1 to 1 correlation between the label of 
                # the segmented image and the crop index
                analysed_image_obj.Add_Crop_Output(
                    image_crop_hs_signal, scaled_reference_coords, 
                    image_relative_coords, [],[], [])
                
        # Add the analysed image object to the list of analysed images 
        # objects to gather them and compare them with the
        # image in dataset list of images, the analysed_image_obj already 
        # holds the image_in_dataset where the crops are taken from
        list_analysed_images.append(analysed_image_obj)
              
        

# Get phases from every crop made within the image, just 1 Analysed_Image object

analysed_image_only = list_analysed_images[0]


crop_outputs_dict = analysed_image_only.Crop_outputs

# dictionary keywords

for crop_index_i in range(3,analysed_image_only.crop_index):
    
    image_crop_hs_signal = crop_outputs_dict[str(crop_index_i) + '_hs_signal']
    crop_list_refined_cryst_spots = crop_outputs_dict[str(crop_index_i) + '_list_refined_cryst_spots']
    refined_pixels = crop_outputs_dict[str(crop_index_i) + '_refined_pixels']
    spots_int_reference = crop_outputs_dict[str(crop_index_i) + '_spots_int_reference']
    
    print('Crop index ' + str(crop_index_i)  + '\n')
    print('Image size')
    print(image_crop_hs_signal.axes_manager['x'].size)
    print('refined_pixels')
    print(refined_pixels)
    print('spots_int_reference')
    print(spots_int_reference)
    
    
    
    for cryst_spot in crop_list_refined_cryst_spots:
    
        print('\n Cryst \n')
        print('spot list', cryst_spot.spots)
        # print('spot pairs', cryst_spot.spot_pairs_obj)
        print('phase name', cryst_spot.phase_name)
        print('ZA of cryst', cryst_spot.ZA)
        # print('ZA priv index', cryst_spot.ZA_priv_index)
        
        for spot in cryst_spot.spot_pairs_obj:
            print('\n Scored Spot Pair info')
            print('ZA of scored spot pair: spot.ZA')
            print(spot.ZA)
            # print('spot.score')
            # print(spot.score)
            # print('pahse spotscore pair')
            # print(spot.phase_name)
            # print('1internalref')
            # print(spot.spot1_int_ref)
            # print('2internalref')
            # print(spot.spot2_int_ref)
            hkl1_reference = spot.hkl1_reference
            hkl1_angle_to_x = spot.spot1_angle_to_x
            hkl2_reference = spot.hkl2_reference
            hkl2_angle_to_x = spot.spot2_angle_to_x
        
            print('spot.hkl1_reference')
            print(spot.hkl1_reference)
            print('spot.spot1_angle_to_x')
            print(spot.spot1_angle_to_x)
            print('spot.hkl2_reference')
            print(hkl2_reference)
            print('spot.spot2_angle_to_x')
            print(hkl2_angle_to_x)
            print('spot.angle_between')
            print(spot.angle_between)
        
   
        
#%%



# FFT indexation interactive printing

fft_info_data = dict()

for crop_index_i in range(3,analysed_image_only.crop_index):
    
    
    crop_key_for_dict = 'Crop_segment_' + str(crop_index_i)
    
    image_crop_hs_signal = crop_outputs_dict[str(crop_index_i) + '_hs_signal']
    FFT_image_array, _ =  ImCalTrans.Compute_FFT_ImageArray(np.asarray(image_crop_hs_signal))
    crop_list_refined_cryst_spots = crop_outputs_dict[str(crop_index_i) + '_list_refined_cryst_spots']
    refined_pixels = crop_outputs_dict[str(crop_index_i) + '_refined_pixels']
    spots_int_reference = crop_outputs_dict[str(crop_index_i) + '_spots_int_reference']
    
    if len(refined_pixels) > 0:
        fft_info_data[crop_key_for_dict] = FFT_indexer.Collect_data(
            FFT_image_array, refined_pixels, spots_int_reference, crop_list_refined_cryst_spots)

    
    
    
    # # this is the function that must go in 
    
    # # info = ['' for i in range(len(self.spot_int_refs))]
    # info = ['' for i in range(len(refined_pixels))]
    
    
    # for spot_int_ref in spots_int_reference:
    # # for spot_int_ref in self.spots_int_reference:
        
    #     info[int(spot_int_ref)] = info[int(spot_int_ref)] + 'SPOT NUMBER ' + str(int(spot_int_ref)) + '\n'
        
    #     for cryst_spot in crop_list_refined_cryst_spots:
    #         # for every phase
            
    #         cryst_spot_phase = cryst_spot.phase_name
    #         cryst_spot_spots_int_ref = cryst_spot.spots
    #         cryst_spot_phase_name = cryst_spot.phase_name
    #         cryst_spot_ZA = cryst_spot.ZA
    #         cryst_spot_ZA_priv_index = cryst_spot.ZA_priv_index
            
    #         # meaning that spot was identified in that cryst spot object
    #         if spot_int_ref in cryst_spot_spots_int_ref:
                
    #             info[int(spot_int_ref)] = info[int(spot_int_ref)] + '--Crystal identified with:\n'
    #             info[int(spot_int_ref)] = info[int(spot_int_ref)] + 'Crystal phase: ' + str(cryst_spot_phase_name) + '\n'
    #             info[int(spot_int_ref)] = info[int(spot_int_ref)] + 'Zone axis: ' + str(cryst_spot_ZA) + '\n'
    #             info[int(spot_int_ref)] = info[int(spot_int_ref)] + 'Internal phase + ZA reference: ' + str(cryst_spot_ZA_priv_index) + '\n'
                
    #             for spot_pair in cryst_spot.spot_pairs_obj:
                    
    #                 # internal refereces of the spots defininng that spot pair
    #                 spot_pair_ints_i = [spot_pair.spot1_int_ref, spot_pair.spot2_int_ref]

    #                 # if that spot is present in the spot pair checked, append this information
    #                 if spot_int_ref in spot_pair_ints_i:
                        
    #                     index_found = spot_pair_ints_i.index(spot_int_ref)
                        
    #                     # if index_found = 0, the spot_int_ref is spot1
    #                     # if index_found = 1, the spot_int_ref is spot2
                        
    #                     if index_found == 0:
    #                         other_index = 1
    #                     else:
    #                         other_index = 0
    #                         # and of course other_index = 1
                        
    #                     hkl_refs = [spot_pair.hkl1_reference, spot_pair.hkl2_reference]
    #                     dists_refs = [spot_pair.spot1_dist , spot_pair.spot2_dist]
    #                     angle_between_spots = spot_pair.angle_between
                        
    #                     info[int(spot_int_ref)] = info[int(spot_int_ref)] + 'Forming spot pair with spot number:' + str(spot_pair_ints_i[other_index]) +'\n'
    #                     info[int(spot_int_ref)] = info[int(spot_int_ref)] + 'where spot '+str(int(spot_int_ref))+ ' is identified as: ' + str(hkl_refs[index_found]) +'\n'
    #                     info[int(spot_int_ref)] = info[int(spot_int_ref)] + 'and spot '+ str(spot_pair_ints_i[other_index]) + ' is identified as: ' + str(hkl_refs[other_index]) +'\n'
                        
    #                     info[int(spot_int_ref)] = info[int(spot_int_ref)] + 'Spot '+str(int(spot_int_ref))+ ' has a distance of: ' + str(dists_refs[index_found]) +' nm\n'
    #                     info[int(spot_int_ref)] = info[int(spot_int_ref)] + 'and spot '+str(spot_pair_ints_i[other_index])+ ' has a distance of: ' + str(dists_refs[other_index]) +' nm\n'
    #                     info[int(spot_int_ref)] = info[int(spot_int_ref)] + 'forming an angle of :' + str(angle_between_spots) +' degrees\n'
    #                     info[int(spot_int_ref)] = info[int(spot_int_ref)] + '(debug) ZA found :' + str(spot_pair.ZA) +'\n'
    #                     info[int(spot_int_ref)] = info[int(spot_int_ref)] + '(debug) phase found :' + str(spot_pair.phase_name) +'\n'
                    

FFT_indexer.fft_info_data = fft_info_data
# Open the app based on the info in fft_info_data

FFT_indexer.main()



        
#%%




# from the segmented image, skip some contours to smooth the global contour down
conts_vertx_per_region_skept = SegmWrap.Skip_n_contours_region_region_intercon(
    conts_vertxs_per_region_segmented[0], images_segmented[0], skip_n = 2)

# Convert to relative coordinates (with respect the lowest mag image) the vertexs and contours
conts_vertx_per_region = Segment.Relative_Vertexs_Contours(
    images_segmented[0], conts_vertx_per_region_skept, 
    relative_positions_to_segment[0], pixel_sizes_to_segment[0])
  

z_thickness_model = 1 #nm

    
# Build a folder named model_cells in the path before the unit cells are checked
# to place the conversion from uce to cif that are needed to build the models
model_cells_filepath = unit_cells_path[:unit_cells_path.find('unit_cells')] + '\\' + 'model_cells' + '\\'
path_model_cells = os.path.isdir(model_cells_filepath)
if path_model_cells == False:
    os.mkdir(model_cells_filepath)
    
# Folder to contain the models created, inside this model_cells, based
# on the name of the image
atom_models_filepath = model_cells_filepath + analysed_image_only.image_in_dataset.name + '\\'
path_atom_models = os.path.isdir(atom_models_filepath)
if path_atom_models == False:
    os.mkdir(atom_models_filepath)


# Information from all the crops done to the image
crop_outputs_dict = analysed_image_only.Crop_outputs

# Loop through the regions and build the atomic model from each region
for label_segm_region in range(1, analysed_image_only.crop_index):
    
    print(label_segm_region)
    image_crop_hs_signal = crop_outputs_dict[str(label_segm_region) + '_hs_signal']
    crop_list_refined_cryst_spots = crop_outputs_dict[str(label_segm_region) + '_list_refined_cryst_spots']
    
    
    # if no crystal is found, either amorphous or bad identified
    # in any case, do not consider it for the atomistic model
    # or build an amorphous one as next step    
    if len(crop_list_refined_cryst_spots) != 0:
    
        # most likely crystal found
        best_cryst_spot = crop_list_refined_cryst_spots[0]
        zone_axis_found =  best_cryst_spot.ZA
        scored_spot_pair = best_cryst_spot.spot_pairs_obj[0]
        hkl1_reference = scored_spot_pair.hkl1_reference
        hkl1_angle_to_x = scored_spot_pair.spot1_angle_to_x
        hkl2_reference = scored_spot_pair.hkl2_reference
        hkl2_angle_to_x = scored_spot_pair.spot2_angle_to_x
        found_phase_name = best_cryst_spot.phase_name
        
        
        # inconsistency between the ZA specified in best_cryst_spot.ZA and scored_spot_pair.ZA
        print('ZA found')
        print(zone_axis_found)
        print('ZA found in scored spot pair')
        print(scored_spot_pair.ZA)
        print('hkl1_reference')
        print(hkl1_reference)
        print(hkl1_angle_to_x)
        print('hkl2_reference')
        print(hkl2_reference)
        print(hkl2_angle_to_x)
        print('angle between')
        print(scored_spot_pair.angle_between)
        
        # convert the file with phase_name in the 
        # unit_cells_path to cif in the model_cells folder  
        cif_cell_filepath = model_cells_filepath + found_phase_name + '.cif'
        
        # Check if the cif file exists in the model_cells directory
        cif_cell_filepath_Path = Path(cif_cell_filepath)
        cif_file_already_created = cif_cell_filepath_Path.is_file()
        
        if cif_file_already_created == False:
            
            # convert the .uce to .cif as it does not exist yet
            uce_cell_filepath = unit_cells_path + '\\' + found_phase_name + '.uce'
            
            cif_cell_filepath = AtomBuild.uce_to_cif(
                uce_cell_filepath, found_phase_name, model_cells_filepath)
            
        
        # Find the rotation we need to induce to the default atomistic model 
        # to rotate it to the found orientation
        final_in_surf_plane_rotation = AtomBuild.Adjust_in_surface_plane_rotation(
            cif_cell_filepath, scored_spot_pair, suface_basis_choice = 'plane')
        
        # Save the models in the folder inside model_cells folder created with
        # the name of the folder 
        # function that does the full shaping and orientation 
        
        
        # check that the cntour adjustment is the same format as the one
        # which was with the trials in the previous checkings while 
        # developing the atomistic_model_builder
        
        # it fails in the make supercell functoin make sure w ereally need the function
        # that finds the optimal cubic shape as i think it deos more bad than good
        
        AtomBuild.Build_shaped_atomistic(
            cif_cell_filepath, zone_axis_found, final_in_surf_plane_rotation, 
            z_thickness_model, conts_vertx_per_region, label_segm_region, 
            atom_models_filepath, adjust_y_bottomleft = True)
    
    else:
        # len(crop_list_refined_cryst_spots) == 0
        # This means no phase was found, so it should be amorphous and 
        
        
        # !!! AMORPHOUS (RANDOM POSITONS) MATERIAL 
        # Address its chemical type by some complementary checmical information
        # like with all the maps check which is the label that coincides the most
        # with that given chemical map and then take this element as the element:
        # to make amorhous...same for binary compounds
        # If it is amorphous, then either do not do anything or build an amorphous:
        # model by just randomly scattering positions in the space defined by
        # that label and its contours
        
        continue
        

# combine the cells altoghether to form the bigger single atomistic model    
AtomBuild.Combine_xyz_supercells(atom_models_filepath)


# final_global_device_supcell = read(temp_xyz_files_folder_directory + 'global_device_supercell.xyz')
# show_atoms(final_global_device_supcell, plane='xy')
# show_atoms(final_global_device_supcell, plane='xz')
# show_atoms(final_global_device_supcell, plane='yz')





