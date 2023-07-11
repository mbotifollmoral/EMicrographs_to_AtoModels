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
from PIL import Image
import gdspy
import random
import ase
import ase.io
import ase.visualize

# !!! NEED to set the path to 
# Alg_Comb_Single_Image_Strain.py
# as the console working directory
Project_main_path = os.getcwd()
if 'EMicrographs_to_AtoModels' in Project_main_path:
    Project_main_path = Project_main_path[:Project_main_path.find('EMicrographs_to_AtoModels')-1]
# Project_main_path has the EMicrographs_to_AtoModels folder
sys.path.append(Project_main_path)

import EMicrographs_to_AtoModels

from EMicrographs_to_AtoModels.Functions.General_functions import HighToLowTM
from EMicrographs_to_AtoModels.Functions.General_functions import Segmentation_1stAprox as Segment
from EMicrographs_to_AtoModels.Functions.General_functions import Phase_Identificator as PhaseIdent
from EMicrographs_to_AtoModels.Functions.General_functions import ImageCalibTransf as ImCalTrans
from EMicrographs_to_AtoModels.Functions.General_functions import Segmentation_Wrapped as SegmWrap
from EMicrographs_to_AtoModels.Functions.General_functions import FFT_indexer
from EMicrographs_to_AtoModels.Functions.General_functions import FEM_input_generator as FEMBuild
from EMicrographs_to_AtoModels.Atomistic_model_builder import Atomistic_Model_Builder as AtomBuild
from EMicrographs_to_AtoModels.Functions.General_functions import GPA_atomistic_combiner as GPA_AtoMod
from EMicrographs_to_AtoModels.Functions.General_functions import GPA_specific as GPA_sp


#%%
'''
Hyperparameter definition
'''
# High to Low hyperparams
real_calibration_factor=0.97  # change calibration of the image
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
crop_setting = 'mask' # 'mask' or 'crop', FFT from the mask of the segmentation or from crop inside

#%%
'''
Template matching and generation of the relative coordinates per image over the lowest mag one
'''


# Emerging windows to choose the path to folder with the images
# root = tkinter.Tk()
# root.attributes("-topmost", True)
# root.withdraw()
# dataset_system_path_name = filedialog.askdirectory(parent=root,initialdir="/",title='Folder with the images dataset')

dataset_system_path_name = r'E:\Arxius varis\PhD\4rth_year\Global_ML_Results\InSb_Sn_VLS2\Micrographs\\'
dataset_system_path_name = r'E:\Arxius varis\PhD\4rth_year\Global_ML_Results\GeQW2\Micrographs\\'
dataset_system_path_name = r'E:\Arxius varis\PhD\4rth_year\Global_ML_Results\InSb_InP_TransvNW_3\Micrographs\\'

# Browse the images in the folder and also calibrate
# !!! CALIBRATION CORRECTION DONE HERE --> NO NEED TO CHANGE THE CALIBRATION OF THE IMAGES AT ANY POINT
# !!! UNLESS IMAGES ARE BINNED


images_in_dataset_list, pixel_sizes = HighToLowTM.Browse_Dataset_Images_and_Recalibrate(
    dataset_system_path_name, real_calibration_factor)

# Sort the dataset with a list of lists in which each list has images with the same pixel size
images_in_dataset_by_pixel_size = HighToLowTM.Sort_Dataset_by_PixelSize(
    images_in_dataset_list, pixel_sizes)

# Delete images which are too low magnificaiton (lamella appearence)
images_in_dataset_by_pixel_size, pixel_sizes = HighToLowTM.TooLow_Magnification_Cutoff(
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
flat_images_in_dataset_by_pixel_size, relative_positions, flat_pixel_sizes = HighToLowTM.Images_Position_PixelSizes_Arrays(
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
        crop_image_array_ups, image_relative_coords = HighToLowTM.Correlate_Segmented_Atomic_q_t_Images(
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
            # settning 'crop' or 'mask' (for single image analysis, mask)
            image_crop_hs_signal, image_crop_array, total_pixels_crop, crop_FOV, scaled_reference_coords = HighToLowTM.Get_Atomic_Crop_from_Segmented_Crop(
                crop_image_array_ups, label, image_in_dataset, setting = crop_setting)
            
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

# Image in dataset base information
image_in_dataset_whole = images_in_dataset_list[0]
image_array_whole = image_in_dataset_whole.image_arraynp_st
total_pixels_whole = image_in_dataset_whole.total_pixels
pixel_size_whole = image_in_dataset_whole.x_calibration
FFT_calibration_whole = image_in_dataset_whole.FFT_calibration
FFT_whole, _ =  ImCalTrans.Compute_FFT_ImageArray(np.asarray(image_in_dataset_whole.hyperspy_2Dsignal))


#%%

'''
# FFT indexation interactive printing
'''


fft_info_data = dict()

for crop_index_i in range(1, analysed_image_only.crop_index):
    
    
    crop_key_for_dict = 'Crop_segment_' + str(crop_index_i)
    
    image_crop_hs_signal = crop_outputs_dict[str(crop_index_i) + '_hs_signal']
    FFT_image_array, _ =  ImCalTrans.Compute_FFT_ImageArray(np.asarray(image_crop_hs_signal))
    crop_list_refined_cryst_spots = crop_outputs_dict[str(crop_index_i) + '_list_refined_cryst_spots']
    refined_pixels = crop_outputs_dict[str(crop_index_i) + '_refined_pixels']
    spots_int_reference = crop_outputs_dict[str(crop_index_i) + '_spots_int_reference']
    if len(refined_pixels) > 0:
        fft_info_data[crop_key_for_dict] = FFT_indexer.Collect_data(
            FFT_image_array, refined_pixels, spots_int_reference, crop_list_refined_cryst_spots)

                    

FFT_indexer.fft_info_data = fft_info_data
# Open the app based on the info in fft_info_data

FFT_indexer.main()

   
#%%

'''
# GPA Computation and calculation of the virutal crystal nature   
'''

# Retrieve data extracted from the reference 
analysed_image_only = list_analysed_images[0]
crop_outputs_dict = analysed_image_only.Crop_outputs

# Image in dataset base information
image_in_dataset_whole = images_in_dataset_list[0]
image_array_whole = image_in_dataset_whole.image_arraynp_st
total_pixels_whole = image_in_dataset_whole.total_pixels
pixel_size_whole = image_in_dataset_whole.x_calibration
FFT_calibration_whole = image_in_dataset_whole.FFT_calibration
FFT_whole, _ =  ImCalTrans.Compute_FFT_ImageArray(np.asarray(image_in_dataset_whole.hyperspy_2Dsignal))

# The information from GPA is gonna come from the biggest region that is segmented 
# in the downer half of the segmented region

# This is to automate the definition of the region where the substrate is found
# to take the reference from it
label_of_GPA_ref = GPA_sp.Define_Region_as_GPA_Reference(
    analysed_image_only, images_segmented[0])
label_of_substrate = GPA_sp.Define_Region_as_GPA_Reference(
    analysed_image_only, images_segmented[0])


# With the 1st approximation just define the label  of the reference we want
# label_of_GPA_ref = 4


# Pixels within the whole image in which the crop of the reference is taken, 
# so the box of the reference itself [first_row,last_row,first_col,last_col]

# Diverge whether the mask or crop criteria was used, and find the crop if necessary
# as with mask the whole image is considered the reference, so just use
# the region of what would be computed if "crop" as the reference
if crop_setting == 'mask':
    # re find the crop that would be used if no mask would be used but
    # the crop within the region
    # so fin dhte expected crop for that part
    # and use it
    _, _, _, _, scaled_reference_coords_GPA_ref = HighToLowTM.Get_Atomic_Crop_from_Segmented_Crop(
        images_segmented[0], label_of_GPA_ref, image_in_dataset_whole, setting = 'crop')
    
else:
    # Just use the scaled cords with is the crop itself 
    scaled_reference_coords_GPA_ref = crop_outputs_dict[str(int(label_of_GPA_ref)) + '_pixel_ref_cords']


# Reduce the coordinates of the square taking the reference region
# for the GPA a given factor 
# !!! Hyperparameter factor
GPA_square_factor = -0.20
scaled_reference_coords_GPA_ref = GPA_sp.Mod_GPA_RefRectangle(
    scaled_reference_coords_GPA_ref, GPA_square_factor)


# 
image_crop_hs_signal_GPA_ref = crop_outputs_dict[str(int(label_of_GPA_ref)) + '_hs_signal']
FFT_image_array_GPA_ref, _ =  ImCalTrans.Compute_FFT_ImageArray(np.asarray(image_crop_hs_signal_GPA_ref))

crop_list_refined_cryst_spots_GPA_ref = crop_outputs_dict[str(int(label_of_GPA_ref)) + '_list_refined_cryst_spots']
refined_pixels_GPA_ref = crop_outputs_dict[str(int(label_of_GPA_ref)) + '_refined_pixels']
spots_int_reference_GPA_ref = crop_outputs_dict[str(int(label_of_GPA_ref)) + '_spots_int_reference']

# Get the best cryst spot containing the best crystal phase fitting this region
best_cryst_spot_GPA_ref = crop_list_refined_cryst_spots_GPA_ref[0]


# Retrieve the best spot pair to be considered the GPA g vectors
best_GPA_ref_spot_pair = GPA_sp.Get_GPA_best_g_vects_pair(
    analysed_image_only, label_of_GPA_ref, images_segmented[0])

# Find the best spots considered in that crystal phase, which should 
# be the best ones to constitute the GPA g vectors
# and its information to help build the virutal crystal, although they
#  need to be updated after the refining of the g vectors with the ref
spot1_int_ref_GPA_ref = best_GPA_ref_spot_pair.spot1_int_ref
spot2_int_ref_GPA_ref = best_GPA_ref_spot_pair.spot2_int_ref


# Retrieve the info of the spots acting as g vectors
hkl1_reference_GPA_ref = best_GPA_ref_spot_pair.hkl1_reference
hkl2_reference_GPA_ref = best_GPA_ref_spot_pair.hkl2_reference
spot1_dist_GPA_ref = best_GPA_ref_spot_pair.spot1_dist
spot2_dist_GPA_ref = best_GPA_ref_spot_pair.spot2_dist
angle_between_GPA_ref = best_GPA_ref_spot_pair.angle_between
spot1_angle_x_GPA_ref = best_GPA_ref_spot_pair.spot1_angle_to_x
spot2_angle_x_GPA_ref = best_GPA_ref_spot_pair.spot2_angle_to_x
found_phase_name_GPA_ref = best_cryst_spot_GPA_ref.phase_name


# coords of the best peaks to use as GPA g vectors in coordinartes of the crop
# NOT of the whole image
cord_spot_1_GPA_ref = refined_pixels_GPA_ref[int(spot1_int_ref_GPA_ref)]
cord_spot_2_GPA_ref = refined_pixels_GPA_ref[int(spot2_int_ref_GPA_ref)]



# generate a 9x9 pixel square arround the best coordinate scaled to the whole image
# so from all this pixels we can find the best one that represents the maximum of
# the peak

scaled_cord_spot_1_GPA_y = int(np.round((cord_spot_1_GPA_ref[0]/FFT_image_array_GPA_ref.shape[0])*total_pixels_whole))
scaled_cord_spot_1_GPA_x = int(np.round((cord_spot_1_GPA_ref[1]/FFT_image_array_GPA_ref.shape[1])*total_pixels_whole))
scaled_cord_spot_2_GPA_y = int(np.round((cord_spot_2_GPA_ref[0]/FFT_image_array_GPA_ref.shape[0])*total_pixels_whole))
scaled_cord_spot_2_GPA_x = int(np.round((cord_spot_2_GPA_ref[1]/FFT_image_array_GPA_ref.shape[1])*total_pixels_whole))

if scaled_cord_spot_1_GPA_y-1 >= 0 and scaled_cord_spot_1_GPA_y+2 <= total_pixels_whole and scaled_cord_spot_1_GPA_x-1 >= 0 and scaled_cord_spot_1_GPA_x+2 <= total_pixels_whole:
    
    FFT_crop_spot_1 = FFT_whole[scaled_cord_spot_1_GPA_y-1:scaled_cord_spot_1_GPA_y+2,scaled_cord_spot_1_GPA_x-1:scaled_cord_spot_1_GPA_x+2]
    cords_max1 = np.where(FFT_crop_spot_1 == np.max(FFT_crop_spot_1))

    scaled_cord_spot_1_GPA_y = scaled_cord_spot_1_GPA_y - 1 + cords_max1[0][0]
    scaled_cord_spot_1_GPA_x = scaled_cord_spot_1_GPA_x - 1 + cords_max1[1][0]
    

if scaled_cord_spot_2_GPA_y-1 >= 0 and scaled_cord_spot_2_GPA_y+2 <= total_pixels_whole and scaled_cord_spot_2_GPA_x-1 >= 0 and scaled_cord_spot_2_GPA_x+2 <= total_pixels_whole:
    
    FFT_crop_spot_2 = FFT_whole[scaled_cord_spot_2_GPA_y-1:scaled_cord_spot_2_GPA_y+2,scaled_cord_spot_2_GPA_x-1:scaled_cord_spot_2_GPA_x+2]
    cords_max2 = np.where(FFT_crop_spot_2 == np.max(FFT_crop_spot_2))

    scaled_cord_spot_2_GPA_y = scaled_cord_spot_2_GPA_y - 1 + cords_max2[0][0]
    scaled_cord_spot_2_GPA_x = scaled_cord_spot_2_GPA_x - 1 + cords_max2[1][0]
# else:
#     the coordinates are the same as they were

spot_1_coords = np.array([scaled_cord_spot_1_GPA_y, scaled_cord_spot_1_GPA_x])
spot_2_coords = np.array([scaled_cord_spot_2_GPA_y, scaled_cord_spot_2_GPA_x])

# Plot the image with the square reference if wanted
GPA_sp.Plot_Image_with_GPA_Reference(image_array_whole, scaled_reference_coords_GPA_ref)

# Make a binary image where the substrate is located (= 1, rest 0)
substrate_segment = np.zeros((np.shape(image_array_whole)))
substrate_segment[images_segmented[0] == label_of_substrate] = 1

# Find contours for the interface angle calculation afterwards
substrate_segment_contour = Segment.Contour_draw_All_regions(substrate_segment)

# Compute the correction angle between mathematical x axis and image horizontal planes
rotation_angle_x_GPA = GPA_sp.Compute_GPA_Rotation_X_Axis(
    [spot1_angle_x_GPA_ref, spot2_angle_x_GPA_ref], substrate_segment,
    substrate_segment_contour)


# Compute the features that should define the resolution of the GPA mask
aver_of_aver, aver_of_max, aver_of_min, aver_of_med, aver_of_desvest = GPA_sp.Compute_Feature_defining_GPA_Res(
    images_segmented[0], conts_vertxs_per_region_segmented[0], pixel_size_whole)

# if the heterostructure based search of spot does not work for defining the GPA
# resolution, use the features calculated in Compute_Feature_defining_GPA_Res
# to adjust the resolution (e.g. a fraction of the average of medians)

# compute the resolution of GPA checking if there is an heterostructure
GPA_resolution = GPA_sp.Get_GPA_Res_HeteroStruct_Separated_Spots(
    analysed_image_only, image_in_dataset_whole, best_GPA_ref_spot_pair,
    images_segmented[0], label_of_GPA_ref)

# Compute the mask size
mask_size, GPA_eff_resolution = GPA_sp.Define_GPA_Mask_Size(
    image_array_whole, FFT_calibration_whole, smallest_feature = GPA_resolution)


# Wiener filter the image for GPA smoothing
GPA_Wiener_Filter = True
if GPA_Wiener_Filter == True:
    image_array_whole = GPA_AtoMod.wiener_filt(image_array_whole, power=1)


# Compute GPA and store the images
exx, eyy, exy, eyx, rot, shear, Dispx, Dispy, ref_spot1, ref_spot2 = GPA_AtoMod.GPA_full_AtomisticModel(
    image_array_whole, pixel_size_whole, spot_1_coords, spot_2_coords, 
    mask_size, scaled_reference_coords_GPA_ref, rotation_angle_x_GPA, display=True)



# Build a directory to save the all outputs and store all the images genrated 
# with all the info from the computation 
Results_dir = dataset_system_path_name[:-3][:dataset_system_path_name[:-3].rfind('\\')] + '\\Results_' + image_in_dataset_whole.name + '\\'
path_Results = os.path.isdir(Results_dir)
if path_Results == False:
    os.mkdir(Results_dir)

# Path to save GPA stuff
GPA_save_directory = Results_dir + 'GPA_output' + '\\'
path_GPA = os.path.isdir(GPA_save_directory)
if path_GPA == False:
    os.mkdir(GPA_save_directory)


# Save all the outputs
for result_image_name, result_image_array in zip(['exx','eyy','exy','eyx','rot','shear','Dispx','Dispy'],[exx,eyy,exy,eyx,rot,shear,Dispx,Dispy]):
    save_GPA_element_directory = GPA_save_directory + result_image_name + '.tiff'        
    im = Image.fromarray(result_image_array, mode='F') # float32
    im.save(save_GPA_element_directory, "TIFF")



#%%
'''
Bragg filtering
'''


# Compute the number of pixels per spot to mask
bragg_mask_size = FFT_indexer.Bragg_filter_mask_size(
    GPA_resolution, image_in_dataset_whole, mask_reduction_factor = 10)

# Bragg filter phases as found, for all the phases found for all the crops
# where phases are found
general_bragg_filterings = FFT_indexer.Indexed_FFT_BraggFiltering(
    analysed_image_only, image_in_dataset_whole, bragg_mask_size)

# !!! Hyperparameter, number of phases to Bragg filter per crop
phases_per_crop = 1
# Plot the Bragg filtered images
FFT_indexer.Colour_Mix_BraggFilteredPhases(
    image_in_dataset_whole, general_bragg_filterings, phases_per_crop = 1)



#%%


'''
Translation of GPA information, displacement fields, into the atomistic model
'''


# Convert all found phases into cif files
model_cells_filepath = AtomBuild.cif_from_uce_All_found_phases(
    analysed_image_only, unit_cells_path, Results_dir)


# first make the check on the reference lattice label_of_GPA_ref

cif_cell_filepath_GPA_ref = model_cells_filepath + found_phase_name_GPA_ref + '.cif'


# compute the theoretical distance given that plane hkl reference and phase
theo_interplanar_d_hkl1_ref = AtomBuild.Compute_interplanar_distance(
    cif_cell_filepath_GPA_ref, hkl1_reference_GPA_ref)

theo_interplanar_d_hkl2_ref = AtomBuild.Compute_interplanar_distance(
    cif_cell_filepath_GPA_ref, hkl2_reference_GPA_ref)


ase_unit_cell_GPA_ref = ase.io.read(cif_cell_filepath_GPA_ref)

spacegroup_cell_GPA_ref = ase_unit_cell_GPA_ref.info['spacegroup'].no

a_GPA_ref, b_GPA_ref, c_GPA_ref, alfa_GPA_ref, beta_GPA_ref, gamma_GPA_ref = ase_unit_cell_GPA_ref.cell.cellpar()

GPA_ref_cell_params = a_GPA_ref, b_GPA_ref, c_GPA_ref, alfa_GPA_ref, beta_GPA_ref, gamma_GPA_ref 


# Recompute the distances of the spots from the subpixel refinement of the GPA ref
refined_distances_of_GPA_ref = PhaseIdent.Spot_coord_To_d_spacing_vect(
    np.array([ref_spot1, ref_spot2]), FFT_calibration_whole, total_pixels_whole)

# !!! UNITS: Convert from nm to angstroms
refined_distances_of_GPA_ref = ImCalTrans.nm_to_Angstroms(
    refined_distances_of_GPA_ref)

# compute the virtual cell parameters with the refined distances from the reference area
a_v_cell, b_v_cell, c_v_cell = GPA_AtoMod.Find_virtual_a_cell_c_cell_params(
    GPA_ref_cell_params, hkl1_reference_GPA_ref, hkl2_reference_GPA_ref, 
    refined_distances_of_GPA_ref[0], refined_distances_of_GPA_ref[1])

# Build the virtual cif file for the reference region
path_to_v_unitcell = GPA_AtoMod.Build_virtual_crystal_cif(
    model_cells_filepath, found_phase_name_GPA_ref, label_of_GPA_ref,
    a_v_cell, b_v_cell, c_v_cell)


# ONLY USE IF WANT TO CREATE A SUPERCELL WITH THE PARAMETERS AS FOUND
# EXPERIENTALLY FOR EACH REGION, BUT NOT FOR STRAIN APPLICATION 
# Build all the crystals except the reference one 
# paths_to_virt_ucells = GPA_AtoMod.Build_All_Virtual_Crysts_Except_Ref(
#     analysed_image_only, image_in_dataset_whole, best_GPA_ref_spot_pair,
#     images_segmented[0], label_of_GPA_ref, 
#     GPA_resolution, model_cells_filepath)


# Build the virtual UNIT cells for all the regions that are not the reference     
# making them coincide with the same interplanar distances as the reference
paths_to_virt_ucells, scored_spot_pairs_found, scaled_cords_spots = GPA_AtoMod.Build_All_Virtual_Crysts_SameDistRef(
    analysed_image_only, image_in_dataset_whole, best_GPA_ref_spot_pair,
    refined_distances_of_GPA_ref[0], refined_distances_of_GPA_ref[1], 
    images_segmented[0],  label_of_GPA_ref, GPA_resolution, model_cells_filepath)


#%%

'''
Base atomistic model builder
'''

# from the segmented image, skip some contours to smooth the global contour down
conts_vertx_per_region_skept = SegmWrap.Skip_n_contours_region_region_intercon(
    conts_vertxs_per_region_segmented[0], images_segmented[0], skip_n = 30)


# Convert to relative coordinates (with respect the lowest mag image) the vertexs and contours
conts_vertx_per_region = Segment.Relative_Vertexs_Contours(
    images_segmented[0], conts_vertx_per_region_skept, 
    relative_positions_to_segment[0], pixel_sizes_to_segment[0])


z_thickness_model = 2 # nm


# If wanted, build the device with the perfect unmodified crystals from database    
# atom_models_filepath = AtomBuild.Build_DeviceSupercell_Base_Crystals(
#     analysed_image_only, model_cells_filepath, 
#     z_thickness_model, conts_vertx_per_region)

# Build the device out of the virtual crystals when suitable
atom_models_filepath, labels_equally_oriented_as_ref = AtomBuild.Build_DeviceSupercell_Virtual_To_Distort(
    analysed_image_only, model_cells_filepath, z_thickness_model, 
    conts_vertx_per_region, label_of_GPA_ref, best_GPA_ref_spot_pair, 
    paths_to_virt_ucells, scored_spot_pairs_found, scaled_cords_spots)



#%%
'''
FEM model input file generator
'''

# Whether to include or not the regions where no crystal is found
# False to include regions where no crystal was found
FEM_only_crystals = True

FEM_models_filepath = FEMBuild.Build_FEM_gds(
    analysed_image_only, conts_vertx_per_region, 
    model_cells_filepath, only_crystalline = FEM_only_crystals)


input_FEM_filename = FEMBuild.Extract_FEM_input_parameters(
        analysed_image_only, model_cells_filepath, 
        only_crystalline = FEM_only_crystals)




#%%


'''
Transfer strain map to atomistic model
'''

# # Plotting the displacement field in small regions

B_strain_width = 400
B_strain_height = 400
B_strain_y_i = 1000
B_strain_y_f = B_strain_y_i + B_strain_height
B_strain_x_i = 250
B_strain_x_f = B_strain_x_i + B_strain_width


Box_strain_pixels = [B_strain_y_i, B_strain_y_f, B_strain_x_i, B_strain_x_f] 


# !!! Strain distortion to models Hyperparameters
B_strain_aug_fact = 0.15
min_dist_red_fact = 1/3
purge_interatomic_distance = True
purge_wrong_displacements = False



# Path to the global cell if it is just one cell, named global...
# which is a required input afterwards for that case when multiple phases
# are found in the same orientation/SG, if no phases same orientation/SG
# just ignore it as it will just point to the last label file
# it is not used as we later filter by len(labels_equally_oriented_as_ref)
path_global_strained_purged = GPA_AtoMod.Distort_AtoModel_Region(
    atom_models_filepath, Dispx, Dispy, Box_strain_pixels, 
    pixel_size_whole, total_pixels_whole,
    B_strain_aug_fact = B_strain_aug_fact, min_dist_red_fact = min_dist_red_fact,
    purge_interatomic_distance = purge_interatomic_distance, 
    purge_wrong_displacements = purge_wrong_displacements)



#%%

'''
Post-strain chemistry/segmentation refinement
'''

# Make the difference to how the refinement is done depending on 
# the equally oriented crystals found
if labels_equally_oriented_as_ref > 1:
    # If many crystals are found with the same orientation, use the 
    # path_region_to_strained_purged to the global strained purged model
    atomregmodel_path_final = GPA_AtoMod.Refine_StrainedRegion_SingleAtomBlock_Segmentation(
        analysed_image_only, model_cells_filepath, 
        path_global_strained_purged, label_of_GPA_ref, 
        labels_equally_oriented_as_ref, conts_vertx_per_region, 
        paths_to_virt_ucells, collapse_occupancies = True)
    
else:
    # labels_equally_oriented_as_ref == 1, so only the reference region:
    # if the crystals are in different orientations and belong to different
    # space groups, then merge the multiple atomic models obtained from the
    # different labels and their distortion
    atomregmodel_path_final = GPA_AtoMod.Refine_StrainedRegion_MultiAtomBlock_Segmentation(
        atom_models_filepath, conts_vertx_per_region)
    



# Cut the final model to the real size of the box that was initially defined
# by the Box_strain_pixels which was increased to make sure the distortions
# still placed atoms inside the region of interest which is now returned
path_finalcut_strainedmodel = GPA_AtoMod.Original_BoxSize_StrainedModel(
    B_strain_aug_fact, Box_strain_pixels, pixel_size_whole, 
    total_pixels_whole, atomregmodel_path_final)



