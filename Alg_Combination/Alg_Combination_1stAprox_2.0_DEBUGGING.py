# -*- coding: utf-8 -*-
"""
Created on Fri Mar  4 16:21:11 2022

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
import HighToLowTM 
import Segmentation_1stAprox as Segment
import Filters_Noise as FiltersNoise
import Phase_Identificator as PhaseIdent
import ImageCalibTransf as ImCalTrans
import PeakFinding_1stAprox as PeakFind
import GPA_specific as GPA_sp

# Peak finding functions
sys.path.append(r'E:\Arxius varis\PhD\4rth_year\Code\Functions\Peak_detector_Final')

import PF_FFT_processing as FFT_Procs
import PF_Peaks_detector as Peaks_detector
import PF_ImageTreatment_and_interfaces as PF_II


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

# Segmentation functions in the combined algorihtm 1st aprox

def Images_to_Segment_Cutoff(pixel_size_segment_thresh, flat_images_in_dataset_by_pixel_size,relative_positions,flat_pixel_sizes):
    '''
    Filter and only return these images that have a pixel size larger than the pixel_size_segment_thresh
    as these will be the ones that will need a segmentation and be used to define the different regions

    Parameters
    ----------
    pixel_size_segment_thresh : pixel size threshold in nm/pixel
    flat_images_in_dataset_by_pixel_size : 
    relative_positions : 
    flat_pixel_sizes : 

    Returns
    -------
    None.

    '''
    
    images_to_segment=flat_images_in_dataset_by_pixel_size[flat_pixel_sizes>pixel_size_segment_thresh]
    relative_positions_to_segment=relative_positions[flat_pixel_sizes>pixel_size_segment_thresh,:]
    pixel_sizes_to_segment=flat_pixel_sizes[flat_pixel_sizes>pixel_size_segment_thresh]
    
    
    # ensure that at least one images is used to segment (even if below the threshold)
    if len(images_to_segment)==0:
        images_to_segment=[flat_images_in_dataset_by_pixel_size[0]]
        relative_positions_to_segment=[relative_positions[0]]
        pixel_sizes_to_segment=[flat_pixel_sizes[0]]
    
    
    return images_to_segment, relative_positions_to_segment, pixel_sizes_to_segment



def Relative_Vertexs_Contours(array_to_segment, conts_vertx_per_region, relative_position, pixel_size):
    '''
    Convert the identified vertices and contours (vertices pair) into real coordinates in nm, depending on
    where the image is located according to the relative positions map found by template matching, in order to
    have for each segment the spatial localisation, in nm, of its contour and the region they hold
    The function only modifies the inputed conts_vertx_per_region dictionary by adding the relative pixels
    and modifying the contour_vector objects by adding the initial and final coords in relative terms

    Parameters
    ----------
    array_to_segment : must have a lavel 0 to account for the noise
    conts_vertx_per_region : dictionary
    relative_position : 
    pixel_size

    Returns
    -------
    conts_vertx_per_region : same dictionary

    '''
    
    
    (x_relative_position_start, y_relative_position_start, _ , _)=relative_position
        
    label_values=np.unique(array_to_segment)
    #label 0 is associated to pixels that do not have a label
    label_values=np.sort(label_values)[1:]
    
    #the contour vector seems to work fine but the pixel ones seem to be capturing the integer only
    
    for label in label_values:
          
        rel_vertexs=np.float64(np.copy(conts_vertx_per_region[str(int(label))+'_'+'vertexs']))
        rel_vertexs[:,0]=y_relative_position_start+pixel_size*rel_vertexs[:,0]
        rel_vertexs[:,1]=x_relative_position_start+pixel_size*rel_vertexs[:,1]
        conts_vertx_per_region[str(int(label))+'_'+'rel_vertexs']=rel_vertexs
        
        for contour_vector in conts_vertx_per_region[str(int(label))+'_'+'contours']:
            init_rel_coords_y,init_rel_coords_x =contour_vector.init_coords
            final_rel_coords_y,final_rel_coords_x=contour_vector.final_coords
            
            init_rel_coords_y=y_relative_position_start+pixel_size*init_rel_coords_y
            init_rel_coords_x=x_relative_position_start+pixel_size*init_rel_coords_x
            final_rel_coords_y=y_relative_position_start+pixel_size*final_rel_coords_y
            final_rel_coords_x=x_relative_position_start+pixel_size*final_rel_coords_x
            
            rel_init_coords=(init_rel_coords_y,init_rel_coords_x)
            rel_final_coords=(final_rel_coords_y,final_rel_coords_x)
            
            contour_vector.add_relative_contour(rel_init_coords,rel_final_coords)
            

    return conts_vertx_per_region



def Segment_Images(
        images_to_segment,relative_positions_to_segment, 
        pixel_sizes_to_segment):
    '''
    Takes the images to be segmented (within pixel sizes criteria) and segments them and puts them in the
    images_segmented list, ordered as they entered (lowest mag to highest mag)
    
    It generates the vertexs and contours for the images and their regions both in absolute terms and in
    relative terms according to the template mathcing and its localisation within the big image
    It is then a list of dictionaries, in which each can:
    conts_vertx_per_region[str(int(label))+'_'+'vertexs']  # vertexs of the contour for each region (label)
    conts_vertx_per_region[str(int(label))+'_'+'rel_vertexs'] # vertex of the contour in nm for each label
    conts_vertx_per_region[str(int(label))+'_'+'contours']  # contour_vector objects for every label and with absolute and relative positions

    Parameters
    ----------
    images_to_segment : image_in_dataset object
    relative_positions_to_segment : relative positions of the images to be segmented
    pixel_sizes_to_segment : pixel sizes of the images to be segmented

    the three arrays must have the same nnumber of elements
    Returns
    -------
    images_segmented : list of arrays
    conts_vertxs_per_region_segmented : list of dictionaries

    '''

    #Hyperparameters for the segmentation
    gauss_blur_filter_size_segment=(5,5)  #size of smoothing filter for the segmentation, go to line to change sigma
    red_image_size=64  # downscale image 
    red_image_size_highFOV=128 # downscale images whose FOV is larger than 500nm
    
    # lists containing the segmented images and the conts and vertex of the segmented regions per image
    images_segmented=[]
    conts_vertxs_per_region_segmented=[]
    
    for image_to_segment, relative_position_to_segment, pixel_size_to_segment in zip(images_to_segment, relative_positions_to_segment,pixel_sizes_to_segment):
        array_to_segment=image_to_segment.image_arraynp_st
        
        # Denoise image if wanted, good for segmenting
        array_to_segment=cv2.GaussianBlur(array_to_segment, gauss_blur_filter_size_segment, 1)
        array_to_segment=(array_to_segment-np.min(array_to_segment))/np.max(array_to_segment-np.min(array_to_segment))
    
        # Histogram to find the peaks and correlate with number of clusters, too many peaks found (noise)
        hist_array=plt.hist(array_to_segment.ravel(),256,[np.min(np.array([array_to_segment])),np.max(np.array([array_to_segment]))])
        plt.close()
        # plt.show()
        hist2=np.append(hist_array[1],0)
        hist1=np.append(0,hist_array[1])
        hist_x_vals=((hist2+hist1)/2)[1:-1]
        
        # Smooth down the curve to bypass the noise and only find real maxima
        smoothed_hist=FiltersNoise.savitzky_golay(hist_array[0], 21, 3)
        
        # plt.plot(hist_x_vals, smoothed_hist)
        # plt.show()
        # Define the number of Gaussian units to fit it
        n_gaussian_ref=len(argrelextrema(smoothed_hist, np.greater)[0]) #number of components in the gaussian refinement (like k means but if doubt overstimate)
        
        # check position of values it if wanted the histogram nature and the peaks
        # max_vals=smoothed_hist[argrelextrema(smoothed_hist, np.greater)[0]]
        # max_vals_positions=hist_array[1][argrelextrema(smoothed_hist, np.greater)[0]]
        
        # Downscale image to make segmentation easier (1st approx stuff)
        
        # Downscale the image differently according to the displayed FOV
        
        # downscale the image after finding the peaks of the histogram, as after the downscaling the histogram is too noisy
        if image_to_segment.FOV<500:   # for small FOVs (less than 500nm), downscale more
            downscaling_factor=int(array_to_segment.shape[0]/red_image_size)
            array_to_segment=skimage.measure.block_reduce(array_to_segment, block_size=tuple(np.int32(downscaling_factor*np.ones(len(np.shape(array_to_segment))))), func=np.mean, cval=0)
            array_to_segment=(array_to_segment-np.min(array_to_segment))/np.max(array_to_segment-np.min(array_to_segment))
        else:  # for FOVs that are larger than 500 nm 
            downscaling_factor=int(array_to_segment.shape[0]/red_image_size_highFOV)
            array_to_segment=skimage.measure.block_reduce(array_to_segment, block_size=tuple(np.int32(downscaling_factor*np.ones(len(np.shape(array_to_segment))))), func=np.mean, cval=0)
            array_to_segment=(array_to_segment-np.min(array_to_segment))/np.max(array_to_segment-np.min(array_to_segment))
    
    
        # perform the gaussian pre clustering
        array_to_segment=Segment.Gaussian_pre_clustering(array_to_segment,number_of_gaussians=n_gaussian_ref,variance_threshold=0.001)
    
        # plt.imshow(ds_image_st)
        # plt.show()
        # plt.hist(ds_image_st.ravel(),256,[np.min(np.array([ds_image_st])),np.max(np.array([ds_image_st]))])
        # plt.show()
        
        #K-means clustering
        # define the k means clusters as the number of maxima that has not been set as an individual gaussian component
        indiv_gauss_comps=np.max(array_to_segment)-1
        
        k_means_clusters=int(n_gaussian_ref-indiv_gauss_comps)    #number of components in k means clustering
        
        values, labels, cost = Segment.best_km(array_to_segment, n_clusters =k_means_clusters)
        labels_ms_reshaped = np.choose(labels, values)
        labels_ms_reshaped.shape = array_to_segment.shape
        
        array_to_segment=labels_ms_reshaped
        del(labels_ms_reshaped)
    
        # Group clusters separated and denoise
        # tolerance 0 cannot be as we need some pixels to be noise for the contour drawing
        array_to_segment=Segment.Multiple_Consecutive_Clustered_regions(array_to_segment, criteria='Tolerance', tolerance=0.01)   
    
        # Keep in mind that during the previous steps the image is full size but after the following line then it becomes the original size 
        # plus one pixel in each direction, in this next step, where the contours are computed
        conts_vertx_per_region=Segment.Contour_draw_All_regions(array_to_segment)
        final_extra_pixs=Segment.Denoise_All_Regions(array_to_segment,conts_vertx_per_region)
    
        for label in final_extra_pixs:
            for pixel in final_extra_pixs[label]:  
                array_to_segment[pixel[0],pixel[1]]=int(label[0])   
                
        # Re-scale the image to its original size
        size_down=(array_to_segment.shape[0]*downscaling_factor,array_to_segment.shape[0]*downscaling_factor)
        array_to_segment= cv2.resize(array_to_segment, size_down, interpolation = cv2.INTER_NEAREST)
        
        #add the segmented image to the list of segmented images
        images_segmented.append(array_to_segment)
        
        # Compute the contours and vertexs of the image reshaped
        conts_vertx_per_region=Segment.Contour_draw_All_regions(array_to_segment)
        
        # Convert to relative coordinates (with respect the lowest mag image) the vertexs and contours
        conts_vertx_per_region=Relative_Vertexs_Contours(array_to_segment, conts_vertx_per_region, relative_position_to_segment, pixel_size_to_segment)
        
        # append the dictionary 
        conts_vertxs_per_region_segmented.append(conts_vertx_per_region)
    
    return images_segmented, conts_vertxs_per_region_segmented


#%%
'''
Functions for communicating the segmented regions as querys with the atomic resoltion images as
templates and find atomically resolved crops within the low mag segmented images
'''

class Analysed_Image:
    '''
    Class to store, for every image, the outputs for all of the arising squares 
    where the phase identification is performed
    '''
    
    def __init__(
            self, image_in_dataset):
        '''
        To initialise the class as soon as an image_in_datset is started to be processed
        '''
        
        self.image_in_dataset = image_in_dataset
        self.crop_index = 0
        self.Crop_outputs = dict()
    
    def Add_Crop_Output(
            self,image_crop_hs_signal, scaled_reference_coords, 
            image_relative_coords, list_refined_cryst_spots):
        ''' 
        Parameters
        ----------
        image_crop_hs_signal : complete (full info) hyperspy signal of the crop
        scaled_reference_coords : pixel coordinates of the crop within the image_in_dataset
        image_relative_coords :
             rel global coords (within the large low mag FOV) of the image 
             where the crops come from (i.e. image_in_dataset)
        list_refined_cryst_spots : 

        Returns
        -------
        None.

        '''

        self.Crop_outputs[str(self.crop_index) + '_hs_signal'] = image_crop_hs_signal
        self.Crop_outputs[str(self.crop_index) + '_pixel_ref_cords'] = scaled_reference_coords
        
        crop_FOV = image_crop_hs_signal.axes_manager['x'].scale * image_crop_hs_signal.axes_manager['x'].size
        
        # Compute the relative coords of where the crop is within the global low mag image/FOV
        [first_row,last_row,first_col,last_col] = scaled_reference_coords
        (x_start, y_start, x_end, y_end) = image_relative_coords  # coords of the image where the crops come from
            
        rel_x_start = x_start + (first_col/self.image_in_dataset.total_pixels)*(x_end-x_start)
        rel_y_start = y_start + (first_row/self.image_in_dataset.total_pixels)*(y_end-y_start)
           
        rel_x_end = rel_x_start + crop_FOV
        rel_y_end = rel_y_start + crop_FOV
        #  result relative coordinates of the square in nm with respect the full global FOV
        crop_rel_global_coords = (rel_x_start, rel_y_start, rel_x_end, rel_y_end)
        self.Crop_outputs[str(self.crop_index) + '_rel_cords'] = crop_rel_global_coords
        
        #compute the center of the found square, first x coord and second y
        center_coords = (rel_x_start + crop_FOV/2, rel_y_start + crop_FOV/2)
        
        self.Crop_outputs[str(self.crop_index) + '_center_cords'] = center_coords
        self.Crop_outputs[str(self.crop_index) + '_list_refined_cryst_spots'] = list_refined_cryst_spots
        # add 1 to the index so the following added crop is added as a new one
        self.crop_index = self.crop_index + 1
  

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
Phase identification functions in the combined algorihtm 1st aprox
'''
  
def Atomic_Resolution_WindowsAnalysis_Cutoff(pixel_size):
    '''
    Define the minimum windows size that can be used for doing the atomic resolution image processing,
    using a window size smaller than the defined with the given pixel_size would imply not being able to 
    capture good enough the atomic resolution informations in its FFT
    
    The commented lines are the regression that has been performed to extract the (hyperbolic) regression 
    that is used with the parameters m, c
    To change the regression because further information is added or the pattern is not correct,
    modify the array values of pixel size and FOV or the coefficints, or maybe change the function
    Parameters
    ----------
    pixel_size : nm/pixel

    Returns
    -------
    minimum_window_size : minimum FOV to be used, in nm

    '''
    # Do the regression with the following values to get a funciton of the FOV to use as minimum FOV 
    # for extracting atomic resolution information
    # FOV in nm and pixel sizes in nm/pixel
    # reasonable_FOVs=np.array([4.7,6.68, 6.68,6.68,4.763,6.67,2.6,1.2, 2.414,2.414,1.63,2.2,1.523,1.188])
    # corresponding_pixel_sizes=np.array([0.00469,0.006565,0.006565,0.006565,0.009205,0.01313,0.01313,0.0185, 
    #                                     0.0185,0.0185,0.026,0.037,0.0515,0.0742])
   
    # plt.scatter(corresponding_pixel_sizes,reasonable_FOVs)
    # plt.show()
    
    # fit a HYPERBOLIC FUNCTION between pixel sizes and FOV
    # corresponding_pixel_sizes_inverse=corresponding_pixel_sizes**-1
    # m,c=np.polyfit(corresponding_pixel_sizes_inverse,reasonable_FOVs,1)
    # print(m, c)
    # regression values, m and c, from the pixel_sizes vs FOV values above
    m=0.028854794880372733 
    c=1.2111383321107803
    
    # to make the regression a bit more restrictive, increase the value of c (for isntance, c=c+1)
    minimum_window_size=m*pixel_size**-1+c
    # corresponding_pixel_sizes_reg=m*corresponding_pixel_sizes**-1+c
    # plt.scatter(corresponding_pixel_sizes,corresponding_pixel_sizes_reg)
    # plt.show()
    
    return minimum_window_size     
 
   
def Pixels_peaks_FFT(
        FFT_image_array, crop_FOV):

    # wrap the 1st aprox peak finding in one line
    
    #1st Approx peak finding hyperparameter finding 
    st_distance,_,FFT_perc=PeakFind.FFT_hyperparams(FFT_image_array,crop_FOV)
    FFT_thresholdingG=PeakFind.Threshold_given_percentage(FFT_image_array, FFT_perc)
    
    #1st Approx peak finding hyperparameter fitting
    pixels_of_peaks=stemtool.afit.peaks_vis(FFT_image_array, dist=st_distance, thresh=FFT_thresholdingG, imsize=(15, 15))
    plt.show()
    # plt.close()
 
    return pixels_of_peaks


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
    FFT_calibration,FFT_units=ImCalTrans.FFT_calibration(image_hs_signal)
    
    # Denoise the image if wanted, might help to denoise but typically the peak finding works worse
    #image_array=FiltersNoise.Adaptative_Gaussian_Blur(image_array, im_calibration)
    
    # Compute the FFT 
    FFT_image_array, FFT_image_complex=ImCalTrans.Compute_FFT_ImageArray(image_array)
    
    # Filter the FFT in case we see it is convenient
    FFT_image_array=ImCalTrans.FFT_Gaussian_Convolution_Filter(FFT_image_array)
    
    # Standarise the FFT
    FFT_image_array=ImCalTrans.Standarise_Image(FFT_image_array)
    # plt.imshow(FFT_image_array, cmap=plt.cm.gray, vmin=FFT_image_array.min(), vmax=FFT_image_array.max())
    # plt.show()
        
    return FFT_image_array, FFT_calibration, FFT_units


def FFT_No_Log_from_crop(
        image_hs_signal, image_array):
    
    # wrap the FFT obtaining and calibration based on the crop in one line
    # inputs are the hs and np signals of the CROP
    
    # Calibrate the FFT, which does change the calibration with cropping
    FFT_calibration,FFT_units=ImCalTrans.FFT_calibration(image_hs_signal)
    
    # Denoise the image if wanted, might help to denoise but typically the peak finding works worse
    #image_array=FiltersNoise.Adaptative_Gaussian_Blur(image_array, im_calibration)
    
    # Compute the FFT 
    FFT_image_array_No_Log, FFT_image_complex=ImCalTrans.Compute_FFT_ImageArray_NO_Log(image_array)
    
    # We see that filtering is not convenient for the analysis
    #FFT_image_array=ImCalTrans.FFT_Gaussian_Convolution_Filter(FFT_image_array)
    
    # Standarise the FFT
    FFT_image_array_No_Log = ImCalTrans.Standarise_Image(FFT_image_array_No_Log)
    # plt.imshow(FFT_image_array, cmap=plt.cm.gray, vmin=FFT_image_array.min(), vmax=FFT_image_array.max())
    # plt.show()
        
    return FFT_image_array_No_Log, FFT_calibration, FFT_units


def Ready_Distances_Angles_Pixels(
        pixels_of_peaks, FFT_calibration, total_pixels_image):
    
    # wrap the translation of frequency peaks pixels to distances, angles and pixels in 1 line

    # Extract distances, angles, and pixel positions
    d_distances=PhaseIdent.Spot_coord_To_d_spacing_vect(pixels_of_peaks, FFT_calibration, total_pixels_image)
    angles_to_x=PhaseIdent.Spot_coord_To_Angles_to_X_vect(pixels_of_peaks,total_pixels_image)
    refined_distances, refined_angles_to_x, refined_pixels=PhaseIdent.Ensure_Center_Diff(d_distances, angles_to_x, pixels_of_peaks)
    
    # Set the values of distances in angstroms as required by Prepare_exp_distances_angles_pixels and other funcs
    refined_distances=ImCalTrans.nm_to_Angstroms(refined_distances)
    
    # Refine distances, angles, and pixel positions
    refined_distances, refined_angles_to_x, refined_pixels=PhaseIdent.Prepare_exp_distances_angles_pixels(refined_distances, refined_angles_to_x,refined_pixels, min_d)
    
    # Get only the right half of the FFT and its positions, angles and pixels
    refined_distances, refined_angles_to_x, refined_pixels = Right_Half_FFT_Only(refined_distances, refined_angles_to_x, refined_pixels)  
    
    return refined_distances, refined_angles_to_x, refined_pixels



def Init_Unit_Cells(
        unit_cells_path):
    '''
    For all the .uce unit cells in the cells_path folder inputted, generate a crystal object
    to hold the simulation of the DP

    Parameters
    ----------
    cells_path : 

    Returns
    -------
    crystal_objects_list : list with Crystal objects containing the info of every unit cell in the path

    '''
    
    crystal_objects_list=[]
    space_group_list = []
    
    for unit_cell in os.listdir(unit_cells_path):
        unit_cell_path=unit_cells_path+'\\'+unit_cell
        # the argument must be the bytes to the path not the string (just encode the string)
        crystal_object = PhaseIdent.Crystal(bytes(unit_cell_path.encode()))
        crystal_objects_list.append(crystal_object)   
        
        # store unit cell information: space group
        unit_cell_text = open(unit_cell_path)
        
        for line in unit_cell_text:
            if line[:4] == 'RGNR':
                space_group = int(line[4:])
        
        space_group_list.append(space_group)
                
    return crystal_objects_list, space_group_list




def From_Image_to_ZA_CrystSpots_Wrapped_1st_PeakDetect(
        image_crop_hs_signal, image_crop_array, total_pixels_crop, crop_FOV, 
        crystal_objects_list, space_group_list, forbidden, min_d, tol):
    '''
    From atomic res image get the ZA and crystal spot pairs classified accordingly
    to get the list_refined_cryst_spots
    The input should be the signal/array from the crop of the atomic image within the segmented crop
    The peak finding is with the first approximation done with the regression of the 2D Gaussian fit

    Returns
    -------
    list_refined_cryst_spots.

    '''

    # Get FFT and its calibration from the hs signal  of the crop
    FFT_image_array, FFT_calibration, FFT_units = FFT_from_crop(
        image_crop_hs_signal, image_crop_array)
    
    
    # Find the pixels of the freqs
    pixels_of_peaks = Pixels_peaks_FFT(
        FFT_image_array,crop_FOV)
    
    # Get functional distances, angles, pixel positions out of the peaks of the freqs
    refined_distances, refined_angles_to_x, refined_pixels = Ready_Distances_Angles_Pixels(
        pixels_of_peaks, FFT_calibration, total_pixels_crop)
    
    # Reference array for reference of the spots
    spots_int_reference = np.arange(len(refined_distances))
    
    # All spot pairs are evaluated depending on the ZA that is found for each of them and sorted from
    # better axis to worse
    all_scored_spot_pairs = PhaseIdent.Score_all_spot_pairs(
        crystal_objects_list, space_group_list, forbidden, min_d, tol, 
        refined_distances, refined_angles_to_x)
    
    # Copy the previous array as the following needs to be modified and create a list for storing the crystals
    all_scored_spot_pairs_to_mod = all_scored_spot_pairs.copy()
    
    # Initialise list for storing the outputs
    list_all_cryst_spots=[]
    
    # Evaluate all the scored spot pairs with ZA and phase to find all the crystals that were identified within an FFT
    PhaseIdent.Find_All_Crystals(
        list_all_cryst_spots, all_scored_spot_pairs_to_mod)
    
    # From all phases found and their respective pixels, check spot per spot if it was attributed to a phase,
    # and assign a phase to them until no spot remains unassigned (from best score phases to worse)
    list_refined_cryst_spots = PhaseIdent.Crystal_refinement(
        list_all_cryst_spots,spots_int_reference)
    
    # Remove the [000] axis determined by the no axis found
    list_refined_cryst_spots = PhaseIdent.Discard_Wrong_Crystals(
        list_refined_cryst_spots)
    
    # To print the info from the list_of_refined_crystals
    # for cryst in list_refined_cryst_spots:
    #     print('Cryst')
    #     print('spot list', cryst.spots)
    #     print('spot pairs', cryst.spot_pairs_obj)
    #     print('phase name', cryst.phase_name)
    #     print('ZA', cryst.ZA)
    #     print('ZA priv index', cryst.ZA_priv_index)

    return list_refined_cryst_spots



def Peak_Detection_Wrapped(
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
    
    # First check if the !st approx peak finding finds 1,2 or more pixels
    # as this would tell if we have an amorphous material (1st central spot),
    # a non identifiable material (central and 1 additional), or some crystal identifiable
    
    pixels_of_peaks = Pixels_peaks_FFT(
        FFT_image_array, crop_FOV)
    
    print(pixels_of_peaks)
    
    print('Pixels of peaks, first')
    print(pixels_of_peaks)
    
    if pixels_of_peaks.shape[0] > 1:
        # if it finds more than 1 pixel, the central and others (not amorphous)
        # recompute the found pixels with the scanning method
        
        #Detect the peaks in experimental FFT image without logarithm
        peaks_in_matrix = Peaks_detector.peaks_detector(FFT_image_array_No_Log)
        
        print('FFT no log')
        
        print(FFT_image_array_No_Log, np.shape(FFT_image_array_No_Log))
        
        print('peaks matrix')
        print(peaks_in_matrix, np.shape(peaks_in_matrix))
        
        
        # Extract the pixel coordinates
        pixels_of_peaks = Peaks_detector.peaks_image_to_coords(peaks_in_matrix)
        
        if visualisation == True:
            # For visualisation, overlaping the positions with the FFT, make the peaks bigger
            # peak_visual_size = 1
            # big_peaks = Peaks_detector.cercles(peak_visual_size, peaks_in_matrix)
            # plt.imshow(FFT_image_array, cmap='gray')
            # plt.imshow(big_peaks, cmap='Blues')
            # plt.show()
            
            # Better visualisation with the overlapping of FFT and peaks
            Peaks_detector.show_peaks_detected(
                FFT_image_array_No_Log, peaks_in_matrix, ratio = 1, plot = visualisation)
            
        
        
    #else
        # pixels_of_peaks.shape[0] == 1 then amorphous material, keep the found pixels like this
    
    
    
    # Exception to prevent the detection of useles pixels
    # Define hyperparam regarding the max number of spots that can be detected
    # as it is very unlikely that the more spots are meaningful
    # !!! HYPERPARAM
    max_spots_detected = 30
    if pixels_of_peaks.shape[0] > max_spots_detected:
        # sort them from the closest spots to the center to the furthests
        y_vects = np.abs(pixels_of_peaks[:,0]-int(total_pixels_crop/2))
        x_vects = np.abs(pixels_of_peaks[:,1]-int(total_pixels_crop/2)) 
        
        FFT_distance_pixels=np.sqrt(x_vects**2+y_vects**2)
        
        pixels_of_peaks_sorted=[pixel_coord for _,pixel_coord in sorted(zip(FFT_distance_pixels,pixels_of_peaks), key=lambda pair: pair[0])]
        
        pixels_of_peaks = np.asarray(pixels_of_peaks_sorted[0:max_spots_detected])
        
    


    print('Pixels of peaks, after')
    print(pixels_of_peaks)
    
    # Show the FFT (log) with the overlapping found positions
    
    #Try this functoin from Ivan's built in module
    # Peaks_detector.show_peaks_detected(fft, peaks, ratio = 1, plot = False)
    
    
    return pixels_of_peaks


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
    print('FFT no log')
    print(FFT_image_array_No_Log, np.shape(FFT_image_array_No_Log))
    
    print('FFT log')
    print(FFT_image_array, np.shape(FFT_image_array))
    
    
    # new_FFT = np.zeros((np.shape(FFT_image_array)[0]+1, np.shape(FFT_image_array)[0]+1))
    # new_FFT[:np.shape(FFT_image_array)[0], :np.shape(FFT_image_array)[0]] = FFT_image_array
    
    # FFT_image_array = new_FFT
    
    # print('New FFT')
    # print(FFT_image_array, np.shape(FFT_image_array))
    
    # Load CNN model
    device = torch.device("cuda")
    
    
    CNN_model = torch.load(r'E:\Arxius varis\PhD\4rth_year\Code\Functions\Ivans_Files\DL2.pt')
    print(type(CNN_model))
    

    #CNN prediction
    CNN_prediction, _ = CNN_model.predict(FFT_image_array)
    del CNN_model
    print('CNN_prediction')
    print(CNN_prediction, np.shape(CNN_prediction))
    
    print('CNN prediction plot')
    plt.imshow(CNN_prediction[0])
    plt.show()
    
    

    #Final prediction output
    # CNN hyperparams
    CNN_side = 31 #It has to be a odd number
    CNN_treshold= 0.69 #It has to be less than 1
    
    # Get the binary matrix with the peaks found 
    peaks_in_matrix, prediction_No_background = Peaks_detector.peaks_recognition(
        CNN_side, CNN_treshold, CNN_prediction, np.shape(FFT_image_array)[0])


    print('peaks matrix')
    print(peaks_in_matrix, np.shape(peaks_in_matrix))
    print('peaks in matrix where 1')
    print(np.where(peaks_in_matrix == 1))
    # Extract the pixel coordinates
    pixels_of_peaks = Peaks_detector.peaks_image_to_coords(peaks_in_matrix)

    # FFT_resized = cv2.resize(prediction[0], dsize=(size, size), interpolation=cv2.INTER_LINEAR)

    Peaks_detector.show_peaks_detected(
        FFT_image_array_No_Log, peaks_in_matrix, ratio = 0.8, plot = visualisation)
    # First check if the !st approx peak finding finds 1,2 or more pixels
    # as this would tell if we have an amorphous material (1st central spot),
    # a non identifiable material (central and 1 additional), or some crystal identifiable
   
    
    return pixels_of_peaks






# Find all crystals given the ZAs and scores, function that works fine
def Find_All_Crystals_DEBUG(list_all_cryst_spots, all_scored_spot_pairs):
    '''
    The refined distances and refined angles to x  and all scored spot pairs should be arrays that 
    are a copy of the original arrays and
    can therefore be modified without the fear of permanently modifying the original
    
    '''
    # in case nothing is contained in all_scored_spot_pairs
    if len(all_scored_spot_pairs)==0:
        print('No spots found')
        #end the function
        return
    
    if len(all_scored_spot_pairs)==1:
        # Only one spot pair
        scored_spot_target=all_scored_spot_pairs[0]
        spot_pair_target_index1=scored_spot_target.spot1_int_ref
        spot_pair_target_index2=scored_spot_target.spot2_int_ref
    
        # Initialise the crystal spots object
        crystal_identified=PhaseIdent.Crystal_spots([spot_pair_target_index1,spot_pair_target_index2])
        crystal_identified.Spot_pairs([scored_spot_target])
        spot_pair_target_phase=scored_spot_target.phase_name
        crystal_identified.Phase(spot_pair_target_phase) 
        spot_pair_target_ZA=scored_spot_target.ZA
        crystal_identified.ZA(spot_pair_target_ZA, 0)
        
        list_all_cryst_spots.append(crystal_identified)
        print('Only one spot pair found')
        #end the function
        return
        
    
    else:
        scored_spot_target=all_scored_spot_pairs[0]
    # contains the internal reference for the spots for each ZA and phase
    spot_int_refs_phase_axis=[]
    # spot target characteristics
    spot_pair_target_index1=scored_spot_target.spot1_int_ref
    spot_pair_target_index2=scored_spot_target.spot2_int_ref
    spot_pair_target_ZA=scored_spot_target.ZA
    spot_pair_target_phase=scored_spot_target.phase_name
    spot_int_refs_phase_axis.append(spot_pair_target_index1)
    spot_int_refs_phase_axis.append(spot_pair_target_index2)
    
    all_scored_spot_pairs=all_scored_spot_pairs[1::]
    
    # contains tuples of spot pairs containing the internal references of possible spots referencing to the same crystal
    spot_pairs_int_refs_possibe=[(spot_pair_target_index1,spot_pair_target_index2)]
    
    # the index of the spot pairs from the modified all scored spots, as will be needed to refer back to the 
    # spot pair which contained the given tuple of internal references of spots to delete it from the main list of all
    index_spot_pairs_int_refs_possibe=[]
    
    for index_scored_pair_eval, scored_spot_pair_eval in enumerate(all_scored_spot_pairs):
        # spot evaluated characteristics
        spot_pair_eval_index1=scored_spot_pair_eval.spot1_int_ref
        spot_pair_eval_index2=scored_spot_pair_eval.spot2_int_ref
        spot_pair_eval_ZA=scored_spot_pair_eval.ZA
        spot_pair_eval_phase=scored_spot_pair_eval.phase_name
        
        # if the spot pairs belong to same identified phase
        if spot_pair_target_phase==spot_pair_eval_phase:
            # if the ZA found is the same, as if it is not for sure it is not the same structure
            # meaning that all the h k l of both ZAs are coincident
            if spot_pair_target_ZA[0]==spot_pair_eval_ZA[0] and spot_pair_target_ZA[1]==spot_pair_eval_ZA[1] and spot_pair_target_ZA[2]==spot_pair_eval_ZA[2]:
                # then it is a possible candidate to belong to the same group of spots: aka refer to the same crystal
                
                # then load the information of the spot pairs internal references to check if the added spots
                # correspond to the crystal or not
                tuple_of_candidate_spot_pairs=(spot_pair_eval_index1, spot_pair_eval_index2)
                spot_pairs_int_refs_possibe.append(tuple_of_candidate_spot_pairs)
                index_spot_pairs_int_refs_possibe.append(index_scored_pair_eval)
    
    # evaluate the tuples included in the list of possible spot pairs:
    # Condition: at least two of the total tuples must contain one of the internal references set at the
    # list containing all the internal references for the spots of the crystal: given the internal reference of
    # the target spot1 and 2, to append a new plane (its reference) to the list, it must fultill that the new spot is 
    # contained in at least two other tuples of the list: if we have target spot 1 and 2= 1 and 2, to incorporate
    # a possible spot 3, the following tuples must exist: (1,3) and (2,3). Then we incorporate 3 as a good
    # spot for the crystal: for incorporating 4, we need to have or: (1,4) and (2,4), or (1,4) and (3,4),
    # or (2,4) and (3,4). As this would mean that the new spot correlates with more than one of the spots found
    # before, ensuring this way that if there is a coincident spot with two misaligned crystals, this spot
    # can still lead to both axis and orientations!
    
    #unique elements in the list and remove the spot pair target int refs 1 and 2
    spot_pairs_int_refs_possibe=np.asarray(spot_pairs_int_refs_possibe)
    unique_spot_int_refs_poss=np.unique(spot_pairs_int_refs_possibe)


    # unique_spot_int_refs_poss=np.delete(unique_spot_int_refs_poss, np.where(unique_spot_int_refs_poss==spot_pair_target_index1))
    # unique_spot_int_refs_poss=np.delete(unique_spot_int_refs_poss, np.where(unique_spot_int_refs_poss==spot_pair_target_index2))
    #this array only contains the possible spot candidates and no longer has tuples but arrays
    #list of indexes of list all_scored_spot_pairs that must be deleted after all the checks and dictionary modifications
    # !!!ERROR HERE the pair used as a reference, indexed as 0 must be deleted and included 
    indexes_all_spot_pairs_to_delete=[0]
    #for each possible spot candidate to reference the crystal
    #does not find everything all the spot pairs
    
    #repeat the process to ensure all spots are considered
    #the higher the range surer we are that everything worked well
    #not very elegant solution but does the job
    # for i in range(4):
    #     for spot_int_ref_possible in unique_spot_int_refs_poss:
    #         total_spot_pairs_contain_poss=[]
    #         temp_index_all_scored_spot_pairs=[]
    #         #for each spot pair that may represent the crystal 
    #         for index_all_spot_pair_int_ref,spot_pair_int_ref_poss in zip(index_spot_pairs_int_refs_possibe,spot_pairs_int_refs_possibe):
    #             #if the possible spot candidate is in the possible crystal spot pair
    #             if spot_int_ref_possible in spot_pair_int_ref_poss:
    #                 #AND if any of the values already representing the crystal, in spot_int_refs_phase_axis, is present
    #                 # in the spot pair candidate
    #                 for already_spot_ref in spot_int_refs_phase_axis:
    #                     if already_spot_ref in spot_pair_int_ref_poss:
    #                         total_spot_pairs_contain_poss.append(spot_pair_int_ref_poss)
    #                         temp_index_all_scored_spot_pairs.append(index_all_spot_pair_int_ref)
    #             # !!! we should add here that even if it is not 2, if the total possible is 1 pair of spots 
    #             # and it is fulfilled,  then indclude it anyway in the crystal spot representations
    #         if len(total_spot_pairs_contain_poss)>=2:
    #             spot_int_refs_phase_axis.append(spot_int_ref_possible)
    #             indexes_all_spot_pairs_to_delete.extend(temp_index_all_scored_spot_pairs)
      
    
    #This strategy with the while loop and its reestart does the job as well as the repeated n times for,
    #but this time ensuring the good result without caring about the n, which could make it slower
    val=len(unique_spot_int_refs_poss)
    iter_=0
    while iter_<val:
        #variable to check if the loop must restart to consider new possible values
        breaker=len(indexes_all_spot_pairs_to_delete)
          
        spot_int_ref_possible=unique_spot_int_refs_poss[iter_]
        total_spot_pairs_contain_poss=[]
        temp_index_all_scored_spot_pairs=[]
        #for each spot pair that may represent the crystal 
        for index_all_spot_pair_int_ref,spot_pair_int_ref_poss in zip(index_spot_pairs_int_refs_possibe,spot_pairs_int_refs_possibe):
            #if the possible spot candidate is in the possible crystal spot pair
            if spot_int_ref_possible in spot_pair_int_ref_poss:
                #AND if any of the values already representing the crystal, in spot_int_refs_phase_axis, is present
                # in the spot pair candidate
                for already_spot_ref in spot_int_refs_phase_axis:
                    if already_spot_ref in spot_pair_int_ref_poss:
                        total_spot_pairs_contain_poss.append(spot_pair_int_ref_poss)
                        temp_index_all_scored_spot_pairs.append(index_all_spot_pair_int_ref)
            # !!! we should add here that even if it is not 2, if the total possible is 1 pair of spots 
            # and it is fulfilled,  then indclude it anyway in the crystal spot representations
        if len(total_spot_pairs_contain_poss)>=2:
            spot_int_refs_phase_axis.append(spot_int_ref_possible)
            indexes_all_spot_pairs_to_delete.extend(temp_index_all_scored_spot_pairs) 
            indexes_all_spot_pairs_to_delete=np.unique(np.asarray(indexes_all_spot_pairs_to_delete))
            indexes_all_spot_pairs_to_delete=list(indexes_all_spot_pairs_to_delete)
            #if extra spot pairs have been added to represent this crystal, then restart the loop to consider 
            #this new element added. If no element was added, meaning no new index corresponding to two already
            #present spot pairs, then just continue
            if len(indexes_all_spot_pairs_to_delete)==breaker:
                iter_=iter_+1
            else:
                iter_=0
        else:
            iter_=iter_+1
                
    #add stuff to class about the elements or spot pairs added, identified by the indexes_all_spot_pairs_to_delete
    # which are all the elements of all_scored_spot_pairs that belong to the crystal targeted by scored_spot_target
    # in case there are repeated indices
    indexes_all_spot_pairs_to_delete=np.asarray(indexes_all_spot_pairs_to_delete)
    indexes_all_spot_pairs_to_delete=np.unique(indexes_all_spot_pairs_to_delete)
    spot_int_refs_phase_axis=np.unique(np.asarray(spot_int_refs_phase_axis))

    crystal_identified=PhaseIdent.Crystal_spots(spot_int_refs_phase_axis)
    
    crystal_scored_spot_pairs = [all_scored_spot_pairs[i] for i in indexes_all_spot_pairs_to_delete]
    crystal_identified.Spot_pairs(crystal_scored_spot_pairs)
    crystal_identified.Phase(spot_pair_target_phase)    
    
    #check if the ZA with same phase already exists in the list to set the difference between 
    # same ZA but with different in-plane orientations
    ZA_priv_index_found=[]
    for crystal in list_all_cryst_spots:
        if crystal.phase_name==crystal_identified.phase_name and crystal.ZA_string==str(spot_pair_target_ZA):
            ZA_priv_index_found.append(crystal.ZA_priv_index)
            
    if len(ZA_priv_index_found)==0:
        #no previous axis with this phase was included, then index=0
        crystal_identified.ZA(spot_pair_target_ZA, 0)
    else:
        # there were already some crystals with this orientation and phase, so the new index must be the
        # highest one found (present already)+1
        ZA_priv_index_i=max(ZA_priv_index_found)+1
        crystal_identified.ZA(spot_pair_target_ZA, ZA_priv_index_i)
        
    
    list_all_cryst_spots.append(crystal_identified)
     
    #then delete the elements from the all scored spot pairs list (not with pop as indexes vary as you delete them)
    #first capture all the element sint he list and delete by element not by index
    for crystal_scored_spot_pair in crystal_scored_spot_pairs:
        all_scored_spot_pairs.remove(crystal_scored_spot_pair)      
            
        
        
    if len(all_scored_spot_pairs)==0:
        print('Finished!')
        # end function
        return
    if len(all_scored_spot_pairs)==1:
        #load this last spot pair, that has not been recognised to the global list of crystal objects and then quit function
        scored_spot_target=all_scored_spot_pairs[0]
        spot_pair_target_index1=scored_spot_target.spot1_int_ref
        spot_pair_target_index2=scored_spot_target.spot2_int_ref
        spot_pair_target_ZA=scored_spot_target.ZA
        spot_pair_target_phase=scored_spot_target.phase_name
        
    
        crystal_identified=PhaseIdent.Crystal_spots([spot_pair_target_index1,spot_pair_target_index2])

        crystal_identified.Spot_pairs([scored_spot_target])
        crystal_identified.Phase(spot_pair_target_phase)  
        
        #check if the ZA with same phase already exists in the list to set the difference between 
        # same ZA but with different in-plane orientations
        ZA_priv_index_found=[]
        for crystal in list_all_cryst_spots:
            if crystal.phase_name==crystal_identified.phase_name and crystal.ZA_string==str(spot_pair_target_ZA):
                ZA_priv_index_found.append(crystal.ZA_priv_index)
                
        if len(ZA_priv_index_found)==0:
            #no previous axis with this phase was included, then index=0
            crystal_identified.ZA(spot_pair_target_ZA, 0)
        else:
            # there were already some crystals with this orientation and phase, so the new index must be the
            # highest one found (present already)+1
            ZA_priv_index_i=max(ZA_priv_index_found)+1
            crystal_identified.ZA(spot_pair_target_ZA, ZA_priv_index_i)
            
        
        list_all_cryst_spots.append(crystal_identified)
        all_scored_spot_pairs.remove(scored_spot_target)

        # end function
        print('Finished!')
        return
    else:
        print('iter')
        Find_All_Crystals_DEBUG(list_all_cryst_spots, all_scored_spot_pairs)
    
    #we do not return anything as this fills the empty list provided
    return 
    


def From_Image_to_ZA_CrystSpots_Wrapped(
        image_crop_hs_signal, image_crop_array, total_pixels_crop, crop_FOV, 
        crystal_objects_list, space_group_list, forbidden, min_d, tol):
    '''
    From atomic res image get the ZA and crystal spot pairs classified accordingly
    to get the list_refined_cryst_spots
    The input should be the signal/array from the crop of the atomic image within the segmented crop
    The peak finding is the final one, to

    Returns
    -------
    list_refined_cryst_spots.

    '''
    
    # Get FFT no log and its calibration from the hs signal  of the crop
    FFT_image_array_No_Log, FFT_calibration, FFT_units = FFT_No_Log_from_crop(
        image_crop_hs_signal, image_crop_array)
    print(FFT_image_array_No_Log, np.shape(FFT_image_array_No_Log), FFT_calibration, FFT_units)
    
    # Get FFT and its calibration from the hs signal  of the crop
    FFT_image_array, FFT_calibration, FFT_units = FFT_from_crop(
        image_crop_hs_signal, image_crop_array)
    print(FFT_image_array, np.shape(FFT_image_array), FFT_calibration, FFT_units)
    
    # Find the pixels of the freqs
    pixels_of_peaks = Peak_Detection_Wrapped_CNN(
        FFT_image_array_No_Log, FFT_image_array, total_pixels_crop, crop_FOV, visualisation = True)
    
    # Get functional distances, angles, pixel positions out of the peaks of the freqs
    refined_distances, refined_angles_to_x, refined_pixels = Ready_Distances_Angles_Pixels(
        pixels_of_peaks, FFT_calibration, total_pixels_crop)
    
    # Reference array for reference of the spots
    spots_int_reference = np.arange(len(refined_distances))
    
    # All spot pairs are evaluated depending on the ZA that is found for each of them and sorted from
    # better axis to worse
    all_scored_spot_pairs = PhaseIdent.Score_all_spot_pairs(
        crystal_objects_list, space_group_list, forbidden, min_d, tol, 
        refined_distances, refined_angles_to_x)
    
    # Copy the previous array as the following needs to be modified and create a list for storing the crystals
    all_scored_spot_pairs_to_mod = all_scored_spot_pairs.copy()
    
    # Initialise list for storing the outputs
    list_all_cryst_spots=[]
    
    # Evaluate all the scored spot pairs with ZA and phase to find all the crystals that were identified within an FFT
    Find_All_Crystals_DEBUG(
        list_all_cryst_spots, all_scored_spot_pairs_to_mod)
    
    # From all phases found and their respective pixels, check spot per spot if it was attributed to a phase,
    # and assign a phase to them until no spot remains unassigned (from best score phases to worse)
    list_refined_cryst_spots = PhaseIdent.Crystal_refinement(
        list_all_cryst_spots,spots_int_reference)
    
    # Remove the [000] axis determined by the no axis found
    list_refined_cryst_spots = PhaseIdent.Discard_Wrong_Crystals(
        list_refined_cryst_spots)
    
    # To print the info from the list_of_refined_crystals
    for cryst in list_refined_cryst_spots:
        print('Cryst')
        print('spot list', cryst.spots)
        print('spot pairs', cryst.spot_pairs_obj)
        print('phase name', cryst.phase_name)
        print('ZA', cryst.ZA)
        print('ZA priv index', cryst.ZA_priv_index)
        print('hkl1',cryst.spot_pairs_obj[0].hkl1_reference, 'at', cryst.spot_pairs_obj[0].spot1_angle_to_x,
              'degrees to x axis')
        print('hkl2',cryst.spot_pairs_obj[0].hkl2_reference, 'at', cryst.spot_pairs_obj[0].spot2_angle_to_x,
              'degrees to x axis')
    return list_refined_cryst_spots

#%%
'''
Hyperparameter definition
'''
# High to Low hyperparams
real_calibration_factor=0.98  # change calibration of the image
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

dataset_system_path_name=r'E:\Arxius varis\PhD\4rth_year\Code\data\debug_dataset\\'
# dataset_system_path_name=r'E:\Arxius varis\PhD\2nd_year\Code\trial_images\full_device_STEM_datasets\SQ20-250-2\\'

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
images_to_segment, relative_positions_to_segment, pixel_sizes_to_segment = Images_to_Segment_Cutoff(
    pixel_size_segment_thresh, flat_images_in_dataset_by_pixel_size, relative_positions, flat_pixel_sizes)

# Perform the segmentation to the images that have been defined to be segmented (not too low mag but low
# enough to almost do not have atomic resolution)
images_segmented, conts_vertxs_per_region_segmented = Segment_Images(
    images_to_segment, relative_positions_to_segment, pixel_sizes_to_segment)


#%%

'''
Functions for finding out the high mag image within the segmented map 
'''

# Initialise the possible unit cells before going into the image per image analysis as the step is common
# Get the possible unit cells into a Crystal object to hold a DP simulation
unit_cells_path = r'E:\Arxius varis\PhD\3rd_year\Code\unit_cells'

# Initialise the crystal objects for every possible unit cell in the path of possible cells
crystal_objects_list, space_group_list = Init_Unit_Cells(unit_cells_path)

# Limit the cells for the trials

# crystal_objects_list = crystal_objects_list[0:3]
# space_group_list = space_group_list[0:3]


# list to store all the Analysed_Image objects with all the images, crops and ZA found
list_analysed_images = []

# From the list of images_in_dataset objects, relative positions and pixel sizes, ordered 1 to 1
# we can proceed with the atomic identification    



# set pixel size below which images are considered: for TITAN and 2k images
# setting it to a bit smaller ( < 0.075 or below may help as the preivous is on the edge of atomic res)
max_pixel_size_atom_res = 0.11  # !!! HYPERPARAMETER


print(flat_images_in_dataset_by_pixel_size)

for image_in_dataset in flat_images_in_dataset_by_pixel_size[2:3]:
    
    if image_in_dataset.x_calibration <= max_pixel_size_atom_res:
        
        # BEWARE if all the images within flat images are sent to analysis, the first image, lowest mag,
        # WILL NEVER be a template and the algorithm will fail
        # this is assessed by the following exception
        if image_in_dataset == flat_images_in_dataset_by_pixel_size[0]:
            # if the image considered is the first one in the global dataset, the lowest mag one,
            # which does not have a query and cannot be a template, then skip this iteration
            continue

        # indent the full process here

        # Initialyse the Analysed_Image object to store the realtive coords of the crops within the image in dataset
        # and the found ZA within the crop
        analysed_image_obj = Analysed_Image(image_in_dataset)
        
        # Get query image, which is the lowest mag image that was segmented and is query of the target image        
        query_image = Find_Segmented_Query(
            image_in_dataset, list_of_best_query_template_pairs, images_to_segment)        
            
        
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
            minimum_FOV = Atomic_Resolution_WindowsAnalysis_Cutoff(
                image_in_dataset.x_calibration)
            
            # Only proceed with the phase identification if the minum FOV given the pixel size is met
            # as if it is not met, then no meaningful information would be extracted and the crop would be useless
            
            if crop_FOV >= minimum_FOV:
                # From atomic res image crop get the ZA and crystal spot pairs classified accordingly
                # to get the list_refined_cryst_spots
                print(image_crop_hs_signal, image_crop_array, total_pixels_crop)
                list_refined_cryst_spots = From_Image_to_ZA_CrystSpots_Wrapped(
                    image_crop_hs_signal, image_crop_array, total_pixels_crop, 
                    crop_FOV, crystal_objects_list, space_group_list, 
                    forbidden, min_d, tol)
                
                # After the analysis of the atomic resolution crop, store the info into the initialised Analysed_Image class
                analysed_image_obj.Add_Crop_Output(
                    image_crop_hs_signal, scaled_reference_coords, 
                    image_relative_coords, list_refined_cryst_spots)
                  
        # Add the analysed image object to the list of analysed images objects to correspond 1 to 1 to the image in datset
        list_analysed_images.append(analysed_image_obj)
                




