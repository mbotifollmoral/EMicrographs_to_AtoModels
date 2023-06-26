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
    
    print('Pixels of peaks, first')
    print(pixels_of_peaks)
    
    if pixels_of_peaks.shape[0] > 1:
        # if it finds more than 1 pixel, the central and others (not amorphous)
        # recompute the found pixels with the scanning method
        
        #Detect the peaks in experimental FFT image without logarithm
        peaks_in_matrix = Peaks_detector.peaks_detector(FFT_image_array_No_Log)
        
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


    
    

    #Final prediction output
    # CNN hyperparams
    CNN_side = 31 #It has to be a odd number
    CNN_treshold= 0.69 #It has to be less than 1
    
    # Get the binary matrix with the peaks found 
    peaks_in_matrix, prediction_No_background = Peaks_detector.peaks_recognition(
        CNN_side, CNN_treshold, CNN_prediction, np.shape(FFT_image_array)[0])


    # Extract the pixel coordinates
    pixels_of_peaks = Peaks_detector.peaks_image_to_coords(peaks_in_matrix)

    # FFT_resized = cv2.resize(prediction[0], dsize=(size, size), interpolation=cv2.INTER_LINEAR)
    plt.title('pixel size '+str(crop_FOV/total_pixels_crop)+' FOV '+str(crop_FOV))
    Peaks_detector.show_peaks_detected(
        FFT_image_array_No_Log, peaks_in_matrix, ratio = 0.8, plot = visualisation)
    plt.show()
    # First check if the !st approx peak finding finds 1,2 or more pixels
    # as this would tell if we have an amorphous material (1st central spot),
    # a non identifiable material (central and 1 additional), or some crystal identifiable
   
    
    return pixels_of_peaks
















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
    
    # Get FFT and its calibration from the hs signal  of the crop
    FFT_image_array, FFT_calibration, FFT_units = FFT_from_crop(
        image_crop_hs_signal, image_crop_array)
    
    
    # Find the pixels of the freqs
    # pixels_of_peaks = Peak_Detection_Wrapped(
    #     FFT_image_array_No_Log, FFT_image_array, total_pixels_crop, crop_FOV, visualisation = True)
    
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

dataset_system_path_name=r'E:\Arxius varis\PhD\2nd_year\Code\trial_images\full_device_STEM_datasets\SQ20-250-2\\'
dataset_system_path_name=r'E:\Arxius varis\PhD\2nd_year\Code\trial_images\full_device_STEM_datasets\Qdev439_InAs120nm_Al full 25nm\NW1\\'
dataset_system_path_name=r'E:\Arxius varis\PhD\2nd_year\Code\trial_images\full_device_STEM_datasets\QT543AlNb\\'

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
crystal_objects_list = crystal_objects_list[0:3]
space_group_list = space_group_list[0:3]
# list to store all the Analysed_Image objects with all the images, crops and ZA found
list_analysed_images = []

# From the list of images_in_dataset objects, relative positions and pixel sizes, ordered 1 to 1
# we can proceed with the atomic identification    



# set pixel size below which images are considered: for TITAN and 2k images
# setting it to a bit smaller ( < 0.075 or below may help as the preivous is on the edge of atomic res)
max_pixel_size_atom_res = 0.11  # !!! HYPERPARAMETER

for image_in_dataset in flat_images_in_dataset_by_pixel_size:
    
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
                list_refined_cryst_spots = From_Image_to_ZA_CrystSpots_Wrapped(
                    image_crop_hs_signal, image_crop_array, total_pixels_crop, 
                    crop_FOV, crystal_objects_list, space_group_list, 
                    forbidden, min_d, tol)
                
                # After the analysis of the atomic resolution crop, store the info into the initialised Analysed_Image class
                analysed_image_obj.Add_Crop_Output(
                    image_crop_hs_signal, scaled_reference_coords, 
                    image_relative_coords, list_refined_cryst_spots)
                  
        # Add the analysed image object to the list of analysed images 
        # objects to gather them and compare them with the
        # image in dataset list of images, the analysed_image_obj already 
        # holds the image_in_dataset where the crops are taken from
        list_analysed_images.append(analysed_image_obj)
                




#%%

def Map_Analysed_Phases(
        list_analysed_images, lowest_mag_relative_coords):
    
    # format of coords  
    # (x_relative_position_start, y_relative_position_start, x_relative_position_end, y_relative_position_end)
    
    (x_low_start, y_low_start, x_low_end, y_low_end) = lowest_mag_relative_coords
    
    fig, ax = plt.subplots()
    fig.set_size_inches(16*4, 18*4)
    ax.scatter(np.array([x_low_start,x_low_end]), np.array([y_low_start,y_low_end]))
    
    
    
    for analysed_image in list_analysed_images:
        # loop through the crops
        for crop_ind in range(analysed_image.crop_index):
            crop_rel_cords = analysed_image.Crop_outputs[str(crop_ind)+'_rel_cords']
            crop_phases = analysed_image.Crop_outputs[str(crop_ind)+'_list_refined_cryst_spots']
            # name the phases depending on the name found
            if len(crop_phases) == 0:
                phase_name = 'No phase'
            elif len(crop_phases) == 1:
                phase_name = str(crop_phases[0].phase_name) + str(crop_phases[0].ZA)
            else:
                phase_name = str(crop_phases[0].phase_name) + str(crop_phases[0].ZA) +' and ' + \
                str(crop_phases[1].phase_name) + str(crop_phases[1].ZA)
            
            (x_start, y_start, x_end, y_end) = crop_rel_cords
            x_cord = x_start + (x_end - x_start)/2
            y_cord = y_low_end - (y_start + (y_end - y_start)/2)
            
            ax.scatter(x_cord, y_cord)
            ax.annotate(phase_name, (x_cord, y_cord))
    
    return 


Map_Analysed_Phases(
    list_analysed_images, relative_positions[0])
    

#%%


    
class Segmented_Image_Regions_with_Phase:
    '''
    Class having, for a segmented image, a dictionary in which each element corresponds to one of 
    the found (discretised) regions with its corresponding label and all the list of phases that were 
    found within the bounds of the discretised region of the particular segmented image considered
    The list of phases can come from all the analysed images that found a crop that lies within
    these limits of the label of the region
    The dictionary self adds the labels and the number of list of phases found for a label in
    the followig form: Regions_Phases[str(int(label_of_region)) + '_' + str(new_index) + '_PhasesList']
    '''
    
    
    def __init__(self, image_in_dataset, segmented_array):
        self.image_in_dataset = image_in_dataset
        self.segmented_array = segmented_array
        
        self.Regions_Phases = dict()
    
    def Add_PhasesList_to_Region(
            self, label_of_region, crop_hs_signal_list_refined_cryst_spots):
        ''' 
        Parameters
        ----------
        The dictionary self adds the labels and the number of list of phases found for a label in
        the followig form: Regions_Phases[str(int(label_of_region)) + '_' + str(new_index) + '_PhasesList']
        and each element added is a list with hs signal of the crop (1st element:[0]) and 
        list of the cryst_spots objects (the rest of the elements: [1:])
        '''
        # account for the previous phases added to the region with number label_of_region
        new_index = 0
        for region_phase in self.Regions_Phases:           
            region_number = int(region_phase[:region_phase.find('_')])
            if region_number == label_of_region:
                new_index = new_index + 1
        
        self.Regions_Phases[str(int(label_of_region)) + '_' + str(new_index) + '_PhasesList'] = crop_hs_signal_list_refined_cryst_spots
                



def All_Segmented_Images_and_Regions_with_Phases(
        images_to_segment, images_segmented, 
        relative_positions_to_segment, list_analysed_images):
    
    '''
    The function finds, for all the images that have been segmented, all the 
    lists_of_cryst_phases that lie within a region with a given label N
    This is, if the cropped windows where the crystal phases were identified,
    its central position /coordinate lies within the spatial domains of a given
    label N, then those are included as Segmented_Image_Regions_with_Phase 
    So the output is a list of Segmented_Image_Regions_with_Phase objects, one
    for each segmented image

    Parameters
    ----------
    images_to_segment : image_in_dataset object
    images_segmented : array, segmented (with the labels) of the image_in_dataset 
    relative_positions_to_segment : array with the rel positions of the segmented images
    list_analysed_images : list of all the Analysed_Image objects with all
                        the atomic resolution info
                        
    Returns
    -------
    list_segmented_im_region_phases : list of Segmented_Image_Regions_with_Phase
                                      objects for all the segmented images 

    '''
        
    
    # initialise the Segmented_Image_Regions_with_Phase classes for every 
    # segmented image that is within the list of images to segment
    # to correlate 1 to 1 with the other lists of segmented elements
    list_segmented_im_region_phases = []
     
    for image_to_segment, segmented_array  in zip(images_to_segment, images_segmented):
        
        segmented_im_region_phases = Segmented_Image_Regions_with_Phase(image_to_segment, segmented_array)
        list_segmented_im_region_phases.append(segmented_im_region_phases)
    
       
    # map the found phases through the regions of every segmented images 
    # so that every segment in the image has the phases that were mapped inside it
    
    for analysed_image in list_analysed_images:
        # loop through the crops
        for crop_ind in range(analysed_image.crop_index):
            
            crop_rel_cords = analysed_image.Crop_outputs[str(crop_ind)+'_rel_cords']
            crop_phases = analysed_image.Crop_outputs[str(crop_ind)+'_list_refined_cryst_spots']
            center_cords = analysed_image.Crop_outputs[str(crop_ind) + '_center_cords']
            image_crop_hs_signal = analysed_image.Crop_outputs[str(crop_ind) + '_hs_signal']
            
            
            for image_to_segment, relative_pos_seg, segmented_array, segmented_im_region_phases in zip(
                    images_to_segment, relative_positions_to_segment,
                    images_segmented, list_segmented_im_region_phases):
                
                (x_start_seg, y_start_seg, x_end_seg, y_end_seg) = relative_pos_seg
                
                # select all the segmented images which fit the central position within their coords
                if (center_cords[0] <= x_end_seg and 
                    center_cords[0] >= x_start_seg) and (
                        center_cords[1] <= y_end_seg and center_cords[1] >= y_start_seg):
                    # then the position fits within the segmented image
                    
                    # find the equivalent pixel of the central coordinate in the segmented image
                    # to find the label (region) in which it belongs to
                    # as we want to group the found phases per region and this is only solved in
                    # the single crop per segmented image
                    
                    # the range should go from 0 to total pixels-1 as it is a coordinate not a crop range
                    x_pixel_center_cord = int(round((image_to_segment.total_pixels-1)*(
                        (center_cords[0] - x_start_seg)/(x_end_seg - x_start_seg))))
                    y_pixel_center_cord = int(round((image_to_segment.total_pixels-1)*(
                        (center_cords[1] - y_start_seg)/(y_end_seg - y_start_seg))))
        
                    # get the label from the segmented array in the found position
                    region_value = segmented_array[y_pixel_center_cord, x_pixel_center_cord]
                    
                    # add the phase information to the label called region_value
                    # in the list_segmented_im_region_phases 
                    
                    # To correlate the phases we extracted with the respective crop and its pixel size, FOV
                    # As we will sort the relevance of the extracated phase with the pixel size and FOV of crop
                    crop_phases_hs_signal = [image_crop_hs_signal] + crop_phases 
                    segmented_im_region_phases.Add_PhasesList_to_Region(region_value, crop_phases_hs_signal)
                    
    
    return list_segmented_im_region_phases




class Correlated_Region_t_q:
    
    def __init__(
            self, template_image, query_image):
        self.template = template_image
        self.query = query_image
        self.CorrRegions = []
        
    def AddCorrRegions(
            self, template_region, corr_query_region):
        '''
        Generate a string that names the template region considered with its integer, and relates it 
        to the corresponding integer in the query
        The string is created like this: 't_X_q_Y' where X is the integer in the template, and
        Y is the integer in the query

        Parameters
        ----------
        template_region : integer, referring to a label (region) in the template image
        corr_query_region : integer, referring to a label (region) in the query image

        '''
        string_name = 't' + str(int(template_region)) + '_q' + str(int(corr_query_region))
        
        self.CorrRegions.append(string_name)


 
def Correlate_Regions_between_SegImages(
        images_to_segment, images_segmented, list_of_best_query_template_pairs,
        flat_images_in_dataset_by_pixel_size, relative_positions):
    
    '''
    Function for every template - query pair of segmented images, to correlate
    the regions of the template, labeled as integers, with the corresponding label
    of the query labeled as another integer. 
    It is done by, after doing the template matching of the segmetned template
    within the segmented query, the region in the template defined as N will
    correlate to the region in the query labeled as M because for all the pixels 
    in template with value N, most of he spatially coincident in the query 
    will be of class M, meaning that the region labeled N in template is the 
    same (probably just a crop) but labeled with M in the query
    We create strings with names tN_qM to express this correlation
    We generate a list of Correlated_Region_t_q objects for every image that
    is segmented and that is not the big reference one (lowest mag), as the next
    function considers these correlation with the big ref segmented image

    Parameters
    ----------
    images_to_segment : 
    images_segmented : 
    list_of_best_query_template_pairs : 
    flat_images_in_dataset_by_pixel_size : 
    relative_positions : 

    Returns
    -------
    list_corr_region_t_q : list of Correlated_Region_t_q objects for all
            the images except the big ref segmented one

    '''
    

    list_corr_region_t_q = []
    # for all segmented images except the first one which has no query
    # images_segmented has the arrays, images_to_segment has the object
    for image_to_segment, segmented_image in zip(images_to_segment[1::], images_segmented[1::]):
        
        # find the query of the image to segment
        for query_template_match in list_of_best_query_template_pairs:
            if query_template_match.template == image_to_segment:
                query_image = query_template_match.query
                
        # find the arrays for both the target image and the query one
        # for the target image it is directly segmented_image
        segmented_query = images_segmented[np.where(images_to_segment == query_image)[0][0]]
        
        crop_image_array_ups, image_relative_coords = Correlate_Segmented_Atomic_q_t_Images(
            image_to_segment, query_image, flat_images_in_dataset_by_pixel_size,  
            relative_positions, images_to_segment, images_segmented)
        
        # !!! Limit the array of possible labels as the 0 is attributed to noise
        seg_image_regions = np.unique(segmented_image)[1::]
        
        # initialise the Correlated_Region_t_q class to keep the track on the correlated regions
        # between integers of the template and query
        corr_region_t_q = Correlated_Region_t_q(
            image_to_segment, query_image) 
        
        for region in seg_image_regions:
            
            segmented_image_bin = np.copy(segmented_image)
            
            segmented_image_bin[segmented_image != region] = 0
            segmented_image_bin[segmented_image == region] = 1
    
            seg_query_most_commonval = np.argmax(np.bincount(crop_image_array_ups[segmented_image_bin == 1]))
            
            if seg_query_most_commonval == 0:
                # if most found is 0, get the 2nd most found
                bincarray = np.bincount(crop_image_array_ups[segmented_image_bin == 1])
                bincarray[np.argmax(bincarray)] = 0
                seg_query_most_commonval = np.argmax(bincarray)
                
            corr_region_t_q.AddCorrRegions(
                region, seg_query_most_commonval)
        
        
        list_corr_region_t_q.append(corr_region_t_q)
    
 
    return list_corr_region_t_q



def Correlate_SegRegions_with_ReferenceSeg(
        images_to_segment, list_corr_region_t_q, images_segmented,
        list_segmented_im_region_phases):
    '''
    For all the functions correlating the regions of the segmented regions
    between each other we are expecting that a final big (lowest mag) 
    reference segmented image comes at the end to represent a good segmentation
    of the device. Here in this function it is when all the correlated informations
    from all the crystaline outputs and their relation with the segmented images
    coincide with the big reference image.
    The output is very similar to All_Segmented_Images_and_Regions_with_Phases
    for the big ref image, which contains all cryst phases the windows where they 
    were taken from have a center in the different regions of this image, but here
    it is more precise as considers all the images with the different magnificaitons
    progressively, and it even maps the phases list that come from each image
    in case a sorting or classificatoin based on the image is used later

    Parameters
    ----------
    images_to_segment : 
    list_corr_region_t_q : list of Correlated_Region_t_q objects for all
            the images except the big ref segmented one
    images_segmented : 
    list_segmented_im_region_phases : list of Segmented_Image_Regions_with_Phase
                                      objects for all the segmented images 

    Returns
    -------
    dict_ph_found_region : 
    dict_ph_found_region_with_image : 
        
    # dict_ph_found_region in each keyword it contains the lists of all the phases that 
    # lead to the query N in big ref segm image from all the semgented images
    # dict_ph_found_region_with_image does the same but in every list containg the phases, 
    # the first term of the list is the image_in_dataset object where they come from
    # and the second elememt is the hyperspy signal of the crop to detect the pixel size and FOV
    # where the exact FFT and crystallogrpahic extraction was performed
    # so we have a reference of where they come from (magnification) and we can sort
    # the ouputs given a criteria on the image the were set (i.e. higher importance to 
    # higher magnificatoin images that are segmented)
   
    '''
    

    # Correlate the labels of the different images and the crystal phases found
    # to link the segmented images (all except the big query reference one)
    # acting as templates to their segmented query, and to the final big 
    # reference query so each of the labels in the big ref query matches 1 or more regions 
    # from the other higiher mag segmented images acting like templates
    
    
    
    
    tot_list_of_corr_regions_stacked = []
    # we invert the order to have the templates (more high mag) first
    for image_to_segment in images_to_segment[::-1][:-1]:
        # start with the higher mags segm images, searching for the label in the big segm image of reference
        reference_segm_image = images_to_segment[0]
        
        list_of_corr_regions_stacked = []
        
        for corr_region_t_q in list_corr_region_t_q:
            # loop though all the possible queries
            if corr_region_t_q.template == image_to_segment:
                template_image = corr_region_t_q.template
                query_image = corr_region_t_q.query
                while query_image != reference_segm_image:
                    list_of_corr_regions_stacked.append(corr_region_t_q.CorrRegions)
                    
                    #update query image
                    for corr_region_t_q2 in list_corr_region_t_q:
                        if corr_region_t_q2.template == query_image:
                            query_image = corr_region_t_q2.query
                            corr_region_t_q = corr_region_t_q2
                
                list_of_corr_regions_stacked.append(corr_region_t_q.CorrRegions)     
                            
        
        #here we have the stacked list of corr regions to correlate the label values from the first to the final
        tot_list_of_corr_regions_stacked.append(list_of_corr_regions_stacked)     
    
    #initialise all the dicts 
    # dict_ph_found_region in each keyword it contains the lists of all the phases that 
    # lead to the query N from all the semgented images
    # dict_ph_found_region_with_image does the same but in every list containg the phases, 
    # the first term of the list is the image_in_dataset object where they come from
    # so we have a reference of where they come from (magnification) and we can sort
    # the ouputs given a criteria on the image the were set (i.e. higher importance to 
    # higher magnificatoin images that are segmented)
    dict_ph_found_region = dict()
    dict_ph_found_region_with_image = dict()
    
    seg_image_regions = np.unique(images_segmented[0])
    
    for region in seg_image_regions:
        dict_ph_found_region[str(int(region))] = []
        dict_ph_found_region_with_image[str(int(region))] = []
    
                
    for stacked_Corr_region, first_template_image in zip(
            tot_list_of_corr_regions_stacked, images_to_segment[::-1][:-1]):
        
        starting_reg = stacked_Corr_region[0]            
        
        if len(stacked_Corr_region) == 1:
            # this means the template directly correlates with the query reference big image
            for corr_region_str in starting_reg:
                
                template_reg = int(corr_region_str[corr_region_str.find('t')+1:corr_region_str.find('_')])
                query_reg = int(corr_region_str[corr_region_str.find('q')+1:])
                
                for segm_im_reg_ph in list_segmented_im_region_phases:
                    if segm_im_reg_ph.image_in_dataset == first_template_image:
                        for region_ph in segm_im_reg_ph.Regions_Phases:
                            
                            region_ph_name = int(region_ph[:region_ph.find('_')])
                            
                            if region_ph_name == template_reg:
                                
                                 dict_ph_found_region[str(int(query_reg))].append(
                                   segm_im_reg_ph.Regions_Phases[region_ph][1:])
                                 # we create a list that contains the phases of 
                                 # segm_im_reg_ph.Regions_Phases[region_ph] and then sum 
                                 # the name of the template image in dataset object
                                 # to correlate the phases iwth the image and crop they come from
                                 list_ph_with_image= [first_template_image] + segm_im_reg_ph.Regions_Phases[region_ph]
                                 dict_ph_found_region_with_image[str(int(query_reg))].append(
                                   list_ph_with_image)
                                 
        else:
            # there is a tree of stacked template-query then stack the elements from 
            # the first one (first template), to the last one, the big query
            # the first template is the first_template_image 
            
            # list of stacked strings relating the first template to last query (big ref image)
            # strings like tN_qX-tL_qY-....tM_qZ, meaning that the template from first template query pair
            # region labeled as N correlates to region Z in the last reference big query,
            # i.e. in reference region from reference image big segmented, label Z must include 
            # the crystal phases found in region N from the first image template indicated
            global_stacked_template_query_pairs = []
            
            total_template_query_pairs = len(stacked_Corr_region)
                    
            for corr_region_str in starting_reg:
                # for all the labels in the very first template  
                start_template_reg = int(corr_region_str[corr_region_str.find('t')+1:corr_region_str.find('_')])
                start_query_reg = int(corr_region_str[corr_region_str.find('q')+1:])
                new_string_stacked = corr_region_str
                
                template_reg1 = start_template_reg 
                query_reg1 = start_query_reg
                
                for template_query_pair in stacked_Corr_region[1:]:
                    # for all template query pairs except the first one which is starting_reg
                    
                    for corr_region_str2 in template_query_pair:
                        # for the rest of template pairs that do not contain the very first template
                        template_reg2 = int(corr_region_str2[corr_region_str2.find('t')+1:corr_region_str2.find('_')])
                        query_reg2 = int(corr_region_str2[corr_region_str2.find('q')+1:])
                        
                        if template_reg2 == query_reg1:
                            # the query from the firs t q pair is the template from the second t q pair
                            new_string_stacked = new_string_stacked + '-' + corr_region_str2
                            query_reg1 = query_reg2
                
                global_stacked_template_query_pairs.append(new_string_stacked)  
                
            for stacked_string in global_stacked_template_query_pairs:
                
                template_reg = int(stacked_string[stacked_string.find('t')+1:stacked_string.find('_')])
                query_reg = int(stacked_string[::-1][:stacked_string[::-1].find('q')][::-1])
                    
                for segm_im_reg_ph in list_segmented_im_region_phases:
                    if segm_im_reg_ph.image_in_dataset == first_template_image:
                        for region_ph in segm_im_reg_ph.Regions_Phases:
                            
                            region_ph_name = int(region_ph[:region_ph.find('_')])
                            
                            if region_ph_name == template_reg:
                                # Append only the list of phases and not the crop info
                                dict_ph_found_region[str(int(query_reg))].append(
                                    segm_im_reg_ph.Regions_Phases[region_ph][1:])
                                # we create a list that contains the phases of 
                                # segm_im_reg_ph.Regions_Phases[region_ph] and then sum 
                                # the name of the template image in dataset object
                                # to correlate the phases iwth the image and crop they come from
                                list_ph_with_image= [first_template_image] + segm_im_reg_ph.Regions_Phases[region_ph]
                                dict_ph_found_region_with_image[str(int(query_reg))].append(
                                   list_ph_with_image)
       
    # the resulting lists, dict_ph_found_region will have for all the crops considered a list, 
    # even if it is empty, whihc means no phase was found inside this crop,
    # and in the dict_ph_found_region_with_image these empty lists are not empty but
    # hold the image and crop where they are taken, and no phase as of course no 
    # phase is found on them 
    # basically dict_ph_found_region allows for a faster phase comparison and 
    # dict_ph_found_region_with_image stands for a compared and physically classified 
    # phase extraction or distinction between the possible guesses
        
    return dict_ph_found_region, dict_ph_found_region_with_image
    


# Get Segmented_Image_Regions_with_Phase for all the segmented images
# to locate the atomic res info with the position of the segmented regions
list_segmented_im_region_phases = All_Segmented_Images_and_Regions_with_Phases(
    images_to_segment, images_segmented, 
    relative_positions_to_segment, list_analysed_images)


# Correlate the regions between segm images to later correlate the
# atomic res information extracted for each label in each segm image
list_corr_region_t_q = Correlate_Regions_between_SegImages(
    images_to_segment, images_segmented, list_of_best_query_template_pairs,
    flat_images_in_dataset_by_pixel_size, relative_positions)


# Correlate the segm regions with the reference big/lowest mag segm image
# and get all the crsytal phases found for each label and the segm image
# where they are identified (considering the central position of the windows
# where the FFT was taken at), as the highest mag segm images will be more
# precise to locate the central coord of a windows within a region

dict_ph_found_region, dict_ph_found_region_with_image = Correlate_SegRegions_with_ReferenceSeg(
    images_to_segment, list_corr_region_t_q, images_segmented,
    list_segmented_im_region_phases)
 
    
#%%   
                   
 
# now we have dict_ph_found_region, which is a dictionary with keywords 'N', for
# every label or unique integer in the segmented map of the lowest mag image 
# segment (big semtned image reference) , and for each N we have the crystal phases,extracted from
# the higher mag segmentation, that correlate to that given N (directly extracted from the atomic res images)
# which is more reliable based on the information on the high mag images      
# it should be very close output to the list_segmented_im_region_phases, element 
# of the list for which Segmented_Image_Regions_with_Phase.image_in_dataset == biggest image segmented
                
# we also have have dict_ph_found_region_with_image, which is a dictionary with keywords 'N', 
# as the preivous one, but in each list with the phases, the first element in the imaeg_in_dataset
# object where the phases were taken from, and the second element is the crop hyperspy signal that
# allows us to see which pixel size and FOV the data was taken from
# as this will be the criteria to score or sort the obtained crystallogrpahic output
# as this way we can sort the information
# based on a given criteria based on the image that they were taken from, and then
# for all lists, lists[0] = original image, lists[1] = crop hs signal,  lists[2:] = phases found in this crop
        



#%%

# Segmentation of the lowest mag image of reference with the
# higher mag images, important function: Segmented_Borders_Combiner

def Stack_Correlated_t_q_pairs(
        images_to_segment, images_segmented, list_corr_region_t_q):
    
# the output is a list of lists, in which each list has two elements, first the images that
# are going to be changed and paralelly the other element the label of that image
# that has to be changed, then each first elemnt which contains the images, will have a stack of 
# image template next query next query ... final query and paralley the labels, and again the same
# but this time changing the labels as the first image template will change the template
# of interest to work with a new one                
    
    images_to_change = []
    
    
    # iterate through template
    for image_to_segment, segm_image in zip(
            images_to_segment[::-1][:-1], images_segmented[::-1][:-1]):
        
        images_to_change_per_image = []
        
        labels_to_change = []
        
        for corr_region in list_corr_region_t_q:
            ident = 0
            templimg = corr_region.template
            queryimg = corr_region.query
            
            if templimg == image_to_segment:
                
                # iterate through label in image
                for t_q_label in corr_region.CorrRegions:
                    
                    # templimg = corr_region.template
                    # queryimg = corr_region.query
                    queryimg = corr_region.query
    
                    images_to_change_per_image.append(templimg)
                    images_to_change_per_image.append(queryimg)
                    
                    templ_label = int(t_q_label[1:t_q_label.find('_')])
                    query_label = int(t_q_label[t_q_label.find('q')+1:])
                    
                    labels_to_change.append(templ_label)
                    labels_to_change.append(query_label)
                    
                    while queryimg != images_to_segment[0]:
                        
                        for corr_region2 in list_corr_region_t_q:
                            
                            templimg2 = corr_region2.template
                            queryimg2 = corr_region2.query
                            
                            if templimg2 == queryimg:
                                
                                images_to_change_per_image.append(queryimg2)
                                
                                for t_q_label2 in corr_region2.CorrRegions:
                                    
                                    templ_label2 = int(t_q_label2[1:t_q_label2.find('_')])
                                    
                                    if templ_label2 == query_label:
                                    
                                        query_label2 = int(t_q_label2[t_q_label2.find('q')+1:])
                                        
                                        labels_to_change.append(query_label2)
                                        queryimg = queryimg2
                                        ident=1
                        if ident == 1:
                            break
                        
        images_to_change.append([images_to_change_per_image, labels_to_change])                
                        
                    
        
    #  to read the output images_to_change:
        
    # for ims_labels, image_to_segment in zip(
    #         images_to_change, images_to_segment[::-1][:-1]):
        
        
    #     first_templ = ims_labels[0][0]
        
    #     list_images = np.asarray(ims_labels[0])
    #     indexes_firsttempl = np.where(list_images == first_templ)[0]
    #     indexes_firsttempl = list(indexes_firsttempl) + [len(ims_labels[0])]

    #     for ind_enu, main_image_index in enumerate(indexes_firsttempl[:-1]):
            
    #         next_image_index = indexes_firsttempl[ind_enu + 1]
            
            
    #         images_in_dataset_cons = ims_labels[0][main_image_index:next_image_index]
    #         labels_cons = ims_labels[1][main_image_index:next_image_index]
    
    
    
    return images_to_change
    
                

# Draw the borders of a segmented image for all the regions and borders of 
# all the images, one for each image, and generating a binary image


def Plot_Images_Stack_Scaling_Segmented_Borders(
        images_to_segment, relative_positions_to_segment, images_segmented):
    '''
    Creates a super high res low mag image and inserts the other images in the positions
    where the template matching stated they fit the best, with their corresponding rescaling.
    The rescaling and super upsampling has to be done very limitedly as doing it raw makes
    the memory to explode. e.g. 16TB required to allocate the upscaling of a single very low 
    mag image with a very high mag pixel size

    Parameters
    ----------
    images_in_dataset_by_pixel_size : TYPE
        DESCRIPTION.
    relative_positions : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    
    flat_images_in_dataset_by_pixel_size=images_to_segment
    
    #Delete the so big lowest mag one to try
    # the too low magnification images are limited so the reconstruction is memory reasonable
    relative_positions_copy=relative_positions_to_segment.copy()
    
    # flat_images_in_dataset_by_pixel_size=flat_images_in_dataset_by_pixel_size
    # relative_positions_copy=relative_positions.copy()
    
    new_xref=relative_positions_copy[0][0]
    new_yref=relative_positions_copy[0][1]
    
    coordinates_mod=np.zeros((len(relative_positions_copy),4))
    for coordinate,relative_position in zip(coordinates_mod,relative_positions_copy):
        coordinate[0]=relative_position[0]-new_xref
        coordinate[1]=relative_position[1]-new_yref
        coordinate[2]=relative_position[2]-new_xref
        coordinate[3]=relative_position[3]-new_yref
        
    range_total=coordinates_mod[0][2]-coordinates_mod[0][0]
    step=flat_images_in_dataset_by_pixel_size[len(flat_images_in_dataset_by_pixel_size)-1].x_calibration
   
    lowmag_total_upsampled_pixels=int(np.ceil(range_total/step))
    
    grid_side = np.linspace(0,range_total,lowmag_total_upsampled_pixels)
    
    #downscale images to fit in memory
    fixed_downscale_factor=1
    dimside=int(np.ceil(len(grid_side)/fixed_downscale_factor))
    dim = (dimside, dimside)
    
    # resize image
    resized_global = cv2.resize(images_segmented[0], dim, interpolation = cv2.INTER_NEAREST)
    # plt.figure(figsize=(100,100))
    # plt.imshow(resized_global)
    # plt.show()
    
    # loop through all the images and scale them up to fit the pixels numbers and position given by
    # relative positions array
    
    for coordinate,image_i, segm_im in zip(
            coordinates_mod[1:],flat_images_in_dataset_by_pixel_size[1:],images_segmented[1:]):
        step_i=step*fixed_downscale_factor
        range_total=coordinate[2]-coordinate[0]
        
        total_pixels_im=int(np.ceil(range_total/step_i))
        dim_i= (total_pixels_im, total_pixels_im)

        resized_im = cv2.resize(segm_im, dim_i, interpolation = cv2.INTER_NEAREST)
        
        startX_p=int(np.ceil(coordinate[0]/step_i))
        startY_p=int(np.ceil(coordinate[1]/step_i))
        endX_p=startX_p+total_pixels_im
        endY_p=startY_p+total_pixels_im
        
        resized_global[startY_p:endY_p,startX_p:endX_p]=resized_im
    
    plt.figure(figsize=(100,100))
    plt.imshow(resized_global)  
    plt.show()
    
    return resized_global



def Segmented_Borders_Combiner(
        images_to_segment, relative_positions_to_segment, 
        images_segmented, conts_vertxs_per_region_segmented):
    '''
    Function to generate, from the segmented images and the corresponding
    arrays, the segmentation of the lowest magnification image based on
    the template matching of the images with their borders on the areas
    that have been segmented
    This generates a common low mag map of 0s and 1s with 1s being the borders
    and 0s the background. Then the regions that are separated between these 
    borders are the segmetned regions of the low mag image
    
    Returns:
    the final result of the segmentation, with a given number of pixels
    that depends on the magnficiations of the images used, binary map
    that must be processed to find the regions separated by the 1s
   '''
    borders_images = []
    
    for image_segmented, cont_vert_reg_seg in zip(
            images_segmented, conts_vertxs_per_region_segmented):
        
        image_segm_size = np.shape(image_segmented)[0]
        borders_image = np.zeros((image_segm_size, image_segm_size))
        
        labels_seg_image = np.unique(image_segmented)
        if labels_seg_image[0] == 0:
            labels_seg_image = labels_seg_image[1:]
        
        for label in labels_seg_image:
            
            for contour_vector in cont_vert_reg_seg[str(int(label))+'_'+'contours']:
                
                init_rel_coords_y,init_rel_coords_x = contour_vector.init_coords
                final_rel_coords_y,final_rel_coords_x = contour_vector.final_coords
                
                if final_rel_coords_x != 0 and final_rel_coords_y != 0 and final_rel_coords_x != image_segm_size and final_rel_coords_y != image_segm_size:
                    
                    borders_image[final_rel_coords_y, final_rel_coords_x] = 1
              
                
        # add the pixels in the borders when they belong to an edge
        
        # check North margin, check the 2nd and 3rd rows
        for index_n_p, north_pixel in enumerate(borders_image[1]):
            if north_pixel == 1:
                # if the third in the same position is also a 1, then change to 1
                if borders_image[2, index_n_p] == 1:
                    borders_image[0, index_n_p] = 1
                # if the arround the 2nd one there are 0s, then add it
                if index_n_p - 1 >= 0 and index_n_p + 1 < image_segm_size:
                    if borders_image[1, index_n_p - 1] == 0 and borders_image[1, index_n_p + 1] == 0:
                        borders_image[0, index_n_p] = 1
            
        
        # check South margin, check the before last and and before before last
        for index_s_p, south_pixel in enumerate(borders_image[image_segm_size-2]):
            if south_pixel == 1:
                # if the before before last in the same position is also a 1, then change to 1
                if borders_image[image_segm_size-3, index_s_p] == 1:
                    borders_image[image_segm_size-1, index_s_p] = 1
                # if the arround the before last one there are 0s, then add it
                if index_s_p - 1 >= 0 and index_s_p + 1 < image_segm_size:
                    if borders_image[image_segm_size-2, index_s_p - 1] == 0 and borders_image[image_segm_size-2, index_s_p + 1] == 0:
                        borders_image[image_segm_size-1, index_s_p] = 1
        
        
        # check West margin, check the second and third columns
        for index_w_p, west_pixel in enumerate(borders_image[:,1]):
            if west_pixel == 1:
                # if the third in the same position is also a 1, then change to 1
                if borders_image[index_w_p, 2] == 1:
                    borders_image[index_w_p, 0] = 1
                # if the arround the 2nd one there are 0s, then add it
                if index_w_p - 1 >= 0 and index_w_p + 1 < image_segm_size:
                    if borders_image[index_w_p - 1, 1] == 0 and borders_image[index_w_p + 1, 1] == 0:
                        borders_image[index_w_p, 0] = 1
              
                    
        # the East margin should already be considered as the contours are
        # written acording to pixel len(image) which is the 
        # right top contour of the last pixel      
        # check East margin, check the before and before before last columns
        
        # for index_e_p, east_pixel in enumerate(borders_image[:,image_segm_size-2]):
        #     if east_pixel == 1:
        #         # if the before before last in the same position is also a 1, then change to 1
        #         if borders_image[index_e_p, image_segm_size-3] == 1:
        #             borders_image[index_e_p, image_segm_size-1] = 1
        #         # if the arround the before last one there are 0s, then add it
        #         if index_e_p - 1 >= 0 and index_e_p + 1 < image_segm_size:
        #             if borders_image[index_e_p - 1, image_segm_size-2] == 0 and borders_image[index_e_p + 1, image_segm_size-2] == 0:
        #                 borders_image[index_e_p, image_segm_size-1] = 1
        
        
        borders_images.append(borders_image)
        plt.figure(figsize=(10,10))
        plt.imshow(image_segmented)
        plt.show()        
        plt.imshow(borders_image)
        plt.show()        
                    
        
    resized_global = Plot_Images_Stack_Scaling_Segmented_Borders(
        images_to_segment, relative_positions_to_segment, borders_images)

        
    return resized_global
   
   
            

resized_global = Segmented_Borders_Combiner(
    images_to_segment, relative_positions_to_segment, 
    images_segmented, conts_vertxs_per_region_segmented)



# Missing generate the labels from these images, 
# by checking the regions that are separted form one to each other





#%%





def score_crop(
        crop_phases):
    '''
    From the crop_phases list, which is a list that has as the first element
    the image where the crop was taken from, as image_in_dataset format, the 
    second element is the actual crop information in hyperspy signal format
    and the rest of elements are the list of refined phases for this crop
    sorted accordingly from more likely to less

    Parameters
    ----------
    crop_phases : TYPE
        DESCRIPTION.

    Returns
    -------
    ghost_crop_score : TYPE
        DESCRIPTION.

    '''
    
    def crystalline_score_crop_criteria(
            crop_pixel_size, crop_FOV):
        # define proper criteria por scoring the crops 
        
        
        missing
        
        
        if crop_pixel_size >= 1:
            
            ghost_crop_score = 1
        else:
            ghost_crop_score = 2
            
        return ghost_crop_score
    
    
    def amorphous_score_crop_criteria(
            crop_pixel_size, crop_FOV):
        # define proper criteria por scoring the crops 
        
        missing
        
        
        if crop_pixel_size >= 1:
            
            ghost_crop_score = 1
        else:
            ghost_crop_score = 2
            
        return ghost_crop_score
    
    
    
    image_in_dataset = crop_phases[0]
    crop_hs_signal = crop_phases[1]
    
    if len(crop_phases) == 2:
        # by now this is the amorhpous encoding if nothing changes, 
        # specially if we do not rethink the function Discard_Wrong_Crystals
        ghost_crop_score = 1
        
        
        # check how many empty (no phase detected) arrays are there in the
        # given label of the region
        # if most of the arrays are empty, then likely that it is actually amorphous
        # to double check the score of the identified ones can be high/bad
        # corresponding to wrongly identified spots (noise spots that often
        # arise in the amorphous images)
        # The ideas is the same, to score in terms of the crop size and the field of 
        # view, but just to limit and give extra importance or credit to the
        # pixel sizes that we have checked are good for the amorhphous case
        
        
        # if we find no phases, score them depending on how the amorphous peak finding works:
        # what pixel sizes and FOc ranges work better and so on
            
        missing
        
        
        
        ghost_crop_score = amorphous_score_crop_criteria(crop_pixel_size, crop_FOV)
        
    else:
        # there is, by now, at least a non amorophous phase identified 
        # check the pixel size and apply 
        
        # !!!only get score from the first element (most likely) in list for each crop
        # at least for the 1st Approx
        missing
        
        ghost_crop_score = crystalline_score_crop_criteria(crop_pixel_size, crop_FOV)

    return ghost_crop_score




crop_scores_per_region = dict()

for region in dict_ph_found_region_with_image:
    list_crops_phases_in_region = dict_ph_found_region_with_image[region]
    # give every image and crop a score to be then sorted in terms of likelyhood 
    # to detect the proper phase and that good peak finding was done
    # this list goes 1 to 1 with the list_of_phases_list
    scores_list_per_crop = []
    
    # loop through different crops
    for crop_phases in list_crops_phases_in_region:
        
        

        image_in_dataset = crop_phases[0]
        crop_hs_signal = crop_phases[1]
        
        
        ghost_crop_score = score_crop(crop_phases)
        # YES OR YES we need to attach some info to the scores_list
        scores_list_per_crop.append(ghost_crop_score)
    
    crop_scores_per_region[region] = scores_list_per_crop



def check_amorphous(
        list_crops_phases_in_region, scores_list_per_crop):
    
    missing
    
    taking into accoun how the scores are given when the arrays come form ana amorphous mateiral
    
    check for all crop phase in list_crops_phases_in_region
    
    sort the scores from lowest to highest (more to less likely)
    and then evaluate if they come from a crop with only two elememtns
    and make differnet criteria to consider it amorphous
    
    
    like
    
    if 50% of crops are amorhpous then amorphos:
    if the socre of the three lowest score crops are amorphous then amorphous:
    if the crystalline scores are higher than a typically good score (chekc this possible score):
        then it is amorphous
            
        else:
            
    
    amorphous_bool = True
    
    return amorphous_bool
    
def check_crystalline_phase(
        list_crops_phases_in_region, scores_list_per_crop):
    # !!! this would be a first approximation stuff as only detects a single 
    # per region, and also comes from just a single phase extracted per crop
    # from the score_crop functoin, that only considers the most likely phase
    
    
    missing
    
    return Crystal_spots object final


# once all the scores are given and assigned to every crop and every image, then
# we should decide or try to distinghius first if the region is amorhpous or
# crystalline, and if crystalline, then find the phase
 
final_phase_per_region = dict()

for region in dict_ph_found_region_with_image:
    
    
    list_crops_phases_in_region = dict_ph_found_region_with_image[region]

    scores_list_per_crop = crop_scores_per_region[region]
    
    amorphous_bool = check_amorphous
    
    if amorphous_bool == True:
        # generate Crystal_spots object with just indicating it is amorphous
        # and that only the central spot is detected
        amorphous_crystal_spot = PhaseIdent.Crystal_spots(['Central Spot Only'])
        
        amorphous_crystal_spot.Phase('Amorphous')
        
        final_phase_per_region[region] = amorphous_crystal_spot
        
    else:
        # it is crystalline
        final_crystal_phase_region = check_crystalline_phase()
        # crystal phase must be a Crystal_spots object with all the 3D info
        # of the phase name axis and the crystal orientatoin
        
        final_phase_per_region[region] = final_crystal_phase_region
    

















                


# for all the querys not big image and templates


            
#             working with the query images should already be fine as
#             the other template ones cna be or not queris but the important 
#             ones are the ones that act as query (from whihc we directly work and
#                                                  assess the template images)

            
            
            
    
    

# !!! now we have to put in comon the crystalline information, and get a phase out of this,
# so ideally we should test both the ZA and the indexation of the dp together to see if it is
# the same orientation of the crystla or not

# work with these dictionaries contianing the segmented info and the phases per region
# dict_ph_found_region, dict_ph_found_region_with_image
# the second one constians the filtering depengin on the segmented image the phases  were found from
# so the center of the windows was in that region of that segmented image (image in the list is the segmented)
# and the second  element is the hs signal of the crop where the phases were found, fin the elements by
# for all lists, lists[0] = original image, lists[1] = crop hs signal,  lists[2:] = phases found in this crop
        

# no need to sort them again as they are sorted already inside each list_all_cryst_spots 
# from the most likely crystal to least, and this should be enough...moreover, the comparison of 
# errors from different crystals differnely calibrated is not as straightofrward,
# so just find the regions and correlate them to each other and find the crystal phase that is more common 
# to the different places...

#     cryst_scores=[]
#     for cryst in list_all_cryst_spots:
#         cryst_score=0
#         for spot_pair in cryst.spot_pairs_obj:
#             spot_pair_score=spot_pair.score
#             cryst_score=cryst_score+spot_pair_score
#         total_spot_pairs=len(cryst.spot_pairs_obj)
#         cryst_scores.append(cryst_score/total_spot_pairs**2)
        
        
        
    
# Beware that the function Discard_Wrong_Crystals(list_refined_cryst_spots)
# is removeing all the cyrstlas that are identified asa 000 axis, whihc is no phase detected,
# which indeed could mean they are amorhpous, then adress this as the amorphous identificaitons 
# is also important        
    
    
        
# give higher priority to the images that have been semgente dat lower pixel sizes
# as the location of the phases center at these is much more reliable, as the sizes at big pixel sizes
# make that one pixel is a lot of variation and can make a position to be mislocated and switch towards
# another of the labels/regions

# then, by giving this high priority we can maybe build the segments from the highes mag images segmented, and
# then when a segment is defined we build this one as if it was correct and the we correlate it with 
# the semgentned image of reference to say that this particular region should belong to a phase 
# correlating it by comparing the area that is coinciding between regions and they coidncide above a given 
# percentatge then they represent the same area/region and then we can correlate the phase with the region        
        
 
    
 
    
 

    
 
 
    
