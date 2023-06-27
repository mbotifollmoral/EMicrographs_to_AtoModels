# -*- coding: utf-8 -*-
"""
Created on Wed Jan 11 13:51:45 2023

@author: Marc
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import sys
import skimage.measure
from scipy.signal import argrelextrema
import os
import sys

# !!! NEED to set the path to 
# Alg_Comb_Single_Image_Strain.py
# as the console working directory
Project_main_path = os.getcwd()
if 'EMicrographs_to_AtoModels' in Project_main_path:
    Project_main_path = Project_main_path[:Project_main_path.find('EMicrographs_to_AtoModels')-1]
# Project_main_path has the EMicrographs_to_AtoModels folder
sys.path.append(Project_main_path)

from EMicrographs_to_AtoModels.Functions.General_functions import Segmentation_1stAprox as Segment
from EMicrographs_to_AtoModels.Functions.General_functions import Filters_Noise as FiltersNoise

from EMicrographs_to_AtoModels.Functions.PeakDet_Segment.Segmentation_model import SG_Segmentation_algorithms as SegmAlgs


# Segmentation functions in the combined algorihtm 1st aprox

def Images_to_Segment_Cutoff(
        pixel_size_segment_thresh, flat_images_in_dataset_by_pixel_size,
        relative_positions, flat_pixel_sizes):
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
        conts_vertx_per_region = Segment.Contour_draw_All_regions(array_to_segment)
        
        # Convert to relative coordinates (with respect the lowest mag image) the vertexs and contours
        conts_vertx_per_region= Segment.Relative_Vertexs_Contours(
            array_to_segment, conts_vertx_per_region, 
            relative_position_to_segment, pixel_size_to_segment)
        
        # append the dictionary 
        conts_vertxs_per_region_segmented.append(conts_vertx_per_region)
    
    return images_segmented, conts_vertxs_per_region_segmented


def Ensure_Contiguous_Labels(image_segmented):
    '''
    Function to ensure that the labels of the segmented image go from 1 to n
    when they have n different labels
    The image must have NO 0s in it
    as otherwise the 0 pixels will be considered as a real region
    '''
    
    old_labels = np.unique(image_segmented)
    n_labels = len(np.unique(image_segmented))
    new_labels = np.arange(1,n_labels+1, 1)
    
    for old_lab, new_lab in zip(old_labels, new_labels):
        
        image_segmented[image_segmented==old_lab] = new_lab
        
    
    return image_segmented


def Upscale_Segmented_Steps(
        image_segmented, original_size):
    '''
    Function to upscale the downscaled segmented image in step to increase the
    reliability of the contours originally found

    Parameters
    ----------
    image_segmented : of course, the segmetned version of the image when it is
                        downscaled
    original_size : int, stating the original size of the image before downsampling

    Returns
    -------
    image_resized : rescaled segmente dimage after the stepped upscaling

    '''
    
    downs_size = image_segmented.shape[0]
    step_resize = 2
    
    orig_ov_down = original_size/downs_size
    org_ov_down_steps = int(orig_ov_down//step_resize)
    
    image_resized = np.copy(image_segmented)
    
    for step in range(1, org_ov_down_steps + 1):
        
        partial_size = downs_size*step
        image_resized = cv2.resize(image_resized, (partial_size, partial_size), interpolation = cv2.INTER_NEAREST)
       
        plt.imshow(image_resized)
        plt.show()
        
    image_resized = cv2.resize(image_resized, (original_size, original_size), interpolation = cv2.INTER_NEAREST)
    
    plt.imshow(image_resized)
    plt.show() 
        
    return image_resized



def Segment_Images_ContourBased(
        images_to_segment, relative_positions_to_segment, 
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
    n_pixels_for_contour = 256
    
    # lists containing the segmented images and the conts and vertex of the segmented regions per image
    images_segmented=[]
    conts_vertxs_per_region_segmented=[]
    
    for image_to_segment, relative_position_to_segment, pixel_size_to_segment in zip(
            images_to_segment, relative_positions_to_segment, pixel_sizes_to_segment):
        array_to_segment=image_to_segment.image_arraynp_st
        
        # Automated segmentation
        image_segmented = SegmAlgs.automate_segmentation(
            image_to_segment.hyperspy_2Dsignal, number_of_pixels = n_pixels_for_contour, plot = False)
                     
        # Re-scale the image to its original size
        size_down=(array_to_segment.shape[0],array_to_segment.shape[0])
        image_segmented= cv2.resize(image_segmented, size_down, interpolation = cv2.INTER_NEAREST)
        
        # progressively rescale the image back to its original size
        # DOES NOT HELP, SAME RESULT WITH SINGLE UPSCALING
        # image_segmented = Upscale_Segmented_Steps(image_segmented, image_to_segment.total_pixels)

        # Delete any possible 0 remaining after upscaling
        image_segmented = SegmAlgs.remove_noise(image_segmented)
        
        # Ensure the labels are contiguous
        image_segmented = Ensure_Contiguous_Labels(image_segmented)

        # Append the image into list
        images_segmented.append(image_segmented)
        
        # Compute the contours and vertexs of the image reshaped
        conts_vertx_per_region = Segment.Contour_draw_All_regions(image_segmented)
        
        # Convert to relative coordinates (with respect the lowest mag image) the vertexs and contours
        conts_vertx_per_region = Segment.Relative_Vertexs_Contours(
            image_segmented, conts_vertx_per_region, 
            relative_position_to_segment, pixel_size_to_segment)
        
        
        '''
        # !!! Super SLOW works and should be used as it ensures there is a univoc
        # path thruogh the contours but it is too slow now Denoise_All_Regions_other_labels
        # is too limiting as it is too slow nested loops
        
        # Ensure the path is univoc
        conts_vertx_per_region_shortest, conts_vertx_per_region_longest = Segment.Conts_verts_per_region_shortest_longest_path(
            image_segmented, conts_vertx_per_region)
        
        # Fill possible pixels within the found divergent paths
        final_extra_pixs = Segment.Denoise_All_Regions_other_labels(
            image_segmented, conts_vertx_per_region_longest)

        for label in final_extra_pixs:
            for pixel in final_extra_pixs[label]:
                image_segmented[pixel[0],pixel[1]]=int(label[0]) 
                        
        
        # Compute the contours and vertexs of the image reshaped
        conts_vertx_per_region = Segment.Contour_draw_All_regions(
            image_segmented)
        
        
        # Convert to relative coordinates (with respect the lowest mag image) the vertexs and contours
        conts_vertx_per_region = Segment.Relative_Vertexs_Contours(
            image_segmented, conts_vertx_per_region, 
            relative_position_to_segment, pixel_size_to_segment)
                
        '''
        
        # append the dictionary 
        conts_vertxs_per_region_segmented.append(conts_vertx_per_region)
    
    
    return images_segmented, conts_vertxs_per_region_segmented



#%%

# Functions originally developed for the automated BG stuff

def Merge_two_segm_regions(
        stacked_segmentation, label_1, label_2):
    
    
    smallest_label = min(label_1, label_2)
    largest_label = max(label_1, label_2)
    
    stacked_segmentation[stacked_segmentation == largest_label] = smallest_label
    
    # now all pixels of both regions have the label of the smallest number
    
    # we need to rescale and make the labels continous, with step of 1 int
    
    labels_after = np.unique(stacked_segmentation)
    
    new_stacked_segmentation = np.copy(stacked_segmentation)
    
    for new_label, label_aft in zip(
            range(len(labels_after)), labels_after):
        
        new_stacked_segmentation[stacked_segmentation == label_aft] = new_label+1
        
        
    return new_stacked_segmentation





def Skip_n_contours_region_region_intercon(
        conts_vertx_per_region, segmented_image, skip_n = 2):
    '''
    Given a set of contours of a segmented image with n labels, this function
    generates another subset of contours that skips skip_n contours from one
    to the other, making the contours longer and even diagonal, and what we want,
    smoother!
    It works with all the contours at the same time so it keeps track of the 
    contour merging that has been done on the previous segments/labels
    and on the parts where the contours of two regions match the same
    contour merging is done to avoid inconsistency. The edges of the labels 
    are also considered (triple points or boundaries) so that they are always
    kept as unmodified vertexs (adjusting if necessary the lenght of the contour) 

    Parameters
    ----------
    conts_vertx_per_region :  dict with contours vertex and rel vertexs
    segmented_image : array with segmented image
    skip_n : int, how many vertex are skept, actually from one vertex, to which it
                is next connected
        DESCRIPTION. The default is 2.

    Returns
    -------
    conts_vertx_per_region_skept : dict with contours vertex 
                                    after the merging of the skip_n contours
                                    The rel_vertex need to be computed 
                                    afterwards by providing the relative position 
                                    to the function Segment.Relative_Vertexs_Contours()
    
    '''
    
    # the process must be done for all the labels one after each other
    # as track of what has been done to the previous label is required
    
    labels = np.unique(segmented_image)
    
    conts_vertx_per_region_skept = dict()
    
    # this dictionary is created to on the loop hold the new contours
    # that will be used to link the regions with the contours that are crated
    # for another one in contact with this one
    # initialise it with every possible label to appear in the iterations
    cache_dict_hold_opposite_contours = dict()
    for label in labels:
        cache_dict_hold_opposite_contours[str(int(label))] = []
    # add also None for the outside of the image just in case
    cache_dict_hold_opposite_contours['None'] = []
        
    
    for label in labels:
    
        vertexs_list_label_iter = conts_vertx_per_region[str(int(label))+'_vertexs']
        contours_list_label_iter = conts_vertx_per_region[str(int(label))+'_contours']  
        rel_vertexs_list_label_iter = conts_vertx_per_region[str(int(label))+'_rel_vertexs'] 
            
        # we need to iterate over the contours_list_label_iter setting each as
        # the initial one and finding a new final one, and in the next iteration, 
        # ignore those that have been skept and use the new final one as the next 
        # initial one
        
        # list of the new contour objects elongated given the new contour merge
        new_contours_elongated = []
        
        # list of contours to iterate and pop given the skip requirements
        new_contours_found = contours_list_label_iter.copy() 

        # to avoid shifts in the position of the elongated contours, 
        # make sure that if there are elongated contours available
        # the loop starts with one of them so no shifts happen throught the path
        
        if len(cache_dict_hold_opposite_contours[str(int(label))]) > 0:
            
            # pick the first element in the cache_dict_hold_opposite_contours[str(int(label))]
            
            first_stor_contour_to_check = cache_dict_hold_opposite_contours[str(int(label))][0]
            
            init_cords_first_stor_contour_to_check = first_stor_contour_to_check.init_coords
            
            for ind, contour_to_start in enumerate(new_contours_found):
                if contour_to_start.init_coords == init_cords_first_stor_contour_to_check:
                    index_to_shift = ind
                    
            # now we shift the list so the element with index    index_to_shift
            # is now the element with index 0 
            
            new_contours_found_cache = new_contours_found[index_to_shift:] + new_contours_found[:index_to_shift]
            new_contours_found = new_contours_found_cache.copy()
            


        contour_found_loop_through = True
                
        for contour_found in new_contours_found:
            
            init_cont_m_in = contour_found.m_in
            init_cont_m_out = contour_found.m_out
            init_cont_init_coords = contour_found.init_coords
            init_cont_final_cords = contour_found.final_coords

            
            index_contour_found = new_contours_found.index(contour_found)
            
            # loop just when the list is not empty
            if len(cache_dict_hold_opposite_contours[str(int(label))]) > 0:
                
                found_starting_contour_stored = False
                
                for contour_stored in cache_dict_hold_opposite_contours[str(int(label))]:
                
                    init_cont_stor_m_in = contour_stored.m_in
                    init_cont_stor_m_out = contour_stored.m_out
                    init_cont_stor_init_coords = contour_stored.init_coords
                    init_cont_stor_final_cords = contour_stored.final_coords
                    
                
                    if init_cont_stor_init_coords == init_cont_init_coords:
                        
                        found_starting_contour_stored = True
                        
                        # then these initial coordinates have already been checked
                        # in the opposite direction with alternate materials
                        # if the intial coordiantes coincide then this contour 
                        #     should be added to the new_contours_elongated
                        
                        new_contours_elongated.append(contour_stored)
                        
                        # now delete the new contour founds till the
                        # final coordinates coincide 
                        
                        for l in reversed(range(0, skip_n+1)):
                        
                            # exception when going with indexes that are too high and
                            # would be out of the list end of list
                            
                            if index_contour_found + l <= len(new_contours_found)-1:
                                contour_final_possible = new_contours_found[index_contour_found + l]
                            else:
                                # just skip the iteration as it is leading to a
                                # contour that cannot be tracked afterwards
                                continue
                            
                            final_poss_cont_m_in = contour_final_possible.m_in
                            final_poss_cont_m_out = contour_final_possible.m_out
                            final_poss_cont_init_coords = contour_final_possible.init_coords
                            final_poss_cont_final_cords = contour_final_possible.final_coords
                            
                            if final_poss_cont_final_cords == init_cont_stor_final_cords:
                                
                                index_of_deletion = l
                                
                                for removal_ind in reversed(range(1, index_of_deletion+1)):
                                    new_contours_found.pop(index_contour_found+removal_ind)
                            
                                # break the loop through the contour_stored
                                break
                            
                    if found_starting_contour_stored == True:
                        break
                        
                        
                # if the contour was not stored then work it out normally    
                if found_starting_contour_stored == False:
    
                    for l in reversed(range(0, skip_n+1)):
                        
                        # exception when going with indexes that are too high and
                        # would be out of the list end of list
                        if index_contour_found + l <= len(new_contours_found)-1:
                            contour_final_possible = new_contours_found[index_contour_found + l]
                        else:
                            # just skip the iteration as it is leading to a
                            # contour that cannot be tracked afterwards
                            continue
                        
                        final_poss_cont_m_in = contour_final_possible.m_in
                        final_poss_cont_m_out = contour_final_possible.m_out
                        final_poss_cont_init_coords = contour_final_possible.init_coords
                        final_poss_cont_final_cords = contour_final_possible.final_coords
                        
                        if init_cont_m_out == final_poss_cont_m_out:
                            
                            elongated_contour = Segment.contour_vector(
                                final_poss_cont_m_in, final_poss_cont_m_out, 
                                init_cont_init_coords, final_poss_cont_final_cords)
                            
                            new_contours_elongated.append(elongated_contour)
                            
                            # we make an elongated contour that is in opposite sense
                            # to the main one to gather it as the another label main contour
                            elongated_contour_reversed = Segment.contour_vector(
                                final_poss_cont_m_out, final_poss_cont_m_in,
                                final_poss_cont_final_cords, init_cont_init_coords)
                            
                            # add this contour to the label holder in 
                            # cache_dict_hold_opposite_contours as these inverse
                            # contours will be used when final_poss_cont_m_out is inside
                            # so when label == final_poss_cont_m_out
                            if type(final_poss_cont_m_out) != type(None):
                                cache_dict_hold_opposite_contours[str(int(final_poss_cont_m_out))].append(elongated_contour_reversed)
                            
                            for removal_ind in reversed(range(1, l+1)):
                                
                                    new_contours_found.pop(index_contour_found+removal_ind)
                                
                            break
            else:
                # so the len(cache_dict_hold_opposite_contours[str(int(label))]) == 0
                
                for l in reversed(range(0, skip_n+1)):
                    
                    # exception when going with indexes that are too high and
                    # would be out of the list end of list
                    if index_contour_found + l <= len(new_contours_found)-1:
                        contour_final_possible = new_contours_found[index_contour_found + l]
                    else:
                        # just skip the iteration as it is leading to a
                        # contour that cannot be tracked afterwards
                        continue
                    
                    final_poss_cont_m_in = contour_final_possible.m_in
                    final_poss_cont_m_out = contour_final_possible.m_out
                    final_poss_cont_init_coords = contour_final_possible.init_coords
                    final_poss_cont_final_cords = contour_final_possible.final_coords
                    
                    if init_cont_m_out == final_poss_cont_m_out:
                        
                        elongated_contour = Segment.contour_vector(
                            final_poss_cont_m_in, final_poss_cont_m_out, 
                            init_cont_init_coords, final_poss_cont_final_cords)
                        
                        new_contours_elongated.append(elongated_contour)
                        
                        # we make an elongated contour that is in opposite sense
                        # to the main one to gather it as the another label main contour
                        elongated_contour_reversed = Segment.contour_vector(
                            final_poss_cont_m_out, final_poss_cont_m_in,
                            final_poss_cont_final_cords, init_cont_init_coords)
                        
                        # add this contour to the label holder in 
                        # cache_dict_hold_opposite_contours as these inverse
                        # contours will be used when final_poss_cont_m_out is inside
                        # so when label == final_poss_cont_m_out
                        if type(final_poss_cont_m_out) != type(None):
                            cache_dict_hold_opposite_contours[str(int(final_poss_cont_m_out))].append(elongated_contour_reversed)
                        
                        for removal_ind in reversed(range(1, l+1)):
                            
                                new_contours_found.pop(index_contour_found+removal_ind)
                            
                        break
        
        # fill the dictionary with the new contour information
        conts_vertx_per_region_skept[str(int(label))+'_contours'] = new_contours_elongated     
        vertexs_elongated = list(map(lambda x: x.init_coords, new_contours_elongated))
        vertexs_elongated = np.asarray(vertexs_elongated)
        
        # np unique is swapping values dunno why
        conts_vertx_per_region_skept[str(int(label))+'_vertexs'] = vertexs_elongated     
        
        
    #     once this is looped thourhg all the contours then store the other edges with
    #     intercahgned m out and min corresponding to the other labels as they will be   
    #     needed for the next iteratin with the next label
            
            
    # test first it is doing the skipwell with one single regin and
    # then check if it stores correctly the information for the next label
    
    return conts_vertx_per_region_skept






