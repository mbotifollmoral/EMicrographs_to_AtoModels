# -*- coding: utf-8 -*-
"""
Created on Wed Jan 11 11:53:03 2023

@author: Marc
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import stemtool
import torch
import torch.nn
import os

# !!! NEED to set the path to 
# Alg_Comb_Single_Image_Strain.py
# as the console working directory
Project_main_path = os.getcwd()
if 'EMicrographs_to_AtoModels' in Project_main_path:
    Project_main_path = Project_main_path[:Project_main_path.find('EMicrographs_to_AtoModels')-1]
# Project_main_path has the EMicrographs_to_AtoModels folder
sys.path.append(Project_main_path)


from EMicrographs_to_AtoModels.Functions.General_functions import Phase_Identificator as PhaseIdent
from EMicrographs_to_AtoModels.Functions.General_functions import ImageCalibTransf as ImCalTrans
from EMicrographs_to_AtoModels.Functions.General_functions import PeakFinding_1stAprox as PeakFind

from EMicrographs_to_AtoModels.Functions.Peak_detector_Indep import PF_FFT_processing as FFT_Procs
from EMicrographs_to_AtoModels.Functions.Peak_detector_Indep import PF_Peaks_detector as Peaks_detector


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



def Pixels_peaks_Standarised_FFT(
        image_crop_hs_signal, FFT_image_array, total_pixels_crop, crop_FOV, visualisation = False):
    '''
    Function for peak detection using the 2DGML method and standarising the pixel
    size to an opt scale defined by opt_scale and not changing the hyperparams
    of the 2DG fitting despite the standarisation
    '''

    # wrap the 1st aprox peak finding in one line
    Standarised_FFT_Log, lim_l, lim_r = FFT_Procs.standarize_fft(
        image_crop_hs_signal, FFT_image_array, opt_scale = 0.035)
    
    # the hyperparams change as well, only the total numbe rpixesl in this case not FOV
    # the total number of pixels is defined by the Standarised_FFT_Log_withFOV
    #1st Approx peak finding hyperparameter finding 
    st_distance,_,FFT_perc=PeakFind.FFT_hyperparams(FFT_image_array, crop_FOV)
    FFT_thresholdingG=PeakFind.Threshold_given_percentage(FFT_image_array, FFT_perc)
    
    #1st Approx peak finding hyperparameter fitting
    pixels_of_peaks=stemtool.afit.peaks_vis(
        Standarised_FFT_Log, dist=st_distance, thresh=FFT_thresholdingG, imsize=(15, 15))
    # plt.show()
    plt.close()
    pixels_of_peaks = np.asarray(pixels_of_peaks, dtype=np.int32)
    # Recompute the pixels of the peaks given the relation between
    # standarised image and the original one
    
    peaks_in_matrix = np.zeros(Standarised_FFT_Log.shape)
    
    for pixel_of_peak in pixels_of_peaks:
        y,x = pixel_of_peak
        peaks_in_matrix[int(y),int(x)] = 1
        
    # Extract the pixel coordinates from the standarised binary matrix
    pixels_of_peaks = Peaks_detector.standarised_peaks_image_to_coords(
        peaks_in_matrix, FFT_image_array, lim_l, lim_r)

        
    if visualisation == True:     
        # Better visualisation with the overlapping of FFT and peaks
        
        peaks_in_matrix_visual = np.zeros(FFT_image_array.shape)
        
        for pixel_of_peak in pixels_of_peaks:
            y,x = pixel_of_peak
            
            peaks_in_matrix_visual[y,x] = 1
            
        Peaks_detector.show_peaks_detected(
            FFT_image_array, peaks_in_matrix_visual, ratio = 0.6, plot = visualisation)
        plt.title('2DGML stand no fov:'+str(len(pixels_of_peaks))+' peaks found, pixel size '+str(crop_FOV/total_pixels_crop)+' FOV '+str(crop_FOV))
        plt.show()

        
    return pixels_of_peaks



def Pixels_peaks_Standarised_FFT_withFOV(
        image_crop_hs_signal, FFT_image_array, total_pixels_crop, crop_FOV):
    '''
    Function for peak detection using the 2DGML method and standarising the pixel
    size to an opt scale defined by opt_scale and the FOV by an opt_fov 
    and not changing the hyperparams of the 2DG fitting despite the standarisation
    '''
    # obtain the standarised FFT from which to compute the peaks (no log),  
    Standarised_FFT_Log, lim_l, lim_r = FFT_Procs.standarize_fft(
        image_crop_hs_signal, FFT_image_array, opt_scale = 0.035)

    Standarised_FFT_Log_withFOV, fov_lim_l, fov_lim_r = FFT_Procs.standarize_fft_modFOV(
        image_crop_hs_signal, FFT_image_array, opt_scale = 0.035, opt_fov = 15)
    

    # recompute the FOV at which to apllpy the hyperparams of the 2D gauss
    # the total number of pixels is defined by the Standarised_FFT_Log_withFOV
    # the fov varies the same way the standarised does with respect the orignial
    crop_FOV_stand = crop_FOV*(Standarised_FFT_Log_withFOV.shape[0]/FFT_image_array.shape[0])
    
    #1st Approx peak finding hyperparameter finding 
    st_distance,_,FFT_perc=PeakFind.FFT_hyperparams(FFT_image_array, crop_FOV)
    FFT_thresholdingG=PeakFind.Threshold_given_percentage(FFT_image_array, FFT_perc)
    
    #1st Approx peak finding hyperparameter fitting
    pixels_of_peaks = stemtool.afit.peaks_vis(
        Standarised_FFT_Log_withFOV, dist=st_distance, thresh=FFT_thresholdingG, imsize=(15, 15))
    plt.show()
    # plt.close()
    
    peaks_in_matrix = np.zeros(Standarised_FFT_Log_withFOV.shape)
    
    for pixel_of_peak in pixels_of_peaks:
        print(pixel_of_peak)
        y,x = pixel_of_peak
        peaks_in_matrix[int(y),int(x)] = 1
        
    # Extract the pixel coordinates from the standarised binary matrix
    pixels_of_peaks = Peaks_detector.standarised_FOV_peaks_image_to_coords(
        peaks_in_matrix, Standarised_FFT_Log, FFT_image_array, lim_l, lim_r)
    
    visualisation = True
    if visualisation == True:     
        # Better visualisation with the overlapping of FFT and peaks
        
        peaks_in_matrix_visual = np.zeros(FFT_image_array.shape)
        
        for pixel_of_peak in pixels_of_peaks:
            y,x = pixel_of_peak
            
            peaks_in_matrix_visual[y,x] = 1
            
        Peaks_detector.show_peaks_detected(
            FFT_image_array, peaks_in_matrix_visual, ratio = 0.6, plot = visualisation)
        plt.title('2DGML stand fov:'+str(len(pixels_of_peaks))+' peaks found, pixel size '+str(crop_FOV/total_pixels_crop)+' FOV '+str(crop_FOV))
        plt.show()

         
    return pixels_of_peaks







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
    FFT_image_array, FFT_calibration, FFT_units = ImCalTrans.FFT_from_crop(
        image_crop_hs_signal, image_crop_array)
    
    
    # Find the pixels of the freqs
    pixels_of_peaks = Pixels_peaks_FFT(
        FFT_image_array,crop_FOV)
    
    # Get functional distances, angles, pixel positions out of the peaks of the freqs
    refined_distances, refined_angles_to_x, refined_pixels = PhaseIdent.Ready_Distances_Angles_Pixels(
        pixels_of_peaks, FFT_calibration, total_pixels_crop, min_d)

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



def Peak_Detection_Wrapped_Standarised(
        image_crop_hs_signal, FFT_image_array_No_Log, FFT_image_array, total_pixels_crop, 
        crop_FOV, visualisation = False):
    '''
    Function wrapping the peak detection, with the method being the 1D Scan 
    of the FFT without the logarithm and standarisation of the pixel size to 
    a value defined by opt_scale
    Choose whether to visualise the overlapping of the found peak positions with the FFT with 
    the logairhtm

    Returns
    -------
    list_refined_cryst_spots.

    '''
    
    # Get the standarised verision of the log FFT for plotting purposes
    Standarised_FFT_Log, lim_l, lim_r = FFT_Procs.standarize_fft(
        image_crop_hs_signal, FFT_image_array, opt_scale = 0.05)
    # obtain the standarised FFT from which to compute the peaks (no log),  
    # and where it is extracted from the whole FFT 
    Standarised_FFT_No_Log, lim_l, lim_r = FFT_Procs.standarize_fft(
        image_crop_hs_signal, FFT_image_array_No_Log, opt_scale = 0.05)
    # Extract the peaks of the standarised FFT with No Log applied to it
    peaks_in_matrix = Peaks_detector.peaks_detector(Standarised_FFT_No_Log)
    
    # Remove any peak found in the frontier of the added patch if standarised 
    # image size is bigger than the original FFT
    if Standarised_FFT_Log.shape[0] > FFT_image_array.shape[0]:
        peaks_in_matrix_mod = np.zeros(peaks_in_matrix.shape)
        peaks_in_matrix_mod[lim_l+10 : lim_r-10, lim_l+10 : lim_r-10] = peaks_in_matrix[lim_l+10 : lim_r-10, lim_l+10 : lim_r-10]
        peaks_in_matrix = peaks_in_matrix_mod
    # Extract the pixel coordinates from the standarised binary matrix
    pixels_of_peaks = Peaks_detector.standarised_peaks_image_to_coords(
        peaks_in_matrix, FFT_image_array, lim_l, lim_r)
    
    if visualisation == True:     
        # Better visualisation with the overlapping of FFT and peaks
        Peaks_detector.show_peaks_detected(
            Standarised_FFT_Log, peaks_in_matrix, ratio = 0.6, plot = visualisation)
        plt.title('St1DScan:'+str(len(pixels_of_peaks))+' peaks found, pixel size '+str(crop_FOV/total_pixels_crop)+' FOV '+str(crop_FOV))
        plt.show()


        peaks_in_matrix_visual = np.zeros(FFT_image_array.shape)
        
        for pixel_of_peak in pixels_of_peaks:
            y,x = pixel_of_peak
            
            peaks_in_matrix_visual[int(y),int(x)] = 1
            
        Peaks_detector.show_peaks_detected(
            FFT_image_array, peaks_in_matrix_visual, ratio = 0.6, plot = visualisation)
        plt.title('St1DScan orig:'+str(len(pixels_of_peaks))+' peaks found, pixel size '+str(crop_FOV/total_pixels_crop)+' FOV '+str(crop_FOV))
        plt.show()
    
    return pixels_of_peaks


def Peak_Detection_Wrapped_StandarisedwithFOV(
        image_crop_hs_signal, FFT_image_array_No_Log, FFT_image_array, total_pixels_crop, 
        crop_FOV, visualisation = False):
    '''
    Function wrapping the peak detection, with the method being the 1D Scan 
    of the FFT without the logarithm and standarisation of the pixel size to 
    a value defined by opt_scale and of the FOV by a val defined by opt_fov
    Choose whether to visualise the overlapping of the found peak positions with the FFT with 
    the logairhtm

    Returns
    -------
    list_refined_cryst_spots.

    '''
    
    pixels_of_peaks = Pixels_peaks_FFT(
        FFT_image_array, crop_FOV)
    
    pixels_of_peaks = Pixels_peaks_Standarised_FFT(
        image_crop_hs_signal, FFT_image_array, total_pixels_crop, crop_FOV)
    
    # pixels_of_peaks = Pixels_peaks_Standarised_FFT_withFOV(
    #     image_crop_hs_signal, FFT_image_array, total_pixels_crop, crop_FOV)
    

    if pixels_of_peaks.shape[0] > 1:
        
        # new part of the function with the new peak finding standarised
        # if it finds more than 1 pixel, the central and others (not amorphous)
        # recompute the found pixels with the scanning method
        
        # Get the standarised verision of the log FFT for plotting purposes
        
        Standarised_FFT_Log, lim_l, lim_r = FFT_Procs.standarize_fft(
            image_crop_hs_signal, FFT_image_array, opt_scale = 0.05)
        # obtain the standarised FFT from which to compute the peaks (no log),  
        # and where it is extracted from the whole FFT 
        Standarised_FFT_No_Log, lim_l, lim_r = FFT_Procs.standarize_fft(
            image_crop_hs_signal, FFT_image_array_No_Log, opt_scale = 0.05)
        
        
        Standarised_FFT_Log_withFOV, fov_lim_l, fov_lim_r = FFT_Procs.standarize_fft_modFOV(
            image_crop_hs_signal, FFT_image_array, opt_scale = 0.05, opt_fov = 45)
        # obtain the standarised FFT from which to compute the peaks (no log),  
        # and where it is extracted from the whole FFT 
        Standarised_FFT_No_Log_withFOV, fov_lim_l, fov_lim_r = FFT_Procs.standarize_fft_modFOV(
            image_crop_hs_signal, FFT_image_array_No_Log, opt_scale = 0.05, opt_fov = 45)
        # Extract the peaks of the standarised FFT with No Log applied to it
        peaks_in_matrix = Peaks_detector.peaks_detector(Standarised_FFT_No_Log_withFOV)
        
        # Remove any peak found in the frontier of the added patch if standarised 
        # image size is bigger than the original FFT
        if Standarised_FFT_Log.shape[0] > FFT_image_array.shape[0]:
            peaks_in_matrix_mod = np.zeros(peaks_in_matrix.shape)
            peaks_in_matrix_mod[int(fov_lim_l)+5 : int(fov_lim_r)-5, int(fov_lim_l)+5 : int(fov_lim_r)-5] = peaks_in_matrix[int(fov_lim_l)+5 : int(fov_lim_r)-5, int(fov_lim_l)+5 : int(fov_lim_r)-5]
            peaks_in_matrix = peaks_in_matrix_mod            
            
        # Extract the pixel coordinates from the standarised binary matrix
        pixels_of_peaks = Peaks_detector.standarised_FOV_peaks_image_to_coords(
            peaks_in_matrix, Standarised_FFT_No_Log, FFT_image_array, lim_l, lim_r)
        
        if visualisation == True:     
            # Better visualisation with the overlapping of FFT and peaks
            Peaks_detector.show_peaks_detected(
                Standarised_FFT_Log_withFOV, peaks_in_matrix, ratio = 0.6, plot = visualisation)
            plt.title('St1DScan FOV adjusted:'+str(len(pixels_of_peaks))+' peaks found, pixel size '+str(crop_FOV/total_pixels_crop)+' FOV '+str(crop_FOV))
            plt.show()
            
            peaks_in_matrix_visual = np.zeros(FFT_image_array.shape)
            
            for pixel_of_peak in pixels_of_peaks:
                y,x = pixel_of_peak
                
                peaks_in_matrix_visual[y,x] = 1
                
            Peaks_detector.show_peaks_detected(
                FFT_image_array, peaks_in_matrix_visual, ratio = 0.6, plot = visualisation)
            plt.title('St1DScan stand no fov:'+str(len(pixels_of_peaks))+' peaks found, pixel size '+str(crop_FOV/total_pixels_crop)+' FOV '+str(crop_FOV))
            plt.show()
            
            
            
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
        
    

    return pixels_of_peaks





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
    
    CNN_model = torch.load(r'D:\Marc_Botifoll\Data_ML\CNN_Model_PeakFinding\DL2.pt')
    
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
    Peaks_detector.show_peaks_detected(
        FFT_image_array_No_Log, peaks_in_matrix, ratio = 0.8, plot = visualisation)
    plt.title('CNN:'+str(len(pixels_of_peaks))+' peaks found, pixel size '+str(crop_FOV/total_pixels_crop)+' FOV '+str(crop_FOV))
    plt.show()
    
    # First check if the !st approx peak finding finds 1,2 or more pixels
    # as this would tell if we have an amorphous material (1st central spot),
    # a non identifiable material (central and 1 additional), or some crystal identifiable
   
    return pixels_of_peaks


def Peak_Detection_Wrapped_Standarised_CNN(
        image_crop_hs_signal, FFT_image_array_No_Log, FFT_image_array, total_pixels_crop, 
        crop_FOV, visualisation = False):
    '''
    Function wrapping the peak detection, final function: CNN stnadarised of the FFT, 
    the with the logarithm of the FFT
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
    
    
    CNN_model = torch.load(r'D:\Marc_Botifoll\Data_ML\CNN_Model_PeakFinding\DL2.pt')
    
    # obtain the standarised FFT from which to compute the peaks (no log),  
    # and where it is extracted from the whole FFT 
    Standarised_FFT_No_Log, lim_l, lim_r = FFT_Procs.standarize_fft(
        image_crop_hs_signal, FFT_image_array_No_Log, opt_scale = 0.035)
    # Get the standarised verision of the log FFT for plotting purposes
    Standarised_FFT_Log, lim_l, lim_r = FFT_Procs.standarize_fft(
        image_crop_hs_signal, FFT_image_array, opt_scale = 0.035)
    
    #CNN prediction
    CNN_prediction, _ = CNN_model.predict(Standarised_FFT_Log)
    del CNN_model


    #Final prediction output
    # CNN hyperparams
    CNN_side = 31 #It has to be a odd number
    CNN_treshold= 0.69 #It has to be less than 1
    
    # Get the binary matrix with the peaks found 
    # peaks_in_matrix, prediction_No_background = Peaks_detector.peaks_recognition(
    #     CNN_side, CNN_treshold, CNN_prediction, 
    #     FFT_image_array, np.shape(Standarised_FFT_Log)[0])
    
    peaks_in_matrix, prediction_No_background = Peaks_detector.peaks_recognition(
        CNN_side, CNN_treshold, CNN_prediction, 
        Standarised_FFT_Log, np.shape(Standarised_FFT_Log)[0])
    
    
    

    # Remove any peak found in the frontier of the added patch if standarised 
    # image size is bigger than the original FFT
    if Standarised_FFT_Log.shape[0] > FFT_image_array.shape[0]:
        peaks_in_matrix_mod = np.zeros(peaks_in_matrix.shape)
        peaks_in_matrix_mod[lim_l+10 : lim_r-10, lim_l+10 : lim_r-10] = peaks_in_matrix[lim_l+10 : lim_r-10, lim_l+10 : lim_r-10]
        peaks_in_matrix = peaks_in_matrix_mod
        
    # Extract the pixel coordinates from the standarised binary matrix
    pixels_of_peaks = Peaks_detector.standarised_peaks_image_to_coords(
        peaks_in_matrix, FFT_image_array, lim_l, lim_r)


    # FFT_resized = cv2.resize(prediction[0], dsize=(size, size), interpolation=cv2.INTER_LINEAR)
    if visualisation == True:     
        # Better visualisation with the overlapping of FFT and peaks
        Peaks_detector.show_peaks_detected(
            Standarised_FFT_Log, peaks_in_matrix, ratio = 0.6, plot = visualisation)
        plt.title('CNN stand:'+str(len(pixels_of_peaks))+' peaks found, pixel size '+str(crop_FOV/total_pixels_crop)+' FOV '+str(crop_FOV))
        plt.show()
        
        
        peaks_in_matrix_visual = np.zeros(FFT_image_array.shape)
        
        for pixel_of_peak in pixels_of_peaks:
            y,x = pixel_of_peak
            
            peaks_in_matrix_visual[int(y),int(x)] = 1
            
        Peaks_detector.show_peaks_detected(
            FFT_image_array, peaks_in_matrix_visual, ratio = 0.6, plot = visualisation)
        plt.title('CNN orig after stand:'+str(len(pixels_of_peaks))+' peaks found, pixel size '+str(crop_FOV/total_pixels_crop)+' FOV '+str(crop_FOV))
        plt.show()
   
    
    return pixels_of_peaks



def Peak_Detection_Wrapped_Standarised_withFOV_CNN(
        image_crop_hs_signal, FFT_image_array_No_Log, FFT_image_array, total_pixels_crop, 
        crop_FOV, visualisation = False):
    '''
    Function wrapping the peak detection, final function: 
    CNN standarised with FOV modified of the FFT,
    the with the logarithm of the FFT
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
    
    
    CNN_model = torch.load(r'D:\Marc_Botifoll\Data_ML\CNN_Model_PeakFinding\DL2.pt')
    
    # obtain the standarised FFT from which to compute the peaks (no log),  
    # and where it is extracted from the whole FFT 
    Standarised_FFT_No_Log, lim_l, lim_r = FFT_Procs.standarize_fft(
        image_crop_hs_signal, FFT_image_array_No_Log, opt_scale = 0.035)
    # Get the standarised verision of the log FFT for plotting purposes
    Standarised_FFT_Log, lim_l, lim_r = FFT_Procs.standarize_fft(
        image_crop_hs_signal, FFT_image_array, opt_scale = 0.035)

    Standarised_FFT_Log_withFOV, fov_lim_l, fov_lim_r = FFT_Procs.standarize_fft_modFOV(
        image_crop_hs_signal, FFT_image_array, opt_scale = 0.035, opt_fov = 15)
    # obtain the standarised FFT from which to compute the peaks (no log),  
    # and where it is extracted from the whole FFT 
    Standarised_FFT_No_Log_withFOV, fov_lim_l, fov_lim_r = FFT_Procs.standarize_fft_modFOV(
        image_crop_hs_signal, FFT_image_array_No_Log, opt_scale = 0.035, opt_fov = 15)
    
    
    #CNN prediction
    CNN_prediction, _ = CNN_model.predict(Standarised_FFT_Log_withFOV)
    del CNN_model


    #Final prediction output
    # CNN hyperparams
    CNN_side = 31 #It has to be a odd number
    CNN_treshold= 0.69 #It has to be less than 1
    
    # Get the binary matrix with the peaks found 
    peaks_in_matrix, prediction_No_background = Peaks_detector.peaks_recognition(
        CNN_side, CNN_treshold, CNN_prediction, np.shape(Standarised_FFT_Log_withFOV)[0])
    
    # Remove any peak found in the frontier of the added patch if standarised 
    # image size is bigger than the original FFT
    if Standarised_FFT_Log.shape[0] > FFT_image_array.shape[0]:
        peaks_in_matrix_mod = np.zeros(peaks_in_matrix.shape)
        peaks_in_matrix_mod[int(fov_lim_l)+10 : int(fov_lim_r)-10, int(fov_lim_l)+10 : int(fov_lim_r)-10] = peaks_in_matrix[int(fov_lim_l)+10 : int(fov_lim_r)-10, int(fov_lim_l)+10 : int(fov_lim_r)-10]
        peaks_in_matrix = peaks_in_matrix_mod    
    
    # Extract the pixel coordinates from the standarised binary matrix
    pixels_of_peaks = Peaks_detector.standarised_FOV_peaks_image_to_coords(
        peaks_in_matrix, Standarised_FFT_No_Log, FFT_image_array, lim_l, lim_r)


    # FFT_resized = cv2.resize(prediction[0], dsize=(size, size), interpolation=cv2.INTER_LINEAR)
    if visualisation == True:     
        # Better visualisation with the overlapping of FFT and peaks
        Peaks_detector.show_peaks_detected(
            Standarised_FFT_Log_withFOV, peaks_in_matrix, ratio = 0.6, plot = visualisation)
        plt.title('CNN stand with fov:'+str(len(pixels_of_peaks))+' peaks found, pixel size '+str(crop_FOV/total_pixels_crop)+' FOV '+str(crop_FOV))
        plt.show()
        

        peaks_in_matrix_visual = np.zeros(FFT_image_array.shape)
        
        for pixel_of_peak in pixels_of_peaks:
            y,x = pixel_of_peak
            
            peaks_in_matrix_visual[y,x] = 1
            
        Peaks_detector.show_peaks_detected(
            FFT_image_array, peaks_in_matrix_visual, ratio = 0.6, plot = visualisation)
        plt.title('CNN orig after stand:'+str(len(pixels_of_peaks))+' peaks found, pixel size '+str(crop_FOV/total_pixels_crop)+' FOV '+str(crop_FOV))
        plt.show()
        
    # First check if the !st approx peak finding finds 1,2 or more pixels
    # as this would tell if we have an amorphous material (1st central spot),
    # a non identifiable material (central and 1 additional), or some crystal identifiable
    
    return pixels_of_peaks



def Peak_Detection_Multimodal_Wrapped(
        image_crop_hs_signal, FFT_image_array_No_Log, FFT_image_array, 
        total_pixels_crop, crop_FOV, visualisation = False):
    
    '''
    Definitive peak detection function that gathers all the three methods
    developed (2DGML, 1DScan and CNN) and adopts the best one in the scenario
    where it is thought to work better in order to get the global best
    peak detection in every situation
    The idea is that in amorphous regions, 2DGML and CNN mostly always return
    a single peak identified, while 1DScan typicallly finds a lot of peaks being 
    those the background noise. We take advantage of this situation to detect the
    amorphous nature of the compounds.
    When it is crystalline, hardly ever the 1DScan gives noise, for which we
    use this method unless it gives more than Max_peaks_1DScan (which we set
    to be 100 as a reasonable value), where we use the method that gives more
    peaks from the other 2 (2DGML or CNN), as in these cases hardly ever more
    peaks mean worse (very rare to find more spots due to noise)
    !!! We also filter by pixel size as higher pixel sizes mean the number
    of spots to find should be reduced
    but if finding problems with too many spots found even wiht  this just
    reduce the Max_peaks_1DScan to 90
    
    It returns a list of the pixel coordintes of the image FFT at which the 
    peaks are identified and the method used to get it, as afterwards
    it is gonna be useful to compute a figure of merit of the quality of the
    fitting given by each, as for crystalline 1DScan is better than the other two
    but in amorphous the others are much better (and 2DGML seems a bit better
    than the CNN in crystalline)
    '''
    
    # 1DScan max peaks as noise
    Max_peaks_1DScan = 100
    
    # Find the pixels of the freqs by 2DGML
    pixels_of_peaks_2DGML = Pixels_peaks_Standarised_FFT(
        image_crop_hs_signal, FFT_image_array, total_pixels_crop, crop_FOV, visualisation = False)
    
    if len(pixels_of_peaks_2DGML) == 0:
        pixels_of_peaks_2DGML = np.array([[int(FFT_image_array.shape[0]/2), int(FFT_image_array.shape[0]/2)]])
    
    # Find the pixels of the freqs by CNN
    # !!! condition set because my PC ram cannot handle 2k CNN loading
    if total_pixels_crop < 1024:
        pixels_of_peaks_CNN = Peak_Detection_Wrapped_Standarised_CNN(
            image_crop_hs_signal, FFT_image_array_No_Log, FFT_image_array, 
            total_pixels_crop, crop_FOV, visualisation = False)  
    else:
        pixels_of_peaks_CNN = np.array([[int(FFT_image_array.shape[0]/2), int(FFT_image_array.shape[0]/2)]])
    
    if len(pixels_of_peaks_CNN) == 0:
        pixels_of_peaks_CNN = np.array([[int(FFT_image_array.shape[0]/2), int(FFT_image_array.shape[0]/2)]])

    # if both algorithms 2DGML and CNN give a 1, then it is amost 100% amorphous 
    if len(pixels_of_peaks_2DGML) == 1 and len(pixels_of_peaks_CNN) == 1:
        pixels_of_peaks = pixels_of_peaks_2DGML
        method_used = 'Amorphous_2DGML_CNN'
        
    else:
        # the algorithms give more than 1 spot and then it is crystalline, so
        # check the best algorithm for crystalline being the 1DS
        pixels_of_peaks_1DScan = Peak_Detection_Wrapped_Standarised(
            image_crop_hs_signal, FFT_image_array_No_Log, FFT_image_array, 
            total_pixels_crop, crop_FOV, visualisation = False)
        
        if len(pixels_of_peaks_1DScan) == 0:
            pixels_of_peaks_1DScan = np.array([[int(FFT_image_array.shape[0]/2), int(FFT_image_array.shape[0]/2)]])
        
        # if the number of peaks identified is higher than 1DScan_max_peaks
        # then it is probably just noise and better use another method
        
        # !!! beware these conditions as still can let some noise to com in
        # maybe adapt the Max_peaks_1DScan by fov or number of pixels...
        
        
        if len(pixels_of_peaks_1DScan) > Max_peaks_1DScan/2 and crop_FOV/total_pixels_crop > 0.07:
            
            # then just chose the method that gave more peaks identified 
            if len(pixels_of_peaks_2DGML) >= len(pixels_of_peaks_CNN):
                pixels_of_peaks = pixels_of_peaks_2DGML
                method_used = '2DGML'
            else:
                pixels_of_peaks = pixels_of_peaks_CNN
                method_used = 'CNN'       
                
        elif len(pixels_of_peaks_1DScan) > Max_peaks_1DScan and crop_FOV/total_pixels_crop <= 0.07:
            
            # then just chose the method that gave more peaks identified 
            if len(pixels_of_peaks_2DGML) >= len(pixels_of_peaks_CNN):
                pixels_of_peaks = pixels_of_peaks_2DGML
                method_used = '2DGML'
            else:
                pixels_of_peaks = pixels_of_peaks_CNN
                method_used = 'CNN'
        else:
            # the number of peaks found is between 1 and 100 and are 
            # very likely actual peaks
            pixels_of_peaks = pixels_of_peaks_1DScan
            method_used = '1DScan'
            
    if visualisation == True:
        
        peaks_in_matrix_visual = np.zeros(FFT_image_array.shape)
        
        for pixel_of_peak in pixels_of_peaks:
            y,x = pixel_of_peak
            
            peaks_in_matrix_visual[int(y),int(x)] = 1
            
        Peaks_detector.show_peaks_detected(
            FFT_image_array, peaks_in_matrix_visual, ratio = 0.6, plot = visualisation)
        plt.title(method_used + ': ' +str(len(pixels_of_peaks))+' peaks found, pixel size '+str(crop_FOV/total_pixels_crop)+' FOV '+str(crop_FOV))
        plt.show()
    
    return pixels_of_peaks, method_used
    





