# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 19:27:44 2023

@author: Marc
"""


import numpy as np
import os
import imutils
import cv2
import hyperspy.api as hs
import matplotlib.pyplot as plt
import sys
from pathlib import Path
import hyperspy.api as hs
import random

random.seed(2**23)
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
from EMicrographs_to_AtoModels.Functions.General_functions import GPA_atomistic_combiner as GPA_AtoMod



def Available_EELS_data_Checker(
        micrographs_path):
    '''
    Initial function to just check whether there is or not available EELS data
    to be used instead of the segmentation path

    Parameters
    ----------
    micrographs_path : path to the image, Micrographs// folder

    Returns
    -------
    path_EELS_path. boolean indicating whether the path to the EELS file
                    exists or not

    '''
    
    # eels data path expected
    EELS_path = micrographs_path[:micrographs_path.rfind('Micrographs')] + 'EELS\\'
    # Boolean stating whether this path already exists, True, or not, False
    path_EELS_path = os.path.isdir(EELS_path)
        
    return path_EELS_path




def Find_EELS_survey_in_micrograph(
        micrographs_path, image_in_dataset):
    '''
    Find the query-template relation between the EELS survey and the
    main image.
    If is able to check whether, based on the FOVs of both images,
    whether the survey fits within the image, the image within the survey,
    or partially and then the cut the survey to be fit inside the image.
    It always finds a template-query relation even if they are not real
    templates and surveys (one image being taken within the other)
    So ensure the images have this relation

    Parameters
    ----------
    micrographs_path : path to the micrographs
    image_in_dataset : image_in_dataset object of the main image where the
                        analysis is taken or considered

    Raises
    ------
    Exception
        DESCRIPTION.

    Returns
    -------
    query_hs_sign : hyperspy signal of the query, whether it is the image or
                    EELS survey 
    template_hs_sign : hyperspy signal of the template, whether it is the image or
                    EELS survey 
    coordinates_template_EELS : coordiantes of the template matching process, in
                            pixels and relative to the query, in the form of
                            startX, startY, endX, endY = coordinates_template_EELS
    scale_factor : scale factor change, ratio between pixels sizes of both
                    images, so to see how the images where rescaled for
                    the template matching process
    setting : 1,2 or 3, indicating the template-query relation
        if survey_x_FOV > main_image_FOV and survey_y_FOV > main_image_FOV:
            so image fits inside the EELS survey
            setting = 1
        elif survey_x_FOV > main_image_FOV or survey_y_FOV > main_image_FOV:
            so survey is bigger than image in one of the dimensions, so we cut it
            and then fit inside the whole image
            setting = 2
        else:
            if image is bigger in FOV than survey, so survey is within the image
            setting = 3

    '''
    
    
    
    # at the same folder level of the Micrographs/ folder, we should search for
    # the folder EELS/
    EELS_path = micrographs_path[:micrographs_path.rfind('Micrographs')] + 'EELS\\'
    
    # expected path to survey image
    survey_filepaht = EELS_path + 'survey.dm3'
    
    # Check if the survey image exists in path or not
    path_survey_filepaht = os.path.isfile(survey_filepaht)
    if path_survey_filepaht == False:
        raise Exception('No survey found, add the complete survey micrograph, named survey.dm3')
    
    # if there exists, continue with the actual template matching
      
    # Extract info from image in dataset, which will be the query
    main_image_hs = image_in_dataset.hyperspy_2Dsignal
    main_image_arraynp_st = image_in_dataset.image_arraynp_st
    main_image_arraynp_st_int = image_in_dataset.image_arraynp_st_int
    main_image_x_calibration = image_in_dataset.x_calibration
    main_image_total_pixels = image_in_dataset.total_pixels
    main_image_units = image_in_dataset.units
    main_image_FOV = image_in_dataset.FOV

    # Extract info from survey image, which will be the template
    survey_hs = hs.load(survey_filepaht)
    survey_arraynp = np.asarray(survey_hs)
    survey_arraynp_st=(survey_arraynp-np.min(survey_arraynp))/np.max(survey_arraynp-np.min(survey_arraynp))
    survey_arraynp_st_int=np.uint8(255*survey_arraynp_st)
    
    # Get calibration values expecting the maps not to be squared but rectangular
    # and get x an dy calibrations separately, being the same for the rest of maps
    # However, the pixel must be squared so x_calibration = y_calibration
    # therefore the scale factor use must be the same in both directions
    survey_x_calibration = survey_hs.axes_manager['x'].scale
    survey_x_total_pixels = survey_hs.axes_manager['x'].size
    survey_x_units = survey_hs.axes_manager['x'].units
    survey_x_FOV = survey_x_calibration*survey_x_total_pixels  
    
    survey_y_calibration = survey_hs.axes_manager['y'].scale
    survey_y_total_pixels = survey_hs.axes_manager['y'].size
    survey_y_units = survey_hs.axes_manager['y'].units
    survey_y_FOV = survey_y_calibration*survey_y_total_pixels      
    
    # First we find out which one should be the template and which the query
    # ideally the query would be the image_in_dataset, and the EELS would
    # be the template that fits inside. However, it might happen that
    # a low mag EELS available and then the image can fit inside this whole map
    # so take into account both scenearios
    # the way to decide the template and the query is not by pixel size but
    # by FOV, as the EELS maps will normally have a very different number of 
    # pixels when compared to the main image 
    # (generally much smaller number of pixels)
    
    if survey_x_FOV > main_image_FOV and survey_y_FOV > main_image_FOV:
        setting = 1
        query_hs_sign = survey_hs
        template_hs_sign = main_image_hs
        
        query_image_st = survey_arraynp_st_int
        template_image_st = main_image_arraynp_st_int
        
        
    elif survey_x_FOV > main_image_FOV or survey_y_FOV > main_image_FOV:

        setting = 2
        # If it is the x direction the one in which EELS is larger
        if survey_x_FOV > main_image_FOV:
            
            # find number of pixels in survey that gather the main image's FOV
            new_x_size = int(np.floor(main_image_FOV/survey_x_calibration))
            
            # removing the edges of that dimension
            new_survey = survey_arraynp_st[:, int(np.ceil((survey_x_total_pixels - new_x_size)/2)):int(np.ceil((survey_x_total_pixels - new_x_size)/2))+new_x_size]
            
            survey_arraynp_st=(new_survey-np.min(new_survey))/np.max(new_survey-np.min(new_survey))
            survey_arraynp_st_int=np.uint8(255*survey_arraynp_st)
            
            
            survey_x_total_pixels = np.shape(survey_arraynp_st)[1]
            survey_x_FOV = survey_x_calibration*survey_x_total_pixels

            # rebuild the hyperspy signal
            survey_hs = hs.signals.Signal2D(survey_arraynp_st)
            
            survey_hs.axes_manager[0].name = 'x'
            survey_hs.axes_manager[1].name = 'y'
            
            survey_hs.axes_manager['x'].scale = survey_x_calibration
            survey_hs.axes_manager['x'].units = survey_x_units
            
            survey_hs.axes_manager['y'].scale = survey_y_calibration
            survey_hs.axes_manager['y'].units = survey_y_units
            
            
            # assing the new survey their template status
            query_hs_sign = main_image_hs
            template_hs_sign = survey_hs
            
            query_image_st = main_image_arraynp_st_int
            template_image_st = survey_arraynp_st_int
    
            
            
        # If it is the y direction the one in which EELS is larger
        if survey_y_FOV > main_image_FOV:
            
            # find number of pixels in survey that gather the main image's FOV
            new_y_size = int(np.floor(main_image_FOV/survey_y_calibration))
            
            # removing the edges of that dimension
            new_survey = survey_arraynp_st[int(np.ceil((survey_y_total_pixels - new_y_size)/2)):int(np.ceil((survey_y_total_pixels - new_y_size)/2))+new_y_size, :]
            
            
            survey_arraynp_st=(new_survey-np.min(new_survey))/np.max(new_survey-np.min(new_survey))
            survey_arraynp_st_int=np.uint8(255*survey_arraynp_st)
            
            
            survey_y_total_pixels = np.shape(survey_arraynp_st)[0]
            survey_y_FOV = survey_y_calibration*survey_y_total_pixels

            # rebuild the hyperspy signal
            survey_hs = hs.signals.Signal2D(survey_arraynp_st)
            
            survey_hs.axes_manager[0].name = 'x'
            survey_hs.axes_manager[1].name = 'y'
            
            
            survey_hs.axes_manager['x'].scale = survey_x_calibration
            survey_hs.axes_manager['x'].units = survey_x_units
            
            survey_hs.axes_manager['y'].scale = survey_y_calibration
            survey_hs.axes_manager['y'].units = survey_y_units
            
            
            # assing the new survey their template status
            query_hs_sign = main_image_hs
            template_hs_sign = survey_hs
            
            query_image_st = main_image_arraynp_st_int
            template_image_st = survey_arraynp_st_int


        
    else:
        setting = 3

        query_hs_sign = main_image_hs
        template_hs_sign = survey_hs
        
        query_image_st = main_image_arraynp_st_int
        template_image_st = survey_arraynp_st_int
    
    
    # First we get the scale ratio between template and image to rescale it
    # according to the difference in positions
    scale_factor = HighToLowTM.Get_Ideal_Scale(
        query_hs_sign, template_hs_sign)
    
    # Now fit the template within the 
    coordinates_template_EELS, maxVal = HighToLowTM.Multiscale_Template_Matching(
        query_image_st, template_image_st, scale_factor)

    
    # Retrieve the coordinates with image_in_datset as the reference
    # despite it acting as template or query
    
    
    
    # fit the survey in the correct place within the whole image
    
    # maybe the loop can be done in the actual quantification if wants
    # to be split ... think whetehr it can be interesting to here just
    # make the template matching and so on and then the substitution
    
    # # Loop though files to find the elements involved
    # for eels_signal in os.listdir(EELS_path):
        
    
    return query_hs_sign, template_hs_sign, coordinates_template_EELS, scale_factor, setting




def EELS_AtoModel_Substitution(
        path_global_strained_purged, EELS_maps_resized, 
        Box_strain_pixels, image_in_dataset, setting):
    '''
    Function to substitute the atoms list from the strained and purged model
    with the atoms observed chemically by the EELS maps and their template
    matching with the actual main image analysed. 
    It interpolates based on the actual atom coordinate the chemical stoichiometry
    for every given element quantified in the EELS map and then generates 
    new Atoms objects both collapsing these occupancies into 1 occupancies
    based on the probability of every atom being in that position, and also
    not collapsing the occupancies but making each element be manifold with
    its own composition stoichiometry (so not collapsing the occs)

    Parameters
    ----------
    path_global_strained_purged : path to the atomodel strained purged
    
    EELS_maps_resized : dictionary containing the quantitative map for every
                        element present in the EELS folder, prepeared in 
                        a way that the region within the box where the model
                        must be susbsituted can be accssed throuhg 
                        the coordinates defined by Box_strain_pixels
                        
    Box_strain_pixels : coordinatates where to take the compositional maps 
                        from that will be used to substitute the atoms in the
                        model, these are coordinates in pixels of the main 
                        image, image_in_dataset
                        
    image_in_dataset : image_in_dataset, main image used in the analysis

    Returns
    -------
    noncolaps_occ_eelsed_path : path to non colapsed model after substitution 
                                with EELS info
    colaps_occ_eelsed_path : path to colapsed model after substitution 
                            with EELS info

    '''
    
    # Convert the area where the chemistry needs to be applied 
    # from pixels to units of lenght, angstroms     
    pixel_size_whole = image_in_dataset.x_calibration
    total_pixels_whole = image_in_dataset.total_pixels
    
    region_to_strain_atomcords = np.array([Box_strain_pixels[2]*pixel_size_whole,
                                           (total_pixels_whole-Box_strain_pixels[1])*pixel_size_whole,
                                           Box_strain_pixels[3]*pixel_size_whole,
                                           (total_pixels_whole-Box_strain_pixels[0])*pixel_size_whole])
    
    # !!! UNITS CHANGE (nm --> angstroms)
    # Need to change the coordinates from nm to angstroms as the coords 
    # inside the file are in angstroms
    region_to_strain_atomcords = region_to_strain_atomcords*10

    
    # Get the atoms list from the distorted atomodel, this models needs
    # collapsed occupancies so occupancies of 1 (or not having them)
    strain_purg_atoms, _ = GPA_AtoMod.read_xyz(
        path_global_strained_purged, BOX = region_to_strain_atomcords)
    
    
    
    EELS_maps_resised_list = []
    
    for element_i in EELS_maps_resized:
        
        EELS_maps_resised_list.append(EELS_maps_resized[element_i])
        
    
    EELS_maps_crop = np.asarray(EELS_maps_resised_list)
    
    if setting == 1:
        EELS_maps_crop = EELS_maps_crop[:,
                                        Box_strain_pixels[0]:Box_strain_pixels[1],
                                        Box_strain_pixels[2]:Box_strain_pixels[3]]
    
    new_Atoms_list_colapsOcc = []
    new_Atoms_list_NoColapsOcc = []
        
    
    for Atom_orig in strain_purg_atoms:
        
        x_cord = Atom_orig.x
        y_cord = Atom_orig.y
        z_cord = Atom_orig.z
        
        # store the occupancies for that atom position, and for each of
        # the possible elements that can be in that position
        occupancies = []
        elements_subst = []
        
        for elem_i, map_crop in zip(
                EELS_maps_resized, EELS_maps_crop):
            
            # BOX IS region_to_strain_atomcords in real space the same
            # way the atoms are plced in the model, so in angstroms
            

            # Prepeare the interpolation arrays to interpolate the
            # values of the quantification maps
            edg_crop_y = [region_to_strain_atomcords[1], region_to_strain_atomcords[3]]
            edg_crop_x = [region_to_strain_atomcords[0], region_to_strain_atomcords[2]]
            
            cords_lincrop_y = np.linspace(
                edg_crop_x[0], edg_crop_x[1], map_crop.shape[1])
            
            cords_lincrop_x= np.linspace(
                edg_crop_y[0], edg_crop_y[1], map_crop.shape[0])

            comp_el_i = GPA_AtoMod.interp(
                x_cord, y_cord, map_crop, cords_lincrop_y, cords_lincrop_x)

            # Append the compositional values as the occupancies     
            occupancies.append(comp_el_i)
            
            elements_subst.append(elem_i)
            
            
        # Readjust the occupancies by the weights found as might not be
        # exactly summing 1 because of the interpolation
        occupancies = np.asarray(occupancies)
        occupancies = occupancies/np.sum(occupancies)
            
        # Adding the non collapsed version of data  
        for occ_i, elem_i in zip(
                occupancies, elements_subst):
            
            if occ_i == 0:
                continue
            
            new_Atom = GPA_AtoMod.Atom(
                elem_i, x_cord, y_cord, z_cord, occ_i)
            
            new_Atoms_list_NoColapsOcc.append(new_Atom)
            
        
        # Add the colapsed version
        element_chosen = random.choices(
            elements_subst, weights = occupancies, k=1)[0] 
        
        new_Atom = GPA_AtoMod.Atom(
            element_chosen, x_cord, y_cord, z_cord)
        
        new_Atoms_list_colapsOcc.append(new_Atom)
        

    # Save both models
    
    # Non colapsed occupancies model, store the occupancies saved in the model
    noncolaps_occ_eelsed_path = path_global_strained_purged[:path_global_strained_purged.find('.xyz')] + '_EELS_NoColaps.xyz'
    
    GPA_AtoMod.save_xyf(
        new_Atoms_list_NoColapsOcc, noncolaps_occ_eelsed_path, save_occsDW = True)
    
    # Colapsed occupancies model, no need to store the occupancies as always 1
    colaps_occ_eelsed_path = path_global_strained_purged[:path_global_strained_purged.find('.xyz')] + '_EELS_Colaps.xyz'

    GPA_AtoMod.save_xyf(
        new_Atoms_list_colapsOcc, colaps_occ_eelsed_path, save_occsDW = False)
    
        
    return noncolaps_occ_eelsed_path, colaps_occ_eelsed_path







def EELS_Chemical_FullProcessing(
        micrographs_path, Box_strain_pixels, image_in_dataset, 
        query_hs_sign, template_hs_sign, coordinates_template_EELS, setting,
        path_global_strained_purged):
    '''
    Main function doing the whole substitution and generation of the atomistic
    model with the chemical composition as quantified by the maps provided
    It needs the query and template hyperspy signals as found by 
    Find_EELS_survey_in_micrograph, and the setting 1,2,3 (relation between
    template and query, and survey and image), to make the substitutions
    accordingly

    Parameters
    ----------
    micrographs_path : path to the micrographs
    Box_strain_pixels : pixels relative to the main image where the atomodel
                        is generated from, in format 
                        [B_strain_y_i, B_strain_y_f, B_strain_x_i, B_strain_x_f] 
    image_in_dataset : image_in_dataset object of the main image where the
                        analysis is taken or considered
    query_hs_sign : hyperspy signal of the query, whether it is the image or
                    EELS survey 
    template_hs_sign : hyperspy signal of the template, whether it is the image or
                    EELS survey 
    coordinates_template_EELS : coordiantes of the template matching process, in
                            pixels and relative to the query, in the form of
                            startX, startY, endX, endY = coordinates_template_EELS
    setting : 1,2 or 3, indicating the template-query relation
            as extracted directly from Find_EELS_survey_in_micrograph
        if survey_x_FOV > main_image_FOV and survey_y_FOV > main_image_FOV:
            so image fits inside the EELS survey
            setting = 1
        elif survey_x_FOV > main_image_FOV or survey_y_FOV > main_image_FOV:
            so survey is bigger than image in one of the dimensions, so we cut it
            and then fit inside the whole image
            setting = 2
        else:
            if image is bigger in FOV than survey, so survey is within the image
            setting = 3
    path_global_strained_purged : path to the strained purged atomodel, xyz file
                            that will have the substitution of its atoms
                            given the composition of EELS, region_cut_strained_purged.xyz

    Returns
    -------
    noncolaps_occ_eelsed_path : path to non colapsed model after substitution 
                                with EELS info
    colaps_occ_eelsed_path : path to colapsed model after substitution 
                            with EELS info

    '''
    
    
    
    # Box_strain_pixels coodinates where the box was extracted from in pxiels, from
    # the final coords from the model after the final recomputation 
    # Box_strain_pixels in pixels and convert to angstroms
    # pixels of the image indataset not of the eels survey
    
    # transform box coords in image in dataset
    pixel_size_whole = image_in_dataset.x_calibration
    total_pixels_whole = image_in_dataset.total_pixels
    
    region_to_strain_atomcords = np.array([Box_strain_pixels[2]*pixel_size_whole,
                                           (total_pixels_whole-Box_strain_pixels[1])*pixel_size_whole,
                                           Box_strain_pixels[3]*pixel_size_whole,
                                           (total_pixels_whole-Box_strain_pixels[0])*pixel_size_whole])
    
    # !!! UNITS CHANGE (nm --> angstroms)
    # Need to change the coordinates from nm to angstroms as the coords 
    # inside the file are in angstroms
    region_to_strain_atomcords = region_to_strain_atomcords*10
    
    
    # at the same folder level of the Micrographs/ folder, we should search for
    # the folder EELS/
    EELS_path = micrographs_path[:micrographs_path.rfind('Micrographs')] + 'EELS\\'


    orig_survey_filepaht = EELS_path + 'survey.dm3'
    

    orig_survey_hs = hs.load(orig_survey_filepaht)
    orig_survey_arraynp = np.asarray(orig_survey_hs)
    orig_survey_shape_y = np.shape(orig_survey_arraynp)[0]
    orig_survey_shape_x = np.shape(orig_survey_arraynp)[1]
    
    
    # Store the base information from the full quanti maps
    EELS_maps = dict()

    # Loop though files to find the elements involved
    for eels_signal in os.listdir(EELS_path):
        # Loop through all maps except the survey
        if 'survey' in eels_signal:
            continue
        
        
        element = eels_signal[:eels_signal.find('.dm3')].split()[0]
        
        # read the map and check if the size of the original survey has
        # changed from the original
        
        element_hs_quant = hs.load(EELS_path + eels_signal)
        element_np_quant = np.asarray(element_hs_quant)
        
        
        EELS_maps[element] = element_np_quant
        
        
    if setting == 1:
    # if survey_x_FOV > main_image_FOV and survey_y_FOV > main_image_FOV:
    # so the image fits within the EELS maps
        
        survey_hs = query_hs_sign
        main_image_hs = template_hs_sign
        
        # retrieve info from query and template
        survey_arraynp = np.asarray(survey_hs)
        survey_arraynp_st=(survey_arraynp-np.min(survey_arraynp))/np.max(survey_arraynp-np.min(survey_arraynp))
        survey_arraynp_st_int=np.uint8(255*survey_arraynp_st)
        
        survey_x_calibration = survey_hs.axes_manager['x'].scale
        survey_x_total_pixels = survey_hs.axes_manager['x'].size
        survey_x_units = survey_hs.axes_manager['x'].units
        survey_x_FOV = survey_x_calibration*survey_x_total_pixels  
        
        survey_y_calibration = survey_hs.axes_manager['y'].scale
        survey_y_total_pixels = survey_hs.axes_manager['y'].size
        survey_y_units = survey_hs.axes_manager['y'].units
        survey_y_FOV = survey_y_calibration*survey_y_total_pixels      


        main_image_hs = image_in_dataset.hyperspy_2Dsignal
        main_image_arraynp_st = image_in_dataset.image_arraynp_st
        main_image_arraynp_st_int = image_in_dataset.image_arraynp_st_int
        main_image_x_calibration = image_in_dataset.x_calibration
        main_image_total_pixels = image_in_dataset.total_pixels
        main_image_units = image_in_dataset.units
        main_image_FOV = image_in_dataset.FOV


        startX, startY, endX, endY = coordinates_template_EELS
        
        query_survey_crop = survey_arraynp_st_int[startY:endY, startX:endX]
        
        resized_query_survey_crop = cv2.resize(
            query_survey_crop, (main_image_total_pixels, main_image_total_pixels),
            interpolation = cv2.INTER_LINEAR)
        
        
        # store the resised info of the eels maps inside the template area
        EELS_maps_resized = dict()

        for element_i in EELS_maps:
            
            
            quantiEELS_el = EELS_maps[element_i]
            
            quantiEELS_el_crop = quantiEELS_el[startY:endY, startX:endX]
            
            # store the range of values within the crop to reset back the 
            # quantification values in that range after the standarisation
            # and int conversion that is necessary for the cv2.resize()
            min_val_crop = np.min(quantiEELS_el_crop)
            max_val_crop = np.max(quantiEELS_el_crop)
            
            
            quantiEELS_el_crop_st=(quantiEELS_el_crop-np.min(quantiEELS_el_crop))/np.max(quantiEELS_el_crop-np.min(quantiEELS_el_crop))
            quantiEELS_el_crop_int=np.uint8(255*quantiEELS_el_crop_st)
                        
            
            resized_quantiEELS_el_crop_int = cv2.resize(
                quantiEELS_el_crop_int, (main_image_total_pixels, main_image_total_pixels),
                interpolation = cv2.INTER_LINEAR)
            
            
            # set the values back to the original range of values:
            resized_quantiEELS_el_crop_st = resized_quantiEELS_el_crop_int/255
            
            resized_quantiEELS_el_crop = (max_val_crop - min_val_crop)*resized_quantiEELS_el_crop_st + min_val_crop
            
            EELS_maps_resized[element_i] = resized_quantiEELS_el_crop
            


        # Renormalise values from 0 to 1 (occupancies) at every pixel
        quanti_tensor = []
        
        for element_i in EELS_maps_resized:
            
            quanti_tensor.append(EELS_maps_resized[element_i])
            
        # Make sure there are no values below 0, which are unphysical
        quanti_tensor = np.asarray(quanti_tensor)
        quanti_tensor[quanti_tensor < 0] = 0
        
        # put everythin between the 0 and 1 range (for every pixel) 
        # so direct value of the occupancy
        quanti_tensor = quanti_tensor/np.sum(quanti_tensor, axis = 0)
        
        for quantmap, element in zip(
                quanti_tensor, EELS_maps_resized):
            
            EELS_maps_resized[element] = quantmap
            
            
        # Perform the actual substitution based on the elements extracted
        # and the maps used for that
        noncolaps_occ_eelsed_path, colaps_occ_eelsed_path = EELS_AtoModel_Substitution(
            path_global_strained_purged, EELS_maps_resized, 
            Box_strain_pixels, image_in_dataset, setting)
        
            
    # essentially same case as setting 3, but here we crop the EELS maps so it
    # fits within the main image (query)        
    elif setting == 2:    
    # elif survey_x_FOV > main_image_FOV or survey_y_FOV > main_image_FOV:

        # assing the new survey their template status
        main_image_hs = query_hs_sign
        survey_hs = template_hs_sign
        
        # retrieve info from query and template
        survey_arraynp = np.asarray(survey_hs)
        survey_arraynp_st=(survey_arraynp-np.min(survey_arraynp))/np.max(survey_arraynp-np.min(survey_arraynp))
        survey_arraynp_st_int=np.uint8(255*survey_arraynp_st)
        
        survey_x_calibration = survey_hs.axes_manager['x'].scale
        survey_x_total_pixels = survey_hs.axes_manager['x'].size
        survey_x_units = survey_hs.axes_manager['x'].units
        survey_x_FOV = survey_x_calibration*survey_x_total_pixels  
        
        survey_y_calibration = survey_hs.axes_manager['y'].scale
        survey_y_total_pixels = survey_hs.axes_manager['y'].size
        survey_y_units = survey_hs.axes_manager['y'].units
        survey_y_FOV = survey_y_calibration*survey_y_total_pixels      

        main_image_hs = image_in_dataset.hyperspy_2Dsignal
        main_image_arraynp_st = image_in_dataset.image_arraynp_st
        main_image_arraynp_st_int = image_in_dataset.image_arraynp_st_int
        main_image_x_calibration = image_in_dataset.x_calibration
        main_image_total_pixels = image_in_dataset.total_pixels
        main_image_units = image_in_dataset.units
        main_image_FOV = image_in_dataset.FOV



        startX, startY, endX, endY = coordinates_template_EELS        
        
        # crop of the query (main image) where template is located
        query_image_crop_template_size = main_image_arraynp_st_int[startY:endY, 
                                                                   startX:endX]
        
        # compute the overlap with the model and the box of the model
        comn_start_X = np.max([startX, Box_strain_pixels[2]])
        comn_end_X = np.min([endX, Box_strain_pixels[3]])
        comn_start_Y = np.max([startY, Box_strain_pixels[0]])
        comn_end_Y = np.min([endY, Box_strain_pixels[1]])
        
        # new box strain pixels to use to fit the model, in pixels of the
        # main image image_in_dataset
        New_Box_strain_pixels = [comn_start_Y, comn_end_Y, comn_start_X, comn_end_X]

        # cut the query, the main image, to see the new resized shape of 
        # the EELS maps, which have the full original area but different pixel
        # number, based on the overlap of model box and template box
        query_image_crop_comn_box = main_image_arraynp_st_int[comn_start_Y:comn_end_Y, 
                                                              comn_start_X:comn_end_X]
        
        
        # The fraction of the template that fits in the global image and in
        # the modeled region, with the same number of pixels as the main
        # image crop in that region (to have 1 to 1 relation with image and
        # the pixels in the quantitative maps)
        resized_template_to_query_crop = cv2.resize(
            survey_arraynp_st_int, 
            (np.shape(query_image_crop_template_size)[1], np.shape(query_image_crop_template_size)[1]),
            interpolation = cv2.INTER_LINEAR)
        


        # store the resised info of the eels maps inside the template area
        EELS_maps_resized = dict()

        for element_i in EELS_maps:
            
            quantiEELS_el = EELS_maps[element_i]
            
            # crop it the same way the survey was cropped
            quantiEELS_el_orig_x_pixs = np.shape(quantiEELS_el)[1]
            quantiEELS_el_orig_y_pixs = np.shape(quantiEELS_el)[0]
            
            # if it is the pixels in the x direction the ones that are different
            # so the ones that were cut, we need to cut the maps in that direction
            # as well
            if quantiEELS_el_orig_x_pixs != survey_x_total_pixels:
                
                diff_in_x = np.abs(survey_x_total_pixels - quantiEELS_el_orig_x_pixs)
                
                quantiEELS_el = quantiEELS_el[:, int(np.ceil(diff_in_x/2)):int(np.ceil(diff_in_x/2))+survey_x_total_pixels]
                
            # if it is the pixels in the y direction the ones that are different
            # so the ones that were cut, we need to cut the maps in that direction
            # as well
            if quantiEELS_el_orig_y_pixs != survey_y_total_pixels:
            
                diff_in_y = np.abs(survey_y_total_pixels - quantiEELS_el_orig_y_pixs)
                
                quantiEELS_el = quantiEELS_el[int(np.ceil(diff_in_y/2)):int(np.ceil(diff_in_y/2))+survey_y_total_pixels,:]



            # store the range of values within the crop to reset back the 
            # quantification values in that range after the standarisation
            # and int conversion that is necessary for the cv2.resize()
            min_val_EELS_el = np.min(quantiEELS_el)
            max_val_EELS_el = np.max(quantiEELS_el)
            
            
            quantiEELS_el_st=(quantiEELS_el-np.min(quantiEELS_el))/np.max(quantiEELS_el-np.min(quantiEELS_el))
            quantiEELS_el_int=np.uint8(255*quantiEELS_el_st)

            
            # Resize the map to fit within the query where the template 
            # was selected
            resized_quantiEELS_el_to_query_crop = cv2.resize(
                quantiEELS_el_int, 
                (np.shape(query_image_crop_template_size)[1], np.shape(query_image_crop_template_size)[0]),
                interpolation = cv2.INTER_LINEAR)
            
            # We genreate a copy of the main image, we put the rescaled
            # EELS, the tempate, into its position, and then we cut it in
            # the overlapped region with the simulation box
            
            query_image_with_quantiEELS_stacked = np.copy(main_image_arraynp_st_int)
            
            # overwrite the area where the template fits with the actual 
            # EELS quanti map rescaled
            query_image_with_quantiEELS_stacked[startY:endY, 
                                                startX:endX] = resized_quantiEELS_el_to_query_crop
        
        
            # Now cut the image with the EELS map stacked with the simulation
            # box overlapping region
            
            resized_quantiEELS_el_Comn_crop_int = query_image_with_quantiEELS_stacked[comn_start_Y:comn_end_Y, 
                                                                                      comn_start_X:comn_end_X]
            
            # Recover the compositions to get the occupancies as stated in the
            # original quantiEELS_el signal
            
            resized_quantiEELS_el_Comn_crop_st = resized_quantiEELS_el_Comn_crop_int/255
            
            resized_quantiEELS_el_Comn_crop = (max_val_EELS_el - min_val_EELS_el)*resized_quantiEELS_el_Comn_crop_st + min_val_EELS_el
            
            # store the resized value that is common to the model box and EELS survey
            EELS_maps_resized[element_i] = resized_quantiEELS_el_Comn_crop
            
                        
        # Renormalise values from 0 to 1 (occupancies) at every pixel
        quanti_tensor = []
        
        for element_i in EELS_maps_resized:
            
            quanti_tensor.append(EELS_maps_resized[element_i])
            
        # Make sure there are no values below 0, which are unphysical
        quanti_tensor = np.asarray(quanti_tensor)
        quanti_tensor[quanti_tensor < 0] = 0
        
        # put everythin between the 0 and 1 range (for every pixel) 
        # so direct value of the occupancy
        quanti_tensor = quanti_tensor/np.sum(quanti_tensor, axis = 0)
        
        for quantmap, element in zip(
                quanti_tensor, EELS_maps_resized):
            
            EELS_maps_resized[element] = quantmap
            
        
        # Perform the actual substitution based on the elements extracted
        # and the maps used for that, with the coordinates in pixels within
        # the whole image_in_dataset, which now is the query, are the ones
        # where the atomodel box and EELS map within query coincide
        noncolaps_occ_eelsed_path, colaps_occ_eelsed_path = EELS_AtoModel_Substitution(
            path_global_strained_purged, EELS_maps_resized, 
            New_Box_strain_pixels, image_in_dataset, setting)

            
            
    elif setting == 3:
    # if survey_x_FOV < main_image_FOV and survey_y_FOV < main_image_FOV:
        
        
        main_image_hs = query_hs_sign
        survey_hs = template_hs_sign
        
        # retrieve info from query and template
        survey_arraynp = np.asarray(survey_hs)
        survey_arraynp_st=(survey_arraynp-np.min(survey_arraynp))/np.max(survey_arraynp-np.min(survey_arraynp))
        survey_arraynp_st_int=np.uint8(255*survey_arraynp_st)
        
        survey_x_calibration = survey_hs.axes_manager['x'].scale
        survey_x_total_pixels = survey_hs.axes_manager['x'].size
        survey_x_units = survey_hs.axes_manager['x'].units
        survey_x_FOV = survey_x_calibration*survey_x_total_pixels  
        
        survey_y_calibration = survey_hs.axes_manager['y'].scale
        survey_y_total_pixels = survey_hs.axes_manager['y'].size
        survey_y_units = survey_hs.axes_manager['y'].units
        survey_y_FOV = survey_y_calibration*survey_y_total_pixels      


        main_image_hs = image_in_dataset.hyperspy_2Dsignal
        main_image_arraynp_st = image_in_dataset.image_arraynp_st
        main_image_arraynp_st_int = image_in_dataset.image_arraynp_st_int
        main_image_x_calibration = image_in_dataset.x_calibration
        main_image_total_pixels = image_in_dataset.total_pixels
        main_image_units = image_in_dataset.units
        main_image_FOV = image_in_dataset.FOV


        startX, startY, endX, endY = coordinates_template_EELS        
        
        # crop of the query (main image) where template is located
        query_image_crop_template_size = main_image_arraynp_st_int[startY:endY, 
                                                                   startX:endX]
        
        # compute the overlap with the model and the box of the model
        comn_start_X = np.max([startX, Box_strain_pixels[2]])
        comn_end_X = np.min([endX, Box_strain_pixels[3]])
        comn_start_Y = np.max([startY, Box_strain_pixels[0]])
        comn_end_Y = np.min([endY, Box_strain_pixels[1]])
        
        # new box strain pixels to use to fit the model, in pixels of the
        # main image image_in_dataset
        New_Box_strain_pixels = [comn_start_Y, comn_end_Y, comn_start_X, comn_end_X]

        # cut the query, the main image, to see the new resized shape of 
        # the EELS maps, which have the full original area but different pixel
        # number, based on the overlap of model box and template box
        query_image_crop_comn_box = main_image_arraynp_st_int[comn_start_Y:comn_end_Y, 
                                                              comn_start_X:comn_end_X]
        
        
        # The fraction of the template that fits in the global image and in
        # the modeled region, with the same number of pixels as the main
        # image crop in that region (to have 1 to 1 relation with image and
        # the pixels in the quantitative maps)
        resized_template_to_query_crop = cv2.resize(
            survey_arraynp_st_int, 
            (np.shape(query_image_crop_template_size)[1], np.shape(query_image_crop_template_size)[1]),
            interpolation = cv2.INTER_LINEAR)
        
        
        # do the same for the actual EELS maps and store them 
        
        # store the resised info of the eels maps inside the template area
        EELS_maps_resized = dict()

        for element_i in EELS_maps:   
            
            quantiEELS_el = EELS_maps[element_i]
            
            # store the range of values within the crop to reset back the 
            # quantification values in that range after the standarisation
            # and int conversion that is necessary for the cv2.resize()
            min_val_EELS_el = np.min(quantiEELS_el)
            max_val_EELS_el = np.max(quantiEELS_el)
            
            
            quantiEELS_el_st=(quantiEELS_el-np.min(quantiEELS_el))/np.max(quantiEELS_el-np.min(quantiEELS_el))
            quantiEELS_el_int=np.uint8(255*quantiEELS_el_st)

            
            # Resize the map to fit within the query where the template 
            # was selected
            resized_quantiEELS_el_to_query_crop = cv2.resize(
                quantiEELS_el_int, 
                (np.shape(query_image_crop_template_size)[1], np.shape(query_image_crop_template_size)[0]),
                interpolation = cv2.INTER_LINEAR)
            
            plt.imshow(resized_quantiEELS_el_to_query_crop)
            plt.show()
            # We genreate a copy of the main image, we put the rescaled
            # EELS, the tempate, into its position, and then we cut it in
            # the overlapped region with the simulation box
            
            query_image_with_quantiEELS_stacked = np.copy(main_image_arraynp_st_int)
            
            # overwrite the area where the template fits with the actual 
            # EELS quanti map rescaled
            query_image_with_quantiEELS_stacked[startY:endY, 
                                                startX:endX] = resized_quantiEELS_el_to_query_crop
        
        
            # Now cut the image with the EELS map stacked with the simulation
            # box overlapping region
            
            resized_quantiEELS_el_Comn_crop_int = query_image_with_quantiEELS_stacked[comn_start_Y:comn_end_Y, 
                                                                                      comn_start_X:comn_end_X]
            
            # Recover the compositions to get the occupancies as stated in the
            # original quantiEELS_el signal
            
            resized_quantiEELS_el_Comn_crop_st = resized_quantiEELS_el_Comn_crop_int/255
            
            resized_quantiEELS_el_Comn_crop = (max_val_EELS_el - min_val_EELS_el)*resized_quantiEELS_el_Comn_crop_st + min_val_EELS_el
            
            # store the resized value that is common to the model box and EELS survey
            EELS_maps_resized[element_i] = resized_quantiEELS_el_Comn_crop
            
                        
        # Renormalise values from 0 to 1 (occupancies) at every pixel
        quanti_tensor = []
        
        for element_i in EELS_maps_resized:
            
            quanti_tensor.append(EELS_maps_resized[element_i])
            
        # Make sure there are no values below 0, which are unphysical
        quanti_tensor = np.asarray(quanti_tensor)
        quanti_tensor[quanti_tensor < 0] = 0
        
        # put everythin between the 0 and 1 range (for every pixel) 
        # so direct value of the occupancy
        quanti_tensor = quanti_tensor/np.sum(quanti_tensor, axis = 0)
        
        for quantmap, element in zip(
                quanti_tensor, EELS_maps_resized):
            
            EELS_maps_resized[element] = quantmap
            
        
        # Perform the actual substitution based on the elements extracted
        # and the maps used for that, with the coordinates in pixels within
        # the whole image_in_dataset, which now is the query, are the ones
        # where the atomodel box and EELS map within query coincide
        noncolaps_occ_eelsed_path, colaps_occ_eelsed_path = EELS_AtoModel_Substitution(
            path_global_strained_purged, EELS_maps_resized, 
            New_Box_strain_pixels, image_in_dataset, setting)

    
    # # Loop though files to find the elements involved
    # for eels_signal in os.listdir(EELS_path):

    return noncolaps_occ_eelsed_path, colaps_occ_eelsed_path


# # Example code

# real_calibration_factor=0.97  # change calibration of the image

# micrographs_path = r'E:\Arxius varis\PhD\4rth_year\Global_ML_Results\GeQW2\Micrographs\\'

# EELS_Quant_available = Available_EELS_data_Checker(
#     micrographs_path)


# images_in_dataset_list, pixel_sizes = HighToLowTM.Browse_Dataset_Images_and_Recalibrate(
#     micrographs_path, real_calibration_factor)

# query_hs_sign, template_hs_sign, coordinates_template_EELS, scale_factor, setting = Find_EELS_survey_in_micrograph(
#     micrographs_path, images_in_dataset_list[0])


# B_strain_width = 400
# B_strain_height = 400
# B_strain_y_i = 1000
# B_strain_y_f = B_strain_y_i + B_strain_height
# B_strain_x_i = 250
# B_strain_x_f = B_strain_x_i + B_strain_width


# B_strain_width = images_in_dataset_list[0].total_pixels
# B_strain_height = images_in_dataset_list[0].total_pixels
# B_strain_y_i = 0
# B_strain_y_f = B_strain_y_i + B_strain_height
# B_strain_x_i = 0
# B_strain_x_f = B_strain_x_i + B_strain_width



# Box_strain_pixels = [B_strain_y_i, B_strain_y_f, B_strain_x_i, B_strain_x_f] 

# path_global_strained_purged = r'E:\Arxius varis\PhD\4rth_year\Global_Results_per_Device_Workstation\GeQW2\Results_GeQW2\model_cells\GeQW2_strained\region_cut_strained_purged.xyz'

# noncolaps_occ_eelsed_path, colaps_occ_eelsed_path = EELS_Chemical_FullProcessing(
#     micrographs_path, Box_strain_pixels, images_in_dataset_list[0], 
#     query_hs_sign, template_hs_sign, coordinates_template_EELS, setting,
#     path_global_strained_purged)










