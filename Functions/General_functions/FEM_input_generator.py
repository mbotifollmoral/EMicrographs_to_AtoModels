# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 12:21:40 2023

@author: Marc
"""

#TO RUN IN PYTHON >=3.8 (nextnanopy support >=3.8)

'''
CELL intended for linking nextnanopy with the nextnano installation and licenses
REPEAT and EDIT in case the new installation modifies the paths
'''

import numpy as np
import gdspy
from pathlib import Path
import os
import sys

sys.path.append(r'E:\Arxius varis\PhD\4rth_year\Code\Functions\General_functions')
sys.path.append(r'E:\Arxius varis\PhD\4rth_year\Code\Atomistic_model_builder')

import Atomistic_Model_Builder as AtomBuild
import GPA_atomistic_combiner as GPA_AtoMod



def Config_nextnano():
    '''
    Prepare the nextnano environment to link python with software

    Returns
    -------
    None.

    '''
    import nextnanopy as nn

    conf=nn.config
    
    
    nn.config.set('nextnano++','exe',r'C:\Program Files\nextnano\2021_09_03\nextnano++\bin 64bit\nextnano++_Intel_64bit.exe')
    nn.config.set('nextnano++','license',r'C:\Users\Marc\Documents\nextnano\License\License_nnp.lic')
    nn.config.set('nextnano++','database',r'C:\Program Files\nextnano\2021_09_03\nextnano++\Syntax\database_nnp.in')
    nn.config.set('nextnano++','outputdirectory',r'C:\Users\Marc\Documents\nextnano\Output')
    
    nn.config.set('nextnano3','exe',r'C:\Program Files\nextnano\2021_09_03\nextnano3\Intel 64bit\nextnano3_Intel_64bit.exe')
    nn.config.set('nextnano3','license',r'C:\Users\Marc\Documents\nextnano\License\License_nnp.lic')
    nn.config.set('nextnano3','database',r'C:\Program Files\nextnano\2021_09_03\nextnano3\Syntax\database_nn3.in')
    nn.config.set('nextnano3','outputdirectory',r'C:\Users\Marc\Documents\nextnano\Output')
    
    nn.config.set('nextnano.MSB','database',r'C:\Program Files\nextnano\2021_09_03\nextnano.MSB\Syntax\Materials.xml')
                  
    nn.config.save() #save permanently
    
    conf=nn.config



def Build_FEM_gds(
        analysed_image, conts_vertx_per_region, 
        model_cells_filepath, only_crystalline = True):
    '''
    Build the gds file having as coordiantes the vertexs of the contour 
    drawn by the segmentation and posterior contour finding and posterior, if
    wanted, smoothing by vertex skipping

    Parameters
    ----------
    analysed_image : 
    conts_vertx_per_region : 
    model_cells_filepath : 
    only_crystalline : if True, only the regions where a crystal phase was 
                identified will create a polygon and will be considered for 
                the FEM model, if False, all the segmented regions 
        DESCRIPTION. The default is True.

    Returns
    -------
    FEM_models_filepath : str to path were FEM files are located

    '''
    
    
    # create folder for the FEM models files
    FEM_models_filepath = model_cells_filepath + 'FEM_' + analysed_image.image_in_dataset.name + '\\'
    path_FEM_models = os.path.isdir(FEM_models_filepath)
    if path_FEM_models == False:
        os.mkdir(FEM_models_filepath)
    
    # Initialise the gds file library
    # The GDSII file is called a library, which contains multiple cells.
    gds_lib = gdspy.GdsLibrary(unit=1e-9, precision=1e-12)
    
    # Geometry must be placed in cells.
    gds_cell = gds_lib.new_cell(analysed_image.image_in_dataset.name + '_cell')
    
    
    # Information from all the crops done to the image
    crop_outputs_dict = analysed_image.Crop_outputs
    
    # Loop through the regions and build the atomic model from each region
    for label_segm_region in range(1, analysed_image.crop_index):
        
        image_crop_hs_signal = crop_outputs_dict[str(label_segm_region) + '_hs_signal']
        crop_list_refined_cryst_spots = crop_outputs_dict[str(label_segm_region) + '_list_refined_cryst_spots']


        # also draw the contours for the amorphous regions or unfound ones
        # if only_crystalline == False
        
        if only_crystalline == True:
        
            if len(crop_list_refined_cryst_spots) == 0:
                continue
            
        # if doesnt work use [str(int(label_segm_region))+'contours'] rel positions
        rel_vertexs_segm_reg = np.copy(conts_vertx_per_region[str(int(label_segm_region))+'_rel_vertexs'])
        ys_relvertxs = analysed_image.image_in_dataset.FOV - rel_vertexs_segm_reg[:,0]
        xs_relvertxs = rel_vertexs_segm_reg[:,1]
        # swap columns
        rel_vertexs_segm_reg_flip = np.copy(rel_vertexs_segm_reg)
        
        rel_vertexs_segm_reg_flip[:,0] = xs_relvertxs
        rel_vertexs_segm_reg_flip[:,1] = ys_relvertxs
        
        
        
        # convert to list of lists
        rel_vertexs_segm_reg_list = [list(i) for i in rel_vertexs_segm_reg_flip]        
        
        # print(rel_vertexs_segm_reg_list)
        
        layer_label = {"layer": 0, "datatype": int(label_segm_region)}
        polygonshape_label = gdspy.Polygon(rel_vertexs_segm_reg_list, **layer_label)
        gds_cell.add(polygonshape_label)
        
        
        
        
        # list_of_nnvertexs_per_polygon is a list of lists with two elements x and y coord 
        # in this order
        
        # in conts_vertx_per_region[str(int(label_segm_region))+'_rel_vertexs']
        # the order is 0 the y and 1 the x so swap the columns
        # to fit in this format
        
        # probably also correct for shift in the y coordinate
        # it will be upside down
        
        
        
    # Save the library in a file called 'first.gds'.
    gds_lib.write_gds(FEM_models_filepath + 'gds_' + analysed_image.image_in_dataset.name + '.gds')
    
    # Optionally, save an image of the cell as SVG.
    gds_cell.write_svg(FEM_models_filepath + 'svg_' + analysed_image.image_in_dataset.name + '.svg')
    
    # Display all cells using the internal viewer.
    gdspy.LayoutViewer(gds_lib)  
    
    gdspy.current_library = gdspy.GdsLibrary()
    
    del gds_lib
    
    
    return FEM_models_filepath




def Extract_FEM_input_parameters(
        analysed_image, model_cells_filepath, only_crystalline = True):
    '''
    Build the input file information with the orientation of the phases and 
    their labels as found by the crystal finding algorithm
    No special format in this function, just presenting the informaiton 
    in an ordered format

    Parameters
    ----------
    analysed_image : TYPE
        DESCRIPTION.
    model_cells_filepath : TYPE
        DESCRIPTION.
    only_crystalline : TYPE, optional
        DESCRIPTION. The default is True.

    Returns
    -------
    input_FEM_filename : TYPE
        DESCRIPTION.

    '''
    
    
    # create folder for the FEM models files
    FEM_models_filepath = model_cells_filepath + 'FEM_' + analysed_image.image_in_dataset.name + '\\'
    path_FEM_models = os.path.isdir(FEM_models_filepath)
    if path_FEM_models == False:
        os.mkdir(FEM_models_filepath)
        
    
    # Information from all the crops done to the image
    crop_outputs_dict = analysed_image.Crop_outputs
    
    FEM_info_list = ['FEM_input_data_' + analysed_image.image_in_dataset.name + '\n', ]
    FEM_info_list = FEM_info_list + ['\n', 'REGIONS\n', '\n']
    # Loop through the regions and build the atomic model from each region
    for label_segm_region in range(1, analysed_image.crop_index):
        
        image_crop_hs_signal = crop_outputs_dict[str(label_segm_region) + '_hs_signal']
        crop_list_refined_cryst_spots = crop_outputs_dict[str(label_segm_region) + '_list_refined_cryst_spots']
        
        
        # if no crystal is found, either amorphous or bad identified
        # in any case, do not consider it for the atomistic model
        # or build an amorphous one as next step    
        
        if only_crystalline == True:
        
            if len(crop_list_refined_cryst_spots) == 0:
                continue
        
        if len(crop_list_refined_cryst_spots) == 0:
            text_label = 'Label_of_region = ' + str(int(label_segm_region)) + '\n'
            text_crystal = 'Phase_name_found = No_crystal_found\n'
    
            FEM_info_list = FEM_info_list + [text_label, text_crystal, '\n']
            continue
        
        # most likely crystal found
        best_cryst_spot = crop_list_refined_cryst_spots[0]
        zone_axis_found =  best_cryst_spot.ZA
        scored_spot_pair = best_cryst_spot.spot_pairs_obj[0]
        hkl1_reference = scored_spot_pair.hkl1_reference
        hkl1_angle_to_x = scored_spot_pair.spot1_angle_to_x
        hkl2_reference = scored_spot_pair.hkl2_reference
        hkl2_angle_to_x = scored_spot_pair.spot2_angle_to_x
        found_phase_name = best_cryst_spot.phase_name
        
        
        # Here check if there exists the virtual crystal of the found_phase_name
        # and build the cell based on this or not if it does not exist
        cif_cell_filepath = model_cells_filepath + found_phase_name + '.cif'
        
        # Both the angle and atomistic model building needs to be done through 
        # either the modified cif (virtual) or the base one
        # Find the rotation we need to induce to the default atomistic model 
        # to rotate it to the found orientation
        
        plane_final_cartesian_x_lab, direction_final_cartesian_x_lab = AtomBuild.Find_plane_pointing_to_final_cartesian_x_axis(
            cif_cell_filepath, scored_spot_pair, tolerance_diff = 0.3, suface_basis_choice = 'plane')

        # fill the information in the file
        text_label = 'Label_of_region = ' + str(int(label_segm_region)) + '\n'
        text_crystal = 'Phase_name_found = ' +str(found_phase_name) + '\n'
        text_ZA = 'Zone_axis_found = ' + str(zone_axis_found) + '\n'
        text_plane_x = 'Plane_pointing_to_Cartesian_x_axis = ' + str(plane_final_cartesian_x_lab) + '\n'
        text_direct_x = 'Direction_pointing_to_Cartesian_x_axis = ' + str(direction_final_cartesian_x_lab) + '\n'

        FEM_info_list = FEM_info_list + [text_label, text_crystal, text_ZA, 
                                         text_plane_x, text_direct_x]

        FEM_info_list = FEM_info_list + ['\n']
        
    FEM_info_list = FEM_info_list + ['END\n']



    # now do the same with the files containing the label identifiers
    input_FEM_filename = FEM_models_filepath + 'FEM_Input_' + analysed_image.image_in_dataset.name + '.txt'
    
    filename_femfile = Path(input_FEM_filename)
    file_already_created_femfile = filename_femfile.is_file()
    
    if file_already_created_femfile == True:
        # overwrite the file
        with open(input_FEM_filename, "w+") as f:
            f.truncate(0)
            f.writelines(FEM_info_list)
            f.close()
    else:
        # create a new file
        with open(input_FEM_filename, 'w+') as f:
            f.writelines(FEM_info_list)
            f.close()

    return input_FEM_filename




