# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 12:18:22 2022

@author: Marc
"""

'''
From a defined geometry, written by coordinates of vertex that follow 
each other in a sorted way to close a loop defining a closed 2D path that will 
define a transverally invariant segment that can be placed with atoms
'''



import numpy as np
import matplotlib.pyplot as plt
import os
import gdspy
from pathlib import Path
import sys
import ase
from ase.io import read, write
from ase.visualize import view
from abtem.visualize import show_atoms
from ase.build import surface, make_supercell, find_optimal_cell_shape, rotate
import mendeleev as mdl


sys.path.append(r'E:\Arxius varis\PhD\3rd_year\Code\Functions')

# General functions
import Segmentation_1stAprox as Segment


'''
Functions for atomistic modelling
'''


def uce_to_cif(
        path_to_uce_unitcell, phase_name, save_filepath,
        templato_filepath = r'E:\Arxius varis\PhD\3rd_year\Code\Functions\difftool_dll\templato.txt'):
    '''
    Function to convert the .uce files used for the simulation of the diffraction 
    pattern to .cif files containing the exact same information but to be used
    to build the atomistic model, as ASE demands .cif

    Parameters
    ----------
    path_to_uce_unitcell : string to .uce unit cell 
    phase_name : string with name of phase (the filename, which must be coincident 
                                            with the phase itself)
    save_filepath : where to save the .cif file, folder directory without the file name nor extension
    templato_filepath : path to templato folder, change it if changes
        DESCRIPTION. The default is r'E:\Arxius varis\PhD\3rd_year\Code\Functions\difftool_dll\templato.txt'.

    Returns
    -------
    path_to_cif_unitcell : str, path to the newly generated .cif file

    '''
    
    
    
    raw_file_data = open(path_to_uce_unitcell)
    
    raw_datalines = raw_file_data.readlines()
    
    list_split_lines = []
    
    for line in raw_datalines:
        list_split_lines.append(line.split())
        
    
    numb_non_eq_atoms = int(list_split_lines[0][1])
    
    a_cell = list_split_lines[1][1]
    b_cell = list_split_lines[1][2]
    c_cell = list_split_lines[1][3]
    alpha_cell = list_split_lines[1][4] 
    beta_cell = list_split_lines[1][5]
    gamma_cell = list_split_lines[1][6]
    
    
    atoms_descript = list_split_lines[2: 2 + numb_non_eq_atoms]
    
    RGNR = list_split_lines[2 + numb_non_eq_atoms][1]
    
    if len(list_split_lines) >= 2 + numb_non_eq_atoms + 2:
        # !!! CONVENTION: the uce files start with STN from 0 to n -1, while in templato we
        # catch from 1 to n
        STN = str(int(int(list_split_lines[2 + numb_non_eq_atoms + 1][1]) + 1))
        
    else:
        STN = '1'
    
        
        # if not defined then set to 0 as default spaace group setting
        
    
    cif_lines = []
    
    cif_name = 'data_' + phase_name + '\n'
    cif_method = '_computing_structure_solution' + " 'Automodel'" + '\n'
    a_cif = '_cell_length_a   ' + a_cell + '\n'
    b_cif = '_cell_length_b   ' + b_cell + '\n'
    c_cif = '_cell_length_c   ' + c_cell + '\n'
    alpha_cif = '_cell_angle_alpha  ' + alpha_cell + '\n'
    beta_cif = '_cell_angle_beta  ' + beta_cell + '\n'
    gamma_cif = '_cell_angle_gamma  ' + gamma_cell + '\n'
    space_group = '_symmetry_Int_Tables_number ' + RGNR + '\n'
    
    
    # go through the templato file to get throuhg HM symbols
    # and symmetry operations 
    
    raw_templato_data = open(templato_filepath, encoding = 'utf-8')
    
    raw_templato_lines = raw_templato_data.readlines()
    
    
    # gather all the space group info from the templato file
    split_space_groups_info = []
    
    index_start = 0
    index_end = 0
    for index_line, line in enumerate(raw_templato_lines):
        
        if line[:8] == 'HMSymbol':
            index_start = index_line
        if line[0] == '#':
            index_end = index_line
            
            
        if index_end >=  index_start +2 and index_line == index_end: 
            split_space_groups_info.append(raw_templato_lines[index_start:index_end])   
        
        
    hmsymbols = []    
        
    for sg in split_space_groups_info:
        
        hmsymb = sg[0][::-1][sg[0][::-1].find('\n')+1:sg[0][::-1].find('|')][::-1]
        hmsymbols.append(hmsymb)
        
    templ_space_groups_numb = []    
    
    for sg in split_space_groups_info:
        
        sgn = sg[0][9: 9+ sg[0][9:].find('|')]
        templ_space_groups_numb.append(sgn)
        
        
    symmetry_ops = []
    
    for sg in split_space_groups_info:
        symops = sg[1:-1]
        symmetry_ops.append(symops)
        
        
    templato_STNs = []   
     
    for sg in split_space_groups_info:
        
        sgn = sg[0][9: 9+ sg[0][9:].find('|')]
    
        num_sym_ops = sg[0][sg[0].find(sgn) + len(sgn) +1 : ]
        
        stn = num_sym_ops[num_sym_ops.find('|')+1: num_sym_ops.find('|') +1 + num_sym_ops[num_sym_ops.find('|')+1:].find('|')]
        
        templato_STNs.append(stn)
        
    
    # find the sym ops, hg symb
    
    symmetry_operations_formatted = []
    
    for sgn, stn, hmsymb, symops in zip(
            templ_space_groups_numb, templato_STNs, hmsymbols, symmetry_ops):
        # filter by space group number
        if sgn == RGNR:
            
            # check if there are more than 1 setting in that given RGNR
            # as there are some discrepancies with where STN starts from, either 0
            # or 1, then better to avoid this if there is just one STN then chose the
            # only one and if more, then for sure there will be a 1
            all_rgnrs_inlist = [el for el in templ_space_groups_numb if el == RGNR]
            if len(all_rgnrs_inlist) == 1:
                stn = STN
            
            # filter by the STN 
            if stn == STN:
                # get the herman maguin symbol
                hmsymbol_found = hmsymb
                
                # get the symmetry operations
                symop_index = 1
                for symop in symops:
                    
                    if symop_index < 10:
                        
                        symop_formatted = '  ' + str(int(symop_index)) + '   '
                    
                        coords = symop[3+len(str(symop_index -1)) + 2 : symop.find('\n')] 
                        
                        symop_formatted = symop_formatted + "'" + coords + "'" + '\n'
                        
                    elif symop_index == 10:
                        symop_formatted = ' ' + str(int(symop_index)) + '   '
                        
                        coords = symop[3+len(str(symop_index -1)) + 2 : symop.find('\n')] 
                        
                        symop_formatted = symop_formatted + "'" + coords + "'" + '\n'
                    else:
                        symop_formatted = ' ' + str(int(symop_index)) + '   '
                        
                        coords = symop[3+len(str(symop_index -1)) + 2 : symop.find('\n')] 
                        
                        symop_formatted = symop_formatted + "' " + coords + "'" + '\n'
                        
                    symmetry_operations_formatted.append(symop_formatted)
                    
                    symop_index = symop_index +1
                
    # Find Bravais cell from HM symbol
    bravais_type =  hmsymbol_found[0]   
    
    # Adjust the symmetry operations to include in the cif file depending on
    # its Bravais cell type       
    
    if bravais_type == 'P':
        #primitive cell, us the base sym ops
        total_symmetry_operations_formatted = symmetry_operations_formatted
        
    elif bravais_type == 'A':
        # store the newly generated symops
        additional_symops = []
        
        for symop in symmetry_operations_formatted:
            
            original_string = symop[symop.find("'") + 1 : symop.find("'") + 1 + symop[symop.find("'") + 1:].find("'")]
            
            x_pos = original_string[:original_string.find(',')]
            y_pos = original_string[original_string.find(',') + 1 : original_string.find(',') + 1 + original_string[original_string.find(',') + 1 : ].find(',')]
            z_pos = original_string[original_string.find(y_pos):][original_string[original_string.find(y_pos):].find(',')+1:]
        
            # only variation
            x_pos_n = x_pos 
            y_pos_n = y_pos + '+1/2'
            z_pos_n = z_pos + '+1/2'
            
            symop_formatted = ' ' + str(int(symop_index)) + '   '
            coords = x_pos_n + ',' + y_pos_n + ',' + z_pos_n
            symop_formatted = symop_formatted + "'" + coords + "'" + '\n'
            
            additional_symops.append(symop_formatted)
            
            symop_index = symop_index +1
        
        
        total_symmetry_operations_formatted = symmetry_operations_formatted + additional_symops
        
    elif bravais_type == 'B':
        # store the newly generated symops
        additional_symops = []
        
        for symop in symmetry_operations_formatted:
            
            original_string = symop[symop.find("'") + 1 : symop.find("'") + 1 + symop[symop.find("'") + 1:].find("'")]
            
            x_pos = original_string[:original_string.find(',')]
            y_pos = original_string[original_string.find(',') + 1 : original_string.find(',') + 1 + original_string[original_string.find(',') + 1 : ].find(',')]
            z_pos = original_string[original_string.find(y_pos):][original_string[original_string.find(y_pos):].find(',')+1:]
        
            # only variation
            x_pos_n = x_pos + '+1/2' 
            y_pos_n = y_pos
            z_pos_n = z_pos + '+1/2'
            
            symop_formatted = ' ' + str(int(symop_index)) + '   '
            coords = x_pos_n + ',' + y_pos_n + ',' + z_pos_n
            symop_formatted = symop_formatted + "'" + coords + "'" + '\n'
            
            additional_symops.append(symop_formatted)
            
            symop_index = symop_index +1
        

        
        total_symmetry_operations_formatted = symmetry_operations_formatted + additional_symops
    
    elif bravais_type == 'C':
        # store the newly generated symops
        additional_symops = []
        
        for symop in symmetry_operations_formatted:
            
            original_string = symop[symop.find("'") + 1 : symop.find("'") + 1 + symop[symop.find("'") + 1:].find("'")]
            
            x_pos = original_string[:original_string.find(',')]
            y_pos = original_string[original_string.find(',') + 1 : original_string.find(',') + 1 + original_string[original_string.find(',') + 1 : ].find(',')]
            z_pos = original_string[original_string.find(y_pos):][original_string[original_string.find(y_pos):].find(',')+1:]
        
            # only variation
            x_pos_n = x_pos + '+1/2' 
            y_pos_n = y_pos + '+1/2'
            z_pos_n = z_pos
            
            symop_formatted = ' ' + str(int(symop_index)) + '   '
            coords = x_pos_n + ',' + y_pos_n + ',' + z_pos_n
            symop_formatted = symop_formatted + "'" + coords + "'" + '\n'
            
            additional_symops.append(symop_formatted)
            
            symop_index = symop_index +1
        

        total_symmetry_operations_formatted = symmetry_operations_formatted + additional_symops
        
    elif bravais_type == 'F':
        # store the newly generated symops
        additional_symops = []
        
        for symop in symmetry_operations_formatted:
            
            original_string = symop[symop.find("'") + 1 : symop.find("'") + 1 + symop[symop.find("'") + 1:].find("'")]
            
            x_pos = original_string[:original_string.find(',')]
            y_pos = original_string[original_string.find(',') + 1 : original_string.find(',') + 1 + original_string[original_string.find(',') + 1 : ].find(',')]
            z_pos = original_string[original_string.find(y_pos):][original_string[original_string.find(y_pos):].find(',')+1:]
        
            # first variation
            x_pos_n = x_pos + '+1/2'
            y_pos_n = y_pos + '+1/2'
            z_pos_n = z_pos
            
            symop_formatted = ' ' + str(int(symop_index)) + '   '
            coords = x_pos_n + ',' + y_pos_n + ',' + z_pos_n
            symop_formatted = symop_formatted + "'" + coords + "'" + '\n'
            
            additional_symops.append(symop_formatted)
            
            symop_index = symop_index +1
            
            # second variation
            x_pos_n = x_pos 
            y_pos_n = y_pos + '+1/2'
            z_pos_n = z_pos + '+1/2'
            
            symop_formatted = ' ' + str(int(symop_index)) + '   '
            coords = x_pos_n + ',' + y_pos_n + ',' + z_pos_n
            symop_formatted = symop_formatted + "'" + coords + "'" + '\n'
            
            additional_symops.append(symop_formatted)
        
            symop_index = symop_index +1
            
            # third variation
            x_pos_n = x_pos + '+1/2'
            y_pos_n = y_pos 
            z_pos_n = z_pos + '+1/2'
            
            symop_formatted = ' ' + str(int(symop_index)) + '   '
            coords = x_pos_n + ',' + y_pos_n + ',' + z_pos_n
            symop_formatted = symop_formatted + "'" + coords + "'" + '\n'
            
            additional_symops.append(symop_formatted)
            
            symop_index = symop_index +1
        
        
        total_symmetry_operations_formatted = symmetry_operations_formatted + additional_symops
        
    elif bravais_type == 'I':
        # store the newly generated symops
        additional_symops = []
        
        for symop in symmetry_operations_formatted:
            
            original_string = symop[symop.find("'") + 1 : symop.find("'") + 1 + symop[symop.find("'") + 1:].find("'")]
            
            x_pos = original_string[:original_string.find(',')]
            y_pos = original_string[original_string.find(',') + 1 : original_string.find(',') + 1 + original_string[original_string.find(',') + 1 : ].find(',')]
            z_pos = original_string[original_string.find(y_pos):][original_string[original_string.find(y_pos):].find(',')+1:]
        
            # only variation
            x_pos_n = x_pos + '+1/2' 
            y_pos_n = y_pos + '+1/2'
            z_pos_n = z_pos + '+1/2'
            
            symop_formatted = ' ' + str(int(symop_index)) + '   '
            coords = x_pos_n + ',' + y_pos_n + ',' + z_pos_n
            symop_formatted = symop_formatted + "'" + coords + "'" + '\n'
            
            additional_symops.append(symop_formatted)
            
            symop_index = symop_index +1
        

        total_symmetry_operations_formatted = symmetry_operations_formatted + additional_symops
        
    else:    
    # elif bravais_type == 'R':
        
        # check if it is hexagonal or normal 
        # so check if gamma == 120
        # if so add the syms if not just keep it as originally
        
        additional_symops = []  
        
        if int(float(gamma_cell)) == 120:
            # then add the extra sym ops relative to rombohedral hexagonal setting
            for symop in symmetry_operations_formatted:
                
                original_string = symop[symop.find("'") + 1 : symop.find("'") + 1 + symop[symop.find("'") + 1:].find("'")]
                
                x_pos = original_string[:original_string.find(',')]
                y_pos = original_string[original_string.find(',') + 1 : original_string.find(',') + 1 + original_string[original_string.find(',') + 1 : ].find(',')]
                z_pos = original_string[original_string.find(y_pos):][original_string[original_string.find(y_pos):].find(',')+1:]
            
                # first variation
                x_pos_n = x_pos + '+2/3'
                y_pos_n = y_pos + '+1/3'
                z_pos_n = z_pos + '+1/3'
                
                symop_formatted = ' ' + str(int(symop_index)) + '   '
                coords = x_pos_n + ',' + y_pos_n + ',' + z_pos_n
                symop_formatted = symop_formatted + "'" + coords + "'" + '\n'
                
                additional_symops.append(symop_formatted)
                
                symop_index = symop_index +1
                
                # second variation
                x_pos_n = x_pos + '+1/3'
                y_pos_n = y_pos + '+2/3'
                z_pos_n = z_pos + '+2/3'
                
                symop_formatted = ' ' + str(int(symop_index)) + '   '
                coords = x_pos_n + ',' + y_pos_n + ',' + z_pos_n
                symop_formatted = symop_formatted + "'" + coords + "'" + '\n'
                
                additional_symops.append(symop_formatted)
            
                symop_index = symop_index +1
                
            
            total_symmetry_operations_formatted = symmetry_operations_formatted + additional_symops
            
        else:
            # do not add anything (add empty list)
            total_symmetry_operations_formatted = symmetry_operations_formatted + additional_symops
        

             
    # complete the lines with the correct formatting            
                
    hmsymb_line =  '_symmetry_space_group_name_H-M ' + "'" + hmsymbol_found + "'" + '\n'
    inter_lines_1 = ['loop_\n',
                     '_symmetry_equiv_pos_site_id\n',
                     '_symmetry_equiv_pos_as_xyz\n']   
            
    
    inter_lines_2 = ["_atom_site_thermal_displace_type 'Uiso'\n",
                     'loop_\n',
                     '_atom_site_label\n',
                     '_atom_site_fract_x\n',
                     '_atom_site_fract_y\n',
                     '_atom_site_fract_z\n',
                     '_atom_site_occupancy\n',
                     '_atom_site_U_iso_or_equiv\n',
                     '_atom_site_absorption\n']         
        
    
    # default lines of cif
    # default lines of the space group
    # positions form templato.txt
        
        
    cif_lines = [cif_name,
                 cif_method,
                 a_cif,
                 b_cif,
                 c_cif,
                 alpha_cif,
                 beta_cif,
                 gamma_cif,
                 space_group,
                 hmsymb_line] + inter_lines_1 + total_symmetry_operations_formatted + inter_lines_2
    
    
    atoms_cif = []
    
    for atom in atoms_descript:
        
        element = atom[0]
        
        if len(element) == 1:
            # if element is just one letter
            space_in = '             '
        else:
            space_in = '            '
            
        element_line = element +space_in + atom[2] + '  ' + atom[3] + '  ' + atom[4] + '  ' + atom[5] + '  ' + atom[6] + '  ' + atom[7] + '\n'
    
        atoms_cif.append(element_line)
    
    cif_lines = cif_lines + atoms_cif
    
    
    # create the .cif file
    path_to_cif_unitcell = save_filepath + '\\' + phase_name + '.cif'
    
    path_to_cif_unitcell_Path = Path(path_to_cif_unitcell)
    file_already_created = path_to_cif_unitcell_Path.is_file()
    
    if file_already_created == True:
        # overwrite the file
        with open(path_to_cif_unitcell, "w+") as f:
            f.truncate(0)
            f.writelines(cif_lines)
            f.close()
    else:
        # create a new file
        with open(path_to_cif_unitcell, 'w+') as f:
            f.writelines(cif_lines)
            f.close()    
                
        
    return path_to_cif_unitcell



def cif_from_uce_All_found_phases(
        analysed_image, unit_cells_path):
    '''
    Ensure that all the phases found, and that will be included in
    the atomistic model, have a cif file representing them so the atomistic 
    model can be built either with the base cif or with the virtual unit cell

    Parameters
    ----------
    analysed_image :  analysed_image object of the only image analysed
    unit_cells_path: str to the uce unit cells
        
    Returns
    -------
    model_cells_filepath: str, path to unit_cells

    '''
    
    # Build a folder named model_cells in the path before the unit cells are checked
    # to place the conversion from uce to cif that are needed to build the models
    model_cells_filepath = unit_cells_path[:unit_cells_path.find('unit_cells')] + '\\' + 'model_cells' + '\\'
    path_model_cells = os.path.isdir(model_cells_filepath)
    if path_model_cells == False:
        os.mkdir(model_cells_filepath)
    
    crop_outputs_dict = analysed_image.Crop_outputs
    # Loop through the regions and build the atomic model from each region
    for label_segm_region in range(1, analysed_image.crop_index):
        
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
            
            
            
            # convert the file with phase_name in the 
            # unit_cells_path to cif in the model_cells folder  
            cif_cell_filepath = model_cells_filepath + found_phase_name + '.cif'
            
            # Check if the cif file exists in the model_cells directory
            cif_cell_filepath_Path = Path(cif_cell_filepath)
            cif_file_already_created = cif_cell_filepath_Path.is_file()
            
            if cif_file_already_created == False:
                
                # convert the .uce to .cif as it does not exist yet
                uce_cell_filepath = unit_cells_path + '\\' + found_phase_name + '.uce'
                
                cif_cell_filepath = uce_to_cif(
                    uce_cell_filepath, found_phase_name, model_cells_filepath)
    
    
    return model_cells_filepath




def Get_Smallest_Rectangular_Cell(
        polygon_vertex_list):
    '''
    Measure the Smallest possible rectangle that can be generated by the drawn
    region with the vertices adn that holds all the vertices inside, so that
    we can crop the sculpture of the desired region by piercing the contour
    accordingly
    No change of units performed

    Parameters
    ----------
    polygon_vertex_list : list with vertices drawing the contours, each element
        is a list with [x,y] coordinates
        
    rel_vertex in the contour list is stored in format [y,x]
    here it needs to be swapped to make it [x,y]

    Returns
    -------
    x_min_c, x_max_c, x_length, y_min_c, y_max_c, y_length.

    '''
    
    polygon_vertex_array = np.asarray(polygon_vertex_list)
    
    x_max_c = np.max(polygon_vertex_array[:,0])
    x_min_c = np.min(polygon_vertex_array[:,0])
    y_max_c = np.max(polygon_vertex_array[:,1])
    y_min_c = np.min(polygon_vertex_array[:,1])
    x_length = x_max_c - x_min_c
    y_length = y_max_c - y_min_c
    
    
    return x_min_c, x_max_c, x_length, y_min_c, y_max_c, y_length



def get_biggest_Rect_fitting_RotSCell(
        supercell_before_rotation, rotation_angle):
    
    '''
    After generating the min rectangle that can hold a rectangular cell of
    the wanted size x_length and y_length after a its rotation of 
    rotation_angle degrees, the square supercell is created. However, by security,
    it will become a bit bigger than the x_min and y_min computed by 
    Get_min_Rectangle_Allocating_rotCell(). Then , this function computes the
    actual biggest rectangle that could fit in this cell, which will be a bit
    bigger than the x_length and y_length originally computed by 
    Get_Smallest_Rectangular_Cell() that could already hold the segmented region
    By doing this after reescaling the coordinates of the rectangle within the 
    rotated supercell, we make even more sure the segmented curve fits inside
    '''
    
    x_min_max = supercell_before_rotation.get_cell()[0,0]
    y_min_max = supercell_before_rotation.get_cell()[1,1]
    
    #if 0, 90 exact, 1/0 and tan(90) has poroblems and gets a 
    # box bigger than possible, then adress it with the if statements
     
    if np.round(rotation_angle,10) == 90:
        rotation_angle = 89.999
        
    if np.round(rotation_angle,10) == -90:
        rotation_angle = -89.999
    
    if np.round(rotation_angle,10) == 0:
        rotation_angle = -0.00001
    
    x_length_max = (x_min_max/abs(np.cos(rotation_angle*(np.pi/180))))-(((np.tan(rotation_angle*(np.pi/180)))**2)/((np.tan(rotation_angle*(np.pi/180)))**2-1))*((x_min_max/abs(np.cos(rotation_angle*(np.pi/180))))-(y_min_max/abs(np.sin(rotation_angle*(np.pi/180)))))
    
    y_length_max = ((abs(np.tan(rotation_angle*(np.pi/180))))/((np.tan(rotation_angle*(np.pi/180)))**2-1))*((x_min_max/abs(np.cos(rotation_angle*(np.pi/180))))-(y_min_max/abs(np.sin(rotation_angle*(np.pi/180)))))
    
    return x_length_max, y_length_max
    





def Get_min_Rectangle_Allocating_rotCell(
        rotation_angle, x_length, y_length):
    '''
    After rotation the whole supercell a certain angle, a tilted rectangle is
    generated, which must fit a non-tilted (sides paralel to cartesian axes)
    rectangle that holds the dimensions of the smallest square containing the
    region drawn by the segmentation
    

    Parameters
    ----------
    rotation_angle : angle, in degrees, defined as positive in going from 
        the positive x horizontal axis to the vertical y, from -180º to 180º
    x_length : from Get_Smallest_Rectangular_Cell(), min horizontal side
            of the smallest rectangle holding the segmented region
    y_length : from Get_Smallest_Rectangular_Cell(), min vertical side
            of the smallest rectangle holding the segmented region

    Returns
    -------
    x_min : min horizontal side of the smallest rectangle that when rotated 
            can hold the non-rotated rectangle holding the segmented region
    y_min : min vertical side of the smallest rectangle that when rotated 
            can hold the non-rotated rectangle holding the segmented region
        
    '''
    
    # just make sure the angle goes from [-180, 180]
    if rotation_angle > 180:
        rotation_angle = rotation_angle-360
        
    if rotation_angle < -180:
        rotation_angle = 360 - np.abs(rotation_angle)
    
    x_min = x_length*abs(np.cos(rotation_angle*(np.pi/180)))+y_length*abs(np.sin(rotation_angle*(np.pi/180)))
    
    y_min = x_length*abs(np.sin(rotation_angle*(np.pi/180)))+y_length*abs(np.cos(rotation_angle*(np.pi/180)))
    
    return x_min, y_min
    
    

def Get_starting_coords_Rect_in_RotSCell(
        rotation_angle, x_length, y_length):
    '''
    After rotation the whole supercell a certain angle, a tilted rectangle is
    generated, which must fit a non-tilted (sides paralel to cartesian axes)
    rectangle that holds the dimensions of the smallest square containing the
    region drawn by the segmentation
    

    Parameters
    ----------
    rotation_angle : angle, in degrees, defined as positive in going from 
        the positive x horizontal axis to the vertical y, from -180º to 180º
    x_length : from Get_Smallest_Rectangular_Cell(), min horizontal side
            of the smallest rectangle holding the segmented region
    y_length : from Get_Smallest_Rectangular_Cell(), min vertical side
            of the smallest rectangle holding the segmented region

    Returns
    -------
    x_i, y_i : x and y coordinates of the starting point of the 
                non-rotated rectangle within the rotated one
    '''
    
    # just make sure the angle goes from [-180, 180]
    if rotation_angle > 180:
        rotation_angle = rotation_angle-360
        
    if rotation_angle < -180:
        rotation_angle = 360 - np.abs(rotation_angle)
     
    # adjust the positions of the initial coord vector based on the cuadrant
    # calculation of this coords in notebook
    
    if rotation_angle >= 0 and rotation_angle <= 90:
        
        x_i = x_length*(np.sin(rotation_angle*(np.pi/180)))**2
        y_i = x_length*(np.sin(rotation_angle*(np.pi/180)))*(np.cos(rotation_angle*(np.pi/180)))

        x_i = -x_i
        y_i = y_i
        
    if rotation_angle > 90 and rotation_angle < 180:
        
        phi = rotation_angle - 90
        
        x_i = x_length + (y_length)/((np.tan((90-phi)*(np.pi/180)))*(1+((np.tan(phi*(np.pi/180)))/(np.tan((90-phi)*(np.pi/180))))))
        
        y_i = (y_length*np.tan(phi*(np.pi/180)))/((np.tan((90-phi)*(np.pi/180)))*(1+((np.tan(phi*(np.pi/180)))/(np.tan((90-phi)*(np.pi/180))))))
        
        x_i = -x_i
        y_i = -y_i
        
    if rotation_angle == 180 or  rotation_angle == -180 :
        
        x_i = -x_length
        y_i = -y_length
        
    if rotation_angle <= -90 and rotation_angle > -180:
        
        rotation_angle = 180 - abs(rotation_angle)
        
        x_i = x_length*(np.sin(rotation_angle*(np.pi/180)))**2
        y_i = x_length*(np.sin(rotation_angle*(np.pi/180)))*(np.cos(rotation_angle*(np.pi/180)))

        x_i = x_i - x_length
        y_i = -y_i - y_length 
                
    if rotation_angle > -90 and rotation_angle <0:
        
        rotation_angle = 180 - abs(rotation_angle)
        
        phi = rotation_angle - 90

        x_i = x_length + (y_length)/((np.tan((90-phi)*(np.pi/180)))*(1+((np.tan(phi*(np.pi/180)))/(np.tan((90-phi)*(np.pi/180))))))
        
        y_i = (y_length*np.tan(phi*(np.pi/180)))/((np.tan((90-phi)*(np.pi/180)))*(1+((np.tan(phi*(np.pi/180)))/(np.tan((90-phi)*(np.pi/180))))))

        x_i = x_i - x_length
        y_i = y_i - y_length
        
        
    return x_i, y_i



    
def Is_coordinate_inside_contour(
        atomic_coordinate, conts_vertx_per_region_label):
    '''
    check if an atomic coordinate is within the profile drawn by the 
    segmentation of the region
    Assumes horizontal and vertical contours the way the pixels
    would draw the regions

    Parameters
    ----------
    All inputs in nm

    
    atomic_coordinate: coordinate of the atom, in format x,y (do not include z)
    conts_vertx_per_region_label: list of contours extracted from the dictionary 
        conts_vertxs_per_region_segmented[i (segmented image)][str(label_of_seg_region)+'_contours']
        and be careful as the contours are in format (y,x)


    Returns
    -------
    is_coord_inside boolean: True if the atom coordiante is inside or False if it is outside

    '''
    
    def Check_vector_North(
            atomic_coordinate, conts_vertx_per_region_label):
        '''
        Check if given the atomic coordinates, if the FIRST vector on its north is following the contour direction,
        from right to left (only the first, as it may found a good one afterwards to misleading results)

        Parameters
        ----------
        atomic_coordinate: coordinate of the atom, in format x,y (do not include z)
        conts_vertx_per_region_label: list of contours extracted from the dictionary 
            conts_vertxs_per_region_segmented[index (segmented image)][str(label_of_seg_region)+'_contours']
            and be careful as the contours are in format (y,x)


        Returns
        -------
        to_north: boolean True or False whether the atom is or not below a right to left contour

        '''
        
        x_atom, y_atom = atomic_coordinate
        
        possible_contours = []
        
        for contour_vector in conts_vertx_per_region_label:
            
            # !!! in real contours use the following (contour_vector is the object itself)
            (y_rel_init, x_rel_init) = contour_vector.rel_init_coords
            (y_rel_final, x_rel_final) = contour_vector.rel_final_coords 
            
            #check if the y of the contours is kept as we search for horizontl contours
            if y_rel_init == y_rel_final:
                # check if the y of the atomic position is bigger (is below)
                # the contour
                if y_atom >=y_rel_init:
                    # check the contours that they x segment coincides with the x
                    # check in both senses, as we need the very first on top of coord
                    if (x_atom <= x_rel_init and x_atom >= x_rel_final) or (x_atom >= x_rel_init and x_atom <= x_rel_final):
                        # the coordinate fits below the contour, so it is a candidate
                        possible_contours.append(contour_vector)

                # accept it and store it, and then check which is the one whose y is closest, 
                # from the top (from the atom coord to the north till a contour)
                # as this is the one will rule the contour
        
        # if no contour is found, then false:
        
        if len(possible_contours)==0:
            to_north=False
            
        else:
            
            # !!! for the real case with the contour object
            possible_contours.sort(key=(lambda x: x.rel_init_coords[0]), reverse=True)
            
            
            # check if it is a right to left contour
            
            # !!! for the real case with the contour object
            if possible_contours[0].rel_init_coords[1] > possible_contours[0].rel_final_coords[1]:
                to_north=True
            else:
                to_north=False
        
        return to_north
    


        
    def Check_vector_South(
            atomic_coordinate, conts_vertx_per_region_label):
        '''
        Check if given the atomic coordinates, if the FIRST vector on its south is following the contour direction,
        from lef to right (only the first, as it may found a good one afterwards to misleading results)

        Parameters
        ----------
        atomic_coordinate: coordinate of the atom, in format x,y (do not include z)
        conts_vertx_per_region_label: list of contours extracted from the dictionary 
            conts_vertxs_per_region_segmented[index (segmented image)][str(label_of_seg_region)+'_contours']
            and be careful as the contours are in format (y,x)


        Returns
        -------
        to_south: boolean True or False whether the atom is or not above a left to right contour

        '''
        
        x_atom, y_atom = atomic_coordinate
        
        possible_contours = []
        
        for contour_vector in conts_vertx_per_region_label:
            
            # !!! in real contours use the following (contour_vector is the object itself)
            (y_rel_init, x_rel_init) = contour_vector.rel_init_coords
            (y_rel_final, x_rel_final) = contour_vector.rel_final_coords 
            
            #check if the y of the contours is kept as we search for horizontl contours
            if y_rel_init == y_rel_final:
                # check if the y of the atomic position is bigger (is below)
                # the contour
                if y_atom <= y_rel_init:
                    # check the contours that they x segment coincides with the x
                    # check in both senses, as we need the very first below the coord
                    if (x_atom <= x_rel_init and x_atom >= x_rel_final) or (x_atom >= x_rel_init and x_atom <= x_rel_final):
                        # the coordinate fits below the contour, so it is a candidate
                        possible_contours.append(contour_vector)

                
                # accept it and store it, and then check which is the one whose y is closest, 
                # from the bottom (from the atom coord to the south till a contour)
                # as this is the one will rule the contour
        
        # if no contour is found, then false:
        
        if len(possible_contours)==0:
            to_south=False
            
        else:
            
            # !!! for the real case with the contour object
            possible_contours.sort(key=(lambda x: x.rel_init_coords[0]), reverse=False)
            
            # check if it is a left to right contour
            
            # !!! for the real case with the contour object
            if possible_contours[0].rel_init_coords[1] < possible_contours[0].rel_final_coords[1]:
                to_south=True
            else:
                to_south=False
        
        return to_south
                    




    def Check_vector_East(
            atomic_coordinate, conts_vertx_per_region_label):
        '''
        Check if given the atomic coordinates, if the FIRST vector on its east is following the contour direction,
        from bottom to top (only the first, as it may found a good one afterwards to misleading results)

        Parameters
        ----------
        atomic_coordinate: coordinate of the atom, in format x,y (do not include z)
        conts_vertx_per_region_label: list of contours extracted from the dictionary 
            conts_vertxs_per_region_segmented[index (segmented image)][str(label_of_seg_region)+'_contours']
            and be careful as the contours are in format (y,x)


        Returns
        -------
        to_east: boolean True or False whether the atom is or not at left of a bottom to top contour

        '''
        
        x_atom, y_atom = atomic_coordinate
        
        possible_contours = []
        
        for contour_vector in conts_vertx_per_region_label:
            
            # !!! in real contours use the following (contour_vector is the object itself)
            (y_rel_init, x_rel_init) = contour_vector.rel_init_coords
            (y_rel_final, x_rel_final) = contour_vector.rel_final_coords 
            
            #check if the x of the contours is kept as we search for vertical contours
            if x_rel_init == x_rel_final:
                # check if the x of the atomic position is smaller (is at left of the contour)
                # the contour
                if x_atom <= x_rel_init:
                    # check the contours that they y segment coincides with the y
                    # check in both senses, as we need the very first at right of atom cord
                    if (y_atom <= y_rel_init and y_atom >= y_rel_final) or (y_atom >= y_rel_init and y_atom <= y_rel_final):
                        # the coordinate fits below the contour, so it is a candidate
                        possible_contours.append(contour_vector)

                
                # accept it and store it, and then check which is the one whose x is closest, 
                # from the right (from the atom coord to the east till a contour)
                # as this is the one will rule the contour
        
        # if no contour is found, then false:
        
        if len(possible_contours)==0:
            to_east=False
            
        else:
            
            # !!! for the real case with the contour object
            possible_contours.sort(key=(lambda x: x.rel_init_coords[1]), reverse=False)

            # check if it is a bottom to top contour
            
            # !!! for the real case with the contour object
            if possible_contours[0].rel_init_coords[0] > possible_contours[0].rel_final_coords[0]:
                to_east=True
            else:
                to_east=False
        
        return to_east
    


    def Check_vector_West(
            atomic_coordinate, conts_vertx_per_region_label):
        '''
        Check if given the atomic coordinates, if the FIRST vector on its west is following the contour direction,
        from top to bottom (only the first, as it may found a good one afterwards to misleading results)

        Parameters
        ----------
        atomic_coordinate: coordinate of the atom, in format x,y (do not include z)
        conts_vertx_per_region_label: list of contours extracted from the dictionary 
            conts_vertxs_per_region_segmented[index (segmented image)][str(label_of_seg_region)+'_contours']
            and be careful as the contours are in format (y,x)


        Returns
        -------
        to_west: boolean True or False whether the atom is or not at right of a top to bottom contour

        '''
        
        x_atom, y_atom = atomic_coordinate
        
        possible_contours = []
        
        for contour_vector in conts_vertx_per_region_label:
            
            # !!! in real contours use the following (contour_vector is the object itself)
            (y_rel_init, x_rel_init) = contour_vector.rel_init_coords
            (y_rel_final, x_rel_final) = contour_vector.rel_final_coords 
            
            #check if the x of the contours is kept as we search for vertical contours
            if x_rel_init == x_rel_final:
                # check if the x of the atomic position is bigger (is at right of the contour)
                # the contour
                if x_atom >= x_rel_init:
                    # check the contours that they y segment coincides with the y
                    # check in both senses, as we need the very first at left of atom cord
                    if (y_atom <= y_rel_init and y_atom >= y_rel_final) or (y_atom >= y_rel_init and y_atom <= y_rel_final):
                        # the coordinate fits below the contour, so it is a candidate
                        possible_contours.append(contour_vector)

                
                # accept it and store it, and then check which is the one whose x is closest, 
                # from the left (from the atom coord to the west till a contour)
                # as this is the one will rule the contour
        
        # if no contour is found, then false:
        
        if len(possible_contours)==0:
            to_west=False
            
        else:
            
            # !!! for the real case with the contour object
            possible_contours.sort(key=(lambda x: x.rel_init_coords[1]), reverse=True)

            # check if it is a top to bottom contour
            
            # !!! for the real case with the contour object
            if possible_contours[0].rel_init_coords[0] < possible_contours[0].rel_final_coords[0]:                
                to_west=True
            else:
                to_west=False
        
        return to_west
    

    # evaluate all the directions for the given atomic coordinate
    
    to_north=Check_vector_North(
        atomic_coordinate, conts_vertx_per_region_label)
    to_south=Check_vector_South(
        atomic_coordinate, conts_vertx_per_region_label)
    to_east=Check_vector_East(
        atomic_coordinate, conts_vertx_per_region_label)
    to_west=Check_vector_West(
        atomic_coordinate, conts_vertx_per_region_label)
    
    # only if all four are true, so all contact a vectorially correct contour
    # then the coordinate is considered to be inside, otherwise outside
    if to_north==True and to_south==True and to_east==True and to_west==True:
        is_coord_inside = True
    else:
        is_coord_inside = False

    return is_coord_inside    
    


def Contour_Linear_Interpolation(
        rel_init_coords, rel_final_coords, variable_known, variable_known_val):
    '''
    For a contour with an initial vertex defined by rel_init_coords and a final
    vertex defined by rel_final_coords, this function generates the linear 
    equation necessary to interpolate the real mathematical boundary between
    these two coordinates, as a way to find the exact place where to define
    the segment

    Parameters
    ----------
    rel_init_coords : tuple with (y_rel_init, x_rel_init)
    rel_final_coords : tuple with (y_rel_final, x_rel_final)
    variable_known : string, either 'x' to get the y, or 'y' to retrieve the x
    variable_known_val: float, the value of the specified known variable
    
    Returns
    -------
    variable_unknown : float, interpolated unknown variable as definde by variable_know

    '''
    
    (y_rel_init, x_rel_init) = rel_init_coords
    (y_rel_final, x_rel_final) = rel_final_coords 
    
    # in a linear equation of the shape y = m*x + c
    
    # account for a perfectly vertical contour, which  would be infinite slope
    if x_rel_final == x_rel_init:
        m = 1e15*np.sign(y_rel_final - y_rel_init)
    else:
        m = (y_rel_final - y_rel_init)/(x_rel_final - x_rel_init)
        
    c = y_rel_init - m*x_rel_init
    
    
    if variable_known == 'x':
        
        variable_unknown = m*variable_known_val + c
        
    else: 
        #variable_known == 'y':
        
        variable_unknown = (variable_known_val - c)/m
        
    return variable_unknown



    
def Is_coordinate_inside_contour_diagonal(
        atomic_coordinate, conts_vertx_per_region_label):
    '''
    check if an atomic coordinate is within the profile drawn by the 
    segmentation of the region
    Assumes horizontal and vertical contours the way the pixels
    would draw the regions
    
    Updated to consider any diagonal contour that is interpolated with a function
    given by the interpolation method, by now Linear interpolation following the
    function Contour_Linear_Interpolation, but could be updated to be even
    smoother with any other interpolation between the two vertexs

    Parameters
    ----------
    All inputs in nm
    
    atomic_coordinate: coordinate of the atom, in format x,y (do not include z)
    conts_vertx_per_region_label: list of contours extracted from the dictionary 
        conts_vertxs_per_region_segmented[i (segmented image)][str(label_of_seg_region)+'_contours']
        and be careful as the contours are in format (y,x)


    Returns
    -------
    is_coord_inside boolean: True if the atom coordiante is inside or False if it is outside

    '''
    
    def Check_vector_North_Interp(
            atomic_coordinate, conts_vertx_per_region_label):
        '''
        Check if given the atomic coordinates, if the FIRST vector on its north is following the contour direction,
        from right to left (only the first, as it may found a good one afterwards to misleading results)

        Parameters
        ----------
        atomic_coordinate: coordinate of the atom, in format x,y (do not include z)
        conts_vertx_per_region_label: list of contours extracted from the dictionary 
            conts_vertxs_per_region_segmented[index (segmented image)][str(label_of_seg_region)+'_contours']
            and be careful as the contours are in format (y,x)


        Returns
        -------
        to_north: boolean True or False whether the atom is or not below a right to left contour

        '''
        
        x_atom, y_atom = atomic_coordinate
        
        possible_contours = []
        possible_contours_interpols_y = []
        
        for contour_vector in conts_vertx_per_region_label:
            
            # !!! in real contours use the following (contour_vector is the object itself)
            (y_rel_init, x_rel_init) = contour_vector.rel_init_coords
            (y_rel_final, x_rel_final) = contour_vector.rel_final_coords 
            
            #check if the y of the contours is kept as we search for horizontl contours
            if (x_atom <= x_rel_init and x_atom >= x_rel_final) or (x_atom >= x_rel_init and x_atom <= x_rel_final):
                # check if the interpolated y is on the north of the atom
                interpol_y = Contour_Linear_Interpolation(
                    (y_rel_init, x_rel_init), (y_rel_final, x_rel_final), 'x', x_atom)
                if y_atom >= interpol_y:
                # the coordinate fits below the contour, so it is a candidate
                    possible_contours.append(contour_vector)
                    possible_contours_interpols_y.append(interpol_y)

                # accept it and store it, and then check which is the one whose y is closest, 
                # from the top (from the atom coord to the north till a contour)
                # as this is the one will rule the contour
        
        # if no contour is found, then false:
        
        if len(possible_contours)==0:
            to_north=False
            
        else:
            
            # !!! for the real case with the contour object
            # sort the contours based on the y so the biggest interpolate y is the first
            # as it is the closest one to the atom
            sorted_contours = [cont for cont, _ in sorted(
                zip(possible_contours, possible_contours_interpols_y), key =(
                    lambda x: x[1]), reverse = True)]
                        
            # check if it is a right to left contour
            
            # !!! for the real case with the contour object
            if sorted_contours[0].rel_init_coords[1] > sorted_contours[0].rel_final_coords[1]:
                to_north=True
            else:
                to_north=False
        
        return to_north
    


        
    def Check_vector_South_Interp(
            atomic_coordinate, conts_vertx_per_region_label):
        '''
        Check if given the atomic coordinates, if the FIRST vector on its south is following the contour direction,
        from lef to right (only the first, as it may found a good one afterwards to misleading results)

        Parameters
        ----------
        atomic_coordinate: coordinate of the atom, in format x,y (do not include z)
        conts_vertx_per_region_label: list of contours extracted from the dictionary 
            conts_vertxs_per_region_segmented[index (segmented image)][str(label_of_seg_region)+'_contours']
            and be careful as the contours are in format (y,x)


        Returns
        -------
        to_south: boolean True or False whether the atom is or not above a left to right contour

        '''
        
        x_atom, y_atom = atomic_coordinate
        
        possible_contours = []
        possible_contours_interpols_y = []

        for contour_vector in conts_vertx_per_region_label:
            
            # !!! in real contours use the following (contour_vector is the object itself)
            (y_rel_init, x_rel_init) = contour_vector.rel_init_coords
            (y_rel_final, x_rel_final) = contour_vector.rel_final_coords 
            
            # check the contours that they x segment coincides with the x
            # check in both senses, as we need the very first below the coord
            if (x_atom <= x_rel_init and x_atom >= x_rel_final) or (x_atom >= x_rel_init and x_atom <= x_rel_final):
                # check if the interpolated y is on the south of the atom
                interpol_y = Contour_Linear_Interpolation(
                    (y_rel_init, x_rel_init), (y_rel_final, x_rel_final), 'x', x_atom)
                if y_atom <= interpol_y:
                    # the coordinate fits below the contour, so it is a candidate
                    possible_contours.append(contour_vector)
                    possible_contours_interpols_y.append(interpol_y)
                
                # accept it and store it, and then check which is the one whose y is closest, 
                # from the bottom (from the atom coord to the south till a contour)
                # as this is the one will rule the contour
        
        # if no contour is found, then false:
        
        if len(possible_contours)==0:
            to_south=False
            
        else:
            
            # !!! for the real case with the contour object
            sorted_contours = [cont for cont, _ in sorted(
                zip(possible_contours, possible_contours_interpols_y), key =(
                    lambda x: x[1]), reverse = False)]
            
            # check if it is a left to right contour
            
            # !!! for the real case with the contour object
            if sorted_contours[0].rel_init_coords[1] < sorted_contours[0].rel_final_coords[1]:
                to_south=True
            else:
                to_south=False
        
        return to_south
                    




    def Check_vector_East_Interp(
            atomic_coordinate, conts_vertx_per_region_label):
        '''
        Check if given the atomic coordinates, if the FIRST vector on its east is following the contour direction,
        from bottom to top (only the first, as it may found a good one afterwards to misleading results)

        Parameters
        ----------
        atomic_coordinate: coordinate of the atom, in format x,y (do not include z)
        conts_vertx_per_region_label: list of contours extracted from the dictionary 
            conts_vertxs_per_region_segmented[index (segmented image)][str(label_of_seg_region)+'_contours']
            and be careful as the contours are in format (y,x)


        Returns
        -------
        to_east: boolean True or False whether the atom is or not at left of a bottom to top contour

        '''
        
        x_atom, y_atom = atomic_coordinate
        
        possible_contours = []
        possible_contours_interpols_x = []
        
        for contour_vector in conts_vertx_per_region_label:
            
            # !!! in real contours use the following (contour_vector is the object itself)
            (y_rel_init, x_rel_init) = contour_vector.rel_init_coords
            (y_rel_final, x_rel_final) = contour_vector.rel_final_coords 
            
            # check the contours that they y segment coincides with the y
            # check in both senses, as we need the very first at right of atom cord
            if (y_atom <= y_rel_init and y_atom >= y_rel_final) or (y_atom >= y_rel_init and y_atom <= y_rel_final):
                # check if the interpolated x is on the east of the atom
                interpol_x = Contour_Linear_Interpolation(
                    (y_rel_init, x_rel_init), (y_rel_final, x_rel_final), 'y', y_atom)
                
                # check if the x of the atomic position is smaller (is at left of the contour)
                # the contour
                if x_atom <= interpol_x:
                    # the coordinate fits below the contour, so it is a candidate
                    possible_contours.append(contour_vector)
                    possible_contours_interpols_x.append(interpol_x)
                
                # accept it and store it, and then check which is the one whose x is closest, 
                # from the right (from the atom coord to the east till a contour)
                # as this is the one will rule the contour
        
        # if no contour is found, then false:
        
        if len(possible_contours)==0:
            to_east=False
            
        else:
            
            # !!! for the real case with the contour object
            sorted_contours = [cont for cont, _ in sorted(
                zip(possible_contours, possible_contours_interpols_x), key =(
                    lambda x: x[1]), reverse = False)]

            # check if it is a bottom to top contour
            
            # !!! for the real case with the contour object
            if sorted_contours[0].rel_init_coords[0] > sorted_contours[0].rel_final_coords[0]:
                to_east=True
            else:
                to_east=False
        
        return to_east
    


    def Check_vector_West_Interp(
            atomic_coordinate, conts_vertx_per_region_label):
        '''
        Check if given the atomic coordinates, if the FIRST vector on its west is following the contour direction,
        from top to bottom (only the first, as it may found a good one afterwards to misleading results)

        Parameters
        ----------
        atomic_coordinate: coordinate of the atom, in format x,y (do not include z)
        conts_vertx_per_region_label: list of contours extracted from the dictionary 
            conts_vertxs_per_region_segmented[index (segmented image)][str(label_of_seg_region)+'_contours']
            and be careful as the contours are in format (y,x)


        Returns
        -------
        to_west: boolean True or False whether the atom is or not at right of a top to bottom contour

        '''
        
        x_atom, y_atom = atomic_coordinate
        
        possible_contours = []
        possible_contours_interpols_x = []

        for contour_vector in conts_vertx_per_region_label:
            
            # !!! in real contours use the following (contour_vector is the object itself)
            (y_rel_init, x_rel_init) = contour_vector.rel_init_coords
            (y_rel_final, x_rel_final) = contour_vector.rel_final_coords 
            
            # check the contours that they y segment coincides with the y
            # check in both senses, as we need the very first at left of atom cord
            if (y_atom <= y_rel_init and y_atom >= y_rel_final) or (y_atom >= y_rel_init and y_atom <= y_rel_final):
                # check if the interpolated x is on the east of the atom
                interpol_x = Contour_Linear_Interpolation(
                    (y_rel_init, x_rel_init), (y_rel_final, x_rel_final), 'y', y_atom)
                
                if x_atom >= interpol_x:
                    # the coordinate fits below the contour, so it is a candidate
                    possible_contours.append(contour_vector)
                    possible_contours_interpols_x.append(interpol_x)
                
                # accept it and store it, and then check which is the one whose x is closest, 
                # from the left (from the atom coord to the west till a contour)
                # as this is the one will rule the contour
        
        # if no contour is found, then false:
        
        if len(possible_contours)==0:
            to_west=False
            
        else:
            
            # !!! for the real case with the contour object
            sorted_contours = [cont for cont, _ in sorted(
                zip(possible_contours, possible_contours_interpols_x), key =(
                    lambda x: x[1]), reverse = True)]
            
            # check if it is a top to bottom contour
            
            # !!! for the real case with the contour object
            if sorted_contours[0].rel_init_coords[0] < sorted_contours[0].rel_final_coords[0]:                
                to_west=True
            else:
                to_west=False
        
        return to_west
    

    # evaluate all the directions for the given atomic coordinate
    
    to_north=Check_vector_North_Interp(
        atomic_coordinate, conts_vertx_per_region_label)
    to_south=Check_vector_South_Interp(
        atomic_coordinate, conts_vertx_per_region_label)
    to_east=Check_vector_East_Interp(
        atomic_coordinate, conts_vertx_per_region_label)
    to_west=Check_vector_West_Interp(
        atomic_coordinate, conts_vertx_per_region_label)
    # only if all four are true, so all contact a vectorially correct contour
    # then the coordinate is considered to be inside, otherwise outside
    if to_north==True and to_south==True and to_east==True and to_west==True:
        is_coord_inside = True
    else:
        is_coord_inside = False

    return is_coord_inside    
    


def Build_shaped_atomistic(
        cell_filepath, zone_axis, rotation_angle, z_thickness,
        conts_vert_of_segmented_image, label_of_seg_region, 
        save_filepath, adjust_y_bottomleft = False):
    '''

    Parameters
    ----------
    cell_filepath: directory path to the unit cell info .cif, .xyz ...
    zone_axis: list with 3 elements being [u,v,w]
    rotation_angle: in degrees, angle from -180 to 180º starting in the horizontal 
                    x axis (positive when going to 1st quadrant)
    z_thickness: thickness, in nm, along the z direction (transveral invariance direction)
    conts_vert_of_segmented_image:
    conts_vert_of_segmented_image[str(label_of_seg_region)+'_rel_vertexs'] = list_rel_vertex
        is the list of the vertices in nm
    conts_vert_of_segmented_image[str(label_of_seg_region)+'_contours'] = list_of_contours
        is the list of the contours,and to get them in nm use:
            rel_init_coords = (y_rel_init, x_rel_init) = contour_vector_object.rel_init_coords
            rel_final_coords = (y_rel_final, x_rel_final) = contour_vector_object.rel_final_coords  
    label_of_seg_region: int, indicates the segm region from which to extract the information  
    save_filepath: path to save files of the different regions                           
    adjust_y_bottomleft: the rel vertices and coords set the reference 0,0 coordinate
                in the top left axis, as this may flip vertically the device geometry
                and atomic positions, set adjust_y_bottomleft = True if we want
                the 0,0 reference coords to be in the bottomleft corner 
                (considers the biggest y position and subtract the y coordiantes
                from it, just a vertical flip if needed)
                False by default (not shifting by default)

    Returns
    -------
    None, draws and creates the temp .xyz file containing the atomistic info

    '''
    
    
    # !!! it is computing a specular symetry and places the atoms 
    # at they horizontal mirror plane : rotation_angle --> - rotation_angle
    rotation_angle = - rotation_angle
    # it also changes the polarity of the dumbells
    rotation_angle = rotation_angle + 180
    # this two corrections make the atom placing viewed by 
    # from ase.visualize import view
    # from abtem.visualize import show_atoms
    # be exactly the same
    
    list_rel_vertex = np.copy(conts_vert_of_segmented_image[str(label_of_seg_region)+'_rel_vertexs'])
    # !!! list_rel_vertex is in format of [y,x], and the functions that use them require 
    # format [x,y], so just interchange the columns
    list_rel_vertex_cache = np.copy(list_rel_vertex)
    list_rel_vertex[:,0] = list_rel_vertex_cache[:,1]
    list_rel_vertex[:,1] = list_rel_vertex_cache[:,0]
    list_rel_vertex = list(list_rel_vertex)
    # now   list_rel_vertex is in format [x,y] and list_rel_vertex_cache in [y,x]

    list_of_contours = conts_vert_of_segmented_image[str(label_of_seg_region)+'_contours'].copy()
    
    # load the cif file of the unit cell
    unit_cell = read(cell_filepath)
    
    # orient trhough zone axis
    unit_cell_oriented = surface(
        unit_cell, indices=(zone_axis[0], zone_axis[1], zone_axis[2]), layers=10, periodic=True)
    
    # show atoms if wanted
    show_atoms(unit_cell_oriented, plane='xy')
    
    # get ZA oriented unit cell dimensions
    unit_cell_oriented_dims = unit_cell_oriented.get_cell()
    
    # !!! UNITS: set the unit cell dimensions in nm
    #!!! beware it is not a perennial change to the unit cell but something 
    # occasional for the  unit cell good considereation
    #!!! then the object holding the unit cell is still in angstroms
    unit_cell_oriented_dims = unit_cell_oriented_dims/10
    
    #Get dimensions of the rectangle
    x_min_c, x_max_c, x_length, y_min_c, y_max_c, y_length = Get_Smallest_Rectangular_Cell(
        list_rel_vertex)
    
    x_min, y_min  = Get_min_Rectangle_Allocating_rotCell(
        rotation_angle, x_length, y_length)
    
    # transform into a cubic supercell if the matrix is not a parallepiped
    
    # check if the flattened unit cell vector matrix is different than 0 in more than 3 elements
    if len(np.where(unit_cell_oriented_dims.flatten() != 0 )[0]) != 3:
        
        # if the unit cell rotated is not cubic or paralepiiped like, convert it
        # and afterwards generate the matrix to make it the desired dimensions
        n_unit_cells_cube = unit_cell_oriented.get_global_number_of_atoms()*2  #!!! Hyperparam to change if need
        supercell_cubic_transform = find_optimal_cell_shape(
            unit_cell_oriented.cell, n_unit_cells_cube, 'sc')
        
        cube_trans_supercell = make_supercell(unit_cell_oriented,supercell_cubic_transform)
        cube_trans_supercell_dims = cube_trans_supercell.get_cell()
        show_atoms(cube_trans_supercell)
        # !!! UNITS: set the unit cell dimensions in nm
        #!!! beware it is not a perennial change to the unit cell but something 
        # occasional for the  unit cell good considereation
        #!!! then the object holding the unit cell is still in angstroms
        cube_trans_supercell_dims = cube_trans_supercell_dims/10
    
        x_cartesian_total = cube_trans_supercell_dims[0,0]
        y_cartesian_total = cube_trans_supercell_dims[1,1]
        z_cartesian_total = cube_trans_supercell_dims[2,2]
        
        # later if the z_s are different it does not matter that much, as we will
        # delete all the z that are bigger than the really wanted z
        # sum 1 extra cell just in case
        
        rect_trans_x = int(np.ceil(x_min/x_cartesian_total)+1)
        rect_trans_y = int(np.ceil(y_min/y_cartesian_total)+1)
        rect_trans_z = int(np.ceil(z_thickness/z_cartesian_total)+1)
        
        
        supercell_rectangle_translation = np.array([[rect_trans_x, 0, 0],
                                                    [0, rect_trans_y, 0],
                                                    [0, 0, rect_trans_z]])
        
        supercell = make_supercell(cube_trans_supercell, supercell_rectangle_translation)

        
    else:
            
        # if the cell is paralepided already just multiply thorugh axis normally
        x_cartesian_total = unit_cell_oriented_dims[0,0]
        y_cartesian_total = unit_cell_oriented_dims[1,1]
        z_cartesian_total = unit_cell_oriented_dims[2,2]
        
        rect_trans_x = int(np.ceil(x_min/x_cartesian_total)+1)
        rect_trans_y = int(np.ceil(y_min/y_cartesian_total)+1)
        rect_trans_z = int(np.ceil(z_thickness/z_cartesian_total)+1)
        
        supercell_rectangle_translation = np.array([[rect_trans_x, 0, 0],
                                                    [0, rect_trans_y, 0],
                                                    [0, 0, rect_trans_z]])
        
        supercell = make_supercell(unit_cell_oriented, supercell_rectangle_translation)
        
    
    
    # check if the generated cubic shape is really rectangular 
    
    if len(np.where(supercell.get_cell().flatten() != 0 )[0]) == 3:
        # if it is rectangular, then proceed as normally and rotate and crop
        show_atoms(supercell,  plane='xy')
        
        # !!! UNITS: Convert the x_length and y_length from nm to
        # angstroms for the coordinate calculations
        x_length = x_length*10
        y_length = y_length*10
        
        # coordiantes in which the square within starts (coord of the bottom left corner)
        x_i_small_rect, y_i_small_rect = Get_starting_coords_Rect_in_RotSCell(
                rotation_angle, x_length, y_length)
        
        
        # bug here findin gthe starting poistions in -110 degrees puts it outside
        # as the shape that is being rotated since the begining is not perfectly rectangular but a romboid
        # after the supercell  is created maybe ensure we have a rectangle and center it at 0, 0
        # as it should even be big enough to contain the region anyways, crop the biggest rectangle there and 
        # put it back to the 0,0 coordinates in bottom left
        # beware it is not perfectly rectangular but can hve some angles strange that do not make
        # the function that we already have applicabel
        # this higly sure would depend on the angle defined by the  oriented cell
        # wathc out as cropping would be hard as then we lose the periodic boundary conditions
        # the angle between the two sides of the supercell should be asy to compute by checking at th
        # cartesian decomposition of th esupercell genrated
        # SOLVED
        
        # !!! the following two lines can be computed and used in case the 
        # computation of the normal coordinates does not work fine
        # by now the x_i_small_rect, y_i_small_rect seem to work fine
        
        # as we need to always sum x_length and y_length to the coords (not x,y_length_max)
        # the clearer the coords are the better, and it seems the 
        # x_i_small_rect, y_i_small_rect coords for 1st and 2nd quadrant
        # and 
        # x_i_big_rect, y_i_big_rect would work better for 3rd and 4th quadrant
        # are clearer which means they fit better the x_length and y lenght inside the
        # rotated supercell    
        
        # x_length_max, y_length_max = get_biggest_Rect_fitting_RotSCell(
        #     supercell, rotation_angle)
        # x_i_big_rect, y_i_big_rect = Get_starting_coords_Rect_in_RotSCell(
        #         rotation_angle, x_length_max, y_length_max)
            
    
        # let us rotate the supercell
        supercell.rotate(rotation_angle, 'z', rotate_cell= True)
        show_atoms(supercell,  plane='xy')
    
        # show if wanted
        # show_atoms(supercell, plane='xy')
        # show_atoms(supercell, plane='xz')
        # show_atoms(supercell, plane='yz')
        
        # retrieve positions, elements from supercell
        atomic_abs_positions = supercell.get_positions()
        elements_list_to_pos = supercell.get_atomic_numbers()    
    
        atomic_abs_positions_mod = np.copy(atomic_abs_positions)
        elements_list_to_pos_mod = np.copy(elements_list_to_pos)
        
        # mod the x,y positions
        # !!! modify the x_i_small_rect according to quadrant to x_i_big_rect in case
        # we observe some problem with this in these quadrants
        atomic_abs_positions_mod[:,0] = atomic_abs_positions_mod[:,0] - x_i_small_rect
        atomic_abs_positions_mod[:,1] = atomic_abs_positions_mod[:,1] - y_i_small_rect
        
        # filter positions that are inside the x_length and y_length limit
        
        # filter x positions within 0 and x_length
        elements_list_to_pos_mod = elements_list_to_pos_mod[atomic_abs_positions_mod[:,0]>=0]
        atomic_abs_positions_mod = atomic_abs_positions_mod[atomic_abs_positions_mod[:,0]>=0]
        elements_list_to_pos_mod = elements_list_to_pos_mod[atomic_abs_positions_mod[:,0]<= x_length]
        atomic_abs_positions_mod = atomic_abs_positions_mod[atomic_abs_positions_mod[:,0]<= x_length]
        # filter y positions within 0 and y_length
        elements_list_to_pos_mod = elements_list_to_pos_mod[atomic_abs_positions_mod[:,1]>=0]
        atomic_abs_positions_mod = atomic_abs_positions_mod[atomic_abs_positions_mod[:,1]>=0]
        elements_list_to_pos_mod = elements_list_to_pos_mod[atomic_abs_positions_mod[:,1]<= y_length]
        atomic_abs_positions_mod = atomic_abs_positions_mod[atomic_abs_positions_mod[:,1]<= y_length]
            
        # !!! UNITS: Convert the x_min_c and y_min_c from nm to
        # angstroms for the reescaling of the cell
        x_min_c = x_min_c*10
        y_min_c = y_min_c*10
        
        # adjust the positions based on the real absolute position they have within 
        # the global device
        atomic_abs_positions_mod[:,0] = atomic_abs_positions_mod[:,0] + x_min_c
        atomic_abs_positions_mod[:,1] = atomic_abs_positions_mod[:,1] + y_min_c
    
        # evaluate if the elements coords are within the segmented region and z thickness
        # !!! UNITS: for that, change the units of the atom coords from angs to nm
        # as the contours are expressed in nm
        atomic_abs_positions_mod = atomic_abs_positions_mod/10
        
        # ensure all z are smaller than the z_thickness given
        elements_list_to_pos_mod = elements_list_to_pos_mod[atomic_abs_positions_mod[:,2] >= 0]
        atomic_abs_positions_mod = atomic_abs_positions_mod[atomic_abs_positions_mod[:,2] >= 0]
        elements_list_to_pos_mod = elements_list_to_pos_mod[atomic_abs_positions_mod[:,2] <= z_thickness]
        atomic_abs_positions_mod = atomic_abs_positions_mod[atomic_abs_positions_mod[:,2] <= z_thickness]
    
        indices_coords_inside = []
          
        for index, (atomic_pos, element) in enumerate(
                zip(atomic_abs_positions_mod, elements_list_to_pos_mod)):
            
            atomic_pos_xy = atomic_pos[:2]
            
            # is_coord_inside = Is_coordinate_inside_contour(
            #     atomic_pos_xy, list_of_contours)
            
            is_coord_inside = Is_coordinate_inside_contour_diagonal(
                atomic_pos_xy, list_of_contours)
            
            # if coordiante is inside store the index to keep the coord from the array
            if is_coord_inside == True:
                indices_coords_inside.append(index)
            
        
        # only keep the atomic coordinates that are kept within the region of interest
        atomic_abs_positions_mod = atomic_abs_positions_mod[indices_coords_inside]    
        elements_list_to_pos_mod = elements_list_to_pos_mod[indices_coords_inside]   
        
        # adjust the y position to fit the 0,0 in the bottom left corner instead of 
        # top left that is defined by the contours
        if adjust_y_bottomleft == True:
            max_vals = []
            for element in conts_vert_of_segmented_image:
                if element[element.find('_'):] == '_rel_vertexs':
                    # store the array with the rel vertexs
                    rel_vertexs = np.copy(np.asarray(conts_vert_of_segmented_image[element]))
                    max_y_val_el = np.max(rel_vertexs[:,0])
                    max_vals.append(max_y_val_el)
                    
                    
            y_abs_max = max(max_vals)  
            atomic_abs_positions_mod[:,1] = y_abs_max - atomic_abs_positions_mod[:,1]
            
            # find from the total device and all the regions the biggest y and 
            # (for this browse through conts_vert_of_segmented_image[str(label_of_seg_region)+'_rel_vertexs'] = list_rel_vertex :
            #     for all labels)
            # then all positions the y coord atomic_abs_positions_mod[:,1]
            # will be atomic_abs_positions_mod[:,1] = biggest_y - atomic_abs_positions_mod[:,1]
            
            
        
        # !!! UNITS
        # set the coordiantes from nm back to angstroms  as all the comparisons
        # have been done with nm to fit the contour units
        # as ansgtrom seems the standard unit for .xyz files
        atomic_abs_positions_mod = atomic_abs_positions_mod*10
        
        atomic_abs_positions_mod.round(14)
        
        # transform atomic numbers into atomic symbols
        elements_unique_atnum = np.unique(elements_list_to_pos_mod)    
        elements_unique_symb = []
        
        for element in elements_unique_atnum:
            mend_el = mdl.element(int(element))
            el_symbol = mend_el.symbol
            elements_unique_symb.append(el_symbol)
            
        elements_symb_list_to_pos_mod = np.asarray(np.ones((elements_list_to_pos_mod.shape[0],1), dtype=int)*10, dtype=str)
        
        # for the unique elements involved, overwrite the number with symbol
        for element_numb, element_symb in zip(
                elements_unique_atnum, elements_unique_symb):    
            # change the string into the string of the unique symbol by numpy indexing
            elements_symb_list_to_pos_mod[elements_list_to_pos_mod==element_numb] = element_symb
            
        elements_symb_list_to_pos_mod = np.asarray(elements_symb_list_to_pos_mod)
        elements_symb_list_to_pos_mod.shape = (len(elements_symb_list_to_pos_mod),1)
        
        # combine arrays with symbols and 3D positions
        array_length_rect_supercell_xyz = np.hstack((elements_symb_list_to_pos_mod, atomic_abs_positions_mod))
        # print(array_length_rect_supercell_xyz)
        
        
        # build the final file containing the atomistic info from this supercell of region n
        # first line is the number of atoms in cell
        # 2nd is blank space
        length_rect_supercell_xyz = [str(len(elements_symb_list_to_pos_mod))+'\n','\n']
        
        # from third line stack the lines containing the element symbol and the 3d coords
        for symb_coord in array_length_rect_supercell_xyz:
            str_symb_coord = str(symb_coord[0])+'    '+str(symb_coord[1])+'    '+str(symb_coord[2])+'    '+str(symb_coord[3])+'\n'
            length_rect_supercell_xyz = length_rect_supercell_xyz + [str_symb_coord]
        
        # create the .xyz file
        temp_xyz_filename = save_filepath + 'temp_xyz_rot_supercell_'+str(label_of_seg_region)+'.xyz'
        
        filename = Path(temp_xyz_filename)
        file_already_created = filename.is_file()
        
        if file_already_created == True:
            # overwrite the file
            with open(temp_xyz_filename, "w+") as f:
                f.truncate(0)
                f.writelines(length_rect_supercell_xyz)
                f.close()
        else:
            # create a new file
            with open(temp_xyz_filename, 'w+') as f:
                f.writelines(length_rect_supercell_xyz)
                f.close()

            
        # show the supercell created if wanted    
        length_supercel = read(temp_xyz_filename)
        
        show_atoms(length_supercel, plane='xy')
        show_atoms(length_supercel, plane='xz')
        show_atoms(length_supercel, plane='yz')
        
    else:
        
        # if by no means we achieve to have a rectangular orthorombic cell
        # then just rotate the oriented cell, make it superbig so it for sure
        # can hold the wanted size and then just crop accordingly
        # this is brute force but should be just in exceptional cases
        # where strange zone axis are wanted or harder unit cell geometries
        # if this condition is rising instead of the optimum one in a
        # regular basis, check the function find_optimal_cell_shape as might 
        # be generating this strange volumes to which the routine is based
        
        # let us rotate the supercell
        unit_cell_oriented.rotate(rotation_angle, 'z', rotate_cell= True)
        show_atoms(unit_cell_oriented,  plane='xy')
        
        n_unit_cells_cube = unit_cell_oriented.get_global_number_of_atoms()*3  #!!! Hyperparam to change if need
        supercell_cubic_transform = find_optimal_cell_shape(
            unit_cell_oriented.cell, n_unit_cells_cube, 'sc')
        
        cube_trans_supercell = make_supercell(
            unit_cell_oriented,supercell_cubic_transform)
        cube_trans_supercell_dims = cube_trans_supercell.get_cell()
        
        # !!! UNITS: set the unit cell dimensions in nm
        #!!! beware it is not a perennial change to the unit cell but something 
        # occasional for the  unit cell good considereation
        #!!! then the object holding the unit cell is still in angstroms
        cube_trans_supercell_dims = cube_trans_supercell_dims/10
        
        # find the biggest positive cartesian x,y component from either a1 or a2
        # as the cell multiplicationw ill be done through this axis
        x_posib = cube_trans_supercell_dims[:2,0]
        y_posib = cube_trans_supercell_dims[:2,1]
        
        x_max_ind = np.argmax(abs(x_posib))
        y_max_ind = np.argmax(abs(y_posib))
        
        # ensure the translation operation is an invertible matrix det(A) !=0 
        if x_max_ind != y_max_ind:
            # this means the indices are differnt and the variable invertible
            x_max = np.max(abs(x_posib))
            y_max = np.max(abs(y_posib))
        else:
            # if the indices are the same it would mean a det(A) = 0 and fail
            # avoid this by making the indices different and changing the x_max, y_max
            x_max_prov = np.max(abs(x_posib))
            y_max_prov = np.max(abs(y_posib))
            ind_min_from_xy = np.argmin([x_max_prov, y_max_prov])

            if x_max_prov == y_max_prov:
                # adress the unusual case in which they are equal
                x_max_ind = 0
                y_max_ind = 1
                x_max = abs(x_posib)[x_max_ind]
                y_max = abs(y_posib)[y_max_ind]
                
            elif ind_min_from_xy == 0:

                # this means that the x value is the min from both
                # then keep the x_max as x_max and change y_max to the other val
                x_max = np.max(abs(x_posib))
                x_max_ind = np.argmax(abs(x_posib))
                
                if x_max_ind == 0:
                    y_max_ind = 1
                    y_max = abs(y_posib)[y_max_ind]
                else:
                    y_max_ind = 0
                    y_max = abs(y_posib)[y_max_ind]
                
            else:

                # this means that the y value is the min from both
                # then keep the y_max and change the x_max to the other possibility
                y_max = np.max(abs(y_posib))
                y_max_ind = np.argmax(abs(y_posib))
                
                if y_max_ind == 0:
                    x_max_ind = 1
                    x_max = abs(x_posib)[x_max_ind]
                else:
                    x_max_ind = 0
                    x_max = abs(x_posib)[x_max_ind]
                
                
        z_cartesian_total = cube_trans_supercell_dims[2,2]
        
        # the lenghts and cell info are in nm
        rect_trans_x = int(np.ceil(x_length/x_max)+1)
        rect_trans_y = int(np.ceil(y_length/y_max)+1)
        rect_trans_z = int(np.ceil(z_thickness/z_cartesian_total)+1)
        
        # make sure sure is big enough to hold the region 
        
        # doing the cubi thing even if tdoes not set it rectangular could help making
        # the next steps less time consuming and more direct as would be a bit more rectangular
        
        
        # !!! UNITS: Convert the x_length and y_length from nm to
        # angstroms for the coordinate calculations
        x_length = x_length*10
        y_length = y_length*10
        
        
        # !!! UNITS : Set thickness NEW variable in angstroms to use it in 
        # this comparison with the atomic coordinates in z
        z_thickness_angs = z_thickness*10
        
        supercell_rectangle_translation = np.zeros((3,3))
        
        supercell_rectangle_translation[0, x_max_ind] = rect_trans_x
        supercell_rectangle_translation[1, y_max_ind] = rect_trans_y
        supercell_rectangle_translation[2, 2] = rect_trans_z
    
        supercell = make_supercell(
            cube_trans_supercell, supercell_rectangle_translation)
        
        for iteration in np.arange(0,15,1):
            
            # double the supercell dimensiones/sides after every iteration      
            rect_trans_x_mod = 2
            rect_trans_y_mod = 2
    
            supercell_rectangle_translation = np.zeros((3,3))
            
            supercell_rectangle_translation[0, x_max_ind] = rect_trans_x_mod
            supercell_rectangle_translation[1, y_max_ind] = rect_trans_y_mod
            supercell_rectangle_translation[2, 2] = 2
        
            supercell = make_supercell(
                supercell, supercell_rectangle_translation)
            
            # get positions of the coordiantes and find the center of mass
            positions_of_supermegacell = supercell.get_positions()
            elements_list_of_supermegacell = supercell.get_atomic_numbers() 
            
            x_COM = np.average(positions_of_supermegacell[:,0])
            y_COM = np.average(positions_of_supermegacell[:,1])
            z_COM = np.average(positions_of_supermegacell[:,2])
            
            # Main Plane XY
            # xcom and ycom are in angstroms
            # get x,y length to angstroms
            positions_of_supermegacell[:,2] = positions_of_supermegacell[:,2] - (z_COM-(z_thickness_angs/2))
            positions_of_supermegacell = positions_of_supermegacell[positions_of_supermegacell[:,2] >=0] 
            positions_of_supermegacell = positions_of_supermegacell[positions_of_supermegacell[:,2] <=z_thickness_angs] 
            
            
            top_right_cords = positions_of_supermegacell[positions_of_supermegacell[:,0] >= x_COM+(x_length/2)]
            top_right_cords = top_right_cords[top_right_cords[:,1] >= y_COM+(y_length/2)]
            
            bottom_right_cords = positions_of_supermegacell[positions_of_supermegacell[:,0] >= x_COM+(x_length/2)]
            bottom_right_cords = bottom_right_cords[bottom_right_cords[:,1] <= y_COM-(y_length/2)]
            
            top_left_cords = positions_of_supermegacell[positions_of_supermegacell[:,0] <= x_COM-(x_length/2)]
            top_left_cords = top_left_cords[top_left_cords[:,1] >= y_COM+(y_length/2)]
            
            bottom_left_cords = positions_of_supermegacell[positions_of_supermegacell[:,0] <= x_COM-(x_length/2)]
            bottom_left_cords = bottom_left_cords[bottom_left_cords[:,1] <= y_COM-(y_length/2)]
            
            
            # then check whether the z is inside the region of interest
            # this is checking the conditions in two planes yz and xz
            
            positions_of_supermegacell = supercell.get_positions()
            
            # Plane XZ
            top_right_cords_xz = positions_of_supermegacell[positions_of_supermegacell[:,0] >= x_COM+(x_length/2)]
            top_right_cords_xz = top_right_cords_xz[top_right_cords_xz[:,2] >= z_COM+(z_thickness_angs/2)]
            
            bottom_right_cords_xz = positions_of_supermegacell[positions_of_supermegacell[:,0] >= x_COM+(x_length/2)]
            bottom_right_cords_xz = bottom_right_cords_xz[bottom_right_cords_xz[:,2] <= z_COM-(z_thickness_angs/2)]
            
            top_left_cords_xz = positions_of_supermegacell[positions_of_supermegacell[:,0] <= x_COM-(x_length/2)]
            top_left_cords_xz = top_left_cords_xz[top_left_cords_xz[:,2] >= z_COM+(z_thickness_angs/2)]
            
            bottom_left_cords_xz = positions_of_supermegacell[positions_of_supermegacell[:,0] <= x_COM-(x_length/2)]
            bottom_left_cords_xz = bottom_left_cords_xz[bottom_left_cords_xz[:,2] <= z_COM-(z_thickness_angs/2)]
            
            # Plane YZ
            top_right_cords_yz = positions_of_supermegacell[positions_of_supermegacell[:,1] >= y_COM+(y_length/2)]
            top_right_cords_yz = top_right_cords_yz[top_right_cords_yz[:,2] >= z_COM+(z_thickness_angs/2)]
            
            bottom_right_cords_yz = positions_of_supermegacell[positions_of_supermegacell[:,1] >= y_COM+(y_length/2)]
            bottom_right_cords_yz = bottom_right_cords_yz[bottom_right_cords_yz[:,2] <= z_COM-(z_thickness_angs/2)]
            
            top_left_cords_yz = positions_of_supermegacell[positions_of_supermegacell[:,1] <= y_COM-(y_length/2)]
            top_left_cords_yz = top_left_cords_yz[top_left_cords_yz[:,2] >= z_COM+(z_thickness_angs/2)]
            
            bottom_left_cords_yz = positions_of_supermegacell[positions_of_supermegacell[:,1] <= y_COM-(y_length/2)]
            bottom_left_cords_yz = bottom_left_cords_yz[bottom_left_cords_yz[:,2 ] <= z_COM-(z_thickness_angs/2)]
            
            
            # if there exist atomic coordinates in all the 4 corners of the x length 
            # y lenght rectangle then accept the supercell as it can contain it
            # in all the three planes XY, XZ, YZ
            
            # Plane XY
            if len(top_right_cords) != 0 and len(bottom_right_cords) != 0 and len(top_left_cords) != 0 and len(bottom_left_cords) != 0:
                # Plane XZ
                if len(top_right_cords_xz) != 0 and len(bottom_right_cords_xz) != 0 and len(top_left_cords_xz) != 0 and len(bottom_left_cords_xz) != 0:
                    # Plane YZ
                    if len(top_right_cords_yz) != 0 and len(bottom_right_cords_yz) != 0 and len(top_left_cords_yz) != 0 and len(bottom_left_cords_yz) != 0:
                        # if everything is met break the loop and keep the supcell
                        break
            
            
        show_atoms(supercell)
        # now this supercell can contain the x_lenght and y_length square
        # crop it
        
        # retrieve positions, elements from supercell
        atomic_abs_positions = supercell.get_positions()
        elements_list_to_pos = supercell.get_atomic_numbers()    
    
        atomic_abs_positions_mod = np.copy(atomic_abs_positions)
        elements_list_to_pos_mod = np.copy(elements_list_to_pos)

        elements_list_to_pos_mod = elements_list_to_pos_mod[atomic_abs_positions_mod[:,0] >= x_COM-(x_length/2)]
        atomic_abs_positions_mod = atomic_abs_positions_mod[atomic_abs_positions_mod[:,0] >= x_COM-(x_length/2)]
        elements_list_to_pos_mod = elements_list_to_pos_mod[atomic_abs_positions_mod[:,0] <= x_COM+(x_length/2)]
        atomic_abs_positions_mod = atomic_abs_positions_mod[atomic_abs_positions_mod[:,0] <= x_COM+(x_length/2)]
        
        elements_list_to_pos_mod = elements_list_to_pos_mod[atomic_abs_positions_mod[:,1] >= y_COM-(y_length/2)]
        atomic_abs_positions_mod = atomic_abs_positions_mod[atomic_abs_positions_mod[:,1] >= y_COM-(y_length/2)]
        elements_list_to_pos_mod = elements_list_to_pos_mod[atomic_abs_positions_mod[:,1] <= y_COM+(y_length/2)]
        atomic_abs_positions_mod = atomic_abs_positions_mod[atomic_abs_positions_mod[:,1] <= y_COM+(y_length/2)]

        # readjust positions to center 0,0 at bottom left
        atomic_abs_positions_mod[:,0] = atomic_abs_positions_mod[:,0] - (x_COM-(x_length/2))
        atomic_abs_positions_mod[:,1] = atomic_abs_positions_mod[:,1] - (y_COM-(y_length/2))
        
        # and then readjust again to set the coordinates to the relative position of the device
        
        # !!! UNITS: Convert the x_min_c and y_min_c from nm to
        # angstroms for the reescaling of the cell
        x_min_c = x_min_c*10
        y_min_c = y_min_c*10
        
        # adjust the positions based on the real absolute position they have within 
        # the global device
        atomic_abs_positions_mod[:,0] = atomic_abs_positions_mod[:,0] + x_min_c
        atomic_abs_positions_mod[:,1] = atomic_abs_positions_mod[:,1] + y_min_c
    
        # evaluate if the elements coords are within the segmented region and z thickness
        # !!! UNITS: for that, change the units of the atom coords from angs to nm
        # as the contours are expressed in nm
        atomic_abs_positions_mod = atomic_abs_positions_mod/10
        
        z_average = np.average(atomic_abs_positions_mod[:,2])
        
        atomic_abs_positions_mod[:,2] = atomic_abs_positions_mod[:,2] - z_average
        # ensure all z are smaller than the z_thickness given
        elements_list_to_pos_mod = elements_list_to_pos_mod[atomic_abs_positions_mod[:,2] >=  -(z_thickness/2)]
        atomic_abs_positions_mod = atomic_abs_positions_mod[atomic_abs_positions_mod[:,2] >=  -(z_thickness/2)]
        elements_list_to_pos_mod = elements_list_to_pos_mod[atomic_abs_positions_mod[:,2] <=  +(z_thickness/2)]
        atomic_abs_positions_mod = atomic_abs_positions_mod[atomic_abs_positions_mod[:,2] <=  +(z_thickness/2)]
        
        # put the z back starting from 0 to z_thickness range
        atomic_abs_positions_mod[:,2] = atomic_abs_positions_mod[:,2] + (z_thickness/2)
        
        indices_coords_inside = []
          
        for index, (atomic_pos, element) in enumerate(
                zip(atomic_abs_positions_mod, elements_list_to_pos_mod)):
            
            atomic_pos_xy = atomic_pos[:2]
            
            # is_coord_inside = Is_coordinate_inside_contour(
            #     atomic_pos_xy, list_of_contours)
            
            is_coord_inside = Is_coordinate_inside_contour_diagonal(
                atomic_pos_xy, list_of_contours)            
            
            # if coordiante is inside store the index to keep the coord from the array
            if is_coord_inside == True:
                indices_coords_inside.append(index)
                
            
        # only keep the atomic coordinates that are kept within the region of interest
        atomic_abs_positions_mod = atomic_abs_positions_mod[indices_coords_inside]    
        elements_list_to_pos_mod = elements_list_to_pos_mod[indices_coords_inside]   
        
        # adjust the y position to fit the 0,0 in the bottom left corner instead of 
        # top left that is defined by the contours
        if adjust_y_bottomleft == True:
            max_vals = []
            for element in conts_vert_of_segmented_image:
                if element[element.find('_'):] == '_rel_vertexs':
                    # store the array with the rel vertexs
                    rel_vertexs = np.copy(np.asarray(conts_vert_of_segmented_image[element]))
                    max_y_val_el = np.max(rel_vertexs[:,0])
                    max_vals.append(max_y_val_el)
                    
                    
            y_abs_max = max(max_vals)  
            atomic_abs_positions_mod[:,1] = y_abs_max - atomic_abs_positions_mod[:,1]
            
            # find from the total device and all the regions the biggest y and 
            # (for this browse through conts_vert_of_segmented_image[str(label_of_seg_region)+'_rel_vertexs'] = list_rel_vertex :
            #     for all labels)
            # then all positions the y coord atomic_abs_positions_mod[:,1]
            # will be atomic_abs_positions_mod[:,1] = biggest_y - atomic_abs_positions_mod[:,1]
            
            
        
        # !!! UNITS
        # set the coordiantes from nm back to angstroms  as all the comparisons
        # have been done with nm to fit the contour units
        # as ansgtrom seems the standard unit for .xyz files
        atomic_abs_positions_mod = atomic_abs_positions_mod*10
        
        atomic_abs_positions_mod.round(14)
        
        # transform atomic numbers into atomic symbols
        elements_unique_atnum = np.unique(elements_list_to_pos_mod)    
        elements_unique_symb = []
        
        for element in elements_unique_atnum:
            mend_el = mdl.element(int(element))
            el_symbol = mend_el.symbol
            elements_unique_symb.append(el_symbol)
            
        elements_symb_list_to_pos_mod = np.asarray(np.ones((elements_list_to_pos_mod.shape[0],1), dtype=int)*10, dtype=str)
        
        # for the unique elements involved, overwrite the number with symbol
        for element_numb, element_symb in zip(
                elements_unique_atnum, elements_unique_symb):    
            # change the string into the string of the unique symbol by numpy indexing
            elements_symb_list_to_pos_mod[elements_list_to_pos_mod==element_numb] = element_symb
            
        elements_symb_list_to_pos_mod = np.asarray(elements_symb_list_to_pos_mod)
        elements_symb_list_to_pos_mod.shape = (len(elements_symb_list_to_pos_mod),1)
        
        # combine arrays with symbols and 3D positions
        array_length_rect_supercell_xyz = np.hstack((elements_symb_list_to_pos_mod, atomic_abs_positions_mod))
        # print(array_length_rect_supercell_xyz)
        
        
        # build the final file containing the atomistic info from this supercell of region n
        # first line is the number of atoms in cell
        # 2nd is blank space
        length_rect_supercell_xyz = [str(len(elements_symb_list_to_pos_mod))+'\n','\n']
        
        # from third line stack the lines containing the element symbol and the 3d coords
        for symb_coord in array_length_rect_supercell_xyz:
            str_symb_coord = str(symb_coord[0])+'    '+str(symb_coord[1])+'    '+str(symb_coord[2])+'    '+str(symb_coord[3])+'\n'
            length_rect_supercell_xyz = length_rect_supercell_xyz + [str_symb_coord]
        
        # create the .xyz file
        temp_xyz_filename = save_filepath + 'temp_xyz_rot_supercell_'+str(label_of_seg_region)+'.xyz'
        
        filename = Path(temp_xyz_filename)
        file_already_created = filename.is_file()
        
        if file_already_created == True:
            # overwrite the file
            with open(temp_xyz_filename, "w+") as f:
                f.truncate(0)
                f.writelines(length_rect_supercell_xyz)
                f.close()
        else:
            # create a new file
            with open(temp_xyz_filename, 'w+') as f:
                f.writelines(length_rect_supercell_xyz)
                f.close()
            
        # show the supercell created if wanted    
        length_supercel = read(temp_xyz_filename)
        
        show_atoms(length_supercel, plane='xy')
        show_atoms(length_supercel, plane='xz')
        show_atoms(length_supercel, plane='yz')
                
                
    # Build a complementary .txt file containing, for every atom entry in 
    # 'temp_xyz_rot_supercell_'+str(label_of_seg_region)+'.xyz', a label
    # corresponding to label_of_seg_region to identify the segmented region 
    # where they belong, for keeping track of what has been done to its atoms
        
    atom_identifyer_file = [str(len(elements_symb_list_to_pos_mod))+'\n','\n']
    
    for iteration_ in range(len(elements_symb_list_to_pos_mod)):
        new_line_ = str(label_of_seg_region) + '\n'
        atom_identifyer_file = atom_identifyer_file + [new_line_]
    
    
    # create the .xyz file
    atom_identifyer_filename = save_filepath + 'atom_identifier_'+str(label_of_seg_region)+'.txt'
    
    filename_atid = Path(atom_identifyer_filename)
    fileid_already_created = filename_atid.is_file()
    
    if fileid_already_created == True:
        # overwrite the file
        with open(atom_identifyer_filename, "w+") as f:
            f.truncate(0)
            f.writelines(atom_identifyer_file)
            f.close()
    else:
        # create a new file
        with open(atom_identifyer_filename, 'w+') as f:
            f.writelines(atom_identifyer_file)
            f.close()
        
        
    return

 
   


def Build_Atomistic_Parallelepiped(
        cell_filepath, zone_axis, rotation_angle, z_thickness,
        conts_vert_of_segmented_image, label_of_seg_region, 
        save_filepath, adjust_y_bottomleft = False):
    '''
    Slightly different from the Build_Atomistic_Shaped
    as here we create the box that would contain all the atoms
    that have to be inside the region defined and delimited by the 
    contour for that given label, but we do not apply the filtering
    of the segmented region because we want a bigger regions that not only
    has the atoms inside the segmented region, but slightly outside in their
    borders
    This is of outmost importance when applying the strain curves to
    the atomistic model, as the displacements will need to be applied per file 
    generated (i.e. per label) and then use the segmentation iformation or chemistry
    to filter them and then finally combine the files
    

    Parameters
    ----------
    cell_filepath: directory path to the unit cell info .cif, .xyz ...
    zone_axis: list with 3 elements being [u,v,w]
    rotation_angle: in degrees, angle from -180 to 180º starting in the horizontal 
                    x axis (positive when going to 1st quadrant)
    z_thickness: thickness, in nm, along the z direction (transveral invariance direction)
    conts_vert_of_segmented_image:
    conts_vert_of_segmented_image[str(label_of_seg_region)+'_rel_vertexs'] = list_rel_vertex
        is the list of the vertices in nm
    conts_vert_of_segmented_image[str(label_of_seg_region)+'_contours'] = list_of_contours
        is the list of the contours,and to get them in nm use:
            rel_init_coords = (y_rel_init, x_rel_init) = contour_vector_object.rel_init_coords
            rel_final_coords = (y_rel_final, x_rel_final) = contour_vector_object.rel_final_coords  
    label_of_seg_region: int, indicates the segm region from which to extract the information  
    save_filepath: path to save files of the different regions                           
    adjust_y_bottomleft: the rel vertices and coords set the reference 0,0 coordinate
                in the top left axis, as this may flip vertically the device geometry
                and atomic positions, set adjust_y_bottomleft = True if we want
                the 0,0 reference coords to be in the bottomleft corner 
                (considers the biggest y position and subtract the y coordiantes
                from it, just a vertical flip if needed)
                False by default (not shifting by default)

    Returns
    -------
    None, draws and creates the temp .xyz file containing the atomistic info

    '''
    
    
    # !!! it is computing a specular symetry and places the atoms 
    # at they horizontal mirror plane : rotation_angle --> - rotation_angle
    rotation_angle = - rotation_angle
    # it also changes the polarity of the dumbells
    rotation_angle = rotation_angle + 180
    # this two corrections make the atom placing viewed by 
    # from ase.visualize import view
    # from abtem.visualize import show_atoms
    # be exactly the same
    
    list_rel_vertex = np.copy(conts_vert_of_segmented_image[str(label_of_seg_region)+'_rel_vertexs'])
    # !!! list_rel_vertex is in format of [y,x], and the functions that use them require 
    # format [x,y], so just interchange the columns
    list_rel_vertex_cache = np.copy(list_rel_vertex)
    list_rel_vertex[:,0] = list_rel_vertex_cache[:,1]
    list_rel_vertex[:,1] = list_rel_vertex_cache[:,0]
    list_rel_vertex = list(list_rel_vertex)
    # now   list_rel_vertex is in format [x,y] and list_rel_vertex_cache in [y,x]

    list_of_contours = conts_vert_of_segmented_image[str(label_of_seg_region)+'_contours'].copy()
    
    # load the cif file of the unit cell
    unit_cell = read(cell_filepath)
    
    # orient trhough zone axis
    unit_cell_oriented = surface(
        unit_cell, indices=(zone_axis[0], zone_axis[1], zone_axis[2]), layers=10, periodic=True)
    
    # show atoms if wanted
    show_atoms(unit_cell_oriented, plane='xy')
    
    # get ZA oriented unit cell dimensions
    unit_cell_oriented_dims = unit_cell_oriented.get_cell()
    
    # !!! UNITS: set the unit cell dimensions in nm
    #!!! beware it is not a perennial change to the unit cell but something 
    # occasional for the  unit cell good considereation
    #!!! then the object holding the unit cell is still in angstroms
    unit_cell_oriented_dims = unit_cell_oriented_dims/10
    
    #Get dimensions of the rectangle
    x_min_c, x_max_c, x_length, y_min_c, y_max_c, y_length = Get_Smallest_Rectangular_Cell(
        list_rel_vertex)
    
    x_min, y_min  = Get_min_Rectangle_Allocating_rotCell(
        rotation_angle, x_length, y_length)
    
    # transform into a cubic supercell if the matrix is not a parallepiped
    
    # check if the flattened unit cell vector matrix is different than 0 in more than 3 elements
    if len(np.where(unit_cell_oriented_dims.flatten() != 0 )[0]) != 3:
        
        # if the unit cell rotated is not cubic or paralepiiped like, convert it
        # and afterwards generate the matrix to make it the desired dimensions
        n_unit_cells_cube = unit_cell_oriented.get_global_number_of_atoms()*2  #!!! Hyperparam to change if need
        supercell_cubic_transform = find_optimal_cell_shape(
            unit_cell_oriented.cell, n_unit_cells_cube, 'sc')
        
        cube_trans_supercell = make_supercell(unit_cell_oriented,supercell_cubic_transform)
        cube_trans_supercell_dims = cube_trans_supercell.get_cell()
        show_atoms(cube_trans_supercell)
        # !!! UNITS: set the unit cell dimensions in nm
        #!!! beware it is not a perennial change to the unit cell but something 
        # occasional for the  unit cell good considereation
        #!!! then the object holding the unit cell is still in angstroms
        cube_trans_supercell_dims = cube_trans_supercell_dims/10
    
        x_cartesian_total = cube_trans_supercell_dims[0,0]
        y_cartesian_total = cube_trans_supercell_dims[1,1]
        z_cartesian_total = cube_trans_supercell_dims[2,2]
        
        # later if the z_s are different it does not matter that much, as we will
        # delete all the z that are bigger than the really wanted z
        # sum 1 extra cell just in case
        
        rect_trans_x = int(np.ceil(x_min/x_cartesian_total)+1)
        rect_trans_y = int(np.ceil(y_min/y_cartesian_total)+1)
        rect_trans_z = int(np.ceil(z_thickness/z_cartesian_total)+1)
        
        
        supercell_rectangle_translation = np.array([[rect_trans_x, 0, 0],
                                                    [0, rect_trans_y, 0],
                                                    [0, 0, rect_trans_z]])
        
        supercell = make_supercell(cube_trans_supercell, supercell_rectangle_translation)

        
    else:
            
        # if the cell is paralepided already just multiply thorugh axis normally
        x_cartesian_total = unit_cell_oriented_dims[0,0]
        y_cartesian_total = unit_cell_oriented_dims[1,1]
        z_cartesian_total = unit_cell_oriented_dims[2,2]
        
        rect_trans_x = int(np.ceil(x_min/x_cartesian_total)+1)
        rect_trans_y = int(np.ceil(y_min/y_cartesian_total)+1)
        rect_trans_z = int(np.ceil(z_thickness/z_cartesian_total)+1)
        
        supercell_rectangle_translation = np.array([[rect_trans_x, 0, 0],
                                                    [0, rect_trans_y, 0],
                                                    [0, 0, rect_trans_z]])
        
        supercell = make_supercell(unit_cell_oriented, supercell_rectangle_translation)
        
    
    
    # check if the generated cubic shape is really rectangular 
    
    if len(np.where(supercell.get_cell().flatten() != 0 )[0]) == 3:
        # if it is rectangular, then proceed as normally and rotate and crop
        show_atoms(supercell,  plane='xy')
        
        # !!! UNITS: Convert the x_length and y_length from nm to
        # angstroms for the coordinate calculations
        x_length = x_length*10
        y_length = y_length*10
        
        # coordiantes in which the square within starts (coord of the bottom left corner)
        x_i_small_rect, y_i_small_rect = Get_starting_coords_Rect_in_RotSCell(
                rotation_angle, x_length, y_length)
        
        
        # bug here findin gthe starting poistions in -110 degrees puts it outside
        # as the shape that is being rotated since the begining is not perfectly rectangular but a romboid
        # after the supercell  is created maybe ensure we have a rectangle and center it at 0, 0
        # as it should even be big enough to contain the region anyways, crop the biggest rectangle there and 
        # put it back to the 0,0 coordinates in bottom left
        # beware it is not perfectly rectangular but can hve some angles strange that do not make
        # the function that we already have applicabel
        # this higly sure would depend on the angle defined by the  oriented cell
        # wathc out as cropping would be hard as then we lose the periodic boundary conditions
        # the angle between the two sides of the supercell should be asy to compute by checking at th
        # cartesian decomposition of th esupercell genrated
        # SOLVED
        
        # !!! the following two lines can be computed and used in case the 
        # computation of the normal coordinates does not work fine
        # by now the x_i_small_rect, y_i_small_rect seem to work fine
        
        # as we need to always sum x_length and y_length to the coords (not x,y_length_max)
        # the clearer the coords are the better, and it seems the 
        # x_i_small_rect, y_i_small_rect coords for 1st and 2nd quadrant
        # and 
        # x_i_big_rect, y_i_big_rect would work better for 3rd and 4th quadrant
        # are clearer which means they fit better the x_length and y lenght inside the
        # rotated supercell    
        
        # x_length_max, y_length_max = get_biggest_Rect_fitting_RotSCell(
        #     supercell, rotation_angle)
        # x_i_big_rect, y_i_big_rect = Get_starting_coords_Rect_in_RotSCell(
        #         rotation_angle, x_length_max, y_length_max)
            
    
        # let us rotate the supercell
        supercell.rotate(rotation_angle, 'z', rotate_cell= True)
        show_atoms(supercell,  plane='xy')
    
        # show if wanted
        # show_atoms(supercell, plane='xy')
        # show_atoms(supercell, plane='xz')
        # show_atoms(supercell, plane='yz')
        
        # retrieve positions, elements from supercell
        atomic_abs_positions = supercell.get_positions()
        elements_list_to_pos = supercell.get_atomic_numbers()    
    
        atomic_abs_positions_mod = np.copy(atomic_abs_positions)
        elements_list_to_pos_mod = np.copy(elements_list_to_pos)
        
        # mod the x,y positions
        # !!! modify the x_i_small_rect according to quadrant to x_i_big_rect in case
        # we observe some problem with this in these quadrants
        atomic_abs_positions_mod[:,0] = atomic_abs_positions_mod[:,0] - x_i_small_rect
        atomic_abs_positions_mod[:,1] = atomic_abs_positions_mod[:,1] - y_i_small_rect
        
        # filter positions that are inside the x_length and y_length limit
        
        # filter x positions within 0 and x_length
        elements_list_to_pos_mod = elements_list_to_pos_mod[atomic_abs_positions_mod[:,0]>=0]
        atomic_abs_positions_mod = atomic_abs_positions_mod[atomic_abs_positions_mod[:,0]>=0]
        elements_list_to_pos_mod = elements_list_to_pos_mod[atomic_abs_positions_mod[:,0]<= x_length]
        atomic_abs_positions_mod = atomic_abs_positions_mod[atomic_abs_positions_mod[:,0]<= x_length]
        # filter y positions within 0 and y_length
        elements_list_to_pos_mod = elements_list_to_pos_mod[atomic_abs_positions_mod[:,1]>=0]
        atomic_abs_positions_mod = atomic_abs_positions_mod[atomic_abs_positions_mod[:,1]>=0]
        elements_list_to_pos_mod = elements_list_to_pos_mod[atomic_abs_positions_mod[:,1]<= y_length]
        atomic_abs_positions_mod = atomic_abs_positions_mod[atomic_abs_positions_mod[:,1]<= y_length]
            
        # !!! UNITS: Convert the x_min_c and y_min_c from nm to
        # angstroms for the reescaling of the cell
        x_min_c = x_min_c*10
        y_min_c = y_min_c*10
        
        # adjust the positions based on the real absolute position they have within 
        # the global device
        atomic_abs_positions_mod[:,0] = atomic_abs_positions_mod[:,0] + x_min_c
        atomic_abs_positions_mod[:,1] = atomic_abs_positions_mod[:,1] + y_min_c
    
        # evaluate if the elements coords are within the segmented region and z thickness
        # !!! UNITS: for that, change the units of the atom coords from angs to nm
        # as the contours are expressed in nm
        atomic_abs_positions_mod = atomic_abs_positions_mod/10
        
        # ensure all z are smaller than the z_thickness given
        elements_list_to_pos_mod = elements_list_to_pos_mod[atomic_abs_positions_mod[:,2] >= 0]
        atomic_abs_positions_mod = atomic_abs_positions_mod[atomic_abs_positions_mod[:,2] >= 0]
        elements_list_to_pos_mod = elements_list_to_pos_mod[atomic_abs_positions_mod[:,2] <= z_thickness]
        atomic_abs_positions_mod = atomic_abs_positions_mod[atomic_abs_positions_mod[:,2] <= z_thickness]
    
        
        # !!! Here the difference with the shaped atomistic one,
        # here we do not filter according to the shape of the device
    
        # adjust the y position to fit the 0,0 in the bottom left corner instead of 
        # top left that is defined by the contours
        if adjust_y_bottomleft == True:
            max_vals = []
            for element in conts_vert_of_segmented_image:
                if element[element.find('_'):] == '_rel_vertexs':
                    # store the array with the rel vertexs
                    rel_vertexs = np.copy(np.asarray(conts_vert_of_segmented_image[element]))
                    max_y_val_el = np.max(rel_vertexs[:,0])
                    max_vals.append(max_y_val_el)
                    
                    
            y_abs_max = max(max_vals)  
            atomic_abs_positions_mod[:,1] = y_abs_max - atomic_abs_positions_mod[:,1]
            
            # find from the total device and all the regions the biggest y and 
            # (for this browse through conts_vert_of_segmented_image[str(label_of_seg_region)+'_rel_vertexs'] = list_rel_vertex :
            #     for all labels)
            # then all positions the y coord atomic_abs_positions_mod[:,1]
            # will be atomic_abs_positions_mod[:,1] = biggest_y - atomic_abs_positions_mod[:,1]
            
        
        # !!! UNITS
        # set the coordiantes from nm back to angstroms  as all the comparisons
        # have been done with nm to fit the contour units
        # as ansgtrom seems the standard unit for .xyz files
        atomic_abs_positions_mod = atomic_abs_positions_mod*10
        
        atomic_abs_positions_mod.round(14)
        
        # transform atomic numbers into atomic symbols
        elements_unique_atnum = np.unique(elements_list_to_pos_mod)    
        elements_unique_symb = []
        
        for element in elements_unique_atnum:
            mend_el = mdl.element(int(element))
            el_symbol = mend_el.symbol
            elements_unique_symb.append(el_symbol)
            
        elements_symb_list_to_pos_mod = np.asarray(np.ones((elements_list_to_pos_mod.shape[0],1), dtype=int)*10, dtype=str)
        
        # for the unique elements involved, overwrite the number with symbol
        for element_numb, element_symb in zip(
                elements_unique_atnum, elements_unique_symb):    
            # change the string into the string of the unique symbol by numpy indexing
            elements_symb_list_to_pos_mod[elements_list_to_pos_mod==element_numb] = element_symb
            
        elements_symb_list_to_pos_mod = np.asarray(elements_symb_list_to_pos_mod)
        elements_symb_list_to_pos_mod.shape = (len(elements_symb_list_to_pos_mod),1)
        
        # combine arrays with symbols and 3D positions
        array_length_rect_supercell_xyz = np.hstack((elements_symb_list_to_pos_mod, atomic_abs_positions_mod))
        # print(array_length_rect_supercell_xyz)
        
        
        # build the final file containing the atomistic info from this supercell of region n
        # first line is the number of atoms in cell
        # 2nd is blank space
        length_rect_supercell_xyz = [str(len(elements_symb_list_to_pos_mod))+'\n','\n']
        
        # from third line stack the lines containing the element symbol and the 3d coords
        for symb_coord in array_length_rect_supercell_xyz:
            str_symb_coord = str(symb_coord[0])+'    '+str(symb_coord[1])+'    '+str(symb_coord[2])+'    '+str(symb_coord[3])+'\n'
            length_rect_supercell_xyz = length_rect_supercell_xyz + [str_symb_coord]
        
        # create the .xyz file
        temp_xyz_filename = save_filepath + 'temp_xyz_rot_supercell_'+str(label_of_seg_region)+'.xyz'
        
        filename = Path(temp_xyz_filename)
        file_already_created = filename.is_file()
        
        if file_already_created == True:
            # overwrite the file
            with open(temp_xyz_filename, "w+") as f:
                f.truncate(0)
                f.writelines(length_rect_supercell_xyz)
                f.close()
        else:
            # create a new file
            with open(temp_xyz_filename, 'w+') as f:
                f.writelines(length_rect_supercell_xyz)
                f.close()

            
        # show the supercell created if wanted    
        length_supercel = read(temp_xyz_filename)
        
        show_atoms(length_supercel, plane='xy')
        show_atoms(length_supercel, plane='xz')
        show_atoms(length_supercel, plane='yz')
        
    else:
        
        # if by no means we achieve to have a rectangular orthorombic cell
        # then just rotate the oriented cell, make it superbig so it for sure
        # can hold the wanted size and then just crop accordingly
        # this is brute force but should be just in exceptional cases
        # where strange zone axis are wanted or harder unit cell geometries
        # if this condition is rising instead of the optimum one in a
        # regular basis, check the function find_optimal_cell_shape as might 
        # be generating this strange volumes to which the routine is based
        
        # let us rotate the supercell
        unit_cell_oriented.rotate(rotation_angle, 'z', rotate_cell= True)
        show_atoms(unit_cell_oriented,  plane='xy')
        
        n_unit_cells_cube = unit_cell_oriented.get_global_number_of_atoms()*3  #!!! Hyperparam to change if need
        supercell_cubic_transform = find_optimal_cell_shape(
            unit_cell_oriented.cell, n_unit_cells_cube, 'sc')
        
        cube_trans_supercell = make_supercell(
            unit_cell_oriented,supercell_cubic_transform)
        cube_trans_supercell_dims = cube_trans_supercell.get_cell()
        
        # !!! UNITS: set the unit cell dimensions in nm
        #!!! beware it is not a perennial change to the unit cell but something 
        # occasional for the  unit cell good considereation
        #!!! then the object holding the unit cell is still in angstroms
        cube_trans_supercell_dims = cube_trans_supercell_dims/10
        
        # find the biggest positive cartesian x,y component from either a1 or a2
        # as the cell multiplicationw ill be done through this axis
        x_posib = cube_trans_supercell_dims[:2,0]
        y_posib = cube_trans_supercell_dims[:2,1]
        
        x_max_ind = np.argmax(abs(x_posib))
        y_max_ind = np.argmax(abs(y_posib))
        
        # ensure the translation operation is an invertible matrix det(A) !=0 
        if x_max_ind != y_max_ind:
            # this means the indices are differnt and the variable invertible
            x_max = np.max(abs(x_posib))
            y_max = np.max(abs(y_posib))
        else:
            # if the indices are the same it would mean a det(A) = 0 and fail
            # avoid this by making the indices different and changing the x_max, y_max
            x_max_prov = np.max(abs(x_posib))
            y_max_prov = np.max(abs(y_posib))
            ind_min_from_xy = np.argmin([x_max_prov, y_max_prov])

            if x_max_prov == y_max_prov:
                # adress the unusual case in which they are equal
                x_max_ind = 0
                y_max_ind = 1
                x_max = abs(x_posib)[x_max_ind]
                y_max = abs(y_posib)[y_max_ind]
                
            elif ind_min_from_xy == 0:

                # this means that the x value is the min from both
                # then keep the x_max as x_max and change y_max to the other val
                x_max = np.max(abs(x_posib))
                x_max_ind = np.argmax(abs(x_posib))
                
                if x_max_ind == 0:
                    y_max_ind = 1
                    y_max = abs(y_posib)[y_max_ind]
                else:
                    y_max_ind = 0
                    y_max = abs(y_posib)[y_max_ind]
                
            else:

                # this means that the y value is the min from both
                # then keep the y_max and change the x_max to the other possibility
                y_max = np.max(abs(y_posib))
                y_max_ind = np.argmax(abs(y_posib))
                
                if y_max_ind == 0:
                    x_max_ind = 1
                    x_max = abs(x_posib)[x_max_ind]
                else:
                    x_max_ind = 0
                    x_max = abs(x_posib)[x_max_ind]
                
                
        z_cartesian_total = cube_trans_supercell_dims[2,2]
        
        # the lenghts and cell info are in nm
        rect_trans_x = int(np.ceil(x_length/x_max)+1)
        rect_trans_y = int(np.ceil(y_length/y_max)+1)
        rect_trans_z = int(np.ceil(z_thickness/z_cartesian_total)+1)
        
        # make sure sure is big enough to hold the region 
        
        # doing the cubi thing even if tdoes not set it rectangular could help making
        # the next steps less time consuming and more direct as would be a bit more rectangular
        
        
        # !!! UNITS: Convert the x_length and y_length from nm to
        # angstroms for the coordinate calculations
        x_length = x_length*10
        y_length = y_length*10
        
        
        # !!! UNITS : Set thickness NEW variable in angstroms to use it in 
        # this comparison with the atomic coordinates in z
        z_thickness_angs = z_thickness*10
        
        supercell_rectangle_translation = np.zeros((3,3))
        
        supercell_rectangle_translation[0, x_max_ind] = rect_trans_x
        supercell_rectangle_translation[1, y_max_ind] = rect_trans_y
        supercell_rectangle_translation[2, 2] = rect_trans_z
    
        supercell = make_supercell(
            cube_trans_supercell, supercell_rectangle_translation)
        
        for iteration in np.arange(0,15,1):
            
            # double the supercell dimensiones/sides after every iteration      
            rect_trans_x_mod = 2
            rect_trans_y_mod = 2
    
            supercell_rectangle_translation = np.zeros((3,3))
            
            supercell_rectangle_translation[0, x_max_ind] = rect_trans_x_mod
            supercell_rectangle_translation[1, y_max_ind] = rect_trans_y_mod
            supercell_rectangle_translation[2, 2] = 2
        
            supercell = make_supercell(
                supercell, supercell_rectangle_translation)
            
            # get positions of the coordiantes and find the center of mass
            positions_of_supermegacell = supercell.get_positions()
            elements_list_of_supermegacell = supercell.get_atomic_numbers() 
            
            x_COM = np.average(positions_of_supermegacell[:,0])
            y_COM = np.average(positions_of_supermegacell[:,1])
            z_COM = np.average(positions_of_supermegacell[:,2])
            
            # Main Plane XY
            # xcom and ycom are in angstroms
            # get x,y length to angstroms
            positions_of_supermegacell[:,2] = positions_of_supermegacell[:,2] - (z_COM-(z_thickness_angs/2))
            positions_of_supermegacell = positions_of_supermegacell[positions_of_supermegacell[:,2] >=0] 
            positions_of_supermegacell = positions_of_supermegacell[positions_of_supermegacell[:,2] <=z_thickness_angs] 
            
            
            top_right_cords = positions_of_supermegacell[positions_of_supermegacell[:,0] >= x_COM+(x_length/2)]
            top_right_cords = top_right_cords[top_right_cords[:,1] >= y_COM+(y_length/2)]
            
            bottom_right_cords = positions_of_supermegacell[positions_of_supermegacell[:,0] >= x_COM+(x_length/2)]
            bottom_right_cords = bottom_right_cords[bottom_right_cords[:,1] <= y_COM-(y_length/2)]
            
            top_left_cords = positions_of_supermegacell[positions_of_supermegacell[:,0] <= x_COM-(x_length/2)]
            top_left_cords = top_left_cords[top_left_cords[:,1] >= y_COM+(y_length/2)]
            
            bottom_left_cords = positions_of_supermegacell[positions_of_supermegacell[:,0] <= x_COM-(x_length/2)]
            bottom_left_cords = bottom_left_cords[bottom_left_cords[:,1] <= y_COM-(y_length/2)]
            
            
            # then check whether the z is inside the region of interest
            # this is checking the conditions in two planes yz and xz
            
            positions_of_supermegacell = supercell.get_positions()
            
            # Plane XZ
            top_right_cords_xz = positions_of_supermegacell[positions_of_supermegacell[:,0] >= x_COM+(x_length/2)]
            top_right_cords_xz = top_right_cords_xz[top_right_cords_xz[:,2] >= z_COM+(z_thickness_angs/2)]
            
            bottom_right_cords_xz = positions_of_supermegacell[positions_of_supermegacell[:,0] >= x_COM+(x_length/2)]
            bottom_right_cords_xz = bottom_right_cords_xz[bottom_right_cords_xz[:,2] <= z_COM-(z_thickness_angs/2)]
            
            top_left_cords_xz = positions_of_supermegacell[positions_of_supermegacell[:,0] <= x_COM-(x_length/2)]
            top_left_cords_xz = top_left_cords_xz[top_left_cords_xz[:,2] >= z_COM+(z_thickness_angs/2)]
            
            bottom_left_cords_xz = positions_of_supermegacell[positions_of_supermegacell[:,0] <= x_COM-(x_length/2)]
            bottom_left_cords_xz = bottom_left_cords_xz[bottom_left_cords_xz[:,2] <= z_COM-(z_thickness_angs/2)]
            
            # Plane YZ
            top_right_cords_yz = positions_of_supermegacell[positions_of_supermegacell[:,1] >= y_COM+(y_length/2)]
            top_right_cords_yz = top_right_cords_yz[top_right_cords_yz[:,2] >= z_COM+(z_thickness_angs/2)]
            
            bottom_right_cords_yz = positions_of_supermegacell[positions_of_supermegacell[:,1] >= y_COM+(y_length/2)]
            bottom_right_cords_yz = bottom_right_cords_yz[bottom_right_cords_yz[:,2] <= z_COM-(z_thickness_angs/2)]
            
            top_left_cords_yz = positions_of_supermegacell[positions_of_supermegacell[:,1] <= y_COM-(y_length/2)]
            top_left_cords_yz = top_left_cords_yz[top_left_cords_yz[:,2] >= z_COM+(z_thickness_angs/2)]
            
            bottom_left_cords_yz = positions_of_supermegacell[positions_of_supermegacell[:,1] <= y_COM-(y_length/2)]
            bottom_left_cords_yz = bottom_left_cords_yz[bottom_left_cords_yz[:,2 ] <= z_COM-(z_thickness_angs/2)]
            
            
            # if there exist atomic coordinates in all the 4 corners of the x length 
            # y lenght rectangle then accept the supercell as it can contain it
            # in all the three planes XY, XZ, YZ
            
            # Plane XY
            if len(top_right_cords) != 0 and len(bottom_right_cords) != 0 and len(top_left_cords) != 0 and len(bottom_left_cords) != 0:
                # Plane XZ
                if len(top_right_cords_xz) != 0 and len(bottom_right_cords_xz) != 0 and len(top_left_cords_xz) != 0 and len(bottom_left_cords_xz) != 0:
                    # Plane YZ
                    if len(top_right_cords_yz) != 0 and len(bottom_right_cords_yz) != 0 and len(top_left_cords_yz) != 0 and len(bottom_left_cords_yz) != 0:
                        # if everything is met break the loop and keep the supcell
                        break
            
            
        show_atoms(supercell)
        # now this supercell can contain the x_lenght and y_length square
        # crop it
        
        # retrieve positions, elements from supercell
        atomic_abs_positions = supercell.get_positions()
        elements_list_to_pos = supercell.get_atomic_numbers()    
    
        atomic_abs_positions_mod = np.copy(atomic_abs_positions)
        elements_list_to_pos_mod = np.copy(elements_list_to_pos)

        elements_list_to_pos_mod = elements_list_to_pos_mod[atomic_abs_positions_mod[:,0] >= x_COM-(x_length/2)]
        atomic_abs_positions_mod = atomic_abs_positions_mod[atomic_abs_positions_mod[:,0] >= x_COM-(x_length/2)]
        elements_list_to_pos_mod = elements_list_to_pos_mod[atomic_abs_positions_mod[:,0] <= x_COM+(x_length/2)]
        atomic_abs_positions_mod = atomic_abs_positions_mod[atomic_abs_positions_mod[:,0] <= x_COM+(x_length/2)]
        
        elements_list_to_pos_mod = elements_list_to_pos_mod[atomic_abs_positions_mod[:,1] >= y_COM-(y_length/2)]
        atomic_abs_positions_mod = atomic_abs_positions_mod[atomic_abs_positions_mod[:,1] >= y_COM-(y_length/2)]
        elements_list_to_pos_mod = elements_list_to_pos_mod[atomic_abs_positions_mod[:,1] <= y_COM+(y_length/2)]
        atomic_abs_positions_mod = atomic_abs_positions_mod[atomic_abs_positions_mod[:,1] <= y_COM+(y_length/2)]

        # readjust positions to center 0,0 at bottom left
        atomic_abs_positions_mod[:,0] = atomic_abs_positions_mod[:,0] - (x_COM-(x_length/2))
        atomic_abs_positions_mod[:,1] = atomic_abs_positions_mod[:,1] - (y_COM-(y_length/2))
        
        # and then readjust again to set the coordinates to the relative position of the device
        
        # !!! UNITS: Convert the x_min_c and y_min_c from nm to
        # angstroms for the reescaling of the cell
        x_min_c = x_min_c*10
        y_min_c = y_min_c*10
        
        # adjust the positions based on the real absolute position they have within 
        # the global device
        atomic_abs_positions_mod[:,0] = atomic_abs_positions_mod[:,0] + x_min_c
        atomic_abs_positions_mod[:,1] = atomic_abs_positions_mod[:,1] + y_min_c
    
        # evaluate if the elements coords are within the segmented region and z thickness
        # !!! UNITS: for that, change the units of the atom coords from angs to nm
        # as the contours are expressed in nm
        atomic_abs_positions_mod = atomic_abs_positions_mod/10
        
        z_average = np.average(atomic_abs_positions_mod[:,2])
        
        atomic_abs_positions_mod[:,2] = atomic_abs_positions_mod[:,2] - z_average
        # ensure all z are smaller than the z_thickness given
        elements_list_to_pos_mod = elements_list_to_pos_mod[atomic_abs_positions_mod[:,2] >=  -(z_thickness/2)]
        atomic_abs_positions_mod = atomic_abs_positions_mod[atomic_abs_positions_mod[:,2] >=  -(z_thickness/2)]
        elements_list_to_pos_mod = elements_list_to_pos_mod[atomic_abs_positions_mod[:,2] <=  +(z_thickness/2)]
        atomic_abs_positions_mod = atomic_abs_positions_mod[atomic_abs_positions_mod[:,2] <=  +(z_thickness/2)]
        
        # put the z back starting from 0 to z_thickness range
        atomic_abs_positions_mod[:,2] = atomic_abs_positions_mod[:,2] + (z_thickness/2)
        

        # !!! Difference with the previous function build shaped atomistic
        # we do not filter the distances inside the contour
        
        # adjust the y position to fit the 0,0 in the bottom left corner instead of 
        # top left that is defined by the contours
        if adjust_y_bottomleft == True:
            max_vals = []
            for element in conts_vert_of_segmented_image:
                if element[element.find('_'):] == '_rel_vertexs':
                    # store the array with the rel vertexs
                    rel_vertexs = np.copy(np.asarray(conts_vert_of_segmented_image[element]))
                    max_y_val_el = np.max(rel_vertexs[:,0])
                    max_vals.append(max_y_val_el)
                    
                    
            y_abs_max = max(max_vals)  
            atomic_abs_positions_mod[:,1] = y_abs_max - atomic_abs_positions_mod[:,1]
            
            # find from the total device and all the regions the biggest y and 
            # (for this browse through conts_vert_of_segmented_image[str(label_of_seg_region)+'_rel_vertexs'] = list_rel_vertex :
            #     for all labels)
            # then all positions the y coord atomic_abs_positions_mod[:,1]
            # will be atomic_abs_positions_mod[:,1] = biggest_y - atomic_abs_positions_mod[:,1]
            
            
        
        # !!! UNITS
        # set the coordiantes from nm back to angstroms  as all the comparisons
        # have been done with nm to fit the contour units
        # as ansgtrom seems the standard unit for .xyz files
        atomic_abs_positions_mod = atomic_abs_positions_mod*10
        
        atomic_abs_positions_mod.round(14)
        
        # transform atomic numbers into atomic symbols
        elements_unique_atnum = np.unique(elements_list_to_pos_mod)    
        elements_unique_symb = []
        
        for element in elements_unique_atnum:
            mend_el = mdl.element(int(element))
            el_symbol = mend_el.symbol
            elements_unique_symb.append(el_symbol)
            
        elements_symb_list_to_pos_mod = np.asarray(np.ones((elements_list_to_pos_mod.shape[0],1), dtype=int)*10, dtype=str)
        
        # for the unique elements involved, overwrite the number with symbol
        for element_numb, element_symb in zip(
                elements_unique_atnum, elements_unique_symb):    
            # change the string into the string of the unique symbol by numpy indexing
            elements_symb_list_to_pos_mod[elements_list_to_pos_mod==element_numb] = element_symb
            
        elements_symb_list_to_pos_mod = np.asarray(elements_symb_list_to_pos_mod)
        elements_symb_list_to_pos_mod.shape = (len(elements_symb_list_to_pos_mod),1)
        
        # combine arrays with symbols and 3D positions
        array_length_rect_supercell_xyz = np.hstack((elements_symb_list_to_pos_mod, atomic_abs_positions_mod))
        # print(array_length_rect_supercell_xyz)
        
        
        # build the final file containing the atomistic info from this supercell of region n
        # first line is the number of atoms in cell
        # 2nd is blank space
        length_rect_supercell_xyz = [str(len(elements_symb_list_to_pos_mod))+'\n','\n']
        
        # from third line stack the lines containing the element symbol and the 3d coords
        for symb_coord in array_length_rect_supercell_xyz:
            str_symb_coord = str(symb_coord[0])+'    '+str(symb_coord[1])+'    '+str(symb_coord[2])+'    '+str(symb_coord[3])+'\n'
            length_rect_supercell_xyz = length_rect_supercell_xyz + [str_symb_coord]
        
        # create the .xyz file
        temp_xyz_filename = save_filepath + 'temp_xyz_rot_supercell_'+str(label_of_seg_region)+'.xyz'
        
        filename = Path(temp_xyz_filename)
        file_already_created = filename.is_file()
        
        if file_already_created == True:
            # overwrite the file
            with open(temp_xyz_filename, "w+") as f:
                f.truncate(0)
                f.writelines(length_rect_supercell_xyz)
                f.close()
        else:
            # create a new file
            with open(temp_xyz_filename, 'w+') as f:
                f.writelines(length_rect_supercell_xyz)
                f.close()
            
        # show the supercell created if wanted    
        length_supercel = read(temp_xyz_filename)
        
        show_atoms(length_supercel, plane='xy')
        show_atoms(length_supercel, plane='xz')
        show_atoms(length_supercel, plane='yz')
                
                
    # Build a complementary .txt file containing, for every atom entry in 
    # 'temp_xyz_rot_supercell_'+str(label_of_seg_region)+'.xyz', a label
    # corresponding to label_of_seg_region to identify the segmented region 
    # where they belong, for keeping track of what has been done to its atoms
        
    atom_identifyer_file = [str(len(elements_symb_list_to_pos_mod))+'\n','\n']
    
    for iteration_ in range(len(elements_symb_list_to_pos_mod)):
        new_line_ = str(label_of_seg_region) + '\n'
        atom_identifyer_file = atom_identifyer_file + [new_line_]
    
    
    # create the .xyz file
    atom_identifyer_filename = save_filepath + 'atom_identifier_'+str(label_of_seg_region)+'.txt'
    
    filename_atid = Path(atom_identifyer_filename)
    fileid_already_created = filename_atid.is_file()
    
    if fileid_already_created == True:
        # overwrite the file
        with open(atom_identifyer_filename, "w+") as f:
            f.truncate(0)
            f.writelines(atom_identifyer_file)
            f.close()
    else:
        # create a new file
        with open(atom_identifyer_filename, 'w+') as f:
            f.writelines(atom_identifyer_file)
            f.close()
        
        
    return









def Combine_xyz_supercells(
        temp_xyz_files_folder_directory):
    '''
    After generating the temporary .xyz files containing the shaped supercells
    according to the given sgmented device and phase infor for every region
    then unite them all in one single document containing the full cell

    Parameters
    ----------
    temp_xyz_files_folder_directory: directory to the folder of files containing the supercell 
         for all the regions in the device

    Returns
    -------
    None.

    '''
    
    
    total_num_atoms = 0
    total_list_symb_coords = ['\n']
    combined_atom_idents = ['\n']
    
    
    for file_name in os.listdir(temp_xyz_files_folder_directory):
        file_name_c = temp_xyz_files_folder_directory +'\\'+file_name
        file_extension = file_name_c[::-1][:(file_name_c[::-1]).find('.')][::-1]
        if file_extension == 'xyz' and file_name[:6] != 'global':
            supercell_xyz = open(file_name_c)
            list_lines = supercell_xyz.readlines()
            numb_atoms = int(list_lines[0])
            total_num_atoms = total_num_atoms + numb_atoms
            list_symb_coords = list_lines[2:]
            total_list_symb_coords = total_list_symb_coords + list_symb_coords
            
            # concatenate the label atom identifiers
            # assuming the name of the files is temp_xyz_rot_supercell_
            # coordinated witht the global supercell construction building
            label_of_region = file_name[23:file_name.find('.')]
            path_to_atomident_lab = temp_xyz_files_folder_directory + '\\' + 'atom_identifier_' + label_of_region + '.txt'
            atomidentfile = open(path_to_atomident_lab)
            list_lin_atomid = atomidentfile.readlines()
            list_lin_atomid_ap = list_lin_atomid[2:]
            combined_atom_idents = combined_atom_idents + list_lin_atomid_ap
            
            
    total_list_symb_coords = [str(total_num_atoms)+'\n'] + total_list_symb_coords
    global_supercell_name = temp_xyz_files_folder_directory +'\\' + 'global_device_supercell.xyz'
    
    filename = Path(global_supercell_name)
    file_already_created = filename.is_file()
    
    if file_already_created == True:
        # overwrite the file
        with open(global_supercell_name, "w+") as f:
            f.truncate(0)
            f.writelines(total_list_symb_coords)
            f.close()
    else:
        # create a new file
        with open(global_supercell_name, 'w+') as f:
            f.writelines(total_list_symb_coords)
            f.close()


    # now do the same with the files containing the label identifiers
    combined_atom_idents = [str(total_num_atoms)+'\n'] + combined_atom_idents
    global_atomid_name = temp_xyz_files_folder_directory +'\\' + 'global_atom_identifier.txt'
    
    filename_atid = Path(global_atomid_name)
    file_already_created_atid = filename_atid.is_file()
    
    if file_already_created_atid == True:
        # overwrite the file
        with open(global_atomid_name, "w+") as f:
            f.truncate(0)
            f.writelines(combined_atom_idents)
            f.close()
    else:
        # create a new file
        with open(global_atomid_name, 'w+') as f:
            f.writelines(combined_atom_idents)
            f.close()

    return
        
    
# '''
# Define thickness and in-(ZA) plane rotation
# '''



# # the format the coordinates are is, in nm, 
# '''
# Here we define some custom geometries or morphologies by putting some 
# vertex together, and then we construct some classes and objects 
# that ressemble the classes that are outputted from the global workflow and
# from the segmentation
# This was just for testing the process so it could be directly coupled with
# the full workflow
# '''


# nextnano_vertexs=dict()


# shape_1 =[[20,10],[30,10],[30,17.2],[45.7,17.2],[60,17.2],[60,9],[100,9],
#           [100,13],[90,13],[90,25],[90,31],[95,31],[95,40],[19,40],[19,30],
#           [11.63,30], [11.63,10]]
          
# shape_2 = [[59.9,40],[68,40],[68,90],[68,100],[55,100],[55,82],[46,82],[33,82],
#             [33,92],[25,92],[25,60],[25,40],
#             [30,40],[45,40],[45,42],[59.9,42]]

# shape_3 = [[68,43],[100,43],[100,90],[95,90],[95,70],[80,70],[80,80],
#             [90,80],[90,90],[68,90],[68,75],[68,52]]

# # !!! this is just for this example
# # put the center of coords 0,0 in the top left
# # by subtracting 110


# shape_1 = np.asarray(shape_1)
# shape_2 = np.asarray(shape_2)
# shape_3 = np.asarray(shape_3)

# shape_1[:,1] = 110 - shape_1[:,1]
# shape_2[:,1] = 110 - shape_2[:,1]
# shape_3[:,1] = 110 - shape_3[:,1]

# # shape_1_cache = np.copy(shape_1)
# # shape_1[:,0] = shape_1_cache[:,1]
# # shape_1[:,1] = shape_1_cache[:,0]

# shape_1 = shape_1*0.1
# shape_2 = shape_2*0.1
# shape_3 = shape_3*0.1

# nextnano_vertexs[str(int(0))+'_nnvertexs']=shape_1
# nextnano_vertexs[str(int(1))+'_nnvertexs']=shape_2
# nextnano_vertexs[str(int(2))+'_nnvertexs']=shape_3

# contour_vectors1 = []    
    
# lenshape1_or = len(list(shape_1))

# for index in range(lenshape1_or):
    
#     # !!! this is just for the example used
  
#     # to close the loop 
    
    
    
#     shape_1_list = list(shape_1)+[list(shape_1)[0]]
    
#     y_rel_init =  shape_1_list[index][1]
#     x_rel_init =  shape_1_list[index][0]
    
    
    
#     y_rel_final =  shape_1_list[index+1][1]
#     x_rel_final =  shape_1_list[index+1][0]
    
#     # !!! use this normally
#     # y_rel_init, x_rel_init = rel_contour
    
#     contour_vect = (y_rel_init, x_rel_init, y_rel_final, x_rel_final)
    
#     contour_vectors1.append(contour_vect)
    
# contour_vectors2 = []    
    
# lenshape2_or = len(list(shape_2))

# for index in range(lenshape2_or):
    
#     # !!! this is just for the example used
  
#     # to close the loop 
    
    
    
#     shape_2_list = list(shape_2)+[list(shape_2)[0]]
    
#     y_rel_init =  shape_2_list[index][1]
#     x_rel_init =  shape_2_list[index][0]
    
    
    
#     y_rel_final =  shape_2_list[index+1][1]
#     x_rel_final =  shape_2_list[index+1][0]
    
#     # !!! use this normally
#     # y_rel_init, x_rel_init = rel_contour
    
#     contour_vect = (y_rel_init, x_rel_init, y_rel_final, x_rel_final)
    
#     contour_vectors2.append(contour_vect)    
   
    
# contour_vectors3 = []    
    
# lenshape3_or = len(list(shape_3))

# for index in range(lenshape3_or):
    
#     # !!! this is just for the example used
  
#     # to close the loop 
    
    
    
#     shape_3_list = list(shape_3)+[list(shape_3)[0]]
    
#     y_rel_init =  shape_3_list[index][1]
#     x_rel_init =  shape_3_list[index][0]
    
    
    
#     y_rel_final =  shape_3_list[index+1][1]
#     x_rel_final =  shape_3_list[index+1][0]
    
#     # !!! use this normally
#     # y_rel_init, x_rel_init = rel_contour
    
#     contour_vect = (y_rel_init, x_rel_init, y_rel_final, x_rel_final)
    
#     contour_vectors3.append(contour_vect)    
        
    


# class Contour_vector:
#     def __init__(self, rel_init_coords, rel_final_coords):
#         self.rel_init_coords=rel_init_coords
#         self.rel_final_coords=rel_final_coords


# contours_obj1 = []

# for contour_el in contour_vectors1:
#     contour_obj = Contour_vector(contour_el[:2], contour_el[2:])
#     contours_obj1.append(contour_obj)
    
# contours_obj2 = []

# for contour_el in contour_vectors2:
    
#     contour_obj = Contour_vector(contour_el[:2], contour_el[2:])
#     contours_obj2.append(contour_obj)
    
# contours_obj3 = []

# for contour_el in contour_vectors3:
    
#     contour_obj = Contour_vector(contour_el[:2], contour_el[2:])
#     contours_obj3.append(contour_obj)
    

# conts_vert_of_segmented_image = dict()
# conts_vert_of_segmented_image['0_rel_vertexs'] = shape_1
# conts_vert_of_segmented_image['0_contours'] = contours_obj1
# conts_vert_of_segmented_image['1_rel_vertexs'] = shape_2
# conts_vert_of_segmented_image['1_contours'] = contours_obj2
# conts_vert_of_segmented_image['2_rel_vertexs'] = shape_3
# conts_vert_of_segmented_image['2_contours'] = contours_obj3


# # #  To visualise the cell (Like the finite elements one)

# # # The GDSII file is called a library, which contains multiple cells.
# # lib = gdspy.GdsLibrary()

# # # Geometry must be placed in cells.
# # cell = lib.new_cell('FIRST')

# # #labelling not mandatory but can be useful, datatype is the label
# # #the labels are specially useful for the gds structure but not that much
# # #for the nextnano geometry generation, as the labels are inferred automatically

# # # Create a polygon from a list of vertices
# # label_values = [0,1,2]
# # for label in label_values:
    
# #     list_of_nnvertexs_per_polygon=nextnano_vertexs[str(int(label))+'_'+'nnvertexs']
# #     ld_label = {"layer": 0, "datatype": int(label)}

# #     polygonshape = gdspy.Polygon(list_of_nnvertexs_per_polygon, **ld_label)

    
# #     cell.add(polygonshape)
    

# # # Display all cells using the internal viewer.
# # gdspy.LayoutViewer()    


# '''
# Create and fill the supercell defined by that morphology with atoms
# '''
# # load the unit cell information into ase



# zone_axis = [1,1,3]

# cell_filepath=r'E:\Arxius varis\PhD\2nd_year\Code\unit_cells\inp.cif'
# cell_filepath=r'E:\Arxius varis\PhD\2nd_year\Code\unit_cells\ge_fd-3ms.cif'
# cell_filepath=r'E:\Arxius varis\PhD\2nd_year\Code\unit_cells\inp.cif'
# cell_filepath=r'E:\Arxius varis\PhD\2nd_year\Code\unit_cells\moo2.cif'
# cell_filepath=r'E:\Arxius varis\PhD\2nd_year\Code\unit_cells\inas_wz.cif'

# z_thickness = 1   # nm

# rotation_angle = 170  # degrees


# save_filepaht = r'E:\Arxius varis\PhD\4rth_year\Code\Atomistic_model_builder\trial_cells\\'

# # pick one of the example regions for testing that shape
# label_of_seg_region = 0
 
# unit_cell = read(cell_filepath)

# unit_cell_oriented = surface(
#     unit_cell, indices=(zone_axis[0], zone_axis[1], zone_axis[2]), layers=10, periodic=True)

# unit_cell_oriented *= (3,3,3)

# unit_cell_oriented.rotate(rotation_angle, 'z', rotate_cell= True)

# view(unit_cell_oriented)




# function that does the full shaping and orientation 
# Build_shaped_atomistic(
#     cell_filepath, zone_axis, rotation_angle, z_thickness, 
#     conts_vert_of_segmented_image, label_of_seg_region, 
#     save_filepaht, adjust_y_bottomleft = True)


# # # The global process would loop over the different labels and then combine cells

# # # combine the cells altoghether    
# # temp_xyz_files_folder_directory = r'E:\Arxius varis\PhD\4rth_year\Code\Atomistic_model_builder\\'

# # Combine_xyz_supercells(
# #     temp_xyz_files_folder_directory)


# # final_global_device_supcell = read(temp_xyz_files_folder_directory + 'global_device_supercell.xyz')
# # show_atoms(final_global_device_supcell, plane='xy')
# # show_atoms(final_global_device_supcell, plane='xz')
# # show_atoms(final_global_device_supcell, plane='yz')




#%%

'''
Script to properly orient, a ZA based on a direction/plane
and given a set of planes in that zone separated a given angle between
'''


import numpy as np
import matplotlib.pyplot as plt
import os
import gdspy
import math
import sys
sys.path.append(r'E:\Arxius varis\PhD\3rd_year\Code\Functions')

import Phase_Identificator as PhaseIdent


import ase
from ase.io import read, write
from ase.visualize import view
from abtem.visualize import show_atoms
from ase.build import surface, make_supercell, find_optimal_cell_shape, rotate
import mendeleev as mdl


from math import gcd
import numpy as np
from numpy.linalg import norm, solve

from ase.build import bulk



def get_surface_basis(
        lattice, indices, layers, vacuum=None, 
        tol=1e-10, periodic=False):
    

  
    """
    From ASE
    
    Create surface from a given lattice and Miller indices.

    lattice: Atoms object or str
        Bulk lattice structure of alloy or pure metal.  Note that the
        unit-cell must be the conventional cell - not the primitive cell.
        One can also give the chemical symbol as a string, in which case the
        correct bulk lattice will be generated automatically.
    indices: sequence of three int
        Surface normal in Miller indices (h,k,l).
    layers: int
        Number of equivalent layers of the slab.
    vacuum: float
        Amount of vacuum added on both sides of the slab.
    periodic: bool
        Whether the surface is periodic in the normal to the surface
    """
    
    
    def build(lattice, basis, layers, tol, periodic):
        surf = lattice.copy()
        scaled = solve(basis.T, surf.get_scaled_positions().T).T
        scaled -= np.floor(scaled + tol)
        surf.set_scaled_positions(scaled)
        surf.set_cell(np.dot(basis, surf.cell), scale_atoms=True)
        surf *= (1, 1, layers)
    
        a1, a2, a3 = surf.cell
        surf.set_cell([a1, a2,
                       np.cross(a1, a2) * np.dot(a3, np.cross(a1, a2)) /
                       norm(np.cross(a1, a2))**2])
    
        # Change unit cell to have the x-axis parallel with a surface vector
        # and z perpendicular to the surface:
        a1, a2, a3 = surf.cell
        surf.set_cell([(norm(a1), 0, 0),
                       (np.dot(a1, a2) / norm(a1),
                        np.sqrt(norm(a2)**2 - (np.dot(a1, a2) / norm(a1))**2), 0),
                       (0, 0, norm(a3))],
                      scale_atoms=True)
    
        surf.pbc = (True, True, periodic)
    
        # Move atoms into the unit cell:
        scaled = surf.get_scaled_positions()
        scaled[:, :2] %= 1
        surf.set_scaled_positions(scaled)
    
        if not periodic:
            surf.cell[2] = 0.0
    
        return surf
    
    def build_first_surf_vect(lattice, basis, layers, tol, periodic):
        surf = lattice.copy()
        scaled = solve(basis.T, surf.get_scaled_positions().T).T
        scaled -= np.floor(scaled + tol)
        surf.set_scaled_positions(scaled)
        surf.set_cell(np.dot(basis, surf.cell), scale_atoms=True)
        surf *= (1, 1, layers)
    
        a1, a2, a3 = surf.cell

    
        a1 = a1
        a2 = a2
        a3 = np.cross(a1, a2) * np.dot(a3, np.cross(a1, a2)) / norm(np.cross(a1, a2))**2
    
        return a1, a2, a3
    
    
    
    def ext_gcd(a, b):
        if b == 0:
            return 1, 0
        elif a % b == 0:
            return 0, 1
        else:
            x, y = ext_gcd(b, a % b)
            return y, x - y * (a // b)    
        
    

    indices = np.asarray(indices)

    if indices.shape != (3,) or not indices.any() or indices.dtype != int:
        raise ValueError('%s is an invalid surface type' % indices)

    if isinstance(lattice, str):
        lattice = bulk(lattice, cubic=True)

    h, k, l = indices  # noqa (E741, the variable l)
    h0, k0, l0 = (indices == 0)

    if h0 and k0 or h0 and l0 or k0 and l0:  # if two indices are zero
        if not h0:
            c1, c2, c3 = [(0, 1, 0), (0, 0, 1), (1, 0, 0)]
        if not k0:
            c1, c2, c3 = [(0, 0, 1), (1, 0, 0), (0, 1, 0)]
        if not l0:
            c1, c2, c3 = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
    else:
        p, q = ext_gcd(k, l)
        a1, a2, a3 = lattice.cell

        # constants describing the dot product of basis c1 and c2:
        # dot(c1,c2) = k1+i*k2, i in Z
        k1 = np.dot(p * (k * a1 - h * a2) + q * (l * a1 - h * a3),
                    l * a2 - k * a3)
        k2 = np.dot(l * (k * a1 - h * a2) - k * (l * a1 - h * a3),
                    l * a2 - k * a3)

        if abs(k2) > tol:
            i = -int(round(k1 / k2))  # i corresponding to the optimal basis
            p, q = p + i * l, q - i * k

        a, b = ext_gcd(p * k + q * l, h)

        c1 = np.array([p * k + q * l, -p * h, -q * h])
        c2 = np.array([0, l, -k]) // abs(gcd(l, k))
        c3 = np.array([b, a * p, a * q])

    c_basis = np.array([c1, c2, c3])
        
    a1, a2, a3 = build_first_surf_vect(lattice, np.array([c1, c2, c3]), layers, tol, periodic)
    
    a_surf_vects = (a1, a2, a3)
    
    return c_basis, a_surf_vects



def find_g_vector_plane(
        real_lattice, direct_metric_tensor, a1, tolerance_diff = 0.5):
    '''
    find the hkl plane (indices) that forms an angle of 0 degrees with
    the plane a1 vector (used for the surface vectors)
    Returns
    -------
    None.

    '''
    # first find the direction to which the  
    direction_to_0 = solve(real_lattice.T, a1)
    plane_to_0 = np.dot(direction_to_0, direct_metric_tensor)
    
    # we ensure it is an integer number or very close to it
    plane_to_0_int = np.round(plane_to_0)
    
    diff = np.abs(plane_to_0 - plane_to_0_int)
    if np.sum(diff) > tolerance_diff:
        return find_g_vector_plane(real_lattice, direct_metric_tensor, 2*a1, tolerance_diff)
    else:
        
        gcd_1 = math.gcd(int(plane_to_0_int[0]), int(plane_to_0_int[1]))
        gcd_2 = math.gcd(int(plane_to_0_int[1]), int(plane_to_0_int[2]))
        gcd_3 = math.gcd(int(plane_to_0_int[0]), int(plane_to_0_int[2]))
        
        gcds = np.array([gcd_1, gcd_2, gcd_3])
                
        gcd = np.min(gcds[gcds != 0])
        
        plane_to_0_int = plane_to_0_int/gcd
        
        return plane_to_0_int
    


def Find_plane_paralel_to_direction(
        direction, direct_metric_tensor, tolerance_diff = 1):
    '''
    

    Parameters
    ----------
    direction : list uvw 3 indices of the direction which defines the direct lattice
            vector that is paralle to the g vector in the reciprocal space 
            that must guide the surface
    direct_metric_tensor : 3x3 array matrix representing the direct metric tensor

    Returns
    -------
    plane_paralel : array hkl Miller indices of the plane 
                    paralel to the direction uvw

    '''
    
    plane_paralel = np.dot(direction, direct_metric_tensor)
    
    plane_paralel_int = np.round(plane_paralel)
    
    diff = np.abs(plane_paralel - plane_paralel_int)
    
    if np.sum(diff) > tolerance_diff:
        return Find_plane_paralel_to_direction(
            2*np.array(direction, dtype = np.int64), direct_metric_tensor, tolerance_diff = tolerance_diff)
    else:
        gcd_1 = math.gcd(int(plane_paralel_int[0]), int(plane_paralel_int[1]))
        gcd_2 = math.gcd(int(plane_paralel_int[1]), int(plane_paralel_int[2]))
        gcd_3 = math.gcd(int(plane_paralel_int[0]), int(plane_paralel_int[2]))
        
        gcds = np.array([gcd_1, gcd_2, gcd_3])
                
        gcd = np.min(gcds[gcds != 0])
        
        plane_paralel_int = plane_paralel_int/gcd
        
        return [int(plane_paralel_int[0]), int(plane_paralel_int[1]), int(plane_paralel_int[2])]
    
    
def Find_direction_paralel_to_plane(
        plane, reciprocal_metric_tensor, tolerance_diff = 1):    
    
    '''
    

    Parameters
    ----------
    plane : list hkl 3 indices of the plane which defines the reciprocal lattice
            vector that is parallel to the direct vector in the direct space 
    reciprocal_metric_tensor : 3x3 array matrix representing the reciprocal metric tensor
    tolerance_diff: value indicating how close to int the indices need to be
                    the bigget the more we force the indices to be integers
                    closer to the inputed float indices (more restrictive)
    Returns
    -------
    direction_paralel : array uvw Miller indices of the direction 
                    paralel to the plane hkl inputted

    '''
    direction_paralel = np.dot(plane, reciprocal_metric_tensor)
    
    direction_paralel_smallest = np.min(np.abs(direction_paralel))
    
    val_to_mult = 1/direction_paralel_smallest
    direction_paralel = direction_paralel*val_to_mult
    

    direction_paralel_int = np.round(direction_paralel)
    
    diff = np.abs(direction_paralel - direction_paralel_int)

    if np.sum(diff) > tolerance_diff:
        return Find_direction_paralel_to_plane(
            2*np.array(plane, dtype = np.int64), reciprocal_metric_tensor, tolerance_diff = tolerance_diff)
       
    else:
        gcd_1 = math.gcd(int(direction_paralel_int[0]), int(direction_paralel_int[1]))
        gcd_2 = math.gcd(int(direction_paralel_int[1]), int(direction_paralel_int[2]))
        gcd_3 = math.gcd(int(direction_paralel_int[0]), int(direction_paralel_int[2]))
        
        gcds = np.array([gcd_1, gcd_2, gcd_3])
                
        gcd = np.min(gcds[gcds != 0])
        
        direction_paralel_int = direction_paralel_int/gcd

        return [int(direction_paralel_int[0]), int(direction_paralel_int[1]), int(direction_paralel_int[2])]
    
   
def Use_plane_or_direction(
        direction, plane, choice = 'plane'):
    
    '''
    Function to choose wheter to use the zone axis as
    indicated by the direction directly found from the axis identificator
    or to use its plane equivalent (the theoretically correct is the plane
    therefore the default one is the plane)
    '''
    
    if choice == 'direction':
        indices_for_surface = direction
    else:
        # choice == 'plane'
        indices_for_surface = plane
        
    return indices_for_surface



def angle_between_cartes_vectors(
        vector_1, vector_2):
    '''
    Angle between two vectors in the cartesian coordinates,
    so the three components are the x, y, z components of a cartesian axis
    system

    Parameters
    ----------
    vector_1 : 3 elements array/list
    vector_2 : 3 elements array/list

    Returns
    -------
    angle_between_cartes_vect : angle, float

    '''
    
    
    if (np.dot(vector_1,vector_2))/(norm(vector_1)*norm(vector_2)) > 1:
        angle_between_cartes_vect = 0
    elif (np.dot(vector_1,vector_2))/(norm(vector_1)*norm(vector_2)) < -1:
        angle_between_cartes_vect = 180
    else:
        angle_between_cartes_vect = (180/np.pi)*np.arccos((np.dot(vector_1,vector_2))/(norm(vector_1)*norm(vector_2)))
    
    return angle_between_cartes_vect





# !!! this function does not work as expected
# I think because the direct and reciprocal lattice comparison is wrong...
# not sure why...

def Ensure_plane_to_direction_is_0(
        plane, direction, direct_lattice, reciprocal_lattice):
    
    '''
    Verify that the angle between the plane and the direction is really 0 
    
    '''
    
    g_vector_paralel = np.dot(plane, reciprocal_lattice)
    g_vector_paralel_comps = np.sum(g_vector_paralel, axis = 0)
    
    direction_vector = np.dot(direction, direct_lattice)
    direction_vector_comps = np.sum(direction_vector, axis = 0)
    
    angle_plane_direction = angle_between_cartes_vectors(g_vector_paralel_comps, direction_vector_comps)

    return angle_plane_direction


def angle_between_planes(
        plane_1, plane_2, reciprocal_metric_tensor):
    '''
    Compute the anlge between two planes hkl1 and hkl2,
    which is the same as computing the angle between g vectors paralel to 
    these planes
    
    Parameters
    ----------
    plane_1 : list of 3 Miller indices hkl
    plane_2 : list of 3 Miller indices hkl
    reciprocal_metric_tensor : 3x3 matrix enconding the reciprocal metric tensor
                            to compute the conversions betweeen 
                            vector reference systems

    Returns
    -------
    angle between the two planes in degrees

    '''
    
    # account for the paralel planes with same multiplied indices
    # where this eq has a solution hkl1 = n*hkl2
    
    gcd_1 = math.gcd(int(plane_1[0]), int(plane_1[1]))
    gcd_2 = math.gcd(int(plane_1[1]), int(plane_1[2]))
    gcd_3 = math.gcd(int(plane_1[0]), int(plane_1[2]))
    
    gcds = np.array([gcd_1, gcd_2, gcd_3])
            
    gcd = np.min(gcds[gcds != 0])
    
    plane_1_smallest = plane_1/gcd
    
    
    gcd_1 = math.gcd(int(plane_2[0]), int(plane_2[1]))
    gcd_2 = math.gcd(int(plane_2[1]), int(plane_2[2]))
    gcd_3 = math.gcd(int(plane_2[0]), int(plane_2[2]))
    
    gcds = np.array([gcd_1, gcd_2, gcd_3])
            
    gcd = np.min(gcds[gcds != 0])
    
    plane_2_smallest = plane_2/gcd
    
    
    if plane_1_smallest[0] == plane_2_smallest[0] and plane_1_smallest[1] == plane_2_smallest[1] and plane_1_smallest[2] == plane_2_smallest[2]:
        angle_between = 0
    elif plane_1_smallest[0] == -plane_2_smallest[0] and plane_1_smallest[1] == -plane_2_smallest[1] and plane_1_smallest[2] == -plane_2_smallest[2]:
        angle_between = 180
    else:
        g1_g2_dot = np.dot(plane_1, np.dot(reciprocal_metric_tensor, plane_2))
        
        g1_norm = np.sqrt(np.dot(plane_1, np.dot(reciprocal_metric_tensor, plane_1)))
        g2_norm = np.sqrt(np.dot(plane_2, np.dot(reciprocal_metric_tensor, plane_2)))
        
        
        angle_between = (180/np.pi)*np.arccos(g1_g2_dot/(g1_norm*g2_norm))
    
    
    return angle_between

    

def Compute_interplanar_distance(
        cell_filepath, plane_indices):
    '''
    Compute the interplanar distances of a plane identified by the miller
    indices plane_indices of the cell that is being pointed by cell_filepath
    d = 1/|g*|, being g* the reciprocal space vector with components hkl

    Parameters
    ----------
    cell_filepath : string, the directory to the .cif file of the phase
        DESCRIPTION.
    plane_indices : np array, h,k,l format
        DESCRIPTION.

    Returns
    -------
    interplanar_distance : TYPE
        DESCRIPTION.

    '''
    
    unit_cell = read(cell_filepath)
    # show atoms if wanted
    # show_atoms(unit_cell, plane='xy')
    
    real_lattice = unit_cell.get_cell().round(10)
    
    # get unit cell vectors
    a_1_p = real_lattice[0]
    a_2_p = real_lattice[1]
    a_3_p = real_lattice[2]
    
    # get reciprocal space vectors
    Vol = np.dot(a_1_p, np.cross(a_2_p, a_3_p))
    
    b_1 = np.cross(a_2_p, a_3_p)/Vol
    b_2 = np.cross(a_3_p, a_1_p)/Vol
    b_3 = np.cross(a_1_p, a_2_p)/Vol
    
    reciprocal_lattice = np.array([b_1, b_2, b_3])
    
    dot_rec_plane = np.dot(plane_indices, reciprocal_lattice)
    
    interplanar_distance = 1/np.sqrt((dot_rec_plane[0])**2 + (dot_rec_plane[1])**2 + (dot_rec_plane[2])**2)
    
    # also computable this way
    # get cell params to compute the metric tensors
    # a, b, c, alfa, beta, gamma = unit_cell.cell.cellpar()

    # direct_metric_tensor = np.array([[a**2, a*b*np.cos((np.pi/180)*(gamma)), a*c*np.cos((np.pi/180)*(beta))],
    #                                  [b*a*np.cos((np.pi/180)*(gamma)), b**2, b*c*np.cos((np.pi/180)*(alfa))],
    #                                  [c*a*np.cos((np.pi/180)*(beta)), c*b*np.cos((np.pi/180)*(alfa)), c**2]])
    
    # reciprocal_metric_tensor = np.linalg.inv(direct_metric_tensor)
    

    # interplanar_distance2 = 1/np.sqrt(np.dot(plane_indices, np.dot(reciprocal_metric_tensor,plane_indices )))
    
    
    return interplanar_distance



def Zone_Equation_Checker(
        zone_axis, plane):
    
    '''
    Check if the given plane, the miller indices as a 3 elements list 
    is within the zone defined by the zone axis indicated by a 3 elements list
    Checks if h*u + k*v + l*w = 0 (in which case it is in zone)
    
    '''

    dot_prod = np.dot(zone_axis, plane)

    if dot_prod == 0:
        in_zone = True
    else:
        in_zone = False
    
    return in_zone


def Find_integer_plane(
        plane_indices, tolerance_diff = 0.5):
    
    '''
    plane_indices not integers, but floats and we want to convert
    them in the smalles possible integer defined plane
    '''

    plane_indices_int = np.round(plane_indices)
    
    diff = np.abs(plane_indices - plane_indices_int)
    
    if np.sum(diff) > tolerance_diff:
        return Find_integer_plane(
            2*np.array(plane_indices), tolerance_diff = tolerance_diff)
    else:
        gcd_1 = math.gcd(int(plane_indices_int[0]), int(plane_indices_int[1]))
        gcd_2 = math.gcd(int(plane_indices_int[1]), int(plane_indices_int[2]))
        gcd_3 = math.gcd(int(plane_indices_int[0]), int(plane_indices_int[2]))
        
        gcds = np.array([gcd_1, gcd_2, gcd_3])
                
        gcd = np.min(gcds[gcds != 0])
        
        plane_indices_int = plane_indices_int/gcd
        
        return [int(plane_indices_int[0]), int(plane_indices_int[1]), int(plane_indices_int[2])]


                
def Adjust_in_surface_plane_rotation(
        cell_filepath, scored_spot_pair, suface_basis_choice = 'plane'):
    
    
    
    # load the cif file of the unit cell
    unit_cell = read(cell_filepath)
    # show atoms if wanted
    # show_atoms(unit_cell, plane='xy')
    
    real_lattice = unit_cell.get_cell().round(10)
    
    # get unit cell vectors
    a_1_p = real_lattice[0]
    a_2_p = real_lattice[1]
    a_3_p = real_lattice[2]
    
    # get reciprocal space vectors
    Vol = np.dot(a_1_p, np.cross(a_2_p, a_3_p))
    
    b_1 = np.cross(a_2_p, a_3_p)/Vol
    b_2 = np.cross(a_3_p, a_1_p)/Vol
    b_3 = np.cross(a_1_p, a_2_p)/Vol
    
    reciprocal_lattice = np.array([b_1, b_2, b_3])
    
    # get cell params to compute the metric tensors
    a, b, c, alfa, beta, gamma = unit_cell.cell.cellpar()
     
    direct_metric_tensor = np.array([[a**2, a*b*np.cos((np.pi/180)*(gamma)), a*c*np.cos((np.pi/180)*(beta))],
                                     [b*a*np.cos((np.pi/180)*(gamma)), b**2, b*c*np.cos((np.pi/180)*(alfa))],
                                     [c*a*np.cos((np.pi/180)*(beta)), c*b*np.cos((np.pi/180)*(alfa)), c**2]])
    
    reciprocal_metric_tensor = np.linalg.inv(direct_metric_tensor)
    
    
    # to create a surface we need to input the plane not the direction
    # we convert the direction to the perpendicular plane
    # or equivalently paralel g vector

    zone_axis = scored_spot_pair.ZA

    plane_of_zone_axis = Find_plane_paralel_to_direction(
        zone_axis, direct_metric_tensor, tolerance_diff = 0.5)

    # choose whether to use the plane or the direction, it should be the 
    # plane always, but just to have the option to choose        
    indices_for_surface = Use_plane_or_direction(
        zone_axis, plane_of_zone_axis, choice = suface_basis_choice)

    c_basis, a_surf_vects = get_surface_basis(
        unit_cell, indices_for_surface, 1, vacuum=None, 
        tol=1e-10, periodic=False)

    (a1, a2, a3) = a_surf_vects 
    
    
    # index (plane) paralel to a1   
    # plane_to_0_int = find_g_vector_plane(
    #     real_lattice, direct_metric_tensor, a1, tolerance_diff = 0.5)

    # find vector that is 90 degrees but in surface
    # from a1, as non perfect ZA the a1 and a2 will not be 90º

    new_a2_90_deg_from_a1 = np.cross(a3,a1)

    # get the found planes from the scored spot pair
  
    plane_1_found = np.asarray(scored_spot_pair.hkl1_reference)
    plane_2_found = np.asarray(scored_spot_pair.hkl2_reference)

    
    angle_exp_plane_1_to_x = scored_spot_pair.spot1_angle_to_x
    angle_exp_plane_2_to_x = scored_spot_pair.spot2_angle_to_x

    # ensure that the plane references are planes within the ZA found
    # for that check if they fulfill zone equation
    
    # keep track of the planes that are or not in zone to treat them differently
    in_zone_status = []
    
    for plane_found_i in [plane_1_found, plane_2_found]:
        
        in_zone = Zone_Equation_Checker(
            zone_axis, plane_found_i)
        
        in_zone_status.append(in_zone)
        
        
    # for thsi we need to evaluate all the planes that are equivalent in that crystal system
    # all the possible permutations 
    # check the permutations and evluate all the 
    
    
    # work out different situations whether the planes are in zone or not
    if in_zone_status[0] == True and in_zone_status[1]== True:
        # if both planes are in zone, make sure that the angle between them in
        # theory is the one obtained with # scored_spot_pair.angle_between
        angle_between_theory = angle_between_planes(
            plane_1_found, plane_2_found, reciprocal_metric_tensor)
        if angle_between_theory + angle_between_theory*0.1 >= scored_spot_pair.angle_between and angle_between_theory - angle_between_theory*0.1 <= scored_spot_pair.angle_between:
            # the original permutations can be used to compute the in plane rotation
            
            planes_for_inplane_rot = [plane_1_found, plane_2_found]
            
        else:
            # the angle is not fitting, then keep the first plane permutation and search
            # for another permutation for the second plane that meets the conditions
            list_permutations = PhaseIdent.Crystal_System_Equidistance_Permutations(
                unit_cell.info['spacegroup'].no , plane_2_found)
            list_permutations_copy = list_permutations.copy() 
            # only keep the permutations that are in zone with the given zone axis
            for permutation in list_permutations_copy:
                in_zone = Zone_Equation_Checker(
                    zone_axis, permutation)
                
                if in_zone == False:
                    list_permutations.remove(permutation)
            
            # if there was no permutation in zone, which should not happen as
            # the distances would be different (but for instance if the distances
            # are so similar between non equivalent permutations), just try
            # to find a permutation in zone that allows to keep the calculation
            if len(list_permutations) == 0:
                # force all the permutations possible as if it was a cubic system
                list_permutations = PhaseIdent.Crystal_System_Equidistance_Permutations(
                    227 , plane_2_found)
                
                list_permutations_copy = list_permutations.copy() 
                # only keep the permutations that are in zone with the given zone axis
                for permutation in list_permutations_copy:
                    in_zone = Zone_Equation_Checker(
                        zone_axis, permutation)
                    
                    if in_zone == False:
                        list_permutations.remove(permutation)
            
            if len(list_permutations) == 0:
                raise(ValueError(' No permutation found is in zone'))

            
            # find the permutations whose angle is closest to the observed experimentally
            list_angles_between_theory = []
            
            for permutation in list_permutations:
                angle_between_theory = angle_between_planes(
                    plane_1_found, permutation, reciprocal_metric_tensor)
                
                list_angles_between_theory.append(angle_between_theory)
                
            list_angles_between_theory = np.asarray(list_angles_between_theory)
            new_plane_2_found = list_permutations[np.argmin(np.abs(list_angles_between_theory - scored_spot_pair.angle_between))]
            
            planes_for_inplane_rot = [plane_1_found, new_plane_2_found]
            
    
    # check if only one of the two planes is not in zone
    
    elif in_zone_status[0] == True and in_zone_status[1]== False:
        
        list_permutations = PhaseIdent.Crystal_System_Equidistance_Permutations(
            unit_cell.info['spacegroup'].no , plane_2_found)
        
        list_permutations_copy = list_permutations.copy() 
        # only keep the permutations that are in zone with the given zone axis
        for permutation in list_permutations_copy:
            in_zone = Zone_Equation_Checker(
                zone_axis, permutation)
            
            if in_zone == False:
                list_permutations.remove(permutation)
                
        # if there was no permutation in zone, which should not happen as
        # the distances would be different (but for instance if the distances
        # are so similar between non equivalent permutations), just try
        # to find a permutation in zone that allows to keep the calculation
        if len(list_permutations) == 0:
            # force all the permutations possible as if it was a cubic system
            list_permutations = PhaseIdent.Crystal_System_Equidistance_Permutations(
                227 , plane_2_found)
            
            list_permutations_copy = list_permutations.copy() 
            # only keep the permutations that are in zone with the given zone axis
            for permutation in list_permutations_copy:
                in_zone = Zone_Equation_Checker(
                    zone_axis, permutation)
                
                if in_zone == False:
                    list_permutations.remove(permutation)
    
        if len(list_permutations) == 0:
            raise(ValueError(' No permutation found is in zone'))
 
        # find the permutations whose angle is closest to the observed experimentally
        list_angles_between_theory = []
        
        for permutation in list_permutations:
            angle_between_theory = angle_between_planes(
                plane_1_found, permutation, reciprocal_metric_tensor)
            
            list_angles_between_theory.append(angle_between_theory)
            
        list_angles_between_theory = np.asarray(list_angles_between_theory)
        
        new_plane_2_found = list_permutations[np.argmin(np.abs(list_angles_between_theory - scored_spot_pair.angle_between))]
        
        planes_for_inplane_rot = [plane_1_found, new_plane_2_found]
    
    
    elif in_zone_status[0] == False and in_zone_status[1]== True:
        
        
        list_permutations = PhaseIdent.Crystal_System_Equidistance_Permutations(
            unit_cell.info['spacegroup'].no , plane_1_found)
        
        list_permutations_copy = list_permutations.copy() 
        # only keep the permutations that are in zone with the given zone axis
        for permutation in list_permutations_copy:
            in_zone = Zone_Equation_Checker(
                zone_axis, permutation)
            
            if in_zone == False:
                list_permutations.remove(permutation)
        
        # if there was no permutation in zone, which should not happen as
        # the distances would be different (but for instance if the distances
        # are so similar between non equivalent permutations), just try
        # to find a permutation in zone that allows to keep the calculation
        if len(list_permutations) == 0:
            # force all the permutations possible as if it was a cubic system
            list_permutations = PhaseIdent.Crystal_System_Equidistance_Permutations(
                227 , plane_1_found)
            
            list_permutations_copy = list_permutations.copy() 
            # only keep the permutations that are in zone with the given zone axis
            for permutation in list_permutations_copy:
                in_zone = Zone_Equation_Checker(
                    zone_axis, permutation)
                
                if in_zone == False:
                    list_permutations.remove(permutation)

        
        if len(list_permutations) == 0:
            raise(ValueError(' No permutation found is in zone'))

        
        # find the permutations whose angle is closest to the observed experimentally
        list_angles_between_theory = []
        
        for permutation in list_permutations:
            angle_between_theory = angle_between_planes(
                permutation, plane_2_found, reciprocal_metric_tensor)
                    
            list_angles_between_theory.append(angle_between_theory)
            
        list_angles_between_theory = np.asarray(list_angles_between_theory)
        
        new_plane_1_found = list_permutations[np.argmin(np.abs(list_angles_between_theory - scored_spot_pair.angle_between))]
        
        planes_for_inplane_rot = [new_plane_1_found, plane_2_found]
    
    
    
    
    # check if both planes are out of zone axis
    else:
    # if in_zone_status[0] == False and in_zone_status[1]== False:
        
        list_permutations_1 = PhaseIdent.Crystal_System_Equidistance_Permutations(
            unit_cell.info['spacegroup'].no , plane_1_found)
        
        list_permutations_1_copy = list_permutations_1.copy() 
        # only keep the permutations that are in zone with the given zone axis
        for permutation in list_permutations_1_copy:
            in_zone = Zone_Equation_Checker(
                zone_axis, permutation)
            
            if in_zone == False:
                list_permutations_1.remove(permutation)
        
        # if there was no permutation in zone, which should not happen as
        # the distances would be different (but for instance if the distances
        # are so similar between non equivalent permutations), just try
        # to find a permutation in zone that allows to keep the calculation
        if len(list_permutations_1) == 0:
            # force all the permutations possible as if it was a cubic system
            list_permutations_1 = PhaseIdent.Crystal_System_Equidistance_Permutations(
                227 , plane_1_found)
            
            list_permutations_1_copy = list_permutations_1.copy() 
            # only keep the permutations that are in zone with the given zone axis
            for permutation in list_permutations_1_copy:
                in_zone = Zone_Equation_Checker(
                    zone_axis, permutation)
                
                if in_zone == False:
                    list_permutations_1.remove(permutation)
        
        
        list_permutations_2 = PhaseIdent.Crystal_System_Equidistance_Permutations(
            unit_cell.info['spacegroup'].no , plane_2_found)
        
        list_permutations_2_copy = list_permutations_2.copy() 
        # only keep the permutations that are in zone with the given zone axis
        for permutation in list_permutations_2_copy:
            in_zone = Zone_Equation_Checker(
                zone_axis, permutation)
            
            if in_zone == False:
                list_permutations_2.remove(permutation)
                
        # if there was no permutation in zone, which should not happen as
        # the distances would be different (but for instance if the distances
        # are so similar between non equivalent permutations), just try
        # to find a permutation in zone that allows to keep the calculation
        if len(list_permutations_2) == 0:
            # force all the permutations possible as if it was a cubic system
            list_permutations_2 = PhaseIdent.Crystal_System_Equidistance_Permutations(
                227 , plane_2_found)
            
            list_permutations_2_copy = list_permutations_2.copy() 
            # only keep the permutations that are in zone with the given zone axis
            for permutation in list_permutations_2_copy:
                in_zone = Zone_Equation_Checker(
                    zone_axis, permutation)
                
                if in_zone == False:
                    list_permutations_2.remove(permutation)

                
        # check all the possible combinations between the possible permutations
        # for both planes
        
        # if permut 1 or permut 2 are empty, then raise error 
        # as there is no plane or similar one corresponding to the definition
        if len(list_permutations_1) == 0 or len(list_permutations_2) == 0:
            raise(ValueError(' No permutation found is in zone'))
        
        list_permutation_combs = []
        list_angles_between_theor = []
        for permutation_1 in list_permutations_1:
            
            for permutation_2 in list_permutations_2:
                
                angle_between_theory = angle_between_planes(
                    permutation_1, permutation_2, reciprocal_metric_tensor)
                
                list_permutation_combs.append([permutation_1, permutation_2])
                list_angles_between_theor.append(angle_between_theory)
        
        planes_for_inplane_rot = list_permutation_combs[np.argmin(np.abs(np.asarray(list_angles_between_theor-scored_spot_pair.angle_between)))]                      
    
    # initial assignation of planes for checking multiplicity
    plane_1_found, plane_2_found = planes_for_inplane_rot
    print('plane 1 found')
    print(plane_1_found)
    print('plane 2 found')
    print(plane_2_found)

    
    # !!! first check if the multiplìcity of both planes found is the same or not

    list_permutations_plane_1_found = PhaseIdent.Crystal_System_Equidistance_Permutations(
        unit_cell.info['spacegroup'].no , plane_1_found)
    
    list_permutations_plane_1_found_copy = list_permutations_plane_1_found.copy() 
    # only keep the permutations that are in zone with the given zone axis
    for permutation in list_permutations_plane_1_found_copy:
        in_zone = Zone_Equation_Checker(
            zone_axis, permutation)
        
        if in_zone == False:
            list_permutations_plane_1_found.remove(permutation)

    list_permutations_plane_2_found = PhaseIdent.Crystal_System_Equidistance_Permutations(
        unit_cell.info['spacegroup'].no , plane_2_found)

    list_permutations_plane_2_found_copy = list_permutations_plane_2_found.copy() 
    # only keep the permutations that are in zone with the given zone axis
    for permutation in list_permutations_plane_2_found_copy:
        in_zone = Zone_Equation_Checker(
            zone_axis, permutation)
        
        if in_zone == False:
            list_permutations_plane_2_found.remove(permutation)

    inplane_multipl_plane_1 = len(list_permutations_plane_1_found)
    inplane_multipl_plane_2 = len(list_permutations_plane_2_found)

    inplane_multipl = np.array([inplane_multipl_plane_1, inplane_multipl_plane_2])
    

    # if the multiplicity is the same then check the rotation angle of 
    # assigning the plane 1 to hkl1 and plane 2 to hkl2, and viceversa, 
    # and then keep the one that makes more similar rotation angles
    # in the final computation of the rotation angle to apply
    
    if inplane_multipl_plane_1 == inplane_multipl_plane_2:
        
        # assignation 1
        print('1st assignation')
        plane_1_found, plane_2_found = planes_for_inplane_rot
        print('plane 1 found')
        print(plane_1_found)
        print('plane 2 found')
        print(plane_2_found)

        # turn the planes into g vectors to find the angles between the plane and
        # the vector pointing to the a1 vector
    
        g_vector_plane_1 = np.array([plane_1_found[0]*b_1, plane_1_found[1]*b_2, plane_1_found[2]*b_3])
        g_vector_plane_1 = np.sum(g_vector_plane_1, axis = 0)
    
    
        angle_theo_from_plane_1_g_to_x = angle_between_cartes_vectors(g_vector_plane_1, a1)
    
        g_vector_plane_2 = np.array([plane_2_found[0]*b_1, plane_2_found[1]*b_2, plane_2_found[2]*b_3])
        g_vector_plane_2 = np.sum(g_vector_plane_2, axis = 0)
    
        angle_theo_from_plane_2_g_to_x = angle_between_cartes_vectors(g_vector_plane_2, a1)
    
        #!!! these theo angles should be checked whether they are positive or negative
        
        
        angle_from_new_a2_g_to_hkl_g_1 = angle_between_cartes_vectors(new_a2_90_deg_from_a1, g_vector_plane_1)
        angle_from_new_a2_g_to_hkl_g_2 = angle_between_cartes_vectors(new_a2_90_deg_from_a1, g_vector_plane_2)
        
        print(np.dot(new_a2_90_deg_from_a1,g_vector_plane_2)/(norm(new_a2_90_deg_from_a1)*norm(g_vector_plane_2)))
        print('angle_from_new_a2_g_to_hkl_g_1')
        print(angle_from_new_a2_g_to_hkl_g_1)
        print('angle_from_new_a2_g_to_hkl_g_2')
        print(angle_from_new_a2_g_to_hkl_g_2)
    
        # from a2 that is perfectly perpendicular to a1, just seeing if the angle 
        # from a2 to g hkl is larger than 90 puts it into the negative half 
        if angle_from_new_a2_g_to_hkl_g_1 > 90:
            angle_theo_from_plane_1_g_to_x = -angle_theo_from_plane_1_g_to_x
        if angle_from_new_a2_g_to_hkl_g_2 > 90:
            angle_theo_from_plane_2_g_to_x = -angle_theo_from_plane_2_g_to_x
            
            
        print('angle_theo_from_plane_1_g_to_x')
        print(angle_theo_from_plane_1_g_to_x)
    
        print('angle_theo_from_plane_2_g_to_x')
        print(angle_theo_from_plane_2_g_to_x)
    
        
        angles_exp = [angle_exp_plane_1_to_x, angle_exp_plane_2_to_x]
        angles_theo_1 = [angle_theo_from_plane_1_g_to_x, angle_theo_from_plane_2_g_to_x]


        # !!! IN-PLANE ROTATION ANGLEthis is the rotation we must rotate the surface 
        # object to align the inplane rotation we want to orient 
        # in plane as observed experimentally


        # if multiplicity is the same for both planes, then swap plane 1 with plane 2:
        #     and its g vectors, and then check for the angle of rotation found
        #     get the pair and angle for which both angles are more similar between them
            
        final_in_surf_plane_rotation_1_1 = angles_exp[0] - angles_theo_1[0]
        final_in_surf_plane_rotation_1_2 = angles_exp[1] - angles_theo_1[1]
        
        # the rotation should be the same independently of the planes used to compute it
        # compute how similar these two angles are
        assign_1st_angles_difference = final_in_surf_plane_rotation_1_1 - final_in_surf_plane_rotation_1_2


        # assignation 2
        print('2nd assignation')
        # swap planes with respect the previous assignation
        plane_2_found, plane_1_found = planes_for_inplane_rot
        print('plane 1 found')
        print(plane_1_found)
        print('plane 2 found')
        print(plane_2_found)

    
        # turn the planes into g vectors to find the angles between the plane and
        # the vector pointing to the a1 vector
    
        g_vector_plane_1 = np.array([plane_1_found[0]*b_1, plane_1_found[1]*b_2, plane_1_found[2]*b_3])
        g_vector_plane_1 = np.sum(g_vector_plane_1, axis = 0)
    
    
        angle_theo_from_plane_1_g_to_x = angle_between_cartes_vectors(g_vector_plane_1, a1)
    
        g_vector_plane_2 = np.array([plane_2_found[0]*b_1, plane_2_found[1]*b_2, plane_2_found[2]*b_3])
        g_vector_plane_2 = np.sum(g_vector_plane_2, axis = 0)
    
        angle_theo_from_plane_2_g_to_x = angle_between_cartes_vectors(g_vector_plane_2, a1)
    
        #!!! these theo angles should be checked whether they are positive or negative
        
        
        angle_from_new_a2_g_to_hkl_g_1 = angle_between_cartes_vectors(new_a2_90_deg_from_a1, g_vector_plane_1)
        angle_from_new_a2_g_to_hkl_g_2 = angle_between_cartes_vectors(new_a2_90_deg_from_a1, g_vector_plane_2)
        
        print(np.dot(new_a2_90_deg_from_a1,g_vector_plane_2)/(norm(new_a2_90_deg_from_a1)*norm(g_vector_plane_2)))
        print('angle_from_new_a2_g_to_hkl_g_1')
        print(angle_from_new_a2_g_to_hkl_g_1)
        print('angle_from_new_a2_g_to_hkl_g_2')
        print(angle_from_new_a2_g_to_hkl_g_2)
    
        # from a2 that is perfectly perpendicular to a1, just seeing if the angle 
        # from a2 to g hkl is larger than 90 puts it into the negative half 
        if angle_from_new_a2_g_to_hkl_g_1 > 90:
            angle_theo_from_plane_1_g_to_x = -angle_theo_from_plane_1_g_to_x
        if angle_from_new_a2_g_to_hkl_g_2 > 90:
            angle_theo_from_plane_2_g_to_x = -angle_theo_from_plane_2_g_to_x
            
            
        print('angle_theo_from_plane_1_g_to_x')
        print(angle_theo_from_plane_1_g_to_x)
    
        print('angle_theo_from_plane_2_g_to_x')
        print(angle_theo_from_plane_2_g_to_x)
    
        
        angles_exp = [angle_exp_plane_1_to_x, angle_exp_plane_2_to_x]
        angles_theo_2 = [angle_theo_from_plane_1_g_to_x, angle_theo_from_plane_2_g_to_x]


        # !!! IN-PLANE ROTATION ANGLEthis is the rotation we must rotate the surface 
        # object to align the inplane rotation we want to orient 
        # in plane as observed experimentally


        # if multiplicity is the same for both planes, then swap plane 1 with plane 2:
        #     and its g vectors, and then check for the angle of rotation found
        #     get the pair and angle for which both angles are more similar between them
            
        final_in_surf_plane_rotation_2_1 = angles_exp[0] - angles_theo_2[0]
        final_in_surf_plane_rotation_2_2 = angles_exp[1] - angles_theo_2[1]
        
        # the rotation should be the same independently of the planes used to compute it
        # compute how similar these two angles are
        assign_2nd_angles_difference = final_in_surf_plane_rotation_2_1 - final_in_surf_plane_rotation_2_2

        
        min_angle_diff_index = np.argmin(np.abs(np.array([assign_1st_angles_difference, assign_2nd_angles_difference])))

        # if the minimum difference is from the first assignation
        if min_angle_diff_index == 0:
            final_in_surf_plane_rotation = final_in_surf_plane_rotation_1_1
        # if the minimum difference is from the second assignation
        else:
            final_in_surf_plane_rotation = final_in_surf_plane_rotation_2_1
    
    
    # if the in-zone multiplicity is different, then get the angle from
    # the lowest multiplicity 
    else:        
            
        # make the same assignation as the original assignation for checking multiplicity
        # as multiplicity values are stored given this order of assignation
        plane_1_found, plane_2_found = planes_for_inplane_rot
        
    
        # turn the planes into g vectors to find the angles between the plane and
        # the vector pointing to the a1 vector
    
        g_vector_plane_1 = np.array([plane_1_found[0]*b_1, plane_1_found[1]*b_2, plane_1_found[2]*b_3])
        g_vector_plane_1 = np.sum(g_vector_plane_1, axis = 0)
    
    
        angle_theo_from_plane_1_g_to_x = angle_between_cartes_vectors(g_vector_plane_1, a1)
    
        g_vector_plane_2 = np.array([plane_2_found[0]*b_1, plane_2_found[1]*b_2, plane_2_found[2]*b_3])
        g_vector_plane_2 = np.sum(g_vector_plane_2, axis = 0)
    
        angle_theo_from_plane_2_g_to_x = angle_between_cartes_vectors(g_vector_plane_2, a1)
    
        #!!! these theo angles should be checked whether they are positive or negative
        
        
        angle_from_new_a2_g_to_hkl_g_1 = angle_between_cartes_vectors(new_a2_90_deg_from_a1, g_vector_plane_1)
        angle_from_new_a2_g_to_hkl_g_2 = angle_between_cartes_vectors(new_a2_90_deg_from_a1, g_vector_plane_2)
        
        print(np.dot(new_a2_90_deg_from_a1,g_vector_plane_2)/(norm(new_a2_90_deg_from_a1)*norm(g_vector_plane_2)))
        print('angle_from_new_a2_g_to_hkl_g_1')
        print(angle_from_new_a2_g_to_hkl_g_1)
        print('angle_from_new_a2_g_to_hkl_g_2')
        print(angle_from_new_a2_g_to_hkl_g_2)
    
        # from a2 that is perfectly perpendicular to a1, just seeing if the angle 
        # from a2 to g hkl is larger than 90 puts it into the negative half 
        if angle_from_new_a2_g_to_hkl_g_1 > 90:
            angle_theo_from_plane_1_g_to_x = -angle_theo_from_plane_1_g_to_x
        if angle_from_new_a2_g_to_hkl_g_2 > 90:
            angle_theo_from_plane_2_g_to_x = -angle_theo_from_plane_2_g_to_x
        
        angles_exp = [angle_exp_plane_1_to_x, angle_exp_plane_2_to_x]
        angles_theo = [angle_theo_from_plane_1_g_to_x, angle_theo_from_plane_2_g_to_x]
    
        # !!! IN-PLANE ROTATION ANGLEthis is the rotation we must rotate the surface 
        # object to align the inplane rotation we want to orient 
        # in plane as observed experimentally
    
        final_in_surf_plane_rotation = angles_exp[np.argmin(inplane_multipl)] - angles_theo[np.argmin(inplane_multipl)]
                
        
    # this cannot distinguish polarity...to do so we would need maybe to 
    # porofile trhough the dumbells and distinguish bigger and smaller atom


    return final_in_surf_plane_rotation



def Find_plane_pointing_to_final_cartesian_x_axis(
        cell_filepath, scored_spot_pair, 
        tolerance_diff = 0.5, suface_basis_choice = 'plane'):
    
    
    
    # load the cif file of the unit cell
    unit_cell = read(cell_filepath)
    # show atoms if wanted
    # show_atoms(unit_cell, plane='xy')
    
    real_lattice = unit_cell.get_cell().round(10)
    
    # get unit cell vectors
    a_1_p = real_lattice[0]
    a_2_p = real_lattice[1]
    a_3_p = real_lattice[2]
    
    # get reciprocal space vectors
    Vol = np.dot(a_1_p, np.cross(a_2_p, a_3_p))
    
    b_1 = np.cross(a_2_p, a_3_p)/Vol
    b_2 = np.cross(a_3_p, a_1_p)/Vol
    b_3 = np.cross(a_1_p, a_2_p)/Vol
    
    reciprocal_lattice = np.array([b_1, b_2, b_3])
    
    # get cell params to compute the metric tensors
    a, b, c, alfa, beta, gamma = unit_cell.cell.cellpar()
     
    direct_metric_tensor = np.array([[a**2, a*b*np.cos((np.pi/180)*(gamma)), a*c*np.cos((np.pi/180)*(beta))],
                                     [b*a*np.cos((np.pi/180)*(gamma)), b**2, b*c*np.cos((np.pi/180)*(alfa))],
                                     [c*a*np.cos((np.pi/180)*(beta)), c*b*np.cos((np.pi/180)*(alfa)), c**2]])
    
    reciprocal_metric_tensor = np.linalg.inv(direct_metric_tensor)
    
    
    # to create a surface we need to input the plane not the direction
    # we convert the direction to the perpendicular plane
    # or equivalently paralel g vector

    zone_axis = scored_spot_pair.ZA
    [u,v,w] = zone_axis
    
    plane_of_zone_axis = Find_plane_paralel_to_direction(
        zone_axis, direct_metric_tensor, tolerance_diff = 0.5)

    # choose whether to use the plane or the direction, it should be the 
    # plane always, but just to have the option to choose        
    indices_for_surface = Use_plane_or_direction(
        zone_axis, plane_of_zone_axis, choice = suface_basis_choice)

    c_basis, a_surf_vects = get_surface_basis(
        unit_cell, indices_for_surface, 1, vacuum=None, 
        tol=1e-10, periodic=False)

    (a1, a2, a3) = a_surf_vects 
    
    
    # index (plane) paralel to a1   
    # plane_to_0_int = find_g_vector_plane(
    #     real_lattice, direct_metric_tensor, a1, tolerance_diff = 0.5)

    # find vector that is 90 degrees but in surface
    # from a1, as non perfect ZA the a1 and a2 will not be 90º

    new_a2_90_deg_from_a1 = np.cross(a3,a1)

    # get the found planes from the scored spot pair
  
    plane_1_found = scored_spot_pair.hkl1_reference
    plane_2_found = scored_spot_pair.hkl2_reference

    angle_exp_plane_1_to_x = scored_spot_pair.spot1_angle_to_x
    angle_exp_plane_2_to_x = scored_spot_pair.spot2_angle_to_x
        
    # ensure that the plane references are planes within the ZA found
    # for that check if they fulfill zone equation
    
    # keep track of the planes that are or not in zone to treat them differently
    in_zone_status = []
    
    for plane_found_i in [plane_1_found, plane_2_found]:
        
        in_zone = Zone_Equation_Checker(
            zone_axis, plane_found_i)
        
        in_zone_status.append(in_zone)
        
        
    # for thsi we need to evaluate all the planes that are equivalent in that crystal system
    # all the possible permutations 
    # check the permutations and evluate all the 
    
    
    # work out different situations whether the planes are in zone or not
    if in_zone_status[0] == True and in_zone_status[1]== True:
        # if both planes are in zone, make sure that the angle between them in
        # theory is the one obtained with # scored_spot_pair.angle_between
        
        angle_between_theory = angle_between_planes(
            plane_1_found, plane_2_found, reciprocal_metric_tensor)
        
        if angle_between_theory + angle_between_theory*0.1 >= scored_spot_pair.angle_between and angle_between_theory - angle_between_theory*0.1 <= scored_spot_pair.angle_between:
            
            # the original permutations can be used to compute the  in plane rotation
            
            planes_for_inplane_rot = [plane_1_found, plane_2_found]
            
        else:
            # the angle is not fitting, then keep the first plane permutation and search
            # for another permutation for the second plane that meets the conditions
            list_permutations = PhaseIdent.Crystal_System_Equidistance_Permutations(
                unit_cell.info['spacegroup'].no , plane_2_found)
            
            list_permutations_copy = list_permutations.copy() 
            # only keep the permutations that are in zone with the given zone axis
            for permutation in list_permutations_copy:
                in_zone = Zone_Equation_Checker(
                    zone_axis, permutation)
                
                if in_zone == False:
                    list_permutations.remove(permutation)
            
                
            # if there was no permutation in zone, which should not happen as
            # the distances would be different (but for instance if the distances
            # are so similar between non equivalent permutations), just try
            # to find a permutation in zone that allows to keep the calculation
            if len(list_permutations) == 0:
                # force all the permutations possible as if it was a cubic system
                list_permutations = PhaseIdent.Crystal_System_Equidistance_Permutations(
                    227 , plane_2_found)
                
                list_permutations_copy = list_permutations.copy() 
                # only keep the permutations that are in zone with the given zone axis
                for permutation in list_permutations_copy:
                    in_zone = Zone_Equation_Checker(
                        zone_axis, permutation)
                    
                    if in_zone == False:
                        list_permutations.remove(permutation)
            
            if len(list_permutations) == 0:
                raise(ValueError(' No permutation found is in zone'))

            
            # find the permutations whose angle is closest to the observed experimentally
            list_angles_between_theory = []
            
            for permutation in list_permutations:
                angle_between_theory = angle_between_planes(
                    plane_1_found, permutation, reciprocal_metric_tensor)
                
                list_angles_between_theory.append(angle_between_theory)
                
            list_angles_between_theory = np.asarray(list_angles_between_theory)
            
            new_plane_2_found = list_permutations[np.argmin(np.abs(list_angles_between_theory - scored_spot_pair.angle_between))]
            
            planes_for_inplane_rot = [plane_1_found, new_plane_2_found]
            
    
    # check if only one of the two planes is not in zone
    
    elif in_zone_status[0] == True and in_zone_status[1]== False:
        
        
        list_permutations = PhaseIdent.Crystal_System_Equidistance_Permutations(
            unit_cell.info['spacegroup'].no , plane_2_found)
        
        list_permutations_copy = list_permutations.copy() 
        # only keep the permutations that are in zone with the given zone axis
        for permutation in list_permutations_copy:
            in_zone = Zone_Equation_Checker(
                zone_axis, permutation)
            
            if in_zone == False:
                list_permutations.remove(permutation)
         
        # if there was no permutation in zone, which should not happen as
        # the distances would be different (but for instance if the distances
        # are so similar between non equivalent permutations), just try
        # to find a permutation in zone that allows to keep the calculation
        if len(list_permutations) == 0:
            # force all the permutations possible as if it was a cubic system
            list_permutations = PhaseIdent.Crystal_System_Equidistance_Permutations(
                227 , plane_2_found)
            
            list_permutations_copy = list_permutations.copy() 
            # only keep the permutations that are in zone with the given zone axis
            for permutation in list_permutations_copy:
                in_zone = Zone_Equation_Checker(
                    zone_axis, permutation)
                
                if in_zone == False:
                    list_permutations.remove(permutation)
    
        if len(list_permutations) == 0:
            raise(ValueError(' No permutation found is in zone'))
 
        # find the permutations whose angle is closest to the observed experimentally
        list_angles_between_theory = []
        
        for permutation in list_permutations:
            angle_between_theory = angle_between_planes(
                plane_1_found, permutation, reciprocal_metric_tensor)
            
            list_angles_between_theory.append(angle_between_theory)
            
        list_angles_between_theory = np.asarray(list_angles_between_theory)
        
        new_plane_2_found = list_permutations[np.argmin(np.abs(list_angles_between_theory - scored_spot_pair.angle_between))]
        
        planes_for_inplane_rot = [plane_1_found, new_plane_2_found]
    
    
    elif in_zone_status[0] == False and in_zone_status[1]== True:
        
        
        list_permutations = PhaseIdent.Crystal_System_Equidistance_Permutations(
            unit_cell.info['spacegroup'].no , plane_1_found)
        
        list_permutations_copy = list_permutations.copy() 
        # only keep the permutations that are in zone with the given zone axis
        for permutation in list_permutations_copy:
            in_zone = Zone_Equation_Checker(
                zone_axis, permutation)
            
            if in_zone == False:
                list_permutations.remove(permutation)
        
        # if there was no permutation in zone, which should not happen as
        # the distances would be different (but for instance if the distances
        # are so similar between non equivalent permutations), just try
        # to find a permutation in zone that allows to keep the calculation
        if len(list_permutations) == 0:
            # force all the permutations possible as if it was a cubic system
            list_permutations = PhaseIdent.Crystal_System_Equidistance_Permutations(
                227 , plane_1_found)
            
            list_permutations_copy = list_permutations.copy() 
            # only keep the permutations that are in zone with the given zone axis
            for permutation in list_permutations_copy:
                in_zone = Zone_Equation_Checker(
                    zone_axis, permutation)
                
                if in_zone == False:
                    list_permutations.remove(permutation)

        if len(list_permutations) == 0:
            raise(ValueError(' No permutation found is in zone'))

        
        # find the permutations whose angle is closest to the observed experimentally
        list_angles_between_theory = []
        
        for permutation in list_permutations:
            angle_between_theory = angle_between_planes(
                permutation, plane_2_found, reciprocal_metric_tensor)
                    
            list_angles_between_theory.append(angle_between_theory)
            
        list_angles_between_theory = np.asarray(list_angles_between_theory)
        
        new_plane_1_found = list_permutations[np.argmin(np.abs(list_angles_between_theory - scored_spot_pair.angle_between))]
        
        planes_for_inplane_rot = [new_plane_1_found, plane_2_found]
    
    
    
    
    # check if both planes are out of zone axis
    else:
    # if in_zone_status[0] == False and in_zone_status[1]== False:
        
        list_permutations_1 = PhaseIdent.Crystal_System_Equidistance_Permutations(
            unit_cell.info['spacegroup'].no , plane_1_found)
        
        list_permutations_1_copy = list_permutations_1.copy() 
        # only keep the permutations that are in zone with the given zone axis
        for permutation in list_permutations_1_copy:
            in_zone = Zone_Equation_Checker(
                zone_axis, permutation)
            
            if in_zone == False:
                list_permutations_1.remove(permutation)
        
        # if there was no permutation in zone, which should not happen as
        # the distances would be different (but for instance if the distances
        # are so similar between non equivalent permutations), just try
        # to find a permutation in zone that allows to keep the calculation
        if len(list_permutations_1) == 0:
            # force all the permutations possible as if it was a cubic system
            list_permutations_1 = PhaseIdent.Crystal_System_Equidistance_Permutations(
                227 , plane_1_found)
            
            list_permutations_1_copy = list_permutations_1.copy() 
            # only keep the permutations that are in zone with the given zone axis
            for permutation in list_permutations_1_copy:
                in_zone = Zone_Equation_Checker(
                    zone_axis, permutation)
                
                if in_zone == False:
                    list_permutations_1.remove(permutation)
        
        
        list_permutations_2 = PhaseIdent.Crystal_System_Equidistance_Permutations(
            unit_cell.info['spacegroup'].no , plane_2_found)
        
        list_permutations_2_copy = list_permutations_2.copy() 
        # only keep the permutations that are in zone with the given zone axis
        for permutation in list_permutations_2_copy:
            in_zone = Zone_Equation_Checker(
                zone_axis, permutation)
            
            if in_zone == False:
                list_permutations_2.remove(permutation)
                
        # if there was no permutation in zone, which should not happen as
        # the distances would be different (but for instance if the distances
        # are so similar between non equivalent permutations), just try
        # to find a permutation in zone that allows to keep the calculation
        if len(list_permutations_2) == 0:
            # force all the permutations possible as if it was a cubic system
            list_permutations_2 = PhaseIdent.Crystal_System_Equidistance_Permutations(
                227 , plane_2_found)
            
            list_permutations_2_copy = list_permutations_2.copy() 
            # only keep the permutations that are in zone with the given zone axis
            for permutation in list_permutations_2_copy:
                in_zone = Zone_Equation_Checker(
                    zone_axis, permutation)
                
                if in_zone == False:
                    list_permutations_2.remove(permutation)

        # if permut 1 or permut 2 are empty, then raise error 
        # as there is no plane or similar one corresponding to the definition
        if len(list_permutations_1) == 0 or len(list_permutations_2) == 0:
            raise(ValueError(' No permutation found is in zone'))
            
        # check all the possible combinations between the possible permutations
        # for both planes
        list_permutation_combs = []
        list_angles_between_theor = []
        for permutation_1 in list_permutations_1:
            
            for permutation_2 in list_permutations_2:
                
                angle_between_theory = angle_between_planes(
                    permutation_1, permutation_2, reciprocal_metric_tensor)
                
                list_permutation_combs.append([permutation_1, permutation_2])
                list_angles_between_theor.append(angle_between_theory)
        
        planes_for_inplane_rot = list_permutation_combs[np.argmin(np.abs(np.asarray(list_angles_between_theor-scored_spot_pair.angle_between)))]                      
    
    
    # initial assignation of planes for checking multiplicity
    plane_1_found, plane_2_found = planes_for_inplane_rot
    
    # !!! first check if the multiplìcity of both planes found is the same or not

    list_permutations_plane_1_found = PhaseIdent.Crystal_System_Equidistance_Permutations(
        unit_cell.info['spacegroup'].no , plane_1_found)
    
    list_permutations_plane_1_found_copy = list_permutations_plane_1_found.copy() 
    # only keep the permutations that are in zone with the given zone axis
    for permutation in list_permutations_plane_1_found_copy:
        in_zone = Zone_Equation_Checker(
            zone_axis, permutation)
        
        if in_zone == False:
            list_permutations_plane_1_found.remove(permutation)

    list_permutations_plane_2_found = PhaseIdent.Crystal_System_Equidistance_Permutations(
        unit_cell.info['spacegroup'].no , plane_2_found)

    list_permutations_plane_2_found_copy = list_permutations_plane_2_found.copy() 
    # only keep the permutations that are in zone with the given zone axis
    for permutation in list_permutations_plane_2_found_copy:
        in_zone = Zone_Equation_Checker(
            zone_axis, permutation)
        
        if in_zone == False:
            list_permutations_plane_2_found.remove(permutation)

    inplane_multipl_plane_1 = len(list_permutations_plane_1_found)
    inplane_multipl_plane_2 = len(list_permutations_plane_2_found)

    inplane_multipl = np.array([inplane_multipl_plane_1, inplane_multipl_plane_2])
    

    # if the multiplicity is the same then check the rotation angle of 
    # assigning the plane 1 to hkl1 and plane 2 to hkl2, and viceversa, 
    # and then keep the one that makes more similar rotation angles
    # in the final computation of the rotation angle to apply

    
    if inplane_multipl_plane_1 == inplane_multipl_plane_2:
        
        
        # assignation 1
        plane_1_found, plane_2_found = planes_for_inplane_rot

    
        # turn the planes into g vectors to find the angles between the plane and
        # the vector pointing to the a1 vector
    
        g_vector_plane_1 = np.array([plane_1_found[0]*b_1, plane_1_found[1]*b_2, plane_1_found[2]*b_3])
        g_vector_plane_1 = np.sum(g_vector_plane_1, axis = 0)
    
    
        angle_theo_from_plane_1_g_to_x = angle_between_cartes_vectors(g_vector_plane_1, a1)
    
        g_vector_plane_2 = np.array([plane_2_found[0]*b_1, plane_2_found[1]*b_2, plane_2_found[2]*b_3])
        g_vector_plane_2 = np.sum(g_vector_plane_2, axis = 0)
    
        angle_theo_from_plane_2_g_to_x = angle_between_cartes_vectors(g_vector_plane_2, a1)
    
        #!!! these theo angles should be checked whether they are positive or negative
        
        
        angle_from_new_a2_g_to_hkl_g_1 = angle_between_cartes_vectors(new_a2_90_deg_from_a1, g_vector_plane_1)
        angle_from_new_a2_g_to_hkl_g_2 = angle_between_cartes_vectors(new_a2_90_deg_from_a1, g_vector_plane_2)
        
        # from a2 that is perfectly perpendicular to a1, just seeing if the angle 
        # from a2 to g hkl is larger than 90 puts it into the negative half 
        if angle_from_new_a2_g_to_hkl_g_1 > 90:
            angle_theo_from_plane_1_g_to_x = -angle_theo_from_plane_1_g_to_x
        if angle_from_new_a2_g_to_hkl_g_2 > 90:
            angle_theo_from_plane_2_g_to_x = -angle_theo_from_plane_2_g_to_x
            
        
        angles_exp = [angle_exp_plane_1_to_x, angle_exp_plane_2_to_x]
        angles_theo_1 = [angle_theo_from_plane_1_g_to_x, angle_theo_from_plane_2_g_to_x]


        # !!! IN-PLANE ROTATION ANGLEthis is the rotation we must rotate the surface 
        # object to align the inplane rotation we want to orient 
        # in plane as observed experimentally


        # if multiplicity is the same for both planes, then swap plane 1 with plane 2:
        #     and its g vectors, and then check for the angle of rotation found
        #     get the pair and angle for which both angles are more similar between them
            
        final_in_surf_plane_rotation_1_1 = angles_exp[0] - angles_theo_1[0]
        final_in_surf_plane_rotation_1_2 = angles_exp[1] - angles_theo_1[1]
        
        # the rotation should be the same independently of the planes used to compute it
        # compute how similar these two angles are
        assign_1st_angles_difference = final_in_surf_plane_rotation_1_1 - final_in_surf_plane_rotation_1_2


        # assignation 2
        # swap planes with respect the previous assignation
        plane_2_found, plane_1_found = planes_for_inplane_rot

        # turn the planes into g vectors to find the angles between the plane and
        # the vector pointing to the a1 vector
    
        g_vector_plane_1 = np.array([plane_1_found[0]*b_1, plane_1_found[1]*b_2, plane_1_found[2]*b_3])
        g_vector_plane_1 = np.sum(g_vector_plane_1, axis = 0)
    
    
        angle_theo_from_plane_1_g_to_x = angle_between_cartes_vectors(g_vector_plane_1, a1)
    
        g_vector_plane_2 = np.array([plane_2_found[0]*b_1, plane_2_found[1]*b_2, plane_2_found[2]*b_3])
        g_vector_plane_2 = np.sum(g_vector_plane_2, axis = 0)
    
        angle_theo_from_plane_2_g_to_x = angle_between_cartes_vectors(g_vector_plane_2, a1)
    
        #!!! these theo angles should be checked whether they are positive or negative
        
        
        angle_from_new_a2_g_to_hkl_g_1 = angle_between_cartes_vectors(new_a2_90_deg_from_a1, g_vector_plane_1)
        angle_from_new_a2_g_to_hkl_g_2 = angle_between_cartes_vectors(new_a2_90_deg_from_a1, g_vector_plane_2)
        
    
        # from a2 that is perfectly perpendicular to a1, just seeing if the angle 
        # from a2 to g hkl is larger than 90 puts it into the negative half 
        if angle_from_new_a2_g_to_hkl_g_1 > 90:
            angle_theo_from_plane_1_g_to_x = -angle_theo_from_plane_1_g_to_x
        if angle_from_new_a2_g_to_hkl_g_2 > 90:
            angle_theo_from_plane_2_g_to_x = -angle_theo_from_plane_2_g_to_x
            
    
        
        angles_exp = [angle_exp_plane_1_to_x, angle_exp_plane_2_to_x]
        angles_theo_2 = [angle_theo_from_plane_1_g_to_x, angle_theo_from_plane_2_g_to_x]


        # !!! IN-PLANE ROTATION ANGLEthis is the rotation we must rotate the surface 
        # object to align the inplane rotation we want to orient 
        # in plane as observed experimentally


        # if multiplicity is the same for both planes, then swap plane 1 with plane 2:
        #     and its g vectors, and then check for the angle of rotation found
        #     get the pair and angle for which both angles are more similar between them
            
        final_in_surf_plane_rotation_2_1 = angles_exp[0] - angles_theo_2[0]
        final_in_surf_plane_rotation_2_2 = angles_exp[1] - angles_theo_2[1]
        
        # the rotation should be the same independently of the planes used to compute it
        # compute how similar these two angles are
        assign_2nd_angles_difference = final_in_surf_plane_rotation_2_1 - final_in_surf_plane_rotation_2_2

        
        min_angle_diff_index = np.argmin(np.abs(np.array([assign_1st_angles_difference, assign_2nd_angles_difference])))

        # if the minimum difference is from the first assignation
        if min_angle_diff_index == 0:
            
            plane_found_use = plane_1_found
            angle_exp_use = angles_exp[0]
            angle_theo_use = angles_theo_1[0]
            
            
            final_in_surf_plane_rotation = final_in_surf_plane_rotation_1_1
        # if the minimum difference is from the second assignation
        else:
            # min_angle_diff_index == 1
            # plane_1_found now it has swaped with the 2nd it is not the same
            # as in the other condition
            plane_found_use = plane_1_found
            angle_exp_use = angles_exp[0]
            angle_theo_use = angles_theo_2[0]
            
            
            final_in_surf_plane_rotation = final_in_surf_plane_rotation_2_1
    
    
    # if the in-zone multiplicity is different, then get the angle from
    # the lowest multiplicity 
    else:        
            
        # make the same assignation as the original assignation for checking multiplicity
        # as multiplicity values are stored given this order of assignation
        plane_1_found, plane_2_found = planes_for_inplane_rot
        
    
        # turn the planes into g vectors to find the angles between the plane and
        # the vector pointing to the a1 vector
    
        g_vector_plane_1 = np.array([plane_1_found[0]*b_1, plane_1_found[1]*b_2, plane_1_found[2]*b_3])
        g_vector_plane_1 = np.sum(g_vector_plane_1, axis = 0)
    
    
        angle_theo_from_plane_1_g_to_x = angle_between_cartes_vectors(g_vector_plane_1, a1)
    
        g_vector_plane_2 = np.array([plane_2_found[0]*b_1, plane_2_found[1]*b_2, plane_2_found[2]*b_3])
        g_vector_plane_2 = np.sum(g_vector_plane_2, axis = 0)
    
        angle_theo_from_plane_2_g_to_x = angle_between_cartes_vectors(g_vector_plane_2, a1)
    
        #!!! these theo angles should be checked whether they are positive or negative
        
        
        angle_from_new_a2_g_to_hkl_g_1 = angle_between_cartes_vectors(new_a2_90_deg_from_a1, g_vector_plane_1)
        angle_from_new_a2_g_to_hkl_g_2 = angle_between_cartes_vectors(new_a2_90_deg_from_a1, g_vector_plane_2)
        
        print(np.dot(new_a2_90_deg_from_a1,g_vector_plane_2)/(norm(new_a2_90_deg_from_a1)*norm(g_vector_plane_2)))
        print('angle_from_new_a2_g_to_hkl_g_1')
        print(angle_from_new_a2_g_to_hkl_g_1)
        print('angle_from_new_a2_g_to_hkl_g_2')
        print(angle_from_new_a2_g_to_hkl_g_2)
    
        # from a2 that is perfectly perpendicular to a1, just seeing if the angle 
        # from a2 to g hkl is larger than 90 puts it into the negative half 
        if angle_from_new_a2_g_to_hkl_g_1 > 90:
            angle_theo_from_plane_1_g_to_x = -angle_theo_from_plane_1_g_to_x
        if angle_from_new_a2_g_to_hkl_g_2 > 90:
            angle_theo_from_plane_2_g_to_x = -angle_theo_from_plane_2_g_to_x
            
        print('angle_theo_from_plane_1_g_to_x')
        print(angle_theo_from_plane_1_g_to_x)
    
        print('angle_theo_from_plane_2_g_to_x')
        print(angle_theo_from_plane_2_g_to_x)
    
        angles_exp = [angle_exp_plane_1_to_x, angle_exp_plane_2_to_x]
        angles_theo = [angle_theo_from_plane_1_g_to_x, angle_theo_from_plane_2_g_to_x]
    
        # define the plane and angles that need to be used for obtaining the final hkl
        plane_found_use = planes_for_inplane_rot[np.argmin(inplane_multipl)]
        angle_exp_use = angles_exp[np.argmin(inplane_multipl)]
        angle_theo_use = angles_theo[np.argmin(inplane_multipl)]    
    
        # !!! IN-PLANE ROTATION ANGLEthis is the rotation we must rotate the surface 
        # object to align the inplane rotation we want to orient 
        # in plane as observed experimentally
    
        final_in_surf_plane_rotation = angles_exp[np.argmin(inplane_multipl)] - angles_theo[np.argmin(inplane_multipl)]
                

    print('plane_found_use')
    print(plane_found_use)
    print('final_in_surf_plane_rotation')
    print(final_in_surf_plane_rotation)
    

    # this cannot distinguish polarity...to do so we would need maybe to 
    # porofile trhough the dumbells and distinguish bigger and smaller atom

    
    # terms to define the solution of the plane pointing to the angle exp found
    
    # assign variables to reciprocal metric tensor components
    [[g0,g1,g2],[g3,g4,g5],[g6,g7,g8]] = reciprocal_metric_tensor

    # n times the norm of the new vector is bigger than the one from hkl1
    n = 10
    # iterate through different n with try and except if nans are generated
    for n in np.arange(10,100, 1):
            
        [A,B,C] = np.dot(plane_found_use, reciprocal_metric_tensor)                
        
        # g from plane found to use norm squared
        gf_norm2 = np.dot(np.dot(plane_found_use, reciprocal_metric_tensor), plane_found_use)
        
        # consider the exceptions in the type of ZA found
        if v==0 and w==0:
            # for zone axes of type [u,0,0]
            
            A = 0
            
            W = (n*gf_norm2*np.cos((np.pi/180)*angle_exp_use))/B
            X = -C/B
            
            F = (W**2)*g4 - (n**2)*gf_norm2
            E = 2*W*X*g4 + W*(g7 + g5)
            D = (X**2)*g4 + X*(g7+g5) + g8
            
            lx_1 = (-E+np.sqrt((E**2)-4*D*F))/(2*D)
            lx_2 = (-E-np.sqrt((E**2)-4*D*F))/(2*D)
            
            kx_1 = W + X*lx_1
            kx_2 = W + X*lx_2
            
            hx_1 = 0
            hx_2 = 0
            
        elif u==0 and w==0:
            # for zone axes of type [0,v,0]
            
            B = 0
            
            Y = (n*gf_norm2*np.cos((np.pi/180)*angle_exp_use))/A
            Z = -C/A
            
            F = (Y**2)*g0 - (n**2)*gf_norm2
            E = 2*Y*Z*g0 + Y*(g6 + g2)
            D = (Z**2)*g0 + Z*(g6 + g2) + g8
            
            lx_1 = (-E+np.sqrt((E**2)-4*D*F))/(2*D)
            lx_2 = (-E-np.sqrt((E**2)-4*D*F))/(2*D)
                    
            kx_1 = 0
            kx_2 = 0
            
            hx_1 = Y + Z*lx_1
            hx_2 = Y + Z*lx_2
            
            
        elif u==0 and v==0:
            # for zone axes of type [0,0,w]
            C = 0
            
            H = (n*gf_norm2*np.cos((np.pi/180)*angle_exp_use))/A
            J = -B/A
            
            F = (H**2)*g0 - (n**2)*gf_norm2
            E = 2*H*J*g0 + H*(g3 + g1)
            D = (J**2)*g0 + J*(g3 + g1) + g4
            
            lx_1 = 0
            lx_2 = 0
            
            kx_1 = (-E+np.sqrt((E**2)-4*D*F))/(2*D)
            kx_2 = (-E-np.sqrt((E**2)-4*D*F))/(2*D)
            
            hx_1 = H + J*kx_1
            hx_2 = H + J*kx_2
                    
            
        elif u==0 and v != 0 and w !=0:
            # for zone axes of type [0,v,w]
            
            if abs(v) == abs(w) and abs(B) == abs(C):
                
                R = -(np.sign(v)*np.sign(w))
                
                if A == 0 or A < 1e-15:
                    
                    kx_1 = (n*gf_norm2*np.cos((np.pi/180)*abs(angle_exp_use)))/(2*B)
                    kx_2 = (n*gf_norm2*np.cos((np.pi/180)*abs(angle_exp_use)))/(2*B)
                    
                    lx_1 = R*kx_1
                    lx_2 = R*kx_2
                    
                    F = (kx_1**2)*g4 + lx_1*kx_1*(g7+g5)+(lx_1**2)*g8 - (n**2)*gf_norm2
                    E = kx_1*(g3+g1) + lx_1*(g6+g2)
                    D = g0
                    
                    hx_1 = (-E+np.sqrt((E**2)-4*D*F))/(2*D)
                    hx_2 = (-E-np.sqrt((E**2)-4*D*F))/(2*D)
                    
                else:
                    # A ! = 0
                    
                    M = (n*gf_norm2*np.cos((np.pi/180)*angle_exp_use))/A
                    N = -(2*B)/A
                    
                    T = 1/R
                    G = M
                    Q = N/R
                    
                    F = (G**2)*g0 - (n**2)*gf_norm2
                    E = 2*G*Q*g0 + T*G*(g3 + g1) + G*(g6 + g2)
                    D = (Q**2)*g0 + T*Q*(g3 + g1) + Q*(g6 + g2) + (T**2)*g4 + T*(g7 + g5) + g8
                     
                    lx_1 = (-E+np.sqrt((E**2)-4*D*F))/(2*D)
                    lx_2 = (-E-np.sqrt((E**2)-4*D*F))/(2*D)
                    
                    kx_1 = T*lx_1
                    kx_2 = T*lx_2
                    
                    hx_1 = G + Q*lx_1
                    hx_2 = G + Q*lx_2    
                    
                             
            else:
                
                R = (1/((B/C)-(v/w)))*((n*gf_norm2*np.cos((np.pi/180)*angle_exp_use))/C)
                P = (1/((B/C)-(v/w)))*((u/w)-(A/C))
                
                M = ((n*gf_norm2*np.cos((np.pi/180)*angle_exp_use))/C) - (B*R)/C
                N = -((B*P)/C+(A/C))
                
                F = (R**2)*g4 + M*R*(g7+g5) + (M**2)*g8 - (n**2)*gf_norm2
                E = R*(g3+g1) + M*(g6+g2) + 2*R*P*g4 + M*P*(g7+g5) + N*R*(g7+g5) + 2*M*N*g8
                D = g0 + P*(g3+g1) + N*(g6+g2) + (P**2)*g4 + N*P*(g7+g5) + (N**2)*g8
                
                hx_1 = (-E+np.sqrt((E**2)-4*D*F))/(2*D)
                hx_2 = (-E-np.sqrt((E**2)-4*D*F))/(2*D)
                
                kx_1 = R + P*hx_1
                kx_2 = R + P*hx_2  
                
                lx_1 = M + N*hx_1
                lx_2 = M + N*hx_2  
                
                
            # Exceptions that did not generalise enough        
            
            # elif (A == 0 or abs(A) < 1e-15) and abs(v) != abs(w): 
                
            #     K = (n*gf_norm2*np.cos((np.pi/180)*angle_exp_use))/B
            #     X = -C/B
            #     print('K, X')
            #     print(K, X)    
            #     lx_1 = K/((w/v)-X)
            #     lx_2 = K/((w/v)-X)
            #     print('lx',lx_1)
            #     kx_1 = K + X*lx_1
            #     kx_2 = K + X*lx_2
            #     print('kx',kx_1)
            #     F = (kx_1**2)*g4 + lx_1*kx_1*(g7+g5)+(lx_1**2)*g8 - (n**2)*gf_norm2
            #     E = kx_1*(g3+g1) + lx_1*(g6+g2)
            #     D = g0
                
            #     hx_1 = (-E+np.sqrt((E**2)-4*D*F))/(2*D)
            #     hx_2 = (-E-np.sqrt((E**2)-4*D*F))/(2*D)
                
            #     print('hx1',hx_1)
            #     print('hx2',hx_2)
                
            # else:
            
            #     T = -w/v
                
            #     G = (n*gf_norm2*np.cos((np.pi/180)*angle_exp_use))/A
            #     Q = ((B*w)/(A*v)-(C/A))
                
            #     F = (G**2)*g0 - (n**2)*gf_norm2
            #     E = 2*G*Q*g0 + T*G*(g3 + g1) + G*(g6 + g2)
            #     D = (Q**2)*g0 + T*Q*(g3 + g1) + Q*(g6 + g2) + (T**2)*g4 + T*(g7 + g5) + g8
                 
            #     lx_1 = (-E+np.sqrt((E**2)-4*D*F))/(2*D)
            #     lx_2 = (-E-np.sqrt((E**2)-4*D*F))/(2*D)
                
            #     kx_1 = T*lx_1
            #     kx_2 = T*lx_2
                
            #     hx_1 = G + Q*lx_1
            #     hx_2 = G + Q*lx_2
                
                
            
        elif v==0 and u != 0 and w !=0:
            # for zone axes of type [u,0,w]
            
            
            if abs(u) == abs(w) and abs(A) == abs(C):
                
                R = -(np.sign(u)*np.sign(w))
                
                if B == 0 or B < 1e-15:
            
                    hx_1 = (n*gf_norm2*np.cos((np.pi/180)*abs(angle_exp_use)))/(2*A)
                    hx_2 = (n*gf_norm2*np.cos((np.pi/180)*abs(angle_exp_use)))/(2*A)
                    
                    lx_1 = R*hx_1
                    lx_2 = R*hx_2
                    
                    F = (hx_1**2)*g0 + hx_1*lx_1*(g6+g2)+(lx_1**2)*g8 -(n**2)*gf_norm2
                    E = hx_1*(g3+g1) + lx_1*(g7+g5)
                    D = g4
                    
                    kx_1 = (-E+np.sqrt((E**2)-4*D*F))/(2*D)
                    kx_2 = (-E-np.sqrt((E**2)-4*D*F))/(2*D)
                    
                else:
                    # B ! =0
                    
                    T = 1/R
                    
                    M = (n*gf_norm2*np.cos((np.pi/180)*abs(angle_exp_use)))/(2*A)
                    N = -(B/(2*A))
                    
                    U = -M/N
                    W = T/N
                    
                    F = (U**2)*g4 - (n**2)*gf_norm2
                    E = U*T*(g3 + g1) + 2*U*W*g4 + U*(g7 + g5)
                    D = (T**2)*g0 + W*T*(g3 + g1) + T*(g6 + g2) + (W**2)*g4 + W*(g7 + g5) + g8
                    
                    lx_1 = (-E+np.sqrt((E**2)-4*D*F))/(2*D)
                    lx_2 = (-E-np.sqrt((E**2)-4*D*F))/(2*D)
                    
                    kx_1 = U + W*lx_1
                    kx_2 = U + W*lx_2
                    
                    hx_1 = T*lx_1
                    hx_2 = T*lx_2
                    
            else:        
                    

                R = (1/((C/A)-(w/u)))*((n*gf_norm2*np.cos((np.pi/180)*angle_exp_use))/A)
                P = (1/((C/A)-(w/u)))*((v/u)-(B/A))
                
                M = ((n*gf_norm2*np.cos((np.pi/180)*angle_exp_use))/A) - (C*R)/A
                N = -((B/A) + (C*P)/A)
                
                F = (M**2)*g0 + M*R*(g6+g2) + (R**2)*g8 - (n**2)*gf_norm2
                E = 2*M*N*g0 + M*(g3+g1) + M*P*(g6+g2) + N*R*(g6+g2) + R*(g7+g5) + 2*R*P*g8
                D = (N**2)*g0 + N*(g3+g1) + N*P*(g6+g2) + g4 + P*(g7+g5) + (P**2)*g8
                
                kx_1 = (-E+np.sqrt((E**2)-4*D*F))/(2*D)
                kx_2 = (-E-np.sqrt((E**2)-4*D*F))/(2*D)            
                
                hx_1 = M + N*kx_1
                hx_2 = M + N*kx_2
                                
                lx_1 = R + P*kx_1
                lx_2 = R + P*kx_2


                # Exceptions that did not generalise enough   
                
                # print('u0w')
                # T = -w/u
                
                # U = (n*gf_norm2*np.cos((np.pi/180)*angle_exp_use))/B
                # W = -((A*T)/(B)+(C/B))
                
                # F = (U**2)*g4 - (n**2)*gf_norm2
                # E = U*T*(g3 + g1) + 2*U*W*g4 + U*(g7 + g5)
                # D = (T**2)*g0 + W*T*(g3 + g1) + T*(g6 + g2) + (W**2)*g4 + W*(g7 + g5) + g8
                
                # lx_1 = (-E+np.sqrt((E**2)-4*D*F))/(2*D)
                # lx_2 = (-E-np.sqrt((E**2)-4*D*F))/(2*D)
                
                # kx_1 = U + W*lx_1
                # kx_2 = U + W*lx_2
                
                # hx_1 = T*lx_1
                # hx_2 = T*lx_2
                
        # The general case works better than this specific one, that gives nan errors        
        # elif w==0 and u != 0 and v !=0:
        #     # for zone axes of type [u,v,0]
        #     print('uv0')
        #     T = -v/u
            
        #     L = (n*gf_norm2*np.cos((np.pi/180)*angle_exp_use))/C
        #     S = -((A*T)/(C) + (B/C))
            
        #     F = (L**2)*g8 - (n**2)*gf_norm2
        #     E = T*L*(g6 + g2) + L*(g7 + g5) + 2*L*S*g8
        #     D = (T**2)*g0 + T*(g3 + g1) + T*S*(g6 + g2) + g4 + S*(g7 + g5) + (S**2)*g8
            
        #     kx_1 = (-E+np.sqrt((E**2)-4*D*F))/(2*D)
        #     kx_2 = (-E-np.sqrt((E**2)-4*D*F))/(2*D)
            
        #     lx_1 = L + S*kx_1
        #     lx_2 = L + S*kx_2
            
        #     hx_1 = T*kx_1
        #     hx_2 = T*kx_2
            
            
        else:
            # general case
            
            if A== 0 or A < 1e-15:
                K = (n*gf_norm2*np.cos((np.pi/180)*angle_exp_use))/B
                X = -C/B
                
                M = -(K*v)/u
                N = -((X*v)/u + (w/u))
                
                F = (M**2)*g0 + K*M*(g3+g1) + (K**2)*g4 - gf_norm2*(n**2)
                E = 2*M*N*g0 + K*N*(g3+g1) + X*M*(g3+g1) + M*(g6+g2) + 2*K*X*g4 + K*(g7+g5)
                D = (N**2)*g0 + X*N*(g3+g1) + N*(g6+g2) + (X**2)*g4 + X*(g7+g5) + g8
    
                lx_1 = (-E+np.sqrt((E**2)-4*D*F))/(2*D)
                lx_2 = (-E-np.sqrt((E**2)-4*D*F))/(2*D)
                
                kx_1 = K + X*lx_1
                kx_2 = K + X*lx_2
                
                hx_1 = M + N*lx_1
                hx_2 = M + N*lx_2
    
            else:
            
                
                # complete general version for any axis
                R = (u/(B*u-A*v))*n*gf_norm2*np.cos((np.pi/180)*angle_exp_use)
                P = (A*w-C*u)/(B*u-A*v)
                
                M = ((n*gf_norm2*np.cos((np.pi/180)*angle_exp_use))/A)-(B*R)/A
                N = -((B*P)/A + C/A)
                
                F = g0*(M**2)+R*M*(g3+g1)+g4*(R**2)-gf_norm2*(n**2)
                E = 2*M*N*g0+R*N*(g3+g1)+P*M*(g3+g1)+M*(g6+g2)+2*R*P*g4+R*(g7+g5)
                D = g0*(N**2)+P*N*(g3+g1)+N*(g6+g2)+g4*(P**2)+P*(g7+g5)+g8
                
                # two solutions of two planes being the intersection between the cone formed
                # by the solid angle of angle_exp_use and plane_found_use and the intersection
                # of this cone with the plane defined by the zone axis
                
                lx_1 = (-E+np.sqrt((E**2)-4*D*F))/(2*D)
                lx_2 = (-E-np.sqrt((E**2)-4*D*F))/(2*D)
                
                kx_1 = R+P*lx_1
                kx_2 = R+P*lx_2
                
                hx_1 = M+N*lx_1
                hx_2 = M+N*lx_2
        
        # check if vals are nan and if no nan then use these vals
        if math.isnan(lx_1)==False and math.isnan(lx_2)==False and math.isnan(kx_1)==False and math.isnan(kx_2)==False and math.isnan(hx_1)==False and math.isnan(hx_2)==False:
            break
    
        
    plane_1_in_x = [hx_1, kx_1, lx_1]
    plane_2_in_x = [hx_2, kx_2, lx_2]
    
    # force the values to be integers, and be the smalles possible

    plane_1_in_x_int = Find_integer_plane(
        plane_1_in_x, tolerance_diff = tolerance_diff)
    plane_2_in_x_int = Find_integer_plane(
        plane_2_in_x, tolerance_diff = tolerance_diff)
    
    # to choose between the two, the sign of the angles must be checked
    # get the angles from the a1 and a2 basis to these vectors found
    
    g_vect_1x = np.array([plane_1_in_x_int[0]*b_1, plane_1_in_x_int[1]*b_2, plane_1_in_x_int[2]*b_3])
    g_vect_1x = np.sum(g_vect_1x, axis = 0)
    
    angle_theo_plane_1x_to_a1 = angle_between_cartes_vectors(g_vect_1x, a1)
    
    angle_theo_new_a2_g_to_plane_1x = angle_between_cartes_vectors(new_a2_90_deg_from_a1, g_vect_1x)

    if angle_theo_new_a2_g_to_plane_1x > 90:
        angle_theo_plane_1x_to_a1 = -angle_theo_plane_1x_to_a1

    g_vect_2x = np.array([plane_2_in_x_int[0]*b_1, plane_2_in_x_int[1]*b_2, plane_2_in_x_int[2]*b_3])
    g_vect_2x = np.sum(g_vect_2x, axis = 0)
    
    angle_theo_plane_2x_to_a1 = angle_between_cartes_vectors(g_vect_2x, a1)
    
    angle_theo_new_a2_g_to_plane_2x = angle_between_cartes_vectors(new_a2_90_deg_from_a1, g_vect_2x)

    if angle_theo_new_a2_g_to_plane_2x > 90:
        angle_theo_plane_2x_to_a1 = -angle_theo_plane_2x_to_a1
    
    
    print('plane_1_in_x')
    print(plane_1_in_x)
    print('plane_2_in_x')
    print(plane_2_in_x)
    print('plane1xfound int')
    print(plane_1_in_x_int)
    print('plane2xfound int')
    print(plane_2_in_x_int)
    print(' angle_theo_plane_1x_to_a1')
    print(angle_theo_plane_1x_to_a1)
    print('angle_theo_new_a2_g_to_plane_1x')
    print(angle_theo_new_a2_g_to_plane_1x)
    print(' angle_theo_plane_2x_to_a1')
    print(angle_theo_plane_2x_to_a1)
    print('angle_theo_new_a2_g_to_plane_2x')
    print(angle_theo_new_a2_g_to_plane_2x)
    
    # the plane whose angle with a1 and after being rotted the final rotation
    # we have to apply it is closer to 0, then this one will be the chosen one
    # as it is in the correct position with respect the hkl1 vector
    
    angle_to_0_aft_final_rot_plane_1x = final_in_surf_plane_rotation + angle_theo_plane_1x_to_a1
    angle_to_0_aft_final_rot_plane_2x = final_in_surf_plane_rotation + angle_theo_plane_2x_to_a1
    
    planes_in_x_int = [plane_1_in_x_int, plane_2_in_x_int]
    
    plane_final_cartesian_x = planes_in_x_int[np.argmin(np.abs(np.asarray([angle_to_0_aft_final_rot_plane_1x, angle_to_0_aft_final_rot_plane_2x])))]
    
    # find also the direction in case it is necessary
    direction_final_cartesian_x = Find_direction_paralel_to_plane(
        plane_final_cartesian_x, reciprocal_metric_tensor, tolerance_diff = tolerance_diff)
    
    # direction_final_cartesian_x = 1
    
    return plane_final_cartesian_x, direction_final_cartesian_x



def Build_DeviceSupercell_Base_Crystals(
        analysed_image, model_cells_filepath, 
        z_thickness_model, conts_vertx_per_region):
    
    '''
    Build the device supercell as if all the crystals were perfect and 
    no strain considerations are applied, so just direct application of
    the segmentation and the phase and orientation found
    but no virtual crystal formation, just plain device as if it was perfect
    To set the reference of what is supposed to be the perfect device with
    n perfect lattices forming an FFT of just 1 spot per lattice
    '''
    
    # Folder to contain the models created, inside this model_cells, based
    # on the name of the image
    atom_models_filepath = model_cells_filepath + analysed_image.image_in_dataset.name + '_base'+ '\\'
    path_atom_models = os.path.isdir(atom_models_filepath)
    if path_atom_models == False:
        os.mkdir(atom_models_filepath)
           
    # Information from all the crops done to the image
    crop_outputs_dict = analysed_image.Crop_outputs
    
    
    # Loop through the regions and build the atomic model from each region
    for label_segm_region in range(1, analysed_image.crop_index):
        
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
            
            
            # Here check if there exists the virtual crystal of the found_phase_name
            # and build the cell based on this or not if it does not exist
            cif_cell_filepath = model_cells_filepath + found_phase_name + '.cif'
            
            # Both the angle and atomistic model building needs to be done through 
            # either the modified cif (virtual) or the base one
            # Find the rotation we need to induce to the default atomistic model 
            # to rotate it to the found orientation
            
            final_in_surf_plane_rotation = Adjust_in_surface_plane_rotation(
                cif_cell_filepath, scored_spot_pair, suface_basis_choice = 'plane')
            
            # Save the models in the folder inside model_cells folder created with
            # the name of the folder 
            # function that does the full shaping and orientation 
            
            
            # check that the cntour adjustment is the same format as the one
            # which was with the trials in the previous checkings while 
            # developing the atomistic_model_builder
            
            # it fails in the make supercell functoin make sure w ereally need the function
            # that finds the optimal cubic shape as i think it deos more bad than good
            
            Build_shaped_atomistic(
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
    Combine_xyz_supercells(atom_models_filepath)
    
    
    # final_global_device_supcell = read(temp_xyz_files_folder_directory + 'global_device_supercell.xyz')
    # show_atoms(final_global_device_supcell, plane='xy')
    # show_atoms(final_global_device_supcell, plane='xz')
    # show_atoms(final_global_device_supcell, plane='yz')
    
    return atom_models_filepath



class PrimCellPos:
    '''
    Class to store the primitive cell position entry on a given crystal file
    '''
    def __init__(self, element, x,y,z, occ, DW, absorpt):
        self.element = element
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)
        self.occ = float(occ)
        self.DW = float(DW)
        self.absorpt = float(absorpt)
        


def Store_cif_Relative_PrimitiveCellParams(
        cif_unit_cell_path):
    '''
    Only for CIF files, to store the Wickof positions, coordinates 
    occupancy and atom types
    
    ASSUME!!! That after finding _atom_site_ like entry in the cif file
    only the positions come and come in format like the following
    


    Parameters
    ----------
    cif_unit_cell_path : str to cif filepath

    Returns
    -------
    relative_positions : list of n PrimCellPos where n is the number of
                        relative positions found within the cif file

    '''
    

        
    cif_file = open(cif_unit_cell_path)    
    cif_lines = cif_file.readlines()
    
    ind_start = 0
    for index, line in enumerate(cif_lines):
        if '_atom_site' in line:
            ind_start = index
            break
        
    # ASSUME!!! That after finding _atom_site_ like entry in the cif file
    # only the positions come and come in format like the following
    atoms_lines = cif_lines[ind_start:]  
    
    list_split_lines = []
    
    for line in atoms_lines:
        list_split_lines.append(line.split())
    
    primit_pos = [line for line in list_split_lines if len(line)>=3] 
    
    # !!! oerfect would be to read the _atom_site labels and construct the 
    # info based on that, but it is overkill now
    primitcellposes = []
    
    for pos in primit_pos:
        
        element = pos[0]
        x_r = pos[1]
        y_r = pos[2]
        z_r = pos[3]
        occ = pos[4]
        DW = pos[5]
        absorpt = pos[6]
        
        # build the object PrimCellPos
        primitcellpos = PrimCellPos(
            element, x_r, y_r, z_r, occ, DW, absorpt)
    
        primitcellposes.append(primitcellpos)
        
    return primitcellposes




class Wyckoff_position:
    '''
    Class storing the Wyckoff position and nature 
    '''
    def __init__(self, space_group, symb_letter, 
                 multiplicity, x,y,z, unique):
        self.SG = space_group
        self.symb = symb_letter
        self.multi = multiplicity
        self.x = x
        self.y = y
        self.z = z
        self.unique = unique
            

def Wyckoff_pos_Checker(
        space_group, rel_prim_position,
        wyckoff_filepath = r'E:\Arxius varis\PhD\3rd_year\Code\Functions\difftool_dll\Wyckoff_database.txt'):
    '''
    Given a space group number, and a relative position within the unit cell,
    indicate which is the wyckoff position in terms of multiplicty for it
    to know how many atoms are created in the conventional unit cell based
    on that position in the primitive unit cell

    Parameters
    ----------
    space_group : TYPE
        DESCRIPTION.
    rel_prim_position : relative primitive position within the primitive cell
                        list with 3 floats
    wyckoff_filepath : TYPE, optional
        DESCRIPTION. The default is
        r'E:\Arxius varis\PhD\3rd_year\Code\Functions\difftool_dll\Wyckoff_database.txt'

    Returns
    -------
    wyckoff_multiplictiy
    '''

    # STEP 1: Extract wyckoff info from databse file        
    raw_wyckoff_data = open(wyckoff_filepath, encoding = 'utf-8')
    
    raw_wyckoff_lines = raw_wyckoff_data.readlines()
    
    list_wyckoff_lines = []
    
    for line in raw_wyckoff_lines:
        list_wyckoff_lines.append(line.split())
        

    lines_by_SG = []    
    
    sg_list = []
    for line in raw_wyckoff_lines:
        
        if '//' in line:
            lines_by_SG.append(sg_list)
            sg_list = []
            
        sg_list.append(line)
        
    # now lines_by_SG[space_group] contains the information of the spacegroup
    # as the 0 element is an empty list generated during the process
    
    wyckoff_pos_SG = []
    
    for line in lines_by_SG[space_group]:
        
        if 'wyckPos' not in line:
            continue
        
        # extract features
        line_info = line[line.find('wyckPos') + 8 :]
        letter = line_info[1:line_info.find(',')-1]
        line_info2 = line_info[line_info.find(',')+1:]
        multiplicity = int(line_info2[:line_info2.find(',')])
        
        if 'true' in line_info2:
            unique_str = 'true'
            unique = True
        elif 'false' in line_info2:
            unique_str = 'false'
            unique = False
        
        
        cords = line_info2[line_info2.find('"') +1:line_info2.find(unique_str)-2]
        
        x_c = cords[:cords.find(',')]
        cords2 = cords[cords.find(',')+1:]
        y_c = cords2[:cords2.find(',')]
        z_c = cords2[cords2.find(',')+1:]
        
        # convert to float if the values are unique, so not variable dependent
        # just values
        if unique == True:
            x_c = float(x_c)
            y_c = float(y_c)
            z_c = float(z_c)
            
            
        # build the Wyckoff_position class and store it in wyckoff_pos_SG
        wyckpos = Wyckoff_position(
            space_group, letter, multiplicity, x_c, y_c, z_c, unique)
        
        wyckoff_pos_SG.append(wyckpos)
        
    # STEP 2:        
    # Now that all the Wyckoff positions are stored in wyckoff_pos_SG
    # check the multiplicity of the inputted primitive position 
    
    # The coords to check
    x_in = rel_prim_position[0]
    y_in = rel_prim_position[1]
    z_in = rel_prim_position[2]
    
    # They should be positive, but just in case make it sure
    if x_in < 0:
        x_in = round(x_in + 1, 4)
    if y_in < 0:
        y_in = round(y_in + 1, 4)
    if z_in < 0:
        z_in = round(z_in + 1, 4)
    
    # Format the x,y,z input if they are 0.33333 = 1/3 or 0.666667 = 2/3
    # so they can be compared with the values in 
    if round(x_in, 4)== 0.3333:
        x_in = 0.333333
    if round(y_in, 4)== 0.3333:
        y_in = 0.333333
    if round(z_in, 4)== 0.3333:
        z_in = 0.333333
        
    if round(x_in, 4)== 0.6667:
        x_in = 0.666667
    if round(y_in, 4)== 0.6667:
        y_in = 0.666667
    if round(z_in, 4)== 0.6667:
        z_in = 0.666667
    
    
    for wyckpos in wyckoff_pos_SG:
        # if the positions are unique in the wyckoff position
        if wyckpos.unique == True:
            # if the target one coincides with the wyckoff one
            if wyckpos.x == x_in and wyckpos.y == y_in and wyckpos.z == z_in:
                wyckoff_multiplictiy = wyckpos.multi
                return wyckoff_multiplictiy
        
        else:
            # wyckpos.unique == False:
            # So we work with strings and values and signs
            
            x_ch = wyckpos.x
            y_ch = wyckpos.y
            z_ch = wyckpos.z
            # Check if the strings can be converted to numbers
            # As the lowest multiplicities come first, we can make sure 
            # that the first chosen if it matches is more restrictive
            # and is the adequate Wyckoff (i.e., the wychoff x,y,z would always
            # match, that is why it must be checked the last one as there is
            # maybe a lower multi one that fits the given position)
            
            # These wyckfs in which just floats are present (unique pos
            # of space group with non unique wyckfs posses)
            if 'x' not in x_ch and 'x' not in y_ch and 'x' not in z_ch:
                if 'y' not in x_ch and 'y' not in y_ch and 'y' not in z_ch:
                    if 'z' not in x_ch and 'z' not in y_ch and 'z' not in z_ch:
                        x_ch = float(x_ch)
                        y_ch = float(y_ch)
                        z_ch = float(z_ch)
                        if x_ch == x_in and y_ch == y_in and z_ch == z_in:
                            wyckoff_multiplictiy = wyckpos.multi
                            return wyckoff_multiplictiy
                            
            # compute the cords if  symbol is in x cords            
            if 'x' in x_ch:
                x_ch2 = float(x_ch[:x_ch.find('x')] + str(x_in)) 
                # watch out if string to convert to float is empty string ''
                # meaning that if the resulting string isjust the variable
                # then no need to add anything
                if len(x_ch[x_ch.find('x'):]) > 1:
                    x_ch = x_ch2 + float(x_ch[x_ch.find('x')+1:])
                else:
                    x_ch = x_ch2
                    
            elif 'y' in x_ch:
                x_ch2 = float(x_ch[:x_ch.find('y')] + str(y_in))
                if len(x_ch[x_ch.find('y'):]) > 1:
                    x_ch = x_ch2 + float(x_ch[x_ch.find('y')+1:])
                else:
                    x_ch = x_ch2
                    
            elif 'z' in x_ch:
                x_ch2 = float(x_ch[:x_ch.find('z')] + str(z_in))
                if len(x_ch[x_ch.find('z'):]) > 1:
                    x_ch = x_ch2 + float(x_ch[x_ch.find('z')+1:])
                else:
                    x_ch = x_ch2
                                    
            # compute the cords if symbol is in y cords            
            if 'x' in y_ch:
                y_ch2 = float(y_ch[:y_ch.find('x')] + str(x_in))
                if len(y_ch[y_ch.find('x'):]) > 1:
                    y_ch = y_ch2 + float(y_ch[y_ch.find('x')+1:])
                else:
                    y_ch = y_ch2
                
            elif 'y' in y_ch:
                y_ch2 = float(y_ch[:y_ch.find('y')] + str(y_in))
                if len(y_ch[y_ch.find('y'):]) > 1:
                    y_ch = y_ch2 + float(y_ch[y_ch.find('y')+1:])
                else:
                    y_ch = y_ch2
                
                
            elif 'z' in y_ch:
                y_ch2 = float(y_ch[:y_ch.find('z')] + str(z_in))
                if len(y_ch[y_ch.find('z'):]) > 1:
                    y_ch = y_ch2 + float(y_ch[y_ch.find('z')+1:])
                else:
                    y_ch = y_ch2
                    
            # compute the cords if  symbol is in Z cords  
            if 'x' in z_ch:
                z_ch2 = float(z_ch[:z_ch.find('x')] + str(x_in))
                if len(z_ch[z_ch.find('x'):]) > 1:
                    z_ch = z_ch2 + float(z_ch[z_ch.find('x')+1:])
                else:
                    z_ch = z_ch2
                
            elif 'y' in z_ch:
                z_ch2 = float(z_ch[:z_ch.find('y')] + str(y_in))
                if len(z_ch[z_ch.find('y'):]) > 1:
                    z_ch = z_ch2 + float(z_ch[z_ch.find('y')+1:])
                else:
                    z_ch = z_ch2
                
            elif 'z' in z_ch:
                z_ch2 = float(z_ch[:z_ch.find('z')] + str(z_in))
                if len(z_ch[z_ch.find('z'):]) > 1:
                    z_ch = z_ch2 + float(z_ch[z_ch.find('z')+1:])
                else:
                    z_ch = z_ch2
                    
            # for the entries that are directly numbers, they are str at 
            # this point, so just convert everything to float
            x_ch = float(x_ch)        
            y_ch = float(y_ch)        
            z_ch = float(z_ch)       
            
            # Last, ensure all the x_ch, y_ch, z_ch are positive (translational
            # invariance for all the coordiantes)
            
            if x_ch < 0:
                x_ch = round(x_ch + 1, 4)
            if y_ch < 0:
                y_ch = round(y_ch + 1, 4)
            if z_ch < 0:
                z_ch = round(z_ch + 1, 4)
            # Ready to make the comparison with the input coordinates
            if x_ch == x_in and y_ch == y_in and z_ch == z_in:
                wyckoff_multiplictiy = wyckpos.multi
                return wyckoff_multiplictiy


# Functions to build the models out of the virtual cells built
# and where they do correspond

def Adjust_Virt_Cryst_Rotation_on_VirtualRef(
        analysed_image, best_GPA_ref_spot_pair, label_of_GPA_ref,
        label_of_region, model_cells_filepath,
        paths_to_virt_ucells, scored_spot_pairs_found):
    '''
    Compute the rotation that must be corrected when building the virtual
    crystal heterostructing the virtual crystal in the reference region
    That is, the idea of building the virtual crystal on the other materials
    forming the heterostructure implies making the spots representing these
    other materials coincide with the reference one
    That is, we compute the virtual crytsl of the reference (label_of_GPA_ref), 
    and then compute the virtual crystal in the other material in region 
    enconded with label label_of_region, which will have an interplanar distance
    of that spot exactly the same as the other spot taken as reference,
    and then, to place it in the exact same angle we must correct the rotation
    and rotatethe virtual crystl with an angle computed with the function
    AtomBuild.Adjust_in_surface_plane_rotation()
    and then to this sum the angle computed in this function, as this one
    unites both spots angularly
    real_virtual_crystal_ase_rotation =  
    AtomBuild.Adjust_in_surface_plane_rotation() + rotation_adjustment 
    where rotation_adjustment is what is computed here 
    !!! it is already considered as the adjustment, so just SUM it not substract
    
    Parameters
    ----------
    analysed_image : TYPE
        DESCRIPTION.
    best_GPA_ref_spot_pair : TYPE
        DESCRIPTION.
    label_of_GPA_ref : TYPE
        DESCRIPTION.
    label_of_region : TYPE
        DESCRIPTION.
    model_cells_filepath : TYPE
        DESCRIPTION.
    paths_to_virt_ucells : TYPE
        DESCRIPTION.
    scored_spot_pairs_found : TYPE
        DESCRIPTION.

    Returns
    -------
    rotation_adjustment : TYPE
        DESCRIPTION.

    '''
    
    # Information from all the crops done to the image
    crop_outputs_dict = analysed_image.Crop_outputs


    # information from reference region, label_of_GPA_ref

    # Loop through the regions and build the atomic model from each region

    image_crop_hs_signal_ref = crop_outputs_dict[str(label_of_GPA_ref) + '_hs_signal']
    crop_list_refined_cryst_spots_ref = crop_outputs_dict[str(label_of_GPA_ref) + '_list_refined_cryst_spots']

    # most likely crystal found
    best_cryst_spot_ref = crop_list_refined_cryst_spots_ref[0]
    zone_axis_found_ref =  best_cryst_spot_ref.ZA
    found_phase_name_ref = best_cryst_spot_ref.phase_name
    hkl1_ref = best_GPA_ref_spot_pair.hkl1_reference
    hkl2_ref = best_GPA_ref_spot_pair.hkl2_reference
    hkl1_to_x_exp_ref = best_GPA_ref_spot_pair.spot1_angle_to_x
    hkl2_to_x_exp_ref = best_GPA_ref_spot_pair.spot2_angle_to_x

    
    
    # # Here check if there exists the virtual crystal of the found_phase_name
    # # and build the cell based on this or not if it does not exist
    
    # cif_virtual_cell_filepath = model_cells_filepath + found_phase_name_ref + '_v.cif'
    
    # # Both the angle and atomistic model building needs to be done through 
    # # either the modified cif (virtual) or the base one
    
    # # Find the rotation we need to induce to the default atomistic model 
    # # to rotate it to the found orientation
    
    
    # # if a virutal cell is used, so was built, use the rotation computed on
    # #     the plane of the reference
    # # if cif_cell_filepath ==   cif_virtual_cell_filepath:
    # #     final_in_surf_plane_rotation = .....
        
    
    # final_in_surf_plane_rotation_ref = AtomBuild.Adjust_in_surface_plane_rotation(
    #     cif_virtual_cell_filepath, best_GPA_ref_spot_pair, suface_basis_choice = 'plane')
    
    
    # plane_final_cartesian_x_ref, direction_final_cartesian_x_ref = AtomBuild.Find_plane_pointing_to_final_cartesian_x_axis(
    #     cif_virtual_cell_filepath, best_GPA_ref_spot_pair, tolerance_diff = 0.3, suface_basis_choice = 'plane')
    
    
    # unit_cell_ref = ase.io.read(cif_virtual_cell_filepath)
    
    # # get cell params to compute the metric tensors
    # a, b, c, alfa, beta, gamma = unit_cell_ref.cell.cellpar()
     
    # direct_metric_tensor_ref = np.array([[a**2, a*b*np.cos((np.pi/180)*(gamma)), a*c*np.cos((np.pi/180)*(beta))],
    #                                   [b*a*np.cos((np.pi/180)*(gamma)), b**2, b*c*np.cos((np.pi/180)*(alfa))],
    #                                   [c*a*np.cos((np.pi/180)*(beta)), c*b*np.cos((np.pi/180)*(alfa)), c**2]])
    
    # reciprocal_metric_tensor_ref = np.linalg.inv(direct_metric_tensor_ref)

    
    # angle_x_hkl1_ref = AtomBuild.angle_between_planes(
    #         plane_final_cartesian_x_ref, hkl1_ref, reciprocal_metric_tensor_ref)
    
    # angle_x_hkl2_ref = AtomBuild.angle_between_planes(
    #         plane_final_cartesian_x_ref, hkl2_ref, reciprocal_metric_tensor_ref)
    
    
    # plane_final_cartesian_x, direction_final_cartesian_x = Find_plane_pointing_to_final_cartesian_x_axis(
    #     cell_filepath, scored_spot_pair, tolerance_diff = 0.3, suface_basis_choice = 'plane')

    
    image_crop_hs_signal_lab = crop_outputs_dict[str(label_of_region) + '_hs_signal']
    crop_list_refined_cryst_spots_lab = crop_outputs_dict[str(label_of_region) + '_list_refined_cryst_spots']

    # most likely crystal found
    best_cryst_spot_lab = crop_list_refined_cryst_spots_lab[0]
    zone_axis_found_lab =  best_cryst_spot_lab.ZA
    found_phase_name_lab = best_cryst_spot_lab.phase_name
    
    # this will only be done with the phases that needed a virtual one 
    
    for cell_path, scored_spot_pair in zip(
            paths_to_virt_ucells, scored_spot_pairs_found):
        
        if found_phase_name_lab in cell_path:            
            
            hkl1_lab = scored_spot_pair.hkl1_reference
            hkl2_lab = scored_spot_pair.hkl2_reference
            hkl1_to_x_exp_lab = scored_spot_pair.spot1_angle_to_x
            hkl2_to_x_exp_lab = scored_spot_pair.spot2_angle_to_x
            found_scored_spot_pair = scored_spot_pair
            cif_cell_filepath_lab = cell_path
            
       
    
    # final_in_surf_plane_rotation_lab = AtomBuild.Adjust_in_surface_plane_rotation(
    #     cif_cell_filepath_lab, found_scored_spot_pair, suface_basis_choice = 'plane')
    
    
    # plane_final_cartesian_x_lab, direction_final_cartesian_x_lab = AtomBuild.Find_plane_pointing_to_final_cartesian_x_axis(
    #     cif_cell_filepath_lab, found_scored_spot_pair, tolerance_diff = 0.3, suface_basis_choice = 'plane')
    
       
    # # ge tthe cell from the lab
    
    # unit_cell_lab = ase.io.read(cif_cell_filepath_lab)
    
    # # get cell params to compute the metric tensors
    # a, b, c, alfa, beta, gamma = unit_cell_lab.cell.cellpar()
     
    # direct_metric_tensor_lab = np.array([[a**2, a*b*np.cos((np.pi/180)*(gamma)), a*c*np.cos((np.pi/180)*(beta))],
    #                                   [b*a*np.cos((np.pi/180)*(gamma)), b**2, b*c*np.cos((np.pi/180)*(alfa))],
    #                                   [c*a*np.cos((np.pi/180)*(beta)), c*b*np.cos((np.pi/180)*(alfa)), c**2]])
    
    # reciprocal_metric_tensor_lab = np.linalg.inv(direct_metric_tensor_lab)
    
    
    # angle_x_hkl1_lab = AtomBuild.angle_between_planes(
    #         plane_final_cartesian_x_lab, hkl1_lab, reciprocal_metric_tensor_lab)
    
    # angle_x_hkl2_lab = AtomBuild.angle_between_planes(
    #         plane_final_cartesian_x_lab, hkl2_lab, reciprocal_metric_tensor_lab)
    
    
    
    
    # final_in_surf_plane_rotation_ref
    # final_in_surf_plane_rotation_lab
    # also usable for the check
    
    
    # also 
    
    # hkl1_to_x_exp_ref
    # hkl2_to_x_exp_ref
    
    # hkl1_to_x_exp_lab
    # hkl2_to_x_exp_lab
    
    
    
    # in princicple the angle to correct should be just
    # the difference bwteen angles to x of both planes, whihc should coincide
    
    rotation_adjustment1 = - (hkl1_to_x_exp_lab - hkl1_to_x_exp_ref )
    
    # and should coincide wiht
    
    rotation_adjustment2 = - (hkl2_to_x_exp_lab - hkl2_to_x_exp_ref)
    
    rotation_adjustment = (rotation_adjustment1 + rotation_adjustment2)/2
    
    return rotation_adjustment






def Check_Same_SpaceGroup_Orientation(
        analysed_image, model_cells_filepath, 
        label_of_GPA_ref, label_of_region, best_GPA_ref_spot_pair, 
        paths_to_virt_ucells, scored_spot_pairs_found):
    '''
    Check if, given two phases found encoded in the labels 
    label_of_GPA_ref for the reference material and label_of_region
    for the region we are interested, if they have the same
    space group, same zone axis, and within the mask range drawn,
    the same hkl1 and hkl2 pairs between them, meaning they
    are the same planes but a slight misplacement of the rotation
    If that label has no crystal found, no phase identified, returns False

    Parameters
    ----------
    analysed_image : analysed_image object of the target image
    model_cells_filepath : path to cif cells
    label_of_GPA_ref : label of the region being the GPA reference
    label_of_region : label of the region we are checking if it is 
                    oriented such as the reference one
    best_GPA_ref_spot_pair : best scored_spot_pair representing GPA ref g vects
    paths_to_virt_ucells : list of str with paths to the virtual cells found
    scored_spot_pairs_found : list of scored_spot_pairs paired with 
                            paths_to_virt_ucells with the spots found within 
                            the mask drawn around the GPA mask
    Returns
    -------
    equally_oriented : bool, True if same orientation False if not

    '''
    
    
    # orientation information from the reference region
    equally_oriented = False
    
    # Information from all the crops done to the image
    crop_outputs_dict = analysed_image.Crop_outputs

    # information from reference region, label_of_GPA_ref

    # Loop through the regions and build the atomic model from each region

    image_crop_hs_signal_ref = crop_outputs_dict[str(label_of_GPA_ref) + '_hs_signal']
    crop_list_refined_cryst_spots_ref = crop_outputs_dict[str(label_of_GPA_ref) + '_list_refined_cryst_spots']

    # most likely crystal found
    best_cryst_spot_ref = crop_list_refined_cryst_spots_ref[0]
    zone_axis_found_ref =  best_cryst_spot_ref.ZA
    found_phase_name_ref = best_cryst_spot_ref.phase_name
    hkl1_ref = best_GPA_ref_spot_pair.hkl1_reference
    hkl2_ref = best_GPA_ref_spot_pair.hkl2_reference

    # find the space group of the reference cell
    # in principle there should always be a virtual cell created here
    # just add the security check
    cif_ref_virtual_cell_filepath = model_cells_filepath + found_phase_name_ref + '_v.cif'
    
    cif_ref_virtual_cell_filepath_Path = Path(cif_ref_virtual_cell_filepath)
    cif_ref_virtual_file_already_created = cif_ref_virtual_cell_filepath_Path.is_file()
    
    if cif_ref_virtual_file_already_created == True:
        cif_ref_cell_filepath = cif_ref_virtual_cell_filepath
    else:
        cif_ref_cell_filepath = model_cells_filepath + found_phase_name_ref + '.cif'
            
        
    ase_unit_cell_ref = ase.io.read(cif_ref_cell_filepath)
    spacegroup_cell_ref = ase_unit_cell_ref.info['spacegroup'].no
    
    # get the family of planes of the reference region
    # for the planes and ZA, to make the comparison and check if the target 
    # ones are in the family of planes
    hkl1_ref_family = PhaseIdent.Crystal_System_Equidistance_Permutations(
        spacegroup_cell_ref, hkl1_ref)
    hkl2_ref_family =PhaseIdent.Crystal_System_Equidistance_Permutations(
        spacegroup_cell_ref, hkl2_ref)
    
    # !!! Not 100% correct, but assume the same equidsitance is same equivalent ZA
    ZA_ref_family = PhaseIdent.Crystal_System_Equidistance_Permutations(
        spacegroup_cell_ref, zone_axis_found_ref)
    
    
    # check the space group info from the other region 
    image_crop_hs_signal_lab = crop_outputs_dict[str(label_of_region) + '_hs_signal']
    crop_list_refined_cryst_spots_lab  = crop_outputs_dict[str(label_of_region) + '_list_refined_cryst_spots']
    
    if len(crop_list_refined_cryst_spots_lab) != 0:
        best_cryst_spot_lab = crop_list_refined_cryst_spots_lab[0]
    else:
        equally_oriented = False
        return equally_oriented 
    
    zone_axis_found_lab =  best_cryst_spot_lab.ZA
    found_phase_name_lab = best_cryst_spot_lab.phase_name
    
    
    for cell_filepath, scoredspotpair in zip(
            paths_to_virt_ucells, scored_spot_pairs_found):
        if found_phase_name_lab in cell_filepath:
        
            cif_lab_cell_filepath = cell_filepath
            
            hkl1_lab = scoredspotpair.hkl1_reference
            hkl2_lab = scoredspotpair.hkl2_reference
            
    ase_unit_cell_lab = ase.io.read(cif_lab_cell_filepath)
    spacegroup_cell_lab = ase_unit_cell_lab.info['spacegroup'].no
    
    # do not loop through labels just use label as 
    # function input and hceck this one label_of_region

    # now check if all the planes found for each labela
    # are within the family
    # hkl1 of label within hkl1 ref family 
    # hkl2 of label within hkl2 ref family
    # if ZA coincide and spacegroup coincide
    
    # turn elements into lists for making the if in check
    zone_axis_found_lab = list(zone_axis_found_lab)
    hkl1_lab = list(hkl1_lab)
    hkl2_lab = list(hkl2_lab)
    
    ZA_ref_family = [list(i) for i in ZA_ref_family]
    hkl1_ref_family = [list(i) for i in hkl1_ref_family]
    hkl2_ref_family = [list(i) for i in hkl2_ref_family]
    
    if spacegroup_cell_lab == spacegroup_cell_ref:
        if zone_axis_found_lab in ZA_ref_family:
            if hkl1_lab in hkl1_ref_family:
                if hkl2_lab in hkl2_ref_family:
                    
                    # then they are in "almost" same spatial orientation
                    equally_oriented = True
    
        
    return equally_oriented
    


def Build_DeviceSupercell_Virtual_To_Distort(
        analysed_image, model_cells_filepath, 
        z_thickness_model, conts_vertx_per_region,
        label_of_GPA_ref, best_GPA_ref_spot_pair, 
        paths_to_virt_ucells, scored_spot_pairs_found):
    '''
    Main function building the atomistic model supercell for the whole device
    being this one the function that needs to be called when building the
    first original model that will eventually be strained
    This will build the regions based on the virtual crystals found
    for the reference and for the others checking if equally oriented or not
    
    # !!! STEP 1: Check the orientation for all the crystals and check if
    # they are same, and store the regions where they coincide
    # !!! STEP 2: Now split the methodology in 
    # 1) Building same crystal block for these oriented likewise, and if not likewise
    # just with the base crystal information or virtual but not oriented
        # 1.1)
        # Loop through the regions and build the atomic model from each region
        # first in the regions that are not equally oriented, so work
        # with the base unit cells or virtual but adapt rotation

        # 1.2)
        # Now build the single block for all the regions equally orietned 
        # as the reference region labeled as label_of_GPA_ref

    # 2) Building the device with all of virtual crystals separatedly directly from 
    # the beginning with the segmentation information, by compenseting
    # the crystal distances and angles of the spots surrounding the reference one    

    Parameters
    ----------
    analysed_image : analysed_image object
    model_cells_filepath : path to cif cells
    z_thickness_model : thickness of the model
    conts_vertx_per_region : smoothed contours for all the regions
    label_of_GPA_ref : label of region used as GPA reference
    best_GPA_ref_spot_pair : best scored_spot_pair representing GPA ref g vects
    paths_to_virt_ucells : list of str with paths to the virtual cells found
    scored_spot_pairs_found : list of scored_spot_pairs paired with 
                            paths_to_virt_ucells with the spots found within 
                            the mask drawn around the GPA mask

    Returns
    -------
    atom_models_filepath : path to cells built
    labels_equally_oriented_as_ref : list with the labels that share
                spatial orientation with the reference material
                it can be len() = 0 if no coincidence in spatial orientation,
                or len()>=2, at least the reference and another one (2) share
                the same orientation.
                This information will be important afterwards for applying the 
                segmentation/chemical information

    '''
    
    # Folder to contain the models created, inside this model_cells, based
    # on the name of the image
    atom_models_filepath = model_cells_filepath + analysed_image.image_in_dataset.name + '_strained' + '\\'
    path_atom_models = os.path.isdir(atom_models_filepath)
    if path_atom_models == False:
        os.mkdir(atom_models_filepath)
    
    # !!! STEP 1: Check the orientation for all the crystals and check if
    # they are same, and store the regions where they coincide
    
    # Information from all the crops done to the image
    crop_outputs_dict = analysed_image.Crop_outputs
    
    # store these regions (labels) where the crystals are equally oriented as 
    # the crystal of the reference, same space group
    labels_equally_oriented_as_ref = []
    
    # Loop through the regions and build the atomic model from each region
    for label_segm_region in range(1, analysed_image.crop_index):
        
        if label_segm_region != label_of_GPA_ref:
        
            image_crop_hs_signal = crop_outputs_dict[str(label_segm_region) + '_hs_signal']
            crop_list_refined_cryst_spots = crop_outputs_dict[str(label_segm_region) + '_list_refined_cryst_spots']
            
            
            # if no crystal is found, either amorphous or bad identified
            # in any case, do not consider it for the atomistic model
            # or build an amorphous one as next step    
            if len(crop_list_refined_cryst_spots) == 0:
                continue
            
            equally_oriented_lab = Check_Same_SpaceGroup_Orientation(
                analysed_image, model_cells_filepath, label_of_GPA_ref, 
                label_segm_region, best_GPA_ref_spot_pair, 
                paths_to_virt_ucells, scored_spot_pairs_found)
            
            
            if equally_oriented_lab == True:
                labels_equally_oriented_as_ref.append(label_segm_region)
            
    # !!! STEP 2: Now split the methodology in 
    # 1) Building same crystal block for these oriented likewise, and if not likewise
    # just with the base crystal information or virtual but not oriented
    # 2) Building the device with all of virtual crystals separatedly directly from 
    # the beginning with the segmentation information, by compenseting
    # the crystal distances and angles of the spots surrounding the reference one    
        
    # 1)    
    # for those regions where the same crystal space and orientation is found
    # build just one single block out of it   
    if len(labels_equally_oriented_as_ref) != 0:
        # add the label of the reference as well to merge the contours
        labels_equally_oriented_as_ref.append(label_of_GPA_ref)
        
        # unite the contours for the equally oriented labels
        conts_vertx_per_region_unified = Segment.Unify_Contour_Vectors(
            conts_vertx_per_region, labels_equally_oriented_as_ref)
        
        # 1.1)
        # Loop through the regions and build the atomic model from each region
        # first in the regions that are not equally oriented, so work
        # with the base unit cells or virtual but adapt rotation
        for label_segm_region in range(1, analysed_image.crop_index):
            
            if label_segm_region not in labels_equally_oriented_as_ref:
                
                # build based on base or virtual not equally oriented
        
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
                    
                    
                    # Here check if there exists the virtual crystal of the found_phase_name
                    # and build the cell based on this or not if it does not exist
                    
                    cif_virtual_cell_filepath = model_cells_filepath + found_phase_name + '_v.cif'
                    
                    cif_virtual_cell_filepath_Path = Path(cif_virtual_cell_filepath)
                    cif_virtual_file_already_created = cif_virtual_cell_filepath_Path.is_file()
                    
                    if cif_virtual_file_already_created == True:
                        cif_cell_filepath = cif_virtual_cell_filepath
                    else:
                        cif_cell_filepath = model_cells_filepath + found_phase_name + '.cif'
                    
                    # Both the angle and atomistic model building needs to be done through 
                    # either the modified cif (virtual) or the base one
                    
                    # Find the rotation we need to induce to the default atomistic model 
                    # to rotate it to the found orientation
                    
                    
                    # if a virutal cell is used, so was built, use the rotation computed on
                    #     the plane of the reference
                    # if cif_cell_filepath ==   cif_virtual_cell_filepath:
                    #     final_in_surf_plane_rotation = .....
                    final_in_surf_plane_rotation = Adjust_in_surface_plane_rotation(
                        cif_cell_filepath, scored_spot_pair, suface_basis_choice = 'plane')
                    
                    # if there exists a virtual cell, build the cell based on it
                    # if not build based on base cell
                    # for all virtual crystals that are not the reference region ones                
                    if cif_cell_filepath == cif_virtual_cell_filepath and label_segm_region != label_of_GPA_ref:
                        
                        rotation_adjustment = Adjust_Virt_Cryst_Rotation_on_VirtualRef(
                            analysed_image, best_GPA_ref_spot_pair, label_of_GPA_ref, 
                            label_segm_region, 
                            model_cells_filepath, paths_to_virt_ucells, 
                            scored_spot_pairs_found)
                        
                        # adjust the rotation of the virtual crystal
                        final_in_surf_plane_rotation = final_in_surf_plane_rotation + rotation_adjustment
                    
                    # Save the models in the folder inside model_cells folder created with
                    # the name of the folder 
                    # function that does the full shaping and orientation 
                    
                    
                    # check that the cntour adjustment is the same format as the one
                    # which was with the trials in the previous checkings while 
                    # developing the atomistic_model_builder
                    
                    # it fails in the make supercell functoin make sure w ereally need the function
                    # that finds the optimal cubic shape as i think it deos more bad than good
                    
                    Build_shaped_atomistic(
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
                
        
        # 1.2)
        # Now build the single block for all the regions equally orietned 
        # as the reference region labeled as label_of_GPA_ref
        
        image_crop_hs_signal_ref = crop_outputs_dict[str(label_of_GPA_ref) + '_hs_signal']
        crop_list_refined_cryst_spots_ref = crop_outputs_dict[str(label_of_GPA_ref) + '_list_refined_cryst_spots']
        
        # most likely crystal found
        best_cryst_spot_ref = crop_list_refined_cryst_spots_ref[0]
        zone_axis_found_ref =  best_cryst_spot_ref.ZA
        scored_spot_pair_ref = best_cryst_spot_ref.spot_pairs_obj[0]
        hkl1_reference_ref = scored_spot_pair_ref.hkl1_reference
        hkl1_angle_to_x_ref = scored_spot_pair_ref.spot1_angle_to_x
        hkl2_reference_ref = scored_spot_pair_ref.hkl2_reference
        hkl2_angle_to_x_ref = scored_spot_pair_ref.spot2_angle_to_x
        found_phase_name_ref = best_cryst_spot_ref.phase_name
        
        
        # Here check if there exists the virtual crystal of the found_phase_name
        # and build the cell based on this or not if it does not exist
        
        cif_virtual_cell_filepath_ref = model_cells_filepath + found_phase_name_ref + '_v.cif'
        
        cif_virtual_cell_filepath_Path_ref = Path(cif_virtual_cell_filepath_ref)
        cif_virtual_file_already_created_ref = cif_virtual_cell_filepath_Path_ref.is_file()
        
        if cif_virtual_file_already_created_ref == True:
            cif_cell_filepath_ref = cif_virtual_cell_filepath_ref
        else:
            cif_cell_filepath_ref = model_cells_filepath + found_phase_name_ref + '.cif'
        
        # Both the angle and atomistic model building needs to be done through 
        # either the modified cif (virtual) or the base one
        
        # Find the rotation we need to induce to the default atomistic model 
        # to rotate it to the found orientation
        
        
        # if a virutal cell is used, so was built, use the rotation computed on
        #     the plane of the reference
        # if cif_cell_filepath ==   cif_virtual_cell_filepath:
        #     final_in_surf_plane_rotation = .....
        final_in_surf_plane_rotation_ref = Adjust_in_surface_plane_rotation(
            cif_cell_filepath_ref, scored_spot_pair_ref, suface_basis_choice = 'plane')
        
        # As it is the reference crystal we do not need to apply any correction
        # to the computed rotation                 
        
        # it fails in the make supercell functoin make sure w ereally need the function
        # that finds the optimal cubic shape as i think it deos more bad than good
        # 99 is tha label associated to merged regions
        label_unified = 99
        Build_shaped_atomistic(
            cif_cell_filepath_ref, zone_axis_found_ref, final_in_surf_plane_rotation_ref, 
            z_thickness_model, conts_vertx_per_region_unified, label_unified, 
            atom_models_filepath, adjust_y_bottomleft = True)
    
        # if the Unify_Contour_Vectors approach does not work, then :
        #     apply the contours with the all the labels
        #     need to modify the Build_shaped_atomistic
        #     to check the contours of more than one label
     
        # combine the cells altoghether to form the bigger single atomistic model    
        Combine_xyz_supercells(atom_models_filepath)
    
        # final_global_device_supcell = read(temp_xyz_files_folder_directory + 'global_device_supercell.xyz')
        # show_atoms(final_global_device_supcell, plane='xy')
        # show_atoms(final_global_device_supcell, plane='xz')
        # show_atoms(final_global_device_supcell, plane='yz')
            
    # 2)         
    else:   
        # 2) Building the device with all of virtual crystals separatedly directly from 
        # the beginning with the segmentation information, by compenseting
        # the crystal distances and angles of the spots surrounding the reference one    
    
        for label_segm_region in range(1, analysed_image.crop_index):
            
            # build based on base or virtual not equally oriented
    
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
                
                
                # Here check if there exists the virtual crystal of the found_phase_name
                # and build the cell based on this or not if it does not exist
                
                cif_virtual_cell_filepath = model_cells_filepath + found_phase_name + '_v.cif'
                
                cif_virtual_cell_filepath_Path = Path(cif_virtual_cell_filepath)
                cif_virtual_file_already_created = cif_virtual_cell_filepath_Path.is_file()
                
                if cif_virtual_file_already_created == True:
                    cif_cell_filepath = cif_virtual_cell_filepath
                else:
                    cif_cell_filepath = model_cells_filepath + found_phase_name + '.cif'
                
                # Both the angle and atomistic model building needs to be done through 
                # either the modified cif (virtual) or the base one
                
                # Find the rotation we need to induce to the default atomistic model 
                # to rotate it to the found orientation
                
                
                # if a virutal cell is used, so was built, use the rotation computed on
                #     the plane of the reference
                # if cif_cell_filepath ==   cif_virtual_cell_filepath:
                #     final_in_surf_plane_rotation = .....
                final_in_surf_plane_rotation = Adjust_in_surface_plane_rotation(
                    cif_cell_filepath, scored_spot_pair, suface_basis_choice = 'plane')
                
                # if there exists a virtual cell, build the cell based on it
                # if not build based on base cell
                # for all virtual crystals that are not the reference region ones                
                if cif_cell_filepath == cif_virtual_cell_filepath and label_segm_region != label_of_GPA_ref:
                    
                    rotation_adjustment = Adjust_Virt_Cryst_Rotation_on_VirtualRef(
                        analysed_image, best_GPA_ref_spot_pair, label_of_GPA_ref, 
                        label_segm_region, 
                        model_cells_filepath, paths_to_virt_ucells, 
                        scored_spot_pairs_found)
                    
                    # adjust the rotation of the virtual crystal
                    final_in_surf_plane_rotation = final_in_surf_plane_rotation + rotation_adjustment
                
                # Save the models in the folder inside model_cells folder created with
                # the name of the folder 
                # function that does the full shaping and orientation 
                
                
                # check that the cntour adjustment is the same format as the one
                # which was with the trials in the previous checkings while 
                # developing the atomistic_model_builder
                
                # it fails in the make supercell functoin make sure w ereally need the function
                # that finds the optimal cubic shape as i think it deos more bad than good
                
                Build_Atomistic_Parallelepiped(
                    cif_cell_filepath, zone_axis_found, final_in_surf_plane_rotation, 
                    z_thickness_model, conts_vertx_per_region, label_segm_region, 
                    atom_models_filepath, adjust_y_bottomleft = True)
                
                # The Build_Atomistic_Parallelepiped will generate a big supercell
                # not considering the segmentation and generate a file per label 
                

        
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
        
        # We do not combine the models in this case as they are 
        # parallelepipeds that are heavily overlapped to
        # have that extra space for the strain applications
            
    
    return atom_models_filepath, labels_equally_oriented_as_ref








# class Scored_spot_pair:
    
#     def __init__(self, ZA, spot1_angle_to_x, spot2_angle_to_x,
#                   hkl1_reference, hkl2_reference):
#         self.ZA = ZA
#         self.spot1_angle_to_x = spot1_angle_to_x
#         self.spot2_angle_to_x = spot2_angle_to_x
#         angle_between=np.abs(self.spot2_angle_to_x - self.spot1_angle_to_x)
#         if angle_between>180:
#             angle_between=360-angle_between

#         self.angle_between = angle_between
#         self.hkl1_reference = hkl1_reference
#         self.hkl2_reference = hkl2_reference
  
    
# cell_filepath=r'E:\Arxius varis\PhD\2nd_year\Code\unit_cells\moo2.cif'
# cell_filepath=r'E:\Arxius varis\PhD\2nd_year\Code\unit_cells\ge_fd-3ms.cif'
# cell_filepath=r'E:\Arxius varis\PhD\2nd_year\Code\unit_cells\inas_wz.cif'
# cell_filepath=r'E:\Arxius varis\PhD\3rd_year\Code\model_cells\InSb.cif'
# cell_filepath=r'E:\Arxius varis\PhD\2nd_year\Code\unit_cells\inp.cif'

# zone_axis = [0,1,1] 
# plane_1_found = [ 1, -1, 1]
# plane_2_found = [ 1, -1, 1]

# # plane_1_found = [ 1, -1,  1]
# # plane_2_found = [0, 0, 2]

# angle_exp_plane_1_to_x = 19.5789
# angle_exp_plane_2_to_x = -90
   
# # angle_exp_plane_1_to_x = 18.170101189194654
# # angle_exp_plane_2_to_x = -36.25383773744479


# scored_spot_pair = Scored_spot_pair(
#     zone_axis, angle_exp_plane_1_to_x, angle_exp_plane_2_to_x, 
#     np.asarray(plane_1_found), np.asarray(plane_2_found))
   

# final_in_surf_plane_rotation = Adjust_in_surface_plane_rotation(
#     cell_filepath, scored_spot_pair, suface_basis_choice = 'plane')

# print('angle to rotate the surface') 
# print(final_in_surf_plane_rotation)


# unit_cell = read(cell_filepath)

# unit_cell_oriented = surface(
#     unit_cell, indices=(zone_axis[0], zone_axis[1], zone_axis[2]), layers=10, periodic=True)

# unit_cell_oriented *= (3,3,3)

# unit_cell_oriented.rotate(final_in_surf_plane_rotation, 'z', rotate_cell= True)

# view(unit_cell_oriented)


# plane_final_cartesian_x, direction_final_cartesian_x = Find_plane_pointing_to_final_cartesian_x_axis(
#     cell_filepath, scored_spot_pair, tolerance_diff = 0.3, suface_basis_choice = 'plane')

# print('plane_final_cartesian_x')
# print(plane_final_cartesian_x)
# print('direction_final_cartesian_x')
# print(direction_final_cartesian_x)



#%%

# from ase.visualize import view
# from ase.lattice.cubic import FaceCenteredCubic



# # some test on zone axis and vector equivalences

# zone_axis_new = [1,1,2]
  
# cell_filepath=r'E:\Arxius varis\PhD\2nd_year\Code\unit_cells\moo2.cif'
# cell_filepath=r'E:\Arxius varis\PhD\2nd_year\Code\unit_cells\inas_wz.cif'
# cell_filepath=r'E:\Arxius varis\PhD\2nd_year\Code\unit_cells\ni.cif'
# cell_filepath=r'E:\Arxius varis\PhD\2nd_year\Code\unit_cells\ge_fd-3ms.cif'
# cell_filepath=r'E:\Arxius varis\PhD\2nd_year\Code\unit_cells\insb_full.cif'
# cell_filepath=r'E:\Arxius varis\PhD\3rd_year\Code\unit_cells\InSb.cif'
# cell_filepath=r'E:\Arxius varis\PhD\3rd_year\Code\unit_cells\insb_mid.cif'


# # load the cif file of the unit cell
# unit_cell = read(cell_filepath)


# # show atoms if wanted
# # view(unit_cell)
# show_atoms(unit_cell, plane='xy')
# show_atoms(unit_cell, plane='xy')
# show_atoms(unit_cell, plane='xz')
# show_atoms(unit_cell, plane='yz')
# # show_atoms(unit_cell*(2,2,2), plane='yz')

# real_lattice = unit_cell.get_cell().round(10)

# # get unit cell vectors
# a_1_p = real_lattice[0]
# a_2_p = real_lattice[1]
# a_3_p = real_lattice[2]

# # get reciprocal space vectors
# Vol = np.dot(a_1_p, np.cross(a_2_p, a_3_p))

# b_1 = np.cross(a_2_p, a_3_p)/Vol
# b_2 = np.cross(a_3_p, a_1_p)/Vol
# b_3 = np.cross(a_1_p, a_2_p)/Vol

# reciprocal_lattice = np.array([b_1, b_2, b_3])

# # get cell params to compute the metric tensors
# a, b, c, alfa, beta, gamma = unit_cell.cell.cellpar()
 
# direct_metric_tensor = np.array([[a**2, a*b*np.cos((np.pi/180)*(gamma)), a*c*np.cos((np.pi/180)*(beta))],
#                                   [b*a*np.cos((np.pi/180)*(gamma)), b**2, b*c*np.cos((np.pi/180)*(alfa))],
#                                   [c*a*np.cos((np.pi/180)*(beta)), c*b*np.cos((np.pi/180)*(alfa)), c**2]])

# reciprocal_metric_tensor = np.linalg.inv(direct_metric_tensor)

# plane_of_zone_axis = Find_plane_paralel_to_direction(
#     zone_axis_new, direct_metric_tensor, tolerance_diff = 0.5)

# # choose whether to use the plane or the direction, it should be the 
# # plane always, but just to have the option to choose        
# indices_for_surface = Use_plane_or_direction(
#     zone_axis_new, plane_of_zone_axis, choice = 'direction')

# c_basis, a_surf_vects = get_surface_basis(
#     unit_cell, indices_for_surface, 1, vacuum=None, 
#     tol=1e-10, periodic=False)

# (a1, a2, a3) = a_surf_vects 

# print(a1,a2,a3)


# unit_cell_oriented = surface(
#     unit_cell, indices=(zone_axis_new[0], zone_axis_new[1], zone_axis_new[2]), layers=10, periodic=True)

# unit_cell_oriented *= (3,3,3)

# unit_cell_oriented.rotate(54.28, 'z', rotate_cell= True)
# # view(unit_cell_oriented)


# show_atoms(unit_cell_oriented, plane= 'xy')
# show_atoms(unit_cell_oriented, plane= 'yz')
# show_atoms(unit_cell_oriented, plane= 'zx')




# # generate the face with the presetting by ase default function not general
# atoms = FaceCenteredCubic(directions=[[0,-2,2], [1,-1,-1], [2,1,1]],
#                           size=(4,4,4), symbol='Ni', pbc=(1,1,1))

# show_atoms(atoms, plane= 'xy')

# view(atoms)





