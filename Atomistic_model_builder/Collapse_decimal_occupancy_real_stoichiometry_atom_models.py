# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 12:26:05 2022

@author: mbotifoll
"""


import os
import numpy as np
import random
from pathlib import Path
from ase.io import read


    
filepath = r'E:\Arxius varis\PhD\4rth_year\Code\Atomistic_model_builder\Models_for_DFT\SQ21_160_7\Full_QW_Strained'
filename_cif = 'si_si70ge30_qw_5.4nm_full_strained.cif'
filename_xyz = 'si_si70ge30_qw_5.4nm_full_strained.xyz'

filename  = os.path.join(filepath, filename_xyz)
filename  = os.path.join(filepath, filename_cif)








if filename[-4:]== '.cif':
    # cif formatting 
    

    cif_text = open(filename, mode='r')
    
    textlines = cif_text.readlines()
    
    
    def find_1st_Atom_line_cif(line):
        # as the cif files always act with the relative positions within the supercell
        # we can search for the 0. as the pattern exclusive of atom lines
        
        first_found = line.find('0.')
        
        if first_found == -1:
            bool_f = False
            return bool_f
        else:
        
            new_found = line[first_found + 2:].find('0.')
            
            if new_found == -1:
                bool_f = False
            else:
                bool_f = True
            
            return bool_f
    
    
    first_atom_line = list(map(find_1st_Atom_line_cif, textlines))
    
    atom_lines = [line for line, bool_f in zip(textlines, first_atom_line) if bool_f == True]
    no_atom_lines = [line for line, bool_f in zip(textlines, first_atom_line) if bool_f == False]
    
    
    for line_index, line in enumerate(atom_lines):
        
        
        element = line[:line.find('  ')]
        
        
        temp_line_mod = line[line.find('0.')+2:]
        temp_line_mod = temp_line_mod[temp_line_mod.find('0.')+2:]
        temp_line_mod = temp_line_mod[temp_line_mod.find('0.')+2:]
        
    
        temp_line_mod = temp_line_mod[temp_line_mod.find(' ')+2:]
        
        position_string = line[line.find('0.'):line.find(temp_line_mod)]
        
        occupancy = temp_line_mod[:temp_line_mod.find('  ')]
        
    
        if float(occupancy) == 1:
            continue
        
        else:
            # fractional occupancy, need to collapse            
            
            lines_same_pos = list(filter(lambda x: x.find(position_string)!=-1, atom_lines))
            
            
            occupancies = []
            
            for line_samepos in lines_same_pos:
                
                
                element = line_samepos[:line_samepos.find('  ')]
                
                
                temp_line_mod = line_samepos[line_samepos.find('0.')+2:]
                temp_line_mod = temp_line_mod[temp_line_mod.find('0.')+2:]
                temp_line_mod = temp_line_mod[temp_line_mod.find('0.')+2:]
                
            
                temp_line_mod = temp_line_mod[temp_line_mod.find(' ')+2:]
                
                position_string = line_samepos[line_samepos.find('0.'):line_samepos.find(temp_line_mod)]
                
                occupancy = temp_line_mod[:temp_line_mod.find('  ')]
                
                occupancies.append(float(occupancy))
               
                
            line_chosen = random.choices(lines_same_pos, weights = occupancies, k=1)[0] 
            
    
            
            
            element = line_chosen[:line_chosen.find('  ')]
            
            
            temp_line_mod = line_chosen[line_chosen.find('0.')+2:]
            temp_line_mod = temp_line_mod[temp_line_mod.find('0.')+2:]
            temp_line_mod = temp_line_mod[temp_line_mod.find('0.')+2:]
            
        
            temp_line_mod = temp_line_mod[temp_line_mod.find(' ')+2:]
            
            position_string = line_chosen[line_chosen.find('0.'):line_chosen.find(temp_line_mod)]
            
            occupancy = temp_line_mod[:temp_line_mod.find('  ')]
            
            line_chosen_mod = line_chosen[:line_chosen.find(occupancy)] + '1.000000' + temp_line_mod[temp_line_mod.find('  '):]
    
            atom_lines[line_index] = line_chosen_mod
            
            # delete the past ones, one is already overwriten
            for line_del in lines_same_pos:
                if line_del in atom_lines:
                    atom_lines.remove(line_del)
            
        
    colaps_cif_info = no_atom_lines[:-1] + atom_lines + [no_atom_lines[-1]]
    
    
    # create the .cif file
    col_cif_file = filename[:-4] + '_unit_occ.cif'
    
    col_cif_file_Path = Path(col_cif_file)
    file_already_created = col_cif_file_Path.is_file()
    
    if file_already_created == True:
        # overwrite the file
        with open(col_cif_file, "w+") as f:
            f.truncate(0)
            f.writelines(colaps_cif_info)
            f.close()
    else:
        # create a new file
        with open(col_cif_file, 'w+') as f:
            f.writelines(colaps_cif_info)
            f.close()    
        
        
    
elif filename[-4:] == '.xyz':
    # format of .xyz is first line either total number for atoms
    # second nothing or the supercell dimensions
    # from third we can find the total number of atoms
    
    xyz_text = open(filename, mode='r')
    
    textlines = xyz_text.readlines()
    
    
    
    atom_lines = textlines[2:-2]
    no_atom_lines = textlines[:2] + [textlines[-1]]
    
    atom_lines_split = [line.split() for line in atom_lines]
    
    for line_index, (line, line_split) in enumerate(zip(atom_lines, atom_lines_split)):
        
        element = line_split[0]
        x_coord = line_split[1]
        z_coord = line_split[3]
        occupancy = line_split[4]
        
        if float(occupancy) == 1:
            continue
        
        else:
            # fractional occupancy, need to collapse
            position_string = line[line.find(x_coord):line.find(z_coord)+len(z_coord)]

            lines_same_pos = list(filter(lambda x: x.find(position_string)!=-1, atom_lines))
            
            occupancies = []
            
            for line_samepos in lines_same_pos:
                
                line_samepos_split = line_samepos.split()
                element = line_split[0]
                occupancy = line_split[4] 
                
                occupancies.append(float(occupancy))
               
            line_chosen = random.choices(lines_same_pos, weights = occupancies, k=1)[0] 
            line_chosen_split = line_chosen.split()
            
            element = line_chosen_split[0]
            occupancy = line_chosen_split[4] 
            x_coord = line_chosen_split[1]
            
            line_chosen_mod = line_chosen[:line_chosen.find(occupancy)] + '1.000000' + line_chosen[line_chosen.find(occupancy)+len(occupancy):]
            atom_lines[line_index] = line_chosen_mod
                        
            # delete the past ones, one is already overwriten
            for line_del in lines_same_pos:
                if line_del in atom_lines:
                    atom_lines_split.pop(atom_lines.index(line_del))
                    atom_lines.remove(line_del)    
                        
    colaps_xyz_info = [str(len(atom_lines))+' \n'] + [no_atom_lines[1]] + atom_lines + [no_atom_lines[-1]]
    
    # create the .xyz file
    col_xyz_file = filename[:-4] + '_unit_occ.xyz'
    
    col_xyz_file_Path = Path(col_xyz_file)
    file_already_created = col_xyz_file_Path.is_file()
    
    if file_already_created == True:
        # overwrite the file
        with open(col_xyz_file, "w+") as f:
            f.truncate(0)
            f.writelines(colaps_xyz_info)
            f.close()
    else:
        # create a new file
        with open(col_xyz_file, 'w+') as f:
            f.writelines(colaps_xyz_info)
            f.close()    
    
else:
    raise Exception('File format not supported, select .cif or .xyz file')
    




    
    
    
    
    
    


