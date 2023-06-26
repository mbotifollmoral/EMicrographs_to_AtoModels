# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 17:26:34 2021

@author: Marc
"""

'''
#############
File conversors for the applications
#############
'''

import numpy as np	
import os
from ase import data as asedata



def comp_xyzQSTEM_to_Prismatic_add_R_and_temp(filename):
    '''
    Function to make the output from QSTEM model builder a compatible input for 
    Prismatic software/Pyprismatic: basically setting the second comment line as
    the cell's dimensions and change atomic symbol to atomic number (by ASE)
    Filename is the only input, and is a .xyz file directory
    Returns a .xyz, capable to be read by Prismatic/PyPrismatic
    Add occupancy and RMS thermal vibration displacement, which are typically 1 and
    a value between 0.05 A and 0.1 A. As the value is unkown if it is not provided,
    we set occupancy 1 and 0.075
    '''
    file_real=open(filename)
    data_real=file_real.read()
    list_lines=data_real.splitlines()
    start=list_lines[1].find('lim')
    end=list_lines[1].find('--')
    if end !=-1:
        list_lines[1]=list_lines[1][end+2:]
    new_list_line=list_lines[:2]
    for index, element in enumerate(list_lines[2:]):
        element_=element
        el_line=element.split()
        atomic_number=str(asedata.atomic_numbers[el_line[0]])
        new_el=atomic_number+element_[len(el_line[0]):]
        new_list_line.append(new_el)
    final_string=''
    file_real.close()
    for line in new_list_line[0:2]:
        final_string=final_string+line+'\n'
    for line in new_list_line[2:-1]:
        final_string=final_string+line+'1 0.075 '+'\n'
    final_string=final_string+new_list_line[-1]+'1 0.075 '    
    os.remove(filename)
    pre, ext = os.path.splitext(filename)
    filename_temp=pre+'.txt'
    
    file=open(filename_temp,'w+')
    file.write(final_string)
    file.close()
    
    pre, ext = os.path.splitext(filename_temp)
    new_extension='.xyz'
    finalname=pre+new_extension
    os.rename(filename_temp, finalname)
    
    return 





def ascii_cel_to_uce(filename):  
    file_real=open(filename)     

    
    data_real=file_real.read()
    list_lines=data_real.splitlines()
    print(data_real)
    print('empieza list lines')
    print(list_lines)
    
    list_lines=list_lines[11::]
    print(list_lines)

    #let us find a
    
    start=list_lines[0].find('=')
    end=list_lines[1].find('b')
    
    el_line=list_lines[0].split()
    print(el_line)
    print(el_line[0])
    a=el_line[0]
    a=a[2::]
    print(a)
    print(type(a))
    
    print(len(a))
    
    
    a=float(a)
    print(a)
    print(type(a))
    
    a_final_doc=str(a)
    print(type(a_final_doc))
    
    2nd_line='CELL '+a_final_doc
    print(2nd_line)
    
    
    
    
    
    
    
    


    
    fileout=0
    return fileout




filename=r'E:\Arxius varis\PhD\2nd_year\Code\unit_cells\Ge_SG.ACE'
'''
file_real=open(filename, encoding='ansi')
data_real=file_real.read()
list_lines=data_real.splitlines()
print(file_real)

print(data_real)

print(list_lines)
'''


fileout=ascii_cel_to_uce(filename)
print(fileout)
