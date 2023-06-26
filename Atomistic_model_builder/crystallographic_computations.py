# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 16:34:54 2022

@author: mbotifoll
"""





import numpy as np
import matplotlib.pyplot as plt
import os
import math

import ase
from ase.io import read, write
from ase.visualize import view
from ase.build import surface, make_supercell, find_optimal_cell_shape, rotate


from math import gcd
import numpy as np
from numpy.linalg import norm, solve

from ase.build import bulk



cell_filepath=r'E:\Arxius varis\PhD\2nd_year\Code\unit_cells\inas_wz.cif'
cell_filepath=r'E:\Arxius varis\PhD\2nd_year\Code\unit_cells\moo2.cif'
cell_filepath=r'E:\Arxius varis\PhD\2nd_year\Code\unit_cells\ge_fd-3ms.cif'



# load the cif file of the unit cell
unit_cell = read(cell_filepath)
# show atoms if wanted

real_lattice = unit_cell.get_cell().round(10)

# test angles

a_1_p = real_lattice[0]
a_2_p = real_lattice[1]
a_3_p = real_lattice[2]

# get reciprocal space vectors

Vol = np.dot(a_1_p, np.cross(a_2_p, a_3_p))

b_1 = np.cross(a_2_p, a_3_p)/Vol
b_2 = np.cross(a_3_p, a_1_p)/Vol
b_3 = np.cross(a_1_p, a_2_p)/Vol

reciprocal_lattice = np.array([b_1, b_2, b_3])


print(real_lattice)

a, b, c, alfa, beta, gamma = unit_cell.cell.cellpar()
 

print(a,b,c,alfa,beta,gamma)


direct_metric_tensor = np.array([[a**2, a*b*np.cos((np.pi/180)*(gamma)), a*c*np.cos((np.pi/180)*(beta))],
                                 [b*a*np.cos((np.pi/180)*(gamma)), b**2, b*c*np.cos((np.pi/180)*(alfa))],
                                 [c*a*np.cos((np.pi/180)*(beta)), c*b*np.cos((np.pi/180)*(alfa)), c**2]])


reciprocal_metric_tensor = np.linalg.inv(direct_metric_tensor)

print('direct_metric_tensor')
print(direct_metric_tensor)


print('reciprocal_metric_tensor')
print(reciprocal_metric_tensor)


plane_1 = [-4, 4, 4]


plane_2 = [2, -2, 4]


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
    
    
    g1_g2_dot = np.dot(plane_1, np.dot(reciprocal_metric_tensor, plane_2))
    
    g1_norm = np.sqrt(np.dot(plane_1, np.dot(reciprocal_metric_tensor, plane_1)))
    g2_norm = np.sqrt(np.dot(plane_2, np.dot(reciprocal_metric_tensor, plane_2)))
    
    
    angle_between = (180/np.pi)*np.arccos(g1_g2_dot/(g1_norm*g2_norm))
    
    
    return angle_between



def Find_plane_paralel_to_direction(
        direction, direct_metric_tensor, tolerance_diff = 0.5):
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
            2*np.array(direction), direct_metric_tensor, tolerance_diff = 0.5)
    else:
        print('before reduction plane')
        print(plane_paralel_int)
        gcd_1 = math.gcd(int(plane_paralel_int[0]), int(plane_paralel_int[1]))
        gcd_2 = math.gcd(int(plane_paralel_int[1]), int(plane_paralel_int[2]))
        gcd_3 = math.gcd(int(plane_paralel_int[0]), int(plane_paralel_int[2]))
        
        gcds = np.array([gcd_1, gcd_2, gcd_3])
                
        gcd = np.min(gcds[gcds != 0])
        
        plane_paralel_int = plane_paralel_int/gcd
        
        return plane_paralel_int
    



angle_between = angle_between_planes(
    plane_1, plane_2, reciprocal_metric_tensor)



print('angle between')
print(angle_between)


direction = [1,0,0]



plane_paralel = Find_plane_paralel_to_direction(
    direction, direct_metric_tensor, tolerance_diff = 0.5)




print('plane_paralel')
print(plane_paralel)




# check if the vectors are really paralel

def Ensure_plane_to_direction_is_0(
        plane, direction, direct_lattice, reciprocal_lattice):
    
    '''
    Verify that the angle between the plane and the direction is really 0 
    
    '''
    
    g_vector_paralel = np.dot(plane, reciprocal_lattice)
    g_vector_paralel_comps = np.sum(g_vector_paralel, axis = 0)
    
    direction_vector = np.dot(direction, direct_lattice)
    direction_vector_comps = np.sum(direction_vector, axis = 0)
    
    
    angle_plane_direction = (180/np.pi)*np.arccos((np.dot(g_vector_paralel_comps,direction_vector_comps))/(norm(g_vector_paralel_comps)*norm(direction_vector_comps)))

    return angle_plane_direction



angle_plane_direction = Ensure_plane_to_direction_is_0(
    plane_paralel, direction, real_lattice, reciprocal_lattice)
    
print('angle_plane_direction')
print(angle_plane_direction)




