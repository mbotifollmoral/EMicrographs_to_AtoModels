# -*- coding: utf-8 -*-
"""
Created on Thu Feb 17 18:34:04 2022

@author: Marc
"""


import numpy as np
import ctypes
from sympy.utilities.iterables import multiset_permutations, permute_signs
import os
import sys

sys.path.append(r'E:\Arxius varis\PhD\3rd_year\Code\Functions')
sys.path.append(r'E:\Arxius varis\PhD\4rth_year\Code\Functions\General_functions')

# General functions
import ImageCalibTransf as ImCalTrans
import Peak_Finding_Wrapped as PeakFindWrap
import File_Compatibility as FileComp



# ZA Finding Wrapper

#path to the dll
lib = ctypes.CDLL(r"E:\Arxius varis\PhD\3rd_year\Code\Functions\difftool_dll\diffTools.dll")

CrystalHandle = ctypes.POINTER(ctypes.c_char)
c_int_array = np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS')
c_int_array_2 = np.ctypeslib.ndpointer(dtype=np.int32, ndim=2, flags='C_CONTIGUOUS')
c_float_array=np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags='C_CONTIGUOUS')
c_double_array=np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS')



lib.createCrystal.argtypes = [ctypes.c_char_p]
lib.createCrystal.restype = CrystalHandle

lib.calc_d.argtypes = [CrystalHandle, ctypes.c_bool, ctypes.c_float]
lib.calc_d.restypes = ctypes.c_int

lib.FindZA.argtypes = [CrystalHandle, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float]
lib.FindZA.restype = ctypes.c_int

lib.GetZA.argtypes = [CrystalHandle, ctypes.c_int, c_int_array]
lib.GetZA.restype = None

#custom for getting the possible indexation of the pair of spots considered
lib.Gethkls1.argtypes = [CrystalHandle, ctypes.c_int, c_int_array]
lib.Gethkls1.restype = None

lib.Gethkls2.argtypes = [CrystalHandle, ctypes.c_int, c_int_array]
lib.Gethkls2.restype = None

lib.destroyCrystal.argtypes = [CrystalHandle]
lib.destroyCrystal.restype = None

lib.angle.argtypes = [CrystalHandle, c_int_array, c_int_array]
lib.angle.restype = ctypes.c_float

lib.getF.argtypes = [CrystalHandle, ctypes.c_int]
lib.getF.restype = ctypes.c_double

lib.getDistances.argtypes = [CrystalHandle, ctypes.c_int]
lib.getDistances.restype = ctypes.c_float

lib.getIndexes.argtypes = [CrystalHandle, ctypes.c_int, c_int_array]
lib.getIndexes.restype = None



class Crystal:
      
    def __init__(self,name):
        self.instance = lib.createCrystal(name)
        self.phase_name=name
    
    def __del__(self):
        lib.destroyCrystal(self.instance)
    
    def getZA(self,N):
        n = ctypes.c_int(N)
        hkl = np.empty(3, dtype=np.int32)
        lib.GetZA(self.instance,n,hkl)
        return hkl
    
    def gethkls1(self,N):
        n = ctypes.c_int(N)
        hkl = np.empty(3, dtype=np.int32)
        lib.Gethkls1(self.instance,n,hkl)
        return hkl
    
    def gethkls2(self,N):
        n = ctypes.c_int(N)
        hkl = np.empty(3, dtype=np.int32)
        lib.Gethkls2(self.instance,n,hkl)
        return hkl
        
    def Diff(self,flag,D):
        min_d = ctypes.c_float(D)
        flagd=ctypes.c_bool(flag)
        N = lib.calc_d(self.instance,flagd,min_d)
        return N
    
    def FindZA(self,D1,D2,ANG,TOL):
        d1=ctypes.c_float(D1)
        d2=ctypes.c_float(D2)
        ang=ctypes.c_float(ANG)
        tol=ctypes.c_float(TOL)
        self.n = lib.FindZA(self.instance,d1,d2,ang,tol)
        return self.n
    
    def angle(self, hkl, hkl2):        
        angle = lib.angle(self.instance,hkl,hkl2)
        return angle
    
    def getIndexes(self,N):
        n = ctypes.c_int(N)
        hkl = np.empty(3, dtype=np.int32)
        lib.getIndexes(self.instance,n,hkl)
        return hkl
    
    def getF(self,N):
        n = ctypes.c_int(N)
        F = ctypes.c_double()
        F = lib.getF(self.instance,n)
        return F    
    
    def getDistances(self,N):
        n = ctypes.c_int(N)
        d = ctypes.c_float()
        d = lib.getDistances(self.instance,n)
        return d  
    

#  Functions for prepearing the ZA indexation


def Spot_coord_To_d_spacing_vect(coord_vects, FFT_calibration, FFT_pixels):
    # output distances in nm
    y_vects=coord_vects[:,0]    
    x_vects=coord_vects[:,1] 

    FFT_distance_point_x=np.abs(x_vects-int(FFT_pixels/2))*FFT_calibration
    FFT_distance_point_y=np.abs(y_vects-int(FFT_pixels/2))*FFT_calibration
    
    FFT_distance_total=np.sqrt(FFT_distance_point_x**2+FFT_distance_point_y**2)
    
    FFT_distance_total[FFT_distance_total==0]=0.0001
    
    d_spacing_spot=1/FFT_distance_total
    
    return d_spacing_spot



def Spot_coord_To_Angles_to_X_vect(coord_vects,FFT_pixels):
    y_vects=coord_vects[:,0]    
    x_vects=coord_vects[:,1] 
    
    cont_dist=x_vects-int(FFT_pixels/2)
    opp_dist=int(FFT_pixels/2)-y_vects
    
    angles_to_X=np.arctan2(opp_dist,cont_dist)*180/np.pi
    
    return angles_to_X




def Ensure_Center_Diff(distances_array, angles_to_x_array, pixels_array):
    #input distances in nm
    #define hyperparameter, that should not be very modifyable, as it is the 
    #maximum interplanar distance to consider, in nm, let us say d_int > 1-1.5nm
    #as no plane should be  bigger than 1-1.5nm. Let us say 1.5 to include 1/2 indices 
    
    #do the same but delete the center of the pixels array
    angles_to_x_array_c=angles_to_x_array[distances_array<=1.5]
    pixels_array_c=pixels_array[distances_array<=1.5]    
    distances_array_c=distances_array[distances_array<=1.5]

    
    return distances_array_c, angles_to_x_array_c, pixels_array_c


def Prepare_exp_distances_angles_pixels(
        refined_distances_exp, refined_angles_exp, refined_pixels_exp, min_d):
    '''
    Needs the  refined distances, without the center of diffraction. It is not essential, but it is 
    preferred that it has already been extracted: As the process of extracting the center of the diffraction
    also takes everything that is bigger than 1.5nm out =15 A-> if d>15 A then it is removed (in case it 
    needed this distance can be increased if a very big unit cell is involved)
    IMPORTANT: The units must be in angstroms, as min_d is in Angrstoms
    Takes all the distances smaller than mind and deletes them.
    Then sort both axis from max to min
    '''
    
    more_refined_distances_exp=refined_distances_exp[refined_distances_exp>min_d]
    more_refined_angles_exp=refined_angles_exp[refined_distances_exp>min_d]
    more_refined_pixels_exp=refined_pixels_exp[refined_distances_exp>min_d]
    
    more_refined_angles_exp_f=[angle for _,angle in sorted(zip(more_refined_distances_exp,more_refined_angles_exp), key=lambda pair: pair[0])][::-1]
    more_refined_pixels_exp_f=[pixels for _,pixels in sorted(zip(more_refined_distances_exp,more_refined_pixels_exp), key=lambda pair: pair[0])][::-1]
    more_refined_distances_exp_f=sorted(more_refined_distances_exp)[::-1]
      
    more_refined_pixels_exp_f=np.int32(more_refined_pixels_exp_f)

    return more_refined_distances_exp_f, more_refined_angles_exp_f, more_refined_pixels_exp_f


# Functions for finding the ZA within an image

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
 
    
'''
Phase identification functions in the combined algorihtm 1st aprox that 
already have the crystal system independence/consideration
'''

# Functions to modify with the symmetry considerations:
  
def ZAs_scores_CUBIC(crystal_object, d1,d2,angle,ZAs, hkls1, hkls2, distances_arr, reflections_arr):
    
    #only consider this as necessary if extreme as from the axis the respective distance should be computed,
    #as maybe more than one interplanar distance fitted the tolerance condition and then not only the 
    #distance closest to the experimental was considered
    #distance theo should be extracted then from the indexes and then check which distance fits this index
    # d1_theo=distances_arr[np.argmin(np.abs(distances_arr-d1))]
    # d2_theo=distances_arr[np.argmin(np.abs(distances_arr-d2))]
    
    score=[]
    for ZA, hkl1, hkl2, in zip(ZAs,hkls1,hkls2):
        #if cubic, the assumption can be that any permutation has the same distances
        #if not cubic, then the changes in the permutations and signs is different
        hkl1_family=[]
        hkl1_family_temp=[permutation_1 for permutation_1 in multiset_permutations(hkl1)]
        for permutation in hkl1_family_temp:
            els= list(permute_signs(tuple(permutation)))
            
            for el in els:
                hkl1_family.append(np.asarray(el))
                                                                    
        hkl2_family=[]
        hkl2_family_temp=[permutation_2 for permutation_2 in multiset_permutations(hkl2)]
        for permutation in hkl2_family_temp:
            els= list(permute_signs(tuple(permutation)))
            for el in els:
                hkl2_family.append(np.asarray(el))                       

        #check all the possible permutations for both hkl1 and hkl2 and get the distance associated to this index
        # !!! CUBIC assumption                  
                
        for index_refl, possible_refl in enumerate(reflections_arr):
            href=possible_refl[0]
            kref=possible_refl[1]
            lref=possible_refl[2]
            
            for permutation1 in hkl1_family:
                h1=permutation1[0]
                k1=permutation1[1]
                l1=permutation1[2]
                
                if h1==href and k1==kref and l1==lref:
                    d1_theo=distances_arr[index_refl]
            
            for permutation2 in hkl2_family:
                h2=permutation2[0]
                k2=permutation2[1]
                l2=permutation2[2]
                
                if h2==href and k2==kref and l2==lref:
                    d2_theo=distances_arr[index_refl]
                    
        angle_theo=abs(crystal_object.angle(hkl1,hkl2))
        score_val=np.sqrt((d1-d1_theo)**2+(d2-d2_theo)**2+(angle-angle_theo)**2)
        score.append(score_val)
    
    ZA_score=np.asarray(score)
    #sort the ZA array given the score that has been given, put the lowest score on top as the most important one
    ZA_sorted= [ZA for _,ZA in sorted(zip(ZA_score,ZAs), key=lambda pair: pair[0])]
    ZA_score=sorted(ZA_score)
    
    return ZA_score, ZA_sorted



def Crystal_System_Equidistance_Permutations(
        space_group, reflection_indices):
    '''
    Given a space group number, int, and a list of miller indices, reflection,
    compute all the possible equivalent reflection permutations that would
    lead to the same exact interplanar distance  (value dependent on 
    the crystal system (1 out o 7))
    
    Parameters
    ----------
    space_group : int
    reflection_indices : list of int

    Returns
    -------
    list_of_permutations : list of lists of int

    '''
    # collect the list of permutations
    list_of_permutations = []
    
    h_init, k_init, l_init = reflection_indices
    
    
    # identify the crystal system of interest
    
    if space_group <= 2:
        # triclinic
        # only trivial negative allowed
        list_of_permutations.append(list(reflection_indices))
        list_of_permutations.append([-h_init, -k_init ,-l_init])
        
    elif space_group > 2 and space_group <= 15:
        # monoclinic
        # trivial negative
        list_of_permutations.append(list(reflection_indices))
        list_of_permutations.append([-h_init, -k_init ,-l_init])
        # only h and l paired signs change
        list_of_permutations.append([-h_init, k_init ,-l_init])
        list_of_permutations.append([h_init, -k_init ,l_init])
        
    elif space_group > 15 and space_group <= 74:
        # orthorombic
        # any sign change allowed but no indices swap
        list_of_permutations = list(permute_signs(list(reflection_indices)))
        
    elif space_group > 74 and space_group <= 142:
        # tetragonal
        # swap h and k and then any sign permutation
        list_of_permutations = list(permute_signs(list(reflection_indices)))
        list_of_permutations = list_of_permutations + list(permute_signs([k_init, h_init, l_init]))
        
    elif space_group > 142 and space_group <= 167:
        # trigonal/rhombohedral
        # any order swap
        list_of_permutations = list(multiset_permutations(list(reflection_indices)))
        # and get the trivial sign inversion for all of them
        negative_reflections = []
        for reflection in list_of_permutations:
            h_i, k_i, l_i = reflection
            negative_reflections.append([-h_i, -k_i, -l_i])
        
        list_of_permutations = list_of_permutations + negative_reflections
        
    elif space_group > 167 and space_group <= 194 :
        # hexagonal
        # trivial negative
        
        list_of_permutations.append(list(reflection_indices))
        list_of_permutations.append([-h_init, -k_init ,-l_init])
        
        # h and k swap and paired signs change  
        list_of_permutations.append([-h_init, -k_init ,l_init])
        list_of_permutations.append([h_init, k_init ,-l_init])
        
        list_of_permutations.append([k_init, h_init, l_init])
        list_of_permutations.append([-k_init, -h_init, -l_init])
        
        list_of_permutations.append([-k_init, -h_init, l_init])
        list_of_permutations.append([k_init, h_init, -l_init])
        
        # extra planes of family obtained by miller bravais indices permutation
        # h,k,i,l, swap h,k and i positions
        i = -(h_init + k_init)
        
        # h and i swap and paired signs change  
        list_of_permutations.append([h_init, i, l_init])
        list_of_permutations.append([-h_init, -i ,-l_init])
        
        list_of_permutations.append([-h_init, -i ,l_init])
        list_of_permutations.append([h_init, i ,-l_init])
        
        list_of_permutations.append([i, h_init, l_init])
        list_of_permutations.append([-i, -h_init, -l_init])
        
        list_of_permutations.append([-i, -h_init, l_init])
        list_of_permutations.append([i, h_init, -l_init])
        
        # k and i swap and paired signs change  
        list_of_permutations.append([i, k_init, l_init])
        list_of_permutations.append([-i, -k_init ,-l_init])
        
        list_of_permutations.append([-i, -k_init ,l_init])
        list_of_permutations.append([i, k_init ,-l_init])
        
        list_of_permutations.append([k_init, i, l_init])
        list_of_permutations.append([-k_init, -i, -l_init])
        
        list_of_permutations.append([-k_init, -i, l_init])
        list_of_permutations.append([k_init, i, -l_init])
        
        
    else:  
        # space_group > 194 and space_group <= 230
        # cubic
        # all possible permutations allowed
        list_of_permutations_temp = list(multiset_permutations(reflection_indices))
        for permutation in list_of_permutations_temp:
            refl = list(permute_signs(permutation))
            list_of_permutations = list_of_permutations + refl
    
    # ensure all elements are unique
    list_of_permutations = [list(el) for el in np.unique(np.asarray(list_of_permutations), axis = 0)]
    return list_of_permutations





def ZAs_scores(
        crystal_object, space_group_num, d1, d2, angle, 
        ZAs, hkls1, hkls2, distances_arr, reflections_arr):
    
    #only consider this as necessary if extreme as from the axis the respective distance should be computed,
    #as maybe more than one interplanar distance fitted the tolerance condition and then not only the 
    #distance closest to the experimental was considered
    #distance theo should be extracted then from the indexes and then check which distance fits this index
    # d1_theo=distances_arr[np.argmin(np.abs(distances_arr-d1))]
    # d2_theo=distances_arr[np.argmin(np.abs(distances_arr-d2))]
    score=[]
    hkls1_references = []
    hkls2_references = []
    
    for ZA, hkl1, hkl2, in zip(ZAs,hkls1,hkls2):

        # depending on the crystal system we have, compute the possible 
        # permutations of hkl miller indices that can lead to different 
        # interplanar spacings
        
        hkl1_permutations = Crystal_System_Equidistance_Permutations(
            space_group_num, hkl1)
        hkl2_permutations = Crystal_System_Equidistance_Permutations(
            space_group_num, hkl2)
        
        # check all the possible permutations for both hkl1 and hkl2 
        # and get the distance associated to this index
                 
        # if the reflections array and distances array is reduced to 1
        # repeated distance and its corresponding relfection, 
        # then we get only the same reflection for every distance  
        ident1 = 0
        ident2 = 0
        
        for index_refl, possible_refl in enumerate(reflections_arr):
            href = possible_refl[0]
            kref = possible_refl[1]
            lref = possible_refl[2]
            
            for permutation1 in hkl1_permutations:
                h1 = permutation1[0]
                k1 = permutation1[1]
                l1 = permutation1[2]
                
                if h1==href and k1==kref and l1==lref:
                    d1_theo = distances_arr[index_refl]
                    # the reflection in reflections_arr that is inside all
                    # the possible permutations given the crystal system
                    hkl1_reference = possible_refl
                    ident1 = 1
                    break
                
            if ident1 == 1:
                break
            
        for index_refl, possible_refl in enumerate(reflections_arr):
            href = possible_refl[0]
            kref = possible_refl[1]
            lref = possible_refl[2]
 
            for permutation2 in hkl2_permutations:
                h2 = permutation2[0]
                k2 = permutation2[1]
                l2 = permutation2[2]
                
                if h2==href and k2==kref and l2==lref:
                    d2_theo = distances_arr[index_refl]
                    # the reflection in reflections_arr that is inside all
                    # the possible permutations given the crystal system
                    hkl2_reference = possible_refl
                    ident2 = 1
                    break
                    
            if ident2 == 1:
                break
                    
                    
        # the planes being the hkl1, but as depicted by the
        # firsly obtained distance and indices/reflections array obtained
        hkls1_references.append(hkl1_reference)
        hkls2_references.append(hkl2_reference)
        
        angle_theo = abs(crystal_object.angle(hkl1,hkl2))
        score_val = np.sqrt((d1-d1_theo)**2+(d2-d2_theo)**2+(angle-angle_theo)**2)
        score.append(score_val)
    
    ZA_score = np.asarray(score)
    #sort the ZA array given the score that has been given, put the lowest score on top as the most important one
    ZA_sorted= [ZA for _,ZA in sorted(zip(ZA_score,ZAs), key=lambda pair: pair[0])]
    hkls1_references_sorted = [hkl1 for _,hkl1 in sorted(zip(ZA_score,hkls1_references), key=lambda pair: pair[0])]
    hkls2_references_sorted = [hkl2 for _,hkl2 in sorted(zip(ZA_score,hkls2_references), key=lambda pair: pair[0])]
    ZA_score=sorted(ZA_score)
    
    return ZA_score, ZA_sorted, hkls1_references_sorted, hkls2_references_sorted



class Scored_spot_pair:
    
    def __init__(self, spot1_int_ref, spot2_int_ref, crystal_object):
        self.spot1_int_ref=spot1_int_ref
        self.spot2_int_ref=spot2_int_ref
        self.phase=crystal_object
        
        phase_name_temp=crystal_object.phase_name
        strin=str(phase_name_temp)
        start=strin[::-1].find(os.path.sep)
        end=strin[::-1].find('.')
        phase_mat_name=strin[-start:-end-1]
        self.phase_name=phase_mat_name
        
        
    def Spot_distances_angles(self, refined_distances, refined_angles_to_x):
        self.spot1_dist=refined_distances[self.spot1_int_ref]
        self.spot2_dist=refined_distances[self.spot2_int_ref]
        self.spot1_angle_to_x=refined_angles_to_x[self.spot1_int_ref]
        self.spot2_angle_to_x=refined_angles_to_x[self.spot2_int_ref]
        angle_between=np.abs(self.spot2_angle_to_x - self.spot1_angle_to_x)
        if angle_between>180:
            angle_between=360-angle_between
        self.angle_between=angle_between
        
    def Zone_axis(self, ZA, hkl1_reference, hkl2_reference):
        self.ZA = ZA
        self.hkl1_reference = hkl1_reference
        self.hkl2_reference = hkl2_reference

        
    def Score(self, score):
        self.score=score
    
       
'''   
# Deprecated function used at first to work with cubic systems
'''   
  
def Score_all_spot_pairs_CUBIC(crystal_objects,forbidden, min_d, tolerance, refined_distances, refined_angles_to_x):
    
    all_scored_spot_pairs=[]
    
    spots_int_reference_changes=np.arange(len(refined_distances))
    
    #iterate over all the possible phases, crystal objects, that can be represented in the FFT
    for crystal_object in crystal_objects:
        #compute the theoretical reflections and distances for each phase with the crystal instance loaded
        #as the diffraction is computed inside
        N_tot_refl = crystal_object.Diff(forbidden,min_d)
        distances_refl=[]
        indexes_refl=[]
        
        for index in range(N_tot_refl):
            ds=crystal_object.getDistances(index)
            distances_refl.append(ds)
            hkls=crystal_object.getIndexes(index)
            indexes_refl.append(hkls)
            
        distances_refl=np.asarray(distances_refl)
        indexes_refl=np.asarray(indexes_refl)

        # for every spot in the refined distances array, e.g. every spot with meaningful information, 
        # analyse how the pairs with other spots are: when  done with all the spots, delete this spot from 
        # arrays _changes_pairs
        refined_distances_changes_pairs=np.copy(refined_distances)
        refined_angles_to_x_changes_pairs=np.copy(refined_angles_to_x)
        spots_int_reference_changes_pairs=np.copy(spots_int_reference_changes)
        
        # list for repeated spot pairs
        considered_spot_pairs = []
        
        for spot_int_ref, distance, angle in zip(spots_int_reference_changes, refined_distances, refined_angles_to_x):
            
            refined_distances_changes=np.copy(refined_distances_changes_pairs)
            refined_angles_to_x_changes=np.copy(refined_angles_to_x_changes_pairs)
            spots_int_reference_changes_int=np.copy(spots_int_reference_changes_pairs)
            
            if len(refined_distances_changes)==0:
                #to address amorphous materials, set distance1=0
                distance1=0
                distances2=np.array([0])
                angles=np.array([0])
                angle_x_to_dist1=None
            else:
            
                distance1=distance
                angle_x_to_dist1=angle
                
                distances2=np.delete(refined_distances_changes, 0)
                angles=np.abs(refined_angles_to_x_changes-angle_x_to_dist1)
                angles=np.delete(angles, 0)
                angles[angles>180]=360-angles[angles>180]
                spots_int_reference_changes_int=np.delete(spots_int_reference_changes_int, 0)
    
        
            for d2, angle, spot_int_ref2 in zip(distances2,angles,spots_int_reference_changes_int):
                # address the repeated spots considered
                spot_order1 = [spot_int_ref, spot_int_ref2]
                spot_order2 = [spot_int_ref2, spot_int_ref]
                if (spot_order1 in considered_spot_pairs) or (spot_order2 in considered_spot_pairs):
                    # skip iteration as the spot pair is already considered
                    continue
                else:
                    #work on the spot pair
                    print("Between distance 1 = {:.3f} A and distance 2 = {:.3f} A, there are {:.3f} degrees".format(distance1, d2, angle))
                    #adress paralel directions or equivalent ones: [001], [002], [004], [00-1], [00-2]...
                    #just by giving an angle tolerance arround 0 and 180. Not adding this should not be harmful
                    #but adding it might leave some reflections not considered (assess if it is 100% useful)
                    if abs(angle-180)<0.5 or angle<0.5:
                        print('Parallel direction: Spot pair ignored')
                        considered_spot_pairs.append([spot_int_ref, spot_int_ref2])
                        
                    else:
                        #possibly meaningful spot pairs
                        n = crystal_object.FindZA(distance1,d2,angle,tolerance)
                        print("Found ",n,"possible Zone Axes")
                    
                        ZAs_spot_pair=[]
                        #!!! hkls1 and 2 indexes of spots
                        hkls1=[]
                        hkls2=[]
                        
                        if n==0:
                            # np.array([0,0,0]) will be the reference for indicating that no ZA and then ref plane was found
                            # this array has the same structure of the real ZA and ref planes, then easier to work with
                            ZA=np.array([0,0,0])
                            reference_plane=np.array([0,0,0])
                            #if no ZA is found, give a huge number as score 
                            huge_None_ZA_score=2**20
                            ZA_score=[huge_None_ZA_score]
                        else:
                            for possible_ZA in range(n):
                                
                                # Commented original code without high index ZA exception
                                
                                # ZA_n=crystal_object.getZA(possible_ZA)
                                # print(ZA_n)
                                # ZAs_spot_pair.append(ZA_n)
                                # #!!! hkls1 and 2 indexes of spots d1 and d2
                                # hkl1=crystal_object.gethkls1(possible_ZA)
                                # hkl2=crystal_object.gethkls2(possible_ZA)
                                # hkls1.append(hkl1)
                                # hkls2.append(hkl2)                       
                                
                                # !!! ADD HERE HIGH INDEX LIMITATION !!!
                                
                                # Limit ZA that are of high index:
                                    
                                ZA_n=crystal_object.getZA(possible_ZA)
                                # the 10 can be changed, but I think is reasonable and could be even lowered
                                max_ZA_index_sum = 8
                                if abs(ZA_n[0]) + abs(ZA_n[1]) + abs(ZA_n[2]) < max_ZA_index_sum: 
                                
                                    print(ZA_n)
                                    ZAs_spot_pair.append(ZA_n)
                                    #!!! hkls1 and 2 indexes of spots d1 and d2
                                    hkl1=crystal_object.gethkls1(possible_ZA)
                                    hkl2=crystal_object.gethkls2(possible_ZA)
                                    hkls1.append(hkl1)
                                    hkls2.append(hkl2)
                                    
                            # Consider if finally no ZA was added as all of the found ones are high index:
                            if len(ZAs_spot_pair) == 0:
                                # np.array([0,0,0]) will be the reference for indicating that no ZA was found
                                ZA=np.array([0,0,0])
                                reference_plane=np.array([0,0,0])
                                #if no ZA is found, give a huge number as score 
                                huge_None_ZA_score=2**20
                                ZA_score=[huge_None_ZA_score]
                                # some reasonable ZA are found, analyse/score them
                            else:
                                # get the ZA sorted y the most likely to the least likely to be the correct one
                                # the scoring needs the hkls1 and hkls2 of the possible indices for d1 and d2
                                ZA_score, ZA_sorted=ZAs_scores(crystal_object, distance1,d2,angle,ZAs_spot_pair, hkls1, hkls2, distances_refl, indexes_refl)
                                ZA=ZA_sorted[0]
                                #order the ZA indices for good comparison despite the possible permutaitons of hkl and signs
                                # !!! CUBIC assumption
                                ZA=np.sort(np.abs(ZA))
                                reference_plane=indexes_refl[np.argmin(np.abs(distances_refl-distance1))]
                           
                        # Create object Scored_spot_pair with its properties   
                        scored_spot_pair=Scored_spot_pair(spot_int_ref, spot_int_ref2 ,crystal_object)
                        scored_spot_pair.Spot_distances_angles(refined_distances, refined_angles_to_x)
                        scored_spot_pair.Zone_axis(ZA,reference_plane)
                        scored_spot_pair.Score(ZA_score[0])
                        
                        all_scored_spot_pairs.append(scored_spot_pair)
                        # account for the repetition of spots
                        considered_spot_pairs.append([spot_int_ref, spot_int_ref2])
                    
            refined_distances_changes_pairs=refined_distances_changes_pairs[1::]
            refined_angles_to_x_changes_pairs=refined_angles_to_x_changes_pairs[1::]
            spots_int_reference_changes_pairs=spots_int_reference_changes_pairs[1::]
    
    # Sort the spots pairs regarding their score for the sake of using them later
    all_scored_spot_pairs.sort(key=lambda x: x.score)

    # delete all the spot pairs that appear more than once (redundant and with different phases), 
    # and select only these with lower score
    for list_index1,spot_pair1 in enumerate(all_scored_spot_pairs):
        spot_pair1_index1=spot_pair1.spot1_int_ref
        spot_pair1_index2=spot_pair1.spot2_int_ref
        spot_pair1_score=spot_pair1.score
        all_scored_spot_pairs_mod=all_scored_spot_pairs.copy()
        all_scored_spot_pairs_mod=all_scored_spot_pairs_mod[list_index1+1::]
        
        for spot_pair2 in all_scored_spot_pairs_mod:
            spot_pair2_index1=spot_pair2.spot1_int_ref
            spot_pair2_index2=spot_pair2.spot2_int_ref
            spot_pair2_score=spot_pair2.score
            #check if the spots considered in both pair spots are the same or not
            if spot_pair2_index1==spot_pair1_index1 or spot_pair2_index1==spot_pair1_index2:
                if spot_pair2_index2==spot_pair1_index1 or spot_pair2_index2==spot_pair1_index2:
                    # if the spots are the same, then delete the second spot pair considered,
                    # as it will always have more score (be worse), as they have been sorted previously,
                    # or the same score indicating that it is the permutation of both and gives no extra information 
                    if spot_pair2_score>=spot_pair1_score:
                        all_scored_spot_pairs.remove(spot_pair2)
        
    # The returned list contains only spot pairs that are meaningful: sorted by score (first elements are the better
    # , i.e. lower score), permutations of spots 1 <--> 2 (1,2) <--> (2,1) are deleted and only one is kept
    # and each spot pair has only one associated phase, which scored better meaning it is more likely (same spot
    # pairs but higher score phases have been deleted)
    return all_scored_spot_pairs
  

'''   
# Final function to use, general and for any crystal system
'''

def Score_all_spot_pairs(
        crystal_objects, space_group_list, forbidden, min_d, tolerance, 
        refined_distances, refined_angles_to_x):
    
    all_scored_spot_pairs=[]
    
    spots_int_reference_changes=np.arange(len(refined_distances))
    
    #iterate over all the possible phases, crystal objects, that can be represented in the FFT
    for crystal_object, space_group_num in zip(crystal_objects, space_group_list):
        #compute the theoretical reflections and distances for each phase with the crystal instance loaded
        #as the diffraction is computed inside
        N_tot_refl = crystal_object.Diff(forbidden,min_d)
        distances_refl=[]
        indexes_refl=[]
        
        for index in range(N_tot_refl):
            ds=crystal_object.getDistances(index)
            distances_refl.append(ds)
            hkls=crystal_object.getIndexes(index)
            indexes_refl.append(hkls)   
        
        distances_refl=np.asarray(distances_refl)
        indexes_refl=np.asarray(indexes_refl)  
        
        # for every spot in the refined distances array, e.g. every spot with meaningful information, 
        # analyse how the pairs with other spots are: when  done with all the spots, delete this spot from 
        # arrays _changes_pairs
        refined_distances_changes_pairs=np.copy(refined_distances)
        refined_angles_to_x_changes_pairs=np.copy(refined_angles_to_x)
        spots_int_reference_changes_pairs=np.copy(spots_int_reference_changes)
        
        # list for repeated spot pairs
        considered_spot_pairs = []
        
        for spot_int_ref, distance, angle in zip(spots_int_reference_changes, refined_distances, refined_angles_to_x):
            
            refined_distances_changes=np.copy(refined_distances_changes_pairs)
            refined_angles_to_x_changes=np.copy(refined_angles_to_x_changes_pairs)
            spots_int_reference_changes_int=np.copy(spots_int_reference_changes_pairs)
            
            if len(refined_distances_changes)==0:
                #to address amorphous materials, set distance1=0
                distance1=0
                distances2=np.array([0])
                angles=np.array([0])
                angle_x_to_dist1=None
            else:
            
                distance1=distance
                angle_x_to_dist1=angle
                
                distances2=np.delete(refined_distances_changes, 0)
                angles=np.abs(refined_angles_to_x_changes-angle_x_to_dist1)
                angles=np.delete(angles, 0)
                angles[angles>180]=360-angles[angles>180]
                spots_int_reference_changes_int=np.delete(spots_int_reference_changes_int, 0)
    
            for d2, angle, spot_int_ref2 in zip(distances2,angles,spots_int_reference_changes_int):
                # address the repeated spots considered
                spot_order1 = [spot_int_ref, spot_int_ref2]
                spot_order2 = [spot_int_ref2, spot_int_ref]
                if (spot_order1 in considered_spot_pairs) or (spot_order2 in considered_spot_pairs):
                    # skip iteration as the spot pair is already considered
                    continue
                else:
                    #work on the spot pair
                    print("Between distance 1 = {:.3f} A and distance 2 = {:.3f} A, there are {:.3f} degrees".format(distance1, d2, angle))
                    #adress paralel directions or equivalent ones: [001], [002], [004], [00-1], [00-2]...
                    #just by giving an angle tolerance arround 0 and 180. Not adding this should not be harmful
                    #but adding it might leave some reflections not considered (assess if it is 100% useful)
                    if abs(angle-180)<0.5 or angle<0.5:
                        print('Parallel direction: Spot pair ignored')
                        considered_spot_pairs.append([spot_int_ref, spot_int_ref2])
                        
                    else:
                        #possibly meaningful spot pairs
                        n = crystal_object.FindZA(distance1, d2, angle, tolerance)
                        print("Found ",n,"possible Zone Axes")
                    
                        ZAs_spot_pair=[]
                        #!!! hkls1 and 2 indexes of spots
                        hkls1=[]
                        hkls2=[]
                        
                        if n==0:
                            # np.array([0,0,0]) will be the reference for indicating that no ZA and then ref plane was found
                            # this array has the same structure of the real ZA and ref planes, then easier to work with
                            ZA=np.array([0,0,0])
                            hkl1_reference = np.array([0,0,0])
                            hkl2_reference = np.array([0,0,0])
                            #if no ZA is found, give a huge number as score 
                            huge_None_ZA_score=2**20
                            ZA_score=[huge_None_ZA_score]
                        else:
                            for possible_ZA in range(n):
                                
                                # Commented original code without high index ZA exception
                                
                                # ZA_n=crystal_object.getZA(possible_ZA)
                                # print(ZA_n)
                                # ZAs_spot_pair.append(ZA_n)
                                # #!!! hkls1 and 2 indexes of spots d1 and d2
                                # hkl1=crystal_object.gethkls1(possible_ZA)
                                # hkl2=crystal_object.gethkls2(possible_ZA)
                                # hkls1.append(hkl1)
                                # hkls2.append(hkl2)                       
                                
                                # !!! ADD HERE HIGH INDEX LIMITATION !!! HYPERPARAMETER
                                
                                # Limit ZA that are of high index:
                                    
                                ZA_n=crystal_object.getZA(possible_ZA)
                                # the 10 can be changed, but I think is reasonable and could be even lowered
                                max_ZA_index_sum = 8
                                if abs(ZA_n[0]) + abs(ZA_n[1]) + abs(ZA_n[2]) < max_ZA_index_sum: 
                                
                                    ZAs_spot_pair.append(ZA_n)
                                    #!!! hkls1 and 2 indexes of spots d1 and d2
                                    hkl1=crystal_object.gethkls1(possible_ZA)
                                    hkl2=crystal_object.gethkls2(possible_ZA)
                                    hkls1.append(hkl1)
                                    hkls2.append(hkl2)
                                    
                            # Consider if finally no ZA was added as all of the found ones are high index:
                            if len(ZAs_spot_pair) == 0:
                                # np.array([0,0,0]) will be the reference for indicating that no ZA was found
                                ZA = np.array([0,0,0])
                                hkl1_reference = np.array([0,0,0])
                                hkl2_reference = np.array([0,0,0])
                                #if no ZA is found, give a huge number as score 
                                huge_None_ZA_score=2**20
                                ZA_score=[huge_None_ZA_score]
                                # some reasonable ZA are found, analyse/score them
                            else:
                                # get the ZA sorted y the most likely to the least likely to be the correct one
                                # the scoring needs the hkls1 and hkls2 of the possible indices for d1 and d2
                                ZA_score, ZA_sorted, hkls1_references_sorted, hkls2_references_sorted = ZAs_scores(
                                    crystal_object, space_group_num, distance1, d2, angle,
                                    ZAs_spot_pair, hkls1, hkls2, distances_refl, indexes_refl)
                                ZA=ZA_sorted[0]
                                #order the ZA indices for good comparison despite the possible permutaitons of hkl and signs
                                # !!! CUBIC assumption
                                
                                # ZA=np.sort(np.abs(ZA))
                                # reference_plane=indexes_refl[np.argmin(np.abs(distances_refl-distance1))]
                                
                                # reference_plane = hkls1_references_sorted[0]
                                hkl1_reference = hkls1_references_sorted[0]
                                hkl2_reference = hkls2_references_sorted[0]
                           
                        # Create object Scored_spot_pair with its properties   
                        scored_spot_pair=Scored_spot_pair(spot_int_ref, spot_int_ref2 ,crystal_object)
                        scored_spot_pair.Spot_distances_angles(refined_distances, refined_angles_to_x)
                        scored_spot_pair.Zone_axis(ZA, hkl1_reference, hkl2_reference)
                        scored_spot_pair.Score(ZA_score[0])
                        
                        all_scored_spot_pairs.append(scored_spot_pair)
                        # account for the repetition of spots
                        considered_spot_pairs.append([spot_int_ref, spot_int_ref2])
                    
            refined_distances_changes_pairs=refined_distances_changes_pairs[1::]
            refined_angles_to_x_changes_pairs=refined_angles_to_x_changes_pairs[1::]
            spots_int_reference_changes_pairs=spots_int_reference_changes_pairs[1::]
    
    # Sort the spots pairs regarding their score for the sake of using them later
    all_scored_spot_pairs.sort(key=lambda x: x.score)

    # delete all the spot pairs that appear more than once (redundant and with different phases), 
    # and select only these with lower score
    for list_index1,spot_pair1 in enumerate(all_scored_spot_pairs):
        spot_pair1_index1=spot_pair1.spot1_int_ref
        spot_pair1_index2=spot_pair1.spot2_int_ref
        spot_pair1_score=spot_pair1.score
        all_scored_spot_pairs_mod=all_scored_spot_pairs.copy()
        all_scored_spot_pairs_mod=all_scored_spot_pairs_mod[list_index1+1::]
        
        for spot_pair2 in all_scored_spot_pairs_mod:
            spot_pair2_index1=spot_pair2.spot1_int_ref
            spot_pair2_index2=spot_pair2.spot2_int_ref
            spot_pair2_score=spot_pair2.score
            #check if the spots considered in both pair spots are the same or not
            if spot_pair2_index1==spot_pair1_index1 or spot_pair2_index1==spot_pair1_index2:
                if spot_pair2_index2==spot_pair1_index1 or spot_pair2_index2==spot_pair1_index2:
                    # if the spots are the same, then delete the second spot pair considered,
                    # as it will always have more score (be worse), as they have been sorted previously,
                    # or the same score indicating that it is the permutation of both and gives no extra information 
                    if spot_pair2_score>=spot_pair1_score:
                        all_scored_spot_pairs.remove(spot_pair2)
        
    # The returned list contains only spot pairs that are meaningful: sorted by score (first elements are the better
    # , i.e. lower score), permutations of spots 1 <--> 2 (1,2) <--> (2,1) are deleted and only one is kept
    # and each spot pair has only one associated phase, which scored better meaning it is more likely (same spot
    # pairs but higher score phases have been deleted)
    
    
    # Delete the ZA = [000]
    ZA_000_found = []
        
    for scored_spot in all_scored_spot_pairs:
        if scored_spot.ZA[0]==0 and  scored_spot.ZA[1]==0 and scored_spot.ZA[2]==0:
            ZA_000_found.append(scored_spot)
            
    for spot_to_delete in ZA_000_found:
        all_scored_spot_pairs.remove(spot_to_delete)
            
    return all_scored_spot_pairs



# class containing the crystals identified and its correspondent spots
class Crystal_spots:
    def __init__(self, spot_list):
        # the spots are referenced by the index internal reference, not by position or distance
        self.spots=spot_list
        
    def Spot_pairs(self, list_spot_pairs_obj):
        self.spot_pairs_obj=list_spot_pairs_obj
    
    def Phase(self, phase_string):
        self.phase_name=phase_string
    
    def ZA(self, ZA, ZA_priv_index):
        self.ZA=ZA
        self.ZA_string=str(ZA)
        # index used in case more than one crystal in the same ZA appears, 
        # given same phase, would be differentiated by this index
        self.ZA_priv_index=ZA_priv_index
        
    def PF_method(self, method_used):
        self.method_used = method_used


# Find all crystals given the ZAs and scores, function that works fine
def Find_All_Crystals_ORIGINAL(list_all_cryst_spots, all_scored_spot_pairs):
    '''
    The refined distances and refined angles to x  and all scored spot pairs should be arrays that 
    are a copy of the original arrays and
    can therefore be modified without the fear of permanently modifying the original
    
    '''
    print('start')
    print(len(all_scored_spot_pairs))
    
    
    
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
        crystal_identified=Crystal_spots([spot_pair_target_index1,spot_pair_target_index2])
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
    
    print('spot_pairs_int_refs_possibe')
    print(spot_pairs_int_refs_possibe)
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

    crystal_identified=Crystal_spots(spot_int_refs_phase_axis)
    
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
    
    print('list_all_cryst_spots first append')
    print(list_all_cryst_spots)
    
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
        
    
        crystal_identified=Crystal_spots([spot_pair_target_index1,spot_pair_target_index2])

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
        print('list_all_cryst_spots second append')
        print(list_all_cryst_spots)

        # end function
        print('Finished!')
        return
    else:
        print('iter')
        print(len(all_scored_spot_pairs))
        Find_All_Crystals(list_all_cryst_spots, all_scored_spot_pairs)
        
        
        # removes two elements from the list but should find the other one as better 
        # as has better score
        # chekc both scores maybe the sam and trouble if same
        # do they end up in same crystal spot object? tow objects should be generated..
    
    #we do not return anything as this fills the empty list provided
    return 


# !!! Trying to debug some bugs...
# Find all crystals given the ZAs and scores, function that works fine
'''
!!! USE this following Find_All_Crystals() as seems to be bug free,
if something happens Find_All_Crystals_ORIGINAL() is bugged, but okaish as last resource
'''

def Find_All_Crystals(list_all_cryst_spots, all_scored_spot_pairs):
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
        crystal_identified=Crystal_spots([spot_pair_target_index1,spot_pair_target_index2])
        crystal_identified.Spot_pairs([scored_spot_target])
        spot_pair_target_phase=scored_spot_target.phase_name
        crystal_identified.Phase(spot_pair_target_phase) 
        spot_pair_target_ZA=scored_spot_target.ZA
        crystal_identified.ZA(spot_pair_target_ZA, 0)
        
        list_all_cryst_spots.append(crystal_identified)
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
    
    # !!! MOD
    # all_scored_spot_pairs=all_scored_spot_pairs[1::]
    
    # contains tuples of spot pairs containing the internal references of possible spots referencing to the same crystal
    spot_pairs_int_refs_possibe=[(spot_pair_target_index1,spot_pair_target_index2)]
    
    # the index of the spot pairs from the modified all scored spots, as will be needed to refer back to the 
    # spot pair which contained the given tuple of internal references of spots to delete it from the main list of all
    index_spot_pairs_int_refs_possibe=[]
    # !!! MOD added [1:]
    for index_scored_pair_eval, scored_spot_pair_eval in enumerate(all_scored_spot_pairs[1:]):
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
                index_spot_pairs_int_refs_possibe.append(index_scored_pair_eval+1)
    
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

    crystal_identified=Crystal_spots(spot_int_refs_phase_axis)
    
    crystal_scored_spot_pairs =  [all_scored_spot_pairs[i] for i in indexes_all_spot_pairs_to_delete]
    # crystal_scored_spot_pairs = [all_scored_spot_pairs[i] for i in indexes_all_spot_pairs_to_delete]
    
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
        
    
        crystal_identified=Crystal_spots([spot_pair_target_index1,spot_pair_target_index2])

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
        Find_All_Crystals(list_all_cryst_spots, all_scored_spot_pairs)
        
        
        # removes two elements from the list but should find the other one as better 
        # as has better score
        # chekc both scores maybe the sam and trouble if same
        # do they end up in same crystal spot object? tow objects should be generated..
    
    #we do not return anything as this fills the empty list provided
    return 
    
    
def Crystal_refinement(list_all_cryst_spots,spots_int_reference):
    '''
    This function intends to, given the list of all possible crystal objects and their respective spots,
    filter the possible wrong crystals by checking whether the spots actually correspond to one crystal, or more than
    one or if a single spot or group of them was/were added to an incorrect crystal domain
    I.e: basically check if the crystals obtained share the same spots: if one crystal contains all the possible
    spots, then delete the rest (or given the others a 'bad' score). If one crystal has many spots, and then another
    shares one of them and adds more to this shared one, then probably is another crystal domain with one or more 
    shared spots
    '''
    #get a score for each crytal based on the sum of all the scores of its pair spots, normalised over the number of spot pairs
    
        
    for cryst_spot in list_all_cryst_spots:
    
        print('Cryst')
        print('spot list', cryst_spot.spots)
        # print('spot pairs', cryst_spot.spot_pairs_obj)
        print('phase name', cryst_spot.phase_name)
        print('ZA', cryst_spot.ZA)
        # print('ZA priv index', cryst_spot.ZA_priv_index)
        
        for spot in cryst_spot.spot_pairs_obj:
            print('ZA of scored spot pair')
            print(spot.ZA)
    
    
    
    cryst_scores=[]
    for cryst in list_all_cryst_spots:
        cryst_score=0
        for spot_pair in cryst.spot_pairs_obj:
            spot_pair_score=spot_pair.score
            cryst_score=cryst_score+spot_pair_score
        total_spot_pairs=len(cryst.spot_pairs_obj)
        cryst_scores.append(cryst_score/total_spot_pairs**2)
    #the problem of having a score per spot pair is that maybe the average is lower for a worse crystal,
    #because the main crystal has many spots and some are good but some are not, and the average is worse than 
    #the actually worse crystal
    #1st approx simple solution: instead of cryst_score/total_spot_pairs, use cryst_score/total_spot_pairs**2
    #to give more weight to these crystals that have more spots pairs. This way, we are rewarding the crsytal
    #for having more spots recognised as the same axis, which hardly ever will be something bad
        
    list_refined_cryst_spots=[crysts for _,crysts in sorted(zip(cryst_scores,list_all_cryst_spots), key=lambda x: x[0])]
    spots_int_reference=list(np.copy(spots_int_reference))

    for index_cryst,cryst in enumerate(list_refined_cryst_spots):
        if len(spots_int_reference)==0:
            list_refined_cryst_spots=list_refined_cryst_spots[:index_cryst]
            break
        else:
            cryst_spots=cryst.spots
            spots_found_deleted=0
            for spot in cryst_spots:
                if spot in spots_int_reference:
                    spots_int_reference.remove(spot)
                    spots_found_deleted=spots_found_deleted+1
            if spots_found_deleted==0:        
                list_refined_cryst_spots.remove(cryst)
                    
    return list_refined_cryst_spots


def Discard_Wrong_Crystals(list_refined_cryst_spots):
    '''
    Function to remove the crystals that make no sense or that are just used to deal with no ZA found,
    basically the found  [000] axis in any phase, must be removed.
    The function is open to remove further phases in the future in case we see a pattern of wrongly selected
    phases or axes

    Parameters
    ----------
    list_refined_cryst_spots : directly from crystal_refinement

    Returns
    -------
    list_refined_cryst_spots.

    '''
    for cryst_spot in list_refined_cryst_spots:
        if cryst_spot.ZA[0]==0 and cryst_spot.ZA[1]==0 and cryst_spot.ZA[2]==0: 
            list_refined_cryst_spots.remove(cryst_spot)

    return list_refined_cryst_spots



def Ready_Distances_Angles_Pixels(
        pixels_of_peaks, FFT_calibration, total_pixels_image, min_d):
    
    # wrap the translation of frequency peaks pixels to distances, angles and pixels in 1 line

    # Extract distances, angles, and pixel positions
    d_distances = Spot_coord_To_d_spacing_vect(
        pixels_of_peaks, FFT_calibration, total_pixels_image)
    
    angles_to_x = Spot_coord_To_Angles_to_X_vect(
        pixels_of_peaks, total_pixels_image)
    
    refined_distances, refined_angles_to_x, refined_pixels = Ensure_Center_Diff(
        d_distances, angles_to_x, pixels_of_peaks)
    
    # Set the values of distances in angstroms as required by Prepare_exp_distances_angles_pixels and other funcs
    refined_distances = ImCalTrans.nm_to_Angstroms(refined_distances)
    
    # Refine distances, angles, and pixel positions
    refined_distances, refined_angles_to_x, refined_pixels = Prepare_exp_distances_angles_pixels(
        refined_distances, refined_angles_to_x,refined_pixels, min_d)
    
    # Get only the right half of the FFT and its positions, angles and pixels
    refined_distances, refined_angles_to_x, refined_pixels = ImCalTrans.Right_Half_FFT_Only(
        refined_distances, refined_angles_to_x, refined_pixels)  
    
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
        crystal_object = Crystal(bytes(unit_cell_path.encode()))
        crystal_objects_list.append(crystal_object)   
        
        # store unit cell information: space group
        unit_cell_text = open(unit_cell_path)
        
        for line in unit_cell_text:
            if line[:4] == 'RGNR':
                space_group = int(line[4:])
        
        space_group_list.append(space_group)
                
    return crystal_objects_list, space_group_list

def Init_Unit_Cells_AnyFormat(
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
    
    filepaths_used = []
    crystal_objects_list=[]
    space_group_list = []
    
    for unit_cell in os.listdir(unit_cells_path):
        unit_cell_path=unit_cells_path+'\\'+unit_cell
        unit_cell_extension = unit_cell[-3:]
        
        # check if the unit cell has been used previously
        if unit_cell_path not in filepaths_used:
            
            if unit_cell_extension == 'ACE' or unit_cell_extension == 'ace':
                
                # first check if the conversion was done before so check if there
                # is already a .uce with the same name
                filepaths_used.append(unit_cell_path)
                
                for unit_cell_check in os.listdir(unit_cells_path):
                    
                    if unit_cell_check == unit_cell[:unit_cell.find('.')] + '.uce':
                        unit_cell_path = unit_cells_path+'\\'+unit_cell_check
                        break
                    
                # if it has not found any equally named .uce, then convert to uce      
                if unit_cell_path[-3:] != 'uce':
                    unit_cell_path = FileComp.ACE_to_uce(unit_cells_path, unit_cell)
                    
                    
            else:
                unit_cell_path=unit_cells_path+'\\'+unit_cell
              
                
            # the argument must be the bytes to the path not the string (just encode the string)
            crystal_object = Crystal(bytes(unit_cell_path.encode()))
            
            # store unit cell information: space group
            unit_cell_text = open(unit_cell_path)
            
            for line in unit_cell_text:
                if line[:4] == 'RGNR':
                    space_group = int(line[4:])
            
            space_group_list.append(space_group)
            crystal_objects_list.append(crystal_object)   
            filepaths_used.append(unit_cell_path)
            
    return crystal_objects_list, space_group_list



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
    FFT_image_array_No_Log, FFT_calibration, FFT_units = ImCalTrans.FFT_No_Log_from_crop(
        image_crop_hs_signal, image_crop_array)
    
    # Get FFT and its calibration from the hs signal  of the crop
    FFT_image_array, FFT_calibration, FFT_units = ImCalTrans.FFT_from_crop(
        image_crop_hs_signal, image_crop_array)
    
    # Find pixels of the freqs with each method as a standalone unit
    
    # pixels_of_peaks = PeakFindWrap.Peak_Detection_Wrapped_Standarised(
    #     image_crop_hs_signal, FFT_image_array_No_Log, FFT_image_array, 
    #     total_pixels_crop, crop_FOV, visualisation = True)
    
    # Find the pixels of the freqs
    # pixels_of_peaks = PeakFindWrap.Peak_Detection_Wrapped_StandarisedwithFOV(
    #     image_crop_hs_signal, FFT_image_array_No_Log, FFT_image_array, 
    #     total_pixels_crop, crop_FOV, visualisation = True)
    
    # Find the pixels of the freqs
    # pixels_of_peaks = PeakFindWrap.Peak_Detection_Wrapped(
    #     FFT_image_array_No_Log, FFT_image_array, total_pixels_crop, crop_FOV, visualisation = True)
    
    # pixels_of_peaks = PeakFindWrap.Peak_Detection_Wrapped_CNN(
    #     FFT_image_array_No_Log, FFT_image_array, total_pixels_crop, crop_FOV, visualisation = True)
    
    # pixels_of_peaks = PeakFindWrap.Peak_Detection_Wrapped_Standarised_CNN(
    #     image_crop_hs_signal, FFT_image_array_No_Log, FFT_image_array, 
    #     total_pixels_crop, crop_FOV, visualisation = True)  
    
    # pixels_of_peaks = PeakFindWrap.Peak_Detection_Wrapped_Standarised_withFOV_CNN(
    #     image_crop_hs_signal, FFT_image_array_No_Log, FFT_image_array, 
    #     total_pixels_crop, crop_FOV, visualisation = True)  
    
    # Find the pixels of the freqs with the combination of all the methods
    pixels_of_peaks, method_used = PeakFindWrap.Peak_Detection_Multimodal_Wrapped(
        image_crop_hs_signal, FFT_image_array_No_Log, FFT_image_array, 
        total_pixels_crop, crop_FOV, visualisation = True)
    
    # Get functional distances, angles, pixel positions out of the peaks of the freqs
    refined_distances, refined_angles_to_x, refined_pixels = Ready_Distances_Angles_Pixels(
        pixels_of_peaks, FFT_calibration, total_pixels_crop, min_d)
    
    # Reference array for reference of the spots
    spots_int_reference = np.arange(len(refined_distances))
    
    # All spot pairs are evaluated depending on the ZA that is found 
    # for each of them and sorted from better axis to worse
    all_scored_spot_pairs = Score_all_spot_pairs(
        crystal_objects_list, space_group_list, forbidden, min_d, tol, 
        refined_distances, refined_angles_to_x)
    
    # Copy the previous array as the following needs to be modified 
    # and create a list for storing the crystals
    all_scored_spot_pairs_to_mod = all_scored_spot_pairs.copy()
    
    # Initialise list for storing the outputs
    list_all_cryst_spots=[]
    
    # Evaluate all the scored spot pairs with ZA and phase to find all 
    # the crystals that were identified within an FFT
    Find_All_Crystals(
        list_all_cryst_spots, all_scored_spot_pairs_to_mod)
    
    # From all phases found and their respective pixels, check spot per 
    # spot if it was attributed to a phase,
    # and assign a phase to them until no spot remains 
    # unassigned (from best score phases to worse)
    list_refined_cryst_spots = Crystal_refinement(
        list_all_cryst_spots,spots_int_reference)
    
    # Remove the [000] axis determined by the no axis found
    list_refined_cryst_spots = Discard_Wrong_Crystals(
        list_refined_cryst_spots)
    
    # Add to the cryst spots the method used to find the peaks in the FFT
    for crystal_spot in list_refined_cryst_spots:
        crystal_spot.PF_method(method_used)

    
    # To print the info from the list_of_refined_crystals
    # for cryst in list_refined_cryst_spots:
    #     print('Cryst')
    #     print('spot list', cryst.spots)
    #     print('spot pairs', cryst.spot_pairs_obj)
    #     print('phase name', cryst.phase_name)
    #     print('ZA', cryst.ZA)
    #     print('ZA priv index', cryst.ZA_priv_index)

    return list_refined_cryst_spots, refined_pixels, spots_int_reference



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
        self.crop_index = 1
        self.Crop_outputs = dict()
    
    def Add_Crop_Output(
            self,image_crop_hs_signal, scaled_reference_coords, 
            image_relative_coords, list_refined_cryst_spots,
            refined_pixels, spots_int_reference):
        ''' 
        Parameters
        ----------
        image_crop_hs_signal : complete (full info) hyperspy signal of the crop
        scaled_reference_coords : pixel coordinates of the crop within the image_in_dataset
        image_relative_coords :
             rel global coords (within the large low mag FOV) of the image 
             where the crops come from (i.e. image_in_dataset)
        list_refined_cryst_spots : all cryst spots identified
        refined_pixels: list of coordinates of the FFT of the hs_signal 
                        that are the spots identified 
        spots_int_reference: arbitrary internal reference to the coordinate of the 
                            spots inside the refined_pixels list
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
        # Added features to allow the printing of the indexated FFT
        self.Crop_outputs[str(self.crop_index) + '_refined_pixels'] = refined_pixels
        self.Crop_outputs[str(self.crop_index) + '_spots_int_reference'] = spots_int_reference
        # add 1 to the index so the following added crop is added as a new one
        self.crop_index = self.crop_index + 1



