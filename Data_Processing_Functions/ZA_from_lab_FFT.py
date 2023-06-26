# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 16:38:22 2021

@author: Marc
"""

import ctypes
import os
import numpy as np
import matplotlib.pyplot as plt
import skimage.measure as measure
import cv2
import scipy
import torch
import stemtool
import hyperspy.api as hs
import numba
from Diffraction_simulation_ZA_finding import Crystal
import time
from sympy.utilities.iterables import multiset_permutations, permute_signs



def Find_percentage_of_thresholded(FFT_image, threshold):
    FFT_image[FFT_image<=threshold]=0
    percentage_of_tresholded=np.count_nonzero(FFT_image.ravel())/len(FFT_image.ravel())
    return percentage_of_tresholded

def Threshold_given_percentage(FFT_image, percentage):
    y_pixs,x_pixs=np.shape(FFT_image)
    n_int_pixs=int(round(percentage*y_pixs*x_pixs))
    FFT_ravel=np.sort(np.ravel(FFT_image))[::-1]
    threshold=FFT_ravel[n_int_pixs]
    return threshold

def mpfit_Distance(FFT_image,FOV):
    mpfit_model=[[-2.87175127e-11],
                     [ 8.11320079e-09],
                     [-8.18658056e-07],
                     [ 3.33222163e-05],
                     [-2.02745223e-04],
                     [-2.26140649e-02],
                     [ 5.95346985e-01],
                     [-7.69005862e-01]]
    # without the averages
    mpfit_model_c=[[-3.46636981e-11],
                   [ 1.00423053e-08],
                   [-1.06223267e-06],
                   [ 4.84860471e-05],
                   [-6.82330526e-04],
                   [-1.58450088e-02],
                   [ 5.79540436e-01],
                   [-1.10510783e+00]]
    #set the working limits of the model
    if FOV >=30:
        mpfit_dist=np.array([40])
    else:
        
        fov_vals=np.array([FOV**7,FOV**6,FOV**5,FOV**4,FOV**3,FOV**2,FOV**1,1])
        mpfit_dist=np.e**np.dot(fov_vals,mpfit_model)
        mpfit_dist=np.e**np.dot(fov_vals,mpfit_model_c)
     
    #Adjustments depending on the sizze of the image
    if np.shape(FFT_image)[0]==2048:
        mpfit_dist=mpfit_dist*1.30
    elif np.shape(FFT_image)[0]<256:
        mpfit_dist=mpfit_dist*1.55     
    elif np.shape(FFT_image)[0]==256:
        mpfit_dist=mpfit_dist*1.55     
    elif np.shape(FFT_image)[0]==1024:
        mpfit_dist=mpfit_dist*1.05
    elif np.shape(FFT_image)[0]==512:
        mpfit_dist=mpfit_dist*1.15
    else:
        mpfit_dist=mpfit_dist*1.15
        
    return mpfit_dist[0]

def FFT_threshold(FOV):
    FFT_thresh_model=[[-1.01291174e-11],
                          [ 2.88297492e-09],
                          [-3.01778444e-07],
                          [ 1.44327587e-05],
                          [-3.23378617e-04],
                          [ 3.61163733e-03],
                          [-3.72515413e-02],
                          [-1.96361805e-01]]
    # without the averages
    FFT_thresh_model_c=[[ 1.54099057e-12],
                        [-6.56354380e-10],
                        [ 1.05878669e-07],
                        [-8.09680716e-06],
                        [ 2.96148198e-04],
                        [-4.30807411e-03],
                        [ 1.81389577e-03],
                        [-2.45698182e-01]]
    #set the working limits of the model
    if FOV >=80:
        FFT_thresh=np.array([0.6])
    else:
                
        fov_vals=np.array([FOV**7,FOV**6,FOV**5,FOV**4,FOV**3,FOV**2,FOV**1,1])
        FFT_thresh=np.e**np.dot(fov_vals,FFT_thresh_model)
        FFT_thresh=np.e**np.dot(fov_vals,FFT_thresh_model_c)
      
    
    return FFT_thresh[0]

def FFT_percentage(FFT_image,FOV):
    FFT_perc_model=[[-3.00411834e-11],
                        [ 1.17313244e-08],
                        [-1.81232383e-06],
                        [ 1.40635117e-04],
                        [-5.76020214e-03],
                        [ 1.20704617e-01],
                        [-1.20113823e+00],
                        [-2.14024711e+00]]
    # without the averages
    FFT_perc_model_c=[[ 1.38602821e-11],
                      [-2.46874956e-09],
                      [-1.63526870e-08],
                      [ 2.67725990e-05],
                      [-1.91230990e-03],
                      [ 5.28789844e-02],
                      [-6.40863899e-01],
                      [-3.71037505e+00]]
    #set the working limits of the model
    if FOV >=110:
        FFT_perc=np.array([0.00025])  #In case it is too much for higher FOVs, just delete this and keep the FFT_perc_model for all ranges
    # elif FOV <3:
    #     FFT_perc=np.array([0.01])
    else:
        
        
        fov_vals=np.array([FOV**7,FOV**6,FOV**5,FOV**4,FOV**3,FOV**2,FOV**1,1])
        FFT_perc=np.e**np.dot(fov_vals,FFT_perc_model)
        FFT_perc=np.e**np.dot(fov_vals,FFT_perc_model_c)        
        
        if FOV <4.5:
            FFT_perc=FFT_perc*(10**(np.log(128/np.shape(FFT_image)[0])/np.log(4)))
        elif FOV >=4.5 and FOV <=20 :
            FFT_perc=FFT_perc*(10**(np.log(512/np.shape(FFT_image)[0])/np.log(4)))
        else:
            FFT_perc=FFT_perc*(10**(np.log(2048/np.shape(FFT_image)[0])/np.log(4)))
    
    #Adjustments depending on the sizze of the image
    if np.shape(FFT_image)[0]<256:
        FFT_perc=FFT_perc*0.25
    elif np.shape(FFT_image)[0]==256:
        FFT_perc=FFT_perc*0.45  
    elif np.shape(FFT_image)[0]==512:
        FFT_perc=FFT_perc*0.55
    elif np.shape(FFT_image)[0]==1024:
        FFT_perc=FFT_perc*0.80    
    else:
        FFT_perc=FFT_perc*0.80
        
    return FFT_perc[0]

def FFT_hyperparams(FFT_image,FOV):
    #Return, in order, the mpfit dist, the threshold, and the percentage
    
    mpfit_dist=mpfit_Distance(FFT_image,FOV)
    FFT_thresh=FFT_threshold(FOV)
    FFT_perc=FFT_percentage(FFT_image,FOV)
    
    return mpfit_dist,FFT_thresh,FFT_perc



def FFT_calibration(hyperspy_2D_signal):
    fft_shifted = hyperspy_2D_signal.fft(shift=True)
    FFT_calibration=fft_shifted.axes_manager['x'].scale
    FFT_pixels=fft_shifted.axes_manager['x'].size
    FFT_units=fft_shifted.axes_manager['x'].units
    
    return FFT_calibration,FFT_pixels,FFT_units
    
def FFT_calibration_Correction(hyperspy_2D_signal, real_calibration_factor):
    '''
    distances in real space--> d_real=d_measured*real_calibration_factor
    '''
    x_calibration_original=hyperspy_2D_signal.axes_manager['x'].scale
    y_calibration_original=hyperspy_2D_signal.axes_manager['y'].scale
    hyperspy_2D_signal.axes_manager['x'].scale=x_calibration_original*real_calibration_factor
    hyperspy_2D_signal.axes_manager['y'].scale=y_calibration_original*real_calibration_factor
    
    fft_shifted = hyperspy_2D_signal.fft(shift=True)
    FFT_calibration=fft_shifted.axes_manager['x'].scale
    FFT_pixels=fft_shifted.axes_manager['x'].size
    FFT_units=fft_shifted.axes_manager['x'].units
    
    return FFT_calibration,FFT_pixels,FFT_units
    

def Spot_coord_To_d_spacing_indiv(coords, FFT_calibration, FFT_pixels):
    
    (y_max, x_max)=coords

    FFT_distance_point_x=np.abs(x_max-int(FFT_pixels/2))*FFT_calibration
    FFT_distance_point_y=np.abs(y_max-int(FFT_pixels/2))*FFT_calibration
    
    FFT_distance_total=np.sqrt(FFT_distance_point_x**2+FFT_distance_point_y**2)
    
    if FFT_distance_total==0:
        FFT_distance_total=0.0001
    d_spacing_spot=1/FFT_distance_total
    
    return d_spacing_spot



def Spot_coord_To_d_spacing_vect(coord_vects, FFT_calibration, FFT_pixels):
    y_vects=coord_vects[:,0]    
    x_vects=coord_vects[:,1] 

    FFT_distance_point_x=np.abs(x_vects-int(FFT_pixels/2))*FFT_calibration
    FFT_distance_point_y=np.abs(y_vects-int(FFT_pixels/2))*FFT_calibration
    
    FFT_distance_total=np.sqrt(FFT_distance_point_x**2+FFT_distance_point_y**2)
    
    FFT_distance_total[FFT_distance_total==0]=0.0001
    
    d_spacing_spot=1/FFT_distance_total
    
    return d_spacing_spot


def Spot_coord_To_Angles_to_X_indiv(coords,FFT_pixels):
    
    (y_max, x_max)=coords
    
    cont_dist=x_max-int(FFT_pixels/2)
    opp_dist=int(FFT_pixels/2)-y_max
    
    angles_to_X=np.arctan2(opp_dist,cont_dist)*180/np.pi
    
    return angles_to_X


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





def Take_Distance1_from_y_axis(refined_distances, refined_angles_to_x):
    '''
    Generally speaking, it will be better to start the comparison between the pairs 
    distance 1 and distance 2 by considering the distance 1 to be the one that is directly 
    on top of the center of the diffractogram, i.e. the smallest distance found closer to the
    y axis. This function outputs the highest interplanar distance found between +/-
    tolerance_angle degrees the y axis

    Parameters
    ----------
    refined_distances : refined distances matrix without the center of diffraction 
    refined_angles_to_x : same witht the angles

    Returns
    -------
    distance1 : float 

    '''
    #+/- tolerance angle is the circular section that is analysed arround the y axis
    tolerance_angle=10
    refined_angles_to_x_temp=refined_angles_to_x[refined_angles_to_x<90+tolerance_angle]
    refined_distances_temp=refined_distances[refined_angles_to_x<90+tolerance_angle]
    refined_distances_temp_2=refined_distances_temp[refined_angles_to_x_temp>90-tolerance_angle]
    if len(refined_distances_temp_2)==0:
        distance1=None
    else:
        distance1=np.max(refined_distances_temp_2)
        
    return distance1

def Take_Distance1_from_x_axis(refined_distances, refined_angles_to_x):
    '''
    Generally speaking, it will be better to start the comparison between the pairs 
    distance 1 and distance 2 by considering the distance 1 to be the one that is directly 
    on top of the center of the diffractogram, i.e. the smallest distance found closer to the
    y axis. This function outputs the highest interplanar distance found between +/-
    tolerance_angle degrees the y axis

    Parameters
    ----------
    refined_distances : refined distances matrix without the center of diffraction 
    refined_angles_to_x : same witht the angles

    Returns
    -------
    distance1 : float 

    '''
    #+/- tolerance angle is the circular section that is analysed arround the y axis
    tolerance_angle=10 
    refined_angles_to_x_temp=refined_angles_to_x[refined_angles_to_x<0+tolerance_angle]
    refined_distances_temp=refined_distances[refined_angles_to_x<0+tolerance_angle]
    
    refined_distances_temp_2=refined_distances_temp[refined_angles_to_x_temp>0-tolerance_angle]
    if len(refined_distances_temp_2)==0:
        distance1=None
    else:
        distance1=np.max(refined_distances_temp_2)
        
    return distance1


def Find_ZA_mono(crystal_object,forbidden, min_d,N_tot_refl, tolerance, refined_distances, refined_angles_to_x, high_index):
    '''
    

    Parameters
    ----------
    crystal_object : CrystalHandle object with diffractions already computed .Diff(forbidden, min_d)
    forbidden : Bool for allowing forbidden reflections (True) or not (False)
    min_d : minimum interplanar distance computed in the diffraction simulation
    N_tot_refl : total number of reflections computed with the Diff method of the crystal
    tolerance : tolerance when comparing real and experimental distances and angles
    refined_distances : array containing the distance, in ANGSTROMS, and having applied the 
            Ensure_Center_Diff function to extract the center of diffractogram contributions
    refined_angles_to_x : array containing the angles, in Degrees, and having applied the 
            Ensure_Center_Diff function to extract the center of diffractogram contributions
    high_index: bool, allow to output high index ZA as a valid ZA when True, or not when False
            it can be varied but it is a high index ZA when h+k+l>=10
        
    Returns
    -------
    Information to fully characterise the studied phase in 3D
    ZA_final : most common ZA found, array
    reference_plane : plane used as reference, d1, for computing the rest, float
    angle_x_to_dist1 : angle from reference_plane to x axis, in degrees

     '''
    
    #Choose distance 1: 
    
    distance1_y=Take_Distance1_from_y_axis(refined_distances, refined_angles_to_x)
    distance1_x=Take_Distance1_from_x_axis(refined_distances, refined_angles_to_x)
    
    
    if type(distance1_y)==type(None) and type(distance1_x)==type(None):
        if len(refined_distances)==0:
            #to address amorphous materials, set distance1=0
            distance1=0
        else:
            distance1=np.random.choice(refined_distances)
    elif type(distance1_y)==type(None):
        distance1=distance1_x
    elif type(distance1_x)==type(None):
        distance1=distance1_y
    else:
        distances_array=np.array([distance1_y,distance1_x])
        distance1=distances_array[np.argmax(distances_array)]
        
    
    print('Distance 1')
    print(distance1)
    
    #to address amorphous materials, set distance1=0
    if distance1==0:
        #refined_distances=np.array([0,0])
        #refined_angles_to_x=np.array([0,0])
        distances2=np.array([0])
        angles=np.array([0])
        angle_x_to_dist1=None
    else:
        
        indexdist1=np.argmin(np.abs(refined_distances-distance1)) 
        angle_x_to_dist1=refined_angles_to_x[indexdist1]
        print('Angle to x of distance 1')
        print(angle_x_to_dist1)
        
        distances2=np.delete(refined_distances, indexdist1)
        angles=np.abs(refined_angles_to_x-refined_angles_to_x[indexdist1])
        angles=np.delete(angles, indexdist1)
        angles[angles>180]=360-angles[angles>180]

        

    ZA_list=[]
    
    for d2, angle in zip(distances2,angles):
        
        print("Between distance 1 = {:.3f} A and distance 2 = {:.3f} A, there are {:.3f} degrees".format(distance1, d2, angle))
    
    
        n = crystal_object.FindZA(distance1,d2,angle,tol)
        print("Found ",n,"possible Zone Axes")
        
        # #costum stuff for hkls with angles
        # hkls1=[]
        # hkls2=[]
        # for index in range(n):
        #     hkl1=material_crystal.gethkls1(index)
        #     hkl2=material_crystal.gethkls2(index)
        #     hkls1.append(hkl1)
        #     hkls2.append(hkl2)
            
        # print('hkls1')
        # print(hkls1)
        # print('hkls2')
        # print(hkls2)
        # print('angle')
        # if len(hkls1)!=0 and len(hkls2)!=0:
        #     angle1=abs(crystal_object.angle(hkls1[0],hkls2[0]))
        # else: 
        #     angle1=0
        # print(angle1)
        # print('end hkls')
        

        #Include or not the possibility to output high index axes
        high_index=True
        
        if n==0:
            ZA=None
        else:
            if high_index==high_index:
                ZA=crystal_object.getZA(0)
            else:
                for index in range(n):
                    ZA=crystal_object.getZA(index)
                    if np.sum(np.abs(ZA))< 10: #if sum of indices is higher than 10 is already high indexes
                        break
                    else:
                        ZA=None
                        continue
                    print(crystal_object.getZA(index))
            
        
        print(ZA)
        
        if type(ZA)==type(None):
            ZA_list.append(ZA)
        else:
            #sort them as cubic structure has same axis equivalences
            ZA_list.append(np.sort(np.abs(ZA)))
    
    
    ZA_list_index=[index for index in ZA_list if type(index)!=type(None)]
    ZA_arr_index=np.asarray(ZA_list_index)
    
    if len(ZA_arr_index)==0:
        ZA_final=None
    else:
        temp_array_ZA = np.ascontiguousarray(ZA_arr_index).view(np.dtype((np.void, ZA_arr_index.dtype.itemsize * ZA_arr_index.shape[1])))
        _, idx,counts = np.unique(temp_array_ZA, return_index=True, return_counts=True)
        
        ZA_unique = np.unique(temp_array_ZA).view(ZA_arr_index.dtype).reshape(-1, ZA_arr_index.shape[1])
        ZA_final=ZA_unique[np.argmax(counts)]
        
    
    
    
    #Together with ZA, we need an extra direction to define the structure
    #Use the distance1 and its angle to x to define everything
    
    #Get distances from reflections
    distances=[]
    for index in range(N):
        ds=crystal_object.getDistances(index)
        distances.append(ds)
        
    #Get indexes from reflections
    indexes=[]
    for index in range(N):
        hkls=crystal_object.getIndexes(index)
        indexes.append(hkls)
        
    
    
    reference_plane=indexes[np.argmin(np.abs(distances-distance1))]
    print('Found ZA:', ZA_final)
    print('Plane:', reference_plane, 'at',angle_x_to_dist1, 'degrees from x axis.')
            
      
    return ZA_final, reference_plane, angle_x_to_dist1


def Find_ZA_poly(crystal_object,forbidden, min_d,N_tot_refl, tolerance, refined_distances, refined_angles_to_x, high_index):
    '''
    

    Parameters
    ----------
    crystal_object : CrystalHandle object with diffractions already computed .Diff(forbidden, min_d)
    forbidden : Bool for allowing forbidden reflections (True) or not (False)
    min_d : minimum interplanar distance computed in the diffraction simulation
    N_tot_refl : total number of reflections computed with the Diff method of the crystal
    tolerance : tolerance when comparing real and experimental distances and angles
    refined_distances : array containing the distance, in ANGSTROMS, and having applied the 
            Ensure_Center_Diff function to extract the center of diffractogram contributions
    refined_angles_to_x : array containing the angles, in Degrees, and having applied the 
            Ensure_Center_Diff function to extract the center of diffractogram contributions
    high_index: bool, allow to output high index ZA as a valid ZA when True, or not when False
            it can be varied but it is a high index ZA when h+k+l>=10

    Returns
    -------
    Information to fully characterise the studied phase in 3D
    ZA_final : most common ZA found, array
    reference_plane : plane used as reference, d1, for computing the rest, float
    angle_x_to_dist1 : angle from reference_plane to x axis, in degrees

     '''
    
    
    
    ZA_per_spot=[]
    reference_planes=[]
    angle_x_to_dist1_s=[]
    
    #Get distances from reflections
    distances=[]
    for index in range(N_tot_refl):
        ds=crystal_object.getDistances(index)
        distances.append(ds)
    
    #Get indexes from reflections
    indexes=[]
    for index in range(N_tot_refl):
        hkls=crystal_object.getIndexes(index)
        indexes.append(hkls)
    distances=np.asarray(distances)
    indexes=np.asarray(indexes)
    
    print(distances)
    print(indexes)

    for index,(distance,angle) in enumerate(zip(refined_distances,refined_angles_to_x)):
        
        refined_distances_changes=np.copy(refined_distances)
        refined_angles_to_x_changes=np.copy(refined_angles_to_x)
        
        if len(refined_distances)==0:
            #to address amorphous materials, set distance1=0
            distance1=0
            distances2=np.array([0])
            angles=np.array([0])
            angle_x_to_dist1=None
        else:
        
            distance1=distance
            angle_x_to_dist1=angle
            
            distances2=np.delete(refined_distances_changes, index)
            angles=np.abs(refined_angles_to_x_changes-angle_x_to_dist1)
            angles=np.delete(angles, index)
            angles[angles>180]=360-angles[angles>180]



        ZA_list=[]
        
        for d2, angle in zip(distances2,angles):
            print("Between distance 1 = {:.3f} A and distance 2 = {:.3f} A, there are {:.3f} degrees".format(distance1, d2, angle))
        
        
            n = crystal_object.FindZA(distance1,d2,angle,tol)
            print("Found ",n,"possible Zone Axes")
        
            
            
            # #costum stuff for hkls with angles
            # ZAs=[]
            # hkls1=[]
            # hkls2=[]
            # for index in range(n):
            #     ZA=crystal_object.getZA(index)
            #     hkl1=material_crystal.gethkls1(index)
            #     hkl2=material_crystal.gethkls2(index)
            #     hkls1.append(hkl1)
            #     hkls2.append(hkl2)
            #     ZAs.append(ZA)
            
            # #ZA sorted given score, so ZA_sorted contains already all the ZA in order of the most likely to the least
            # ZA_scores, ZA_sorted=ZA_score(crystal_object, distance1,d2,angle,ZAs, hkls1, hkls2, distances, indexes)
            

                
            # print('hkls1')
            # print(hkls1)
            # print('hkls2')
            # print(hkls2)
            # print('angle')
            # if len(hkls1)!=0 and len(hkls2)!=0:
            #     angle1=abs(crystal_object.angle(hkls1[0],hkls2[0]))
            # else: 
            #     angle1=0
            # print(angle1)
            # print('end hkls')
            
            
            
            #Include or not the possibility to output high index axes
            high_index=True
            
            if n==0:
                ZA=None
            else:
                if high_index==high_index:
                    ZA=crystal_object.getZA(0)
                else:
                    for index in range(n):
                        ZA=crystal_object.getZA(index)
                        if np.sum(np.abs(ZA))< 10: #if sum of indices is higher than 10 is already high indexes
                            break
                        else:
                            ZA=None
                            continue
                        print(crystal_object.getZA(index))
                
            
            print(ZA)
            
            if type(ZA)==type(None):
                ZA_list.append(ZA)
            else:
                #sort them as cubic structure has same axis equivalences
                ZA_list.append(np.sort(np.abs(ZA)))
        
        
        ZA_list_index=[index for index in ZA_list if type(index)!=type(None)]
        ZA_arr_index=np.asarray(ZA_list_index)
        if len(ZA_arr_index)==0:
            ZA_final=None
        else:
            temp_array_ZA = np.ascontiguousarray(ZA_arr_index).view(np.dtype((np.void, ZA_arr_index.dtype.itemsize * ZA_arr_index.shape[1])))
            _, idx,counts = np.unique(temp_array_ZA, return_index=True, return_counts=True)
            
            ZA_unique = np.unique(temp_array_ZA).view(ZA_arr_index.dtype).reshape(-1, ZA_arr_index.shape[1])
            ZA_final=ZA_unique[np.argmax(counts)]
            
        #Together with ZA, we need an extra direction to define the structure
        #Use the distance1 and its angle to x to define everything
        
        

        reference_plane=indexes[np.argmin(np.abs(distances-distance1))]
        
        ZA_per_spot.append(ZA_final)
        reference_planes.append(reference_plane)
        angle_x_to_dist1_s.append(angle_x_to_dist1)
          
        print('Found ZA:', ZA_final)
        print('Plane:', reference_plane, 'at',angle_x_to_dist1, 'degrees from x axis.')
                
    
    ZA_per_spot_arr=np.asarray(ZA_per_spot)
    reference_planes_arr=np.asarray(reference_planes)
    angle_x_to_dist1_s_arr=np.asarray(angle_x_to_dist1_s)
    
    shape_original=list(np.shape(ZA_per_spot_arr))
  
    #change the pairings that have led to None axis, to [000]
    #then the keyword [000] refers to No axis found, and then we keep the positions
    #of these arrays that have no axis
    
    ZA_per_spot_list_new=[]
    for index,ZA in enumerate(ZA_per_spot_arr):
        if type(ZA)==type(None):
            ZA_per_spot_list_new.append([0,0,0])
        else:
            ZA_per_spot_list_new.append(ZA)
    
            
    ZA_per_spot_arr_new=np.asarray(ZA_per_spot_list_new)
    np.reshape(ZA_per_spot_arr_new, (shape_original[0],3))
    if len(ZA_per_spot_arr_new)==0:
        ZA_final_final=None
        reference_planes_final=None
        angle_x_to_dist1_final=None
        print('Amorphous')
    else:
        temp_array_ZA_s = np.ascontiguousarray(ZA_per_spot_arr_new).view(np.dtype((np.void, ZA_per_spot_arr_new.dtype.itemsize * ZA_per_spot_arr_new.shape[1])))
        _, idx,counts = np.unique(temp_array_ZA_s, return_index=True, return_counts=True)
        
        ZA_unique_s=ZA_per_spot_arr_new[idx]
                
        ZA_final_final=ZA_unique_s[np.argmax(counts)]
        
        reference_planes_unique=reference_planes_arr[idx]
        reference_planes_final=reference_planes_unique[np.argmax(counts)]
        
        angle_x_to_dist1_unique=angle_x_to_dist1_s_arr[idx]
        angle_x_to_dist1_final=angle_x_to_dist1_unique[np.argmax(counts)]
        
        #if 000 axis is the most found one, switch to the second most found as ZA final
        if ZA_final_final[0]==0 and ZA_final_final[1]==0 and ZA_final_final[2]==0 and len(idx)>1:
            ZA_unique_s_copy=np.copy(ZA_unique_s)
            reference_planes_unique_copy=np.copy(reference_planes_unique)
            angle_x_to_dist1_unique_copy=np.copy(angle_x_to_dist1_unique)
            counts_copy=np.copy(counts)
            
            ZA_unique_s_copy=np.delete(ZA_unique_s_copy,np.argmax(counts_copy),axis=0)
            reference_planes_unique_copy=np.delete(reference_planes_unique_copy,np.argmax(counts_copy), axis=0)
            angle_x_to_dist1_unique_copy=np.delete(angle_x_to_dist1_unique_copy,np.argmax(counts_copy))
            counts_copy=np.delete(counts_copy,np.argmax(counts_copy))      
            
            ZA_final_final=ZA_unique_s_copy[np.argmax(counts_copy)]
        
            reference_planes_final=reference_planes_unique_copy[np.argmax(counts_copy)]
        
            angle_x_to_dist1_final=angle_x_to_dist1_unique_copy[np.argmax(counts_copy)]
        
        
        print('Most commonly found ZA, its reference plane, and angle to x axis')
        print('ZA:',ZA_final_final)
        print('Ref. plane:',reference_planes_final)
        print('Angle to x:',angle_x_to_dist1_final)
     
        print('ZAs found:')
        for index in idx:
            n_times_found=counts[np.argmin(np.abs(idx-index))]
            ZA_i=ZA_per_spot_arr_new[index]
            reference_plane_i=reference_planes_arr[index]
            angle_x_to_dist1_i=angle_x_to_dist1_s_arr[index]
            print('Found on', n_times_found, 'spots')
            print('ZA:',ZA_i)
            print('Ref. plane:',reference_plane_i)
            print('Angle to x:',angle_x_to_dist1_i)
        print('Note: [000] Means no axis was found')
            
    
    return ZA_final_final, reference_planes_final, angle_x_to_dist1_final

# only for cubic case
def ZAs_scores_OLD(crystal_object, d1,d2,angle,ZAs, hkls1, hkls2, distances_arr, reflections_arr):
    
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
        list_of_permutations.append(reflection_indices)
        list_of_permutations.append([-h_init, -k_init ,-l_init])
        
    elif space_group > 2 and space_group <= 15:
        # monoclinic
        # trivial negative
        list_of_permutations.append(reflection_indices)
        list_of_permutations.append([-h_init, -k_init ,-l_init])
        # only h and l paired signs change
        list_of_permutations.append([-h_init, k_init ,-l_init])
        list_of_permutations.append([h_init, -k_init ,l_init])
        
    elif space_group > 15 and space_group <= 74:
        # orthorombic
        # any sign change allowed but no indices swap
        list_of_permutations = list(permute_signs(reflection_indices))
        
    elif space_group > 74 and space_group <= 142:
        # tetragonal
        # swap h and k and then any sign permutation
        list_of_permutations = list(permute_signs(reflection_indices))
        list_of_permutations = list_of_permutations + list(permute_signs([k_init, h_init, l_init]))
        
    elif space_group > 142 and space_group <= 167:
        # trigonal/rhombohedral
        # any order swap
        list_of_permutations = list(multiset_permutations(reflection_indices))
        # and get the trivial sign inversion for all of them
        negative_reflections = []
        for reflection in list_of_permutations:
            h_i, k_i, l_i = reflection
            negative_reflections.append([-h_i, -k_i, -l_i])
        
        list_of_permutations = list_of_permutations + negative_reflections
        
    elif space_group > 167 and space_group <= 194 :
        # hexagonal
        # trivial negative
        list_of_permutations.append(reflection_indices)
        list_of_permutations.append([-h_init, -k_init ,-l_init])
        
        # h and k swap and paired signs change  
        list_of_permutations.append([-h_init, -k_init ,l_init])
        list_of_permutations.append([h_init, k_init ,-l_init])
        
        list_of_permutations.append([k_init, h_init, l_init])
        list_of_permutations.append([-k_init, -h_init, -l_init])
        
        list_of_permutations.append([-k_init, -h_init, l_init])
        list_of_permutations.append([k_init, h_init, -l_init])
        
    else:  # space_group > 194 and space_group <= 230
        # cubic
        # all possible permutations allowed
        list_of_permutations_temp = list(multiset_permutations(reflection_indices))
        for permutation in list_of_permutations_temp:
            refl = list(permute_signs(permutation))
            list_of_permutations = list_of_permutations + refl
    
    return list_of_permutations



# new function with any space group working
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
        
        hkl1_permutations = Crystal_System_Equidistance_Permutations(space_group_num, hkl1)
        hkl2_permutations = Crystal_System_Equidistance_Permutations(space_group_num, hkl2)
        
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




def Prepare_exp_distances_angles_pixels(refined_distances_exp, refined_angles_exp, refined_pixels_exp, min_d):
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
      
    return more_refined_distances_exp_f, more_refined_angles_exp_f, more_refined_pixels_exp_f



class Scored_spot_pair_OLD:
    
    def __init__(self, spot1_int_ref, spot2_int_ref, crystal_object):
        self.spot1_int_ref=spot1_int_ref
        self.spot2_int_ref=spot2_int_ref
        self.phase=crystal_object
        
        phase_name_temp=crystal_object.phase_name
        strin=str(phase_name_temp)
        # ensure the directory is writen with the os.path.sep delimiter \\ or \ or /
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
        
    def Zone_axis(self, ZA, reference_plane):
        self.ZA=ZA
        self.reference_plane=reference_plane
        
    def Score(self, score):
        self.score=score
        
     
 

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
           
     
     
# Depcrecated version of the function, to keep in case the new one crystla system independent fails at some point
def Score_all_spot_pairs_OLD(crystal_objects,forbidden, min_d, tolerance, refined_distances, refined_angles_to_x):
    
    all_scored_spot_pairs=[]
    
    spots_int_reference_changes=np.arange(len(refined_distances))
    
    #iterate over all the possible phases, crystal objects, that can be represented in the FFT
    for crystal_object in crystal_objects:
        #compute the theoretical reflections and distances for each phase with the crystal instance loaded
        #as the diffraction is computed inside
        N_tot_refl = crystal_object.Diff(forbidden,min_d)
        distances_refl=[]
        indexes_refl=[]
        print(N_tot_refl)
        for index in range(N_tot_refl):
            ds=crystal_object.getDistances(index)
            distances_refl.append(ds)
            hkls=crystal_object.getIndexes(index)
            indexes_refl.append(hkls)
            
        distances_refl=np.asarray(distances_refl)
        indexes_refl=np.asarray(indexes_refl)
        
        for dist, ind in zip(distances_refl[:50],indexes_refl[:50]) :
            print('distances refl')
            print(dist)
            print('indexes refl')
            print(ind)

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
                spot_order1 = [spot_int_ref, spot_int_ref2]
                spot_order2 = [spot_int_ref2, spot_int_ref]
                if (spot_order1 in considered_spot_pairs) or (spot_order2 in considered_spot_pairs):
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
                        n = crystal_object.FindZA(distance1,d2,angle,tol)
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
                                ZA_n=crystal_object.getZA(possible_ZA)
                                print(ZA_n)
                                ZAs_spot_pair.append(ZA_n)
                                #!!! hkls1 and 2 indexes of spots d1 and d2
                                hkl1=crystal_object.gethkls1(possible_ZA)
                                hkl2=crystal_object.gethkls2(possible_ZA)
                                hkls1.append(hkl1)
                                hkls2.append(hkl2)
                            
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


# New function to use with crystal system independence
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
                        print('tlerance', tolerance)
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
        # index used in case more than one crystal in the same ZA appears, given same phase, would be 
        # differentiated by this index
        self.ZA_priv_index=ZA_priv_index
    

# !!! DO NOT USE THIS FUNCTION BECAUSE IT IS BUGGED
def Find_All_Crystals_BUGGED(list_all_cryst_spots, all_scored_spot_pairs):
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
    else:
        scored_spot_target=all_scored_spot_pairs[0]
    # contains the internal reference for the spots for each ZA and phase
    spot_int_refs_phase_axis=[]
    # spot target characteristics
    spot_pair_target_index1=scored_spot_target.spot1_int_ref
    spot_pair_target_index2=scored_spot_target.spot2_int_ref
    print('spot indexes')
    print(spot_pair_target_index1)
    print(spot_pair_target_index2)
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
    print(unique_spot_int_refs_poss)
    unique_spot_int_refs_poss=np.delete(unique_spot_int_refs_poss, np.where(unique_spot_int_refs_poss==spot_pair_target_index1))
    unique_spot_int_refs_poss=np.delete(unique_spot_int_refs_poss, np.where(unique_spot_int_refs_poss==spot_pair_target_index2))
    #this array only contains the possible spot candidates and no longer has tuples but arrays
    #list of indexes of list all_scored_spot_pairs that must be deleted after all the checks and dictionary modifications
    # !!!ERROR HERE the pair used as a reference, indexed as 0 must be deleted and included 
    indexes_all_spot_pairs_to_delete=[0]
    #for each possible spot candidate to reference the crystal
    #does not find everything all the spot pairs
    
    for spot_int_ref_possible in unique_spot_int_refs_poss:
        total_spot_pairs_contain_poss=[]
        temp_index_all_scored_spot_pairs=[]
        #for each spot pair that may represent the crystal spot pairs
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
                        
    #add stuff to class about the elements or spot pairs added, identified by the indexes_all_spot_pairs_to_delete
    # which are all the elements of all_scored_spot_pairs that belong to the crystal targeted by scored_spot_target
    
    crystal_identified=Crystal_spots(spot_int_refs_phase_axis)
    
    crystal_scored_spot_pairs = [all_scored_spot_pairs[i] for i in indexes_all_spot_pairs_to_delete]
    print(crystal_scored_spot_pairs)
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
    
    #we do not return anything as this fills the empty list provided
    return 
    
    
    
# Function that works fine
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


#Adaptative gaussian filtering
def Adaptative_Gaussian_Blur(image_array, image_scale):
    '''
    Denoises the image based on the image size and scale to properly and tailoredly fit the features displayed
    This only depends on the scale of the image, not on the FOV, as the FWHM depends only on the feature size,
    e.g. interplanar distances, and this is defined by the magnification
    image scale must be in nm/pixels
    Eventually we should not need any filtering as the DL model should do all the work
    '''
    #average feature size in a crystal, arround 2-3 angrstoms = 0.2-0.3 nm, larger the more secure
    avg_feature_size=0.3  #nm
    FWHM_gaussian=2*avg_feature_size
    desvest_nm=FWHM_gaussian/(2*np.sqrt(2*np.log(2)))
    desvest_pixels=int(np.ceil(desvest_nm/image_scale))  # in number of pixels
    kernel_size=2*desvest_pixels+1  #too agressive
    kernel_size=3
    gaussian_blurred_image=cv2.GaussianBlur(image_array, (kernel_size, kernel_size), desvest_pixels)
    return gaussian_blurred_image
    


    
   



#Hyperparameters
gauss_blur_filter_size=(5,5)  #size of smoothing filter, go to line to change sigma
downscaling_factor=1 #for trials, n factor of downsampling size of image
FFT_thresholding=0.5  #value above which the pixels are kept
st_distance=30 #distance parameter in the Stem tool method
FFT_thresholdingG=0.6 #value above which the pixels are kept, in the gaussian filtered FFT
window_size=2048  #window size of the sliding windows
tol=0.05 #tolerance: how different from theoretical values the previous values can be to get good output
min_d=0.5    #minimum interplanar distance computed in the diffraction
forbidden = True  #Include (True) or not (False) the forbidden reflections

np.random.seed(int(time.time()))

#dm3 loading, and calibration extraction
imagedm3=hs.load(r'E:\Arxius varis\PhD\2nd_year\Code\trial_images\dm3_atomic_resolution\GeQW2.dm3')
meta1=imagedm3.metadata
meta2=imagedm3.original_metadata.export('parameters')



x_calibration=imagedm3.axes_manager['x'].scale
y_calibration=imagedm3.axes_manager['y'].scale

x_pixels_original=imagedm3.axes_manager['x'].size
y_pixels_original=imagedm3.axes_manager['y'].size

x_units=imagedm3.axes_manager['x'].units
y_units=imagedm3.axes_manager['y'].units


#FFT calibration

#FFT_calibration,FFT_pixels,FFT_units=FFT_calibration(imagedm3)



imagearray=np.asarray(imagedm3)
image=imagearray

plt.imshow(image, cmap=plt.cm.gray, vmin=image.min(), vmax=image.max())
plt.show()

#Crop the image if wanted
if window_size==x_pixels_original:
    init_x=0
    init_y=0
else:
    init_y=np.random.choice(np.arange(0,image.shape[0]-window_size,1)) 
    init_x=np.random.choice(np.arange(0,image.shape[1]-window_size,1)) 

#image=image[init_y:init_y+window_size,init_x:init_x+window_size]
hs_image_cropping=imagedm3.isig[init_y:init_y+window_size,init_x:init_x+window_size]


FFT_calibration,FFT_pixels,FFT_units=FFT_calibration(hs_image_cropping)
#Correct calibration in case it is necessary, if no correction is needed, just us 1
real_calibration_factor=1
FFT_calibration,FFT_pixels,FFT_units=FFT_calibration_Correction(hs_image_cropping, real_calibration_factor)

print(x_calibration)
print(FFT_calibration)
print(FFT_units)
image=np.asarray(hs_image_cropping)
#State that the FFT has the same pixels as the windows created

#First standarisation of the image for filtering/blurring it with gaussian filter
image_st=(image-np.min(image))/np.max(image-np.min(image))

plt.imshow(image_st, cmap=plt.cm.gray, vmin=image_st.min(), vmax=image_st.max())
plt.show()


#Application of Gaussian filter for denoising


denoised_image=Adaptative_Gaussian_Blur(image_st, x_calibration)
#denoised_image=cv2.GaussianBlur(image_st, gauss_blur_filter_size, 1)

#Second standarisation of the image after filtering/blurring it with gaussian filter

image_st=(denoised_image-np.min(denoised_image))/np.max(denoised_image-np.min(denoised_image))

#Print histogram



#For sake of evaluation, better work with an image with less pixels, as only the consecutive pixel evaluation would take
#approximately 6 hours to run for a big region of 250.000 pixels in total.

#Then downsample the image and upsample it posteriorly 
#We select a max pooling method to keep track of the brighter elements and this way keep a higher contrast


ds_image=measure.block_reduce(image_st, block_size=tuple(np.int32(downscaling_factor*np.ones(len(np.shape(image_st))))), func=np.mean, cval=0)

#and standarise it again to ensure 0-1 values

ds_image_st=(ds_image-np.min(ds_image))/np.max(ds_image-np.min(ds_image))


# take the fft of the image
fft_image_w_background = np.fft.fftshift(np.log(np.fft.fft2(ds_image_st)))
fft_abs_image_background = np.abs(fft_image_w_background)

# apply the filter
fft_abs_image_background2=np.copy(fft_abs_image_background)
fft_abs_image_backgroundc=np.copy(fft_abs_image_background)


fft_abs_image_backgroundc=(fft_abs_image_backgroundc-np.min(fft_abs_image_backgroundc))/np.max(fft_abs_image_backgroundc-np.min(fft_abs_image_backgroundc))


fft_abs_image_background2=(fft_abs_image_background2-np.min(fft_abs_image_background2))/np.max(fft_abs_image_background2-np.min(fft_abs_image_background2))


fft_abs_image_background2=cv2.GaussianBlur(fft_abs_image_background2, gauss_blur_filter_size, 1)
fft_abs_image_background2=(fft_abs_image_background2-np.min(fft_abs_image_background2))/np.max(fft_abs_image_background2-np.min(fft_abs_image_background2))


#trial with original FFT
#fft_abs_image_background2=fft_abs_image_backgroundc

#Automatic hyperparameter finding
fov=np.shape(fft_abs_image_background2)[0]*y_calibration
print('fov',fov)
st_distance,_,FFT_perc=FFT_hyperparams(fft_abs_image_background2,fov)
print('mpfit',st_distance,'perc',FFT_perc )
FFT_thresholdingG=Threshold_given_percentage(fft_abs_image_background2, FFT_perc)
print('fft_threhs',FFT_thresholdingG)


print('STEM Tool based method (2D Gaussians)')

center_difractogram=(int(FFT_pixels/2), int(FFT_pixels/2))

print('ST gaussian FFT')
twodfit_blur=stemtool.afit.peaks_vis(fft_abs_image_background2, dist=st_distance, thresh=FFT_thresholdingG, imsize=(15, 15))

d_distances=Spot_coord_To_d_spacing_vect(twodfit_blur, FFT_calibration, FFT_pixels)

angles_to_x=Spot_coord_To_Angles_to_X_vect(twodfit_blur,FFT_pixels)


refined_distances, refined_angles_to_x, refined_pixels=Ensure_Center_Diff(d_distances, angles_to_x,twodfit_blur)
#!!! set the values of distances in angstroms
refined_distances=refined_distances*10    

refined_distances, refined_angles_to_x, refined_pixels=Prepare_exp_distances_angles_pixels(refined_distances, refined_angles_to_x, refined_pixels, min_d)
spots_int_reference=np.arange(len(refined_distances))
print(refined_distances)
print(refined_angles_to_x)
#prepare the distances to be analysed


crystal_objects_list=[]
space_group_list = []
    

unit_cell_path1=r'E:/Arxius varis/PhD/2nd_year/Code/unit_cells/Nb.uce'
# the argument must be the bytes to the path not the string (just encode the string)
crystal_object = Crystal(bytes(unit_cell_path1.encode()))
crystal_objects_list.append(crystal_object)   

# store unit cell information: space group
unit_cell_text = open(unit_cell_path1)

for line in unit_cell_text:
    if line[:4] == 'RGNR':
        space_group = int(line[4:])

space_group_list.append(space_group)
    

unit_cell_path2=r'E:/Arxius varis/PhD/2nd_year/Code/unit_cells/Ge.uce'
# the argument must be the bytes to the path not the string (just encode the string)
crystal_object = Crystal(bytes(unit_cell_path2.encode()))
crystal_objects_list.append(crystal_object)   

# store unit cell information: space group
unit_cell_text = open(unit_cell_path2)

for line in unit_cell_text:
    if line[:4] == 'RGNR':
        space_group = int(line[4:])

space_group_list.append(space_group)
    
   
    
# material_crystal1 = Crystal(b'E:/Arxius varis/PhD/2nd_year/Code/unit_cells/Nb.uce')
# material_crystal2 =Crystal(b'E:/Arxius varis/PhD/2nd_year/Code/unit_cells/Ge.uce')
# #material_crystal1=Crystal(b'E:\Arxius varis\PhD\Supervision\Ivan_Pinto_TFG_Student\Files_cells\Ag2Se_Orto.uce')
# crystal_objects=[material_crystal1, material_crystal2]
# N = material_crystal1.Diff(forbidden,min_d)
# print("Calculated ",N ,"reflections")
# polycrystalline=True
# high_index=True


# if polycrystalline==True:
#     Find_ZA_poly(material_crystal,forbidden, min_d,N, tol, refined_distances, refined_angles_to_x, high_index)
# else:
#     Find_ZA_mono(material_crystal,forbidden, min_d,N, tol, refined_distances, refined_angles_to_x, high_index)
import time
init_t = time.time()
all_scored_spot_pairs=Score_all_spot_pairs(
    crystal_objects_list, space_group_list ,forbidden, min_d, tol, refined_distances, refined_angles_to_x)
# print(all_scored_spot_pairs)
end_t = time.time()
print('total_time', end_t - init_t)
# for scored in all_scored_spot_pairs[:40]:
#     print('Scored')
#     print('score',scored.score)
#     print('phase', scored.phase_name)
#     print('spot1 int', scored.spot1_int_ref)
#     print('dist1', scored.spot1_dist)
#     print('spot2 int', scored.spot2_int_ref)
#     print('dist2', scored.spot2_dist)
#     print('angle', scored.angle_between)
#     print('phase obj', scored.phase)
#     print('ZA', scored.ZA)
#     print('ref plane', scored.reference_plane)
        
all_scored_spot_pairs_to_mod=all_scored_spot_pairs.copy()

list_all_cryst_spots=[]

Find_All_Crystals(list_all_cryst_spots, all_scored_spot_pairs_to_mod)

# print(len(list_all_cryst_spots))

# for cryst in list_all_cryst_spots[:6]:
#     print('Cryst')
#     print('spot list', cryst.spots)
#     print('spot pairs', cryst.spot_pairs_obj)
#     print('phase name', cryst.phase_name)
#     print('ZA', cryst.ZA)
#     print('ZA priv index', cryst.ZA_priv_index)

list_refined_cryst_spots=Crystal_refinement(list_all_cryst_spots,spots_int_reference)

print(len(list_refined_cryst_spots))

for cryst in list_refined_cryst_spots:
    print('Cryst')
    print('spot list', cryst.spots)
    print('spot pairs', cryst.spot_pairs_obj)
    print('phase name', cryst.phase_name)
    print('ZA', cryst.ZA)
    print('ZA priv index', cryst.ZA_priv_index)
    
    

    
    