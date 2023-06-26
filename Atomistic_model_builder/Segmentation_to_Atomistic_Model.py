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

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import os
import gdspy
import matplotlib.pyplot as plt
import skimage
import skimage.measure
import sklearn.cluster
import sklearn.mixture
import cv2

import hyperspy.api as hs

import ase
from ase.io import read, write
from ase.visualize import view
from abtem.visualize import show_atoms
from ase.build import surface, make_supercell, find_optimal_cell_shape, rotate
import mendeleev as mdl
import sys

sys.path.append(r'E:\Arxius varis\PhD\4rth_year\Code\Functions\General_functions')
sys.path.append(r'E:\Arxius varis\PhD\4rth_year\Code\Atomistic_model_builder')
sys.path.append(r'E:\Arxius varis\PhD\3rd_year\Code\Functions')
sys.path.append(r'E:\Arxius varis\PhD\4rth_year\Code\Functions\Ivans_Files_2\IVAN\Segmentation_model')

import Atomistic_Model_Builder as AtomBuild
import Segmentation_Wrapped as SegmWrap
import Segmentation_1stAprox as Segment



'''
Build the segmentation of the image we are working with
'''


#Required functions

#K-means clustering functions
def km_clust(array, n_clusters):
    # Create a line array, the lazy way
    X = array.reshape((-1, 1))
    # Define the k-means clustering problem
    k_m = sklearn.cluster.KMeans(n_clusters=n_clusters, init='random',n_init=4,random_state=2**13)
    # Solve the k-means clustering problem
    k_m.fit(X)
    # Get the coordinates of the clusters centres as a 1D array
    values = k_m.cluster_centers_.squeeze()
    # Get the label of each point
    labels = k_m.labels_
    cost=k_m.inertia_
    return(values, labels, cost)

#Gets the best K-means clustering option from 100 iterations
def best_km(array, n_clusters):
    iterations=100
    kmeans_values=[]
    kmeans_labels=[]
    kmeans_cost=[]
    for index,i in enumerate(range(iterations)):
        values, labels, cost= km_clust(array, n_clusters = n_clusters)
        kmeans_values.append(values)
        kmeans_labels.append(labels)
        kmeans_cost.append(cost)
    kmeans_cost=np.array(kmeans_cost)
    best_cost=kmeans_cost.min()    
    best_values=kmeans_values[kmeans_cost.argmin()]
    best_labels=kmeans_labels[kmeans_cost.argmin()]
    return (best_values, best_labels,best_cost)

#Mean shift clustering algorithm
def Mean_shift(image):
    image_reshaped=np.reshape(image, (image.shape[0]*image.shape[1],1))
    # The following bandwidth can be automatically detected using
    #bandwidth = sklearn.cluster.estimate_bandwidth(image, quantile=0.2, n_samples=500)
    
    ms = sklearn.cluster.MeanShift( bin_seeding=True)
    ms.fit(image_reshaped)
    labels = ms.labels_
    cluster_centers = ms.cluster_centers_
    
    return labels, cluster_centers


#Consecutive pixels for a given label, giving an initial target pixel to start with
def Consecutive_pixels_particular_pixel(image_array_labels, target_coords):
    # Same as previous but giving a starting pixel to start from to do the consecutive value analysis


    class pixels_state:
        def __init__(self, coordinates, value, state):
            '''
            State can be 'Old', referring to pixels that were in the array before and therefore do not need to be considered 
            for checking again, or 'New', meaning that it has just been uploaded to the array and has to act as a point for getting new 
            conscutive pixels
            '''  
            self.coords=coordinates
            self.value=value
            self.state=state
        
    ver_pixs=image_array_labels.shape[0]
    hor_pixs=image_array_labels.shape[1]
    
    contiguous_cluster=[]
           
    target_pixel=image_array_labels[target_coords[0],target_coords[1]] 
    
    #process of converting into the class of pixels and appending them into the cluster's array
    
    
        
    #the list contains all the pixels that are contiguous to the first one, each element is a pixels_state class
    
    
    target_pixel_classed=pixels_state((target_coords[0],target_coords[1]), target_pixel, state='New')
    contiguous_cluster.append(target_pixel_classed) 
    
    
    #a list contains all the new pixels found in each iteration
    
    all_coordinates=[(target_coords[0],target_coords[1])]
    for pixel in contiguous_cluster:
        
        
        if pixel.state=='New':
          
            array_containing_all_new_pixels=[]
            (pixel_y,pixel_x)=pixel.coords
            
            if pixel_y==0 and pixel_x!=0 and pixel_x!=hor_pixs-1:
                pixel_N=None
                pixel_NW=None
                pixel_NE=None
                pixel_W=image_array_labels[pixel_y,pixel_x-1] 
                pixel_SW=image_array_labels[pixel_y+1,pixel_x-1] 
                pixel_S=image_array_labels[pixel_y+1,pixel_x] 
                pixel_SE=image_array_labels[pixel_y+1,pixel_x+1] 
                pixel_E=image_array_labels[pixel_y,pixel_x+1] 
            
            #condition for left upper pixel
            elif pixel_y==0 and pixel_x==0: 
                pixel_N=None
                pixel_NW=None
                pixel_NE=None
                pixel_W=None
                pixel_SW=None
                pixel_S=image_array_labels[pixel_y+1,pixel_x] 
                pixel_SE=image_array_labels[pixel_y+1,pixel_x+1] 
                pixel_E=image_array_labels[pixel_y,pixel_x+1] 
                
            #condition for right upper pixel
            elif pixel_y==0 and pixel_x==hor_pixs-1: 
                pixel_N=None
                pixel_NW=None
                pixel_NE=None
                pixel_E=None
                pixel_SE=None
                pixel_W=image_array_labels[pixel_y,pixel_x-1] 
                pixel_SW=image_array_labels[pixel_y+1,pixel_x-1] 
                pixel_S=image_array_labels[pixel_y+1,pixel_x] 
                
            #condition for bottom limit pixels, not edge ones
            elif pixel_y==ver_pixs-1 and pixel_x!=0 and pixel_x!=hor_pixs-1:
                pixel_S=None
                pixel_SW=None
                pixel_SE=None
                pixel_N=image_array_labels[pixel_y-1,pixel_x] 
                pixel_NW=image_array_labels[pixel_y-1,pixel_x-1] 
                pixel_W=image_array_labels[pixel_y,pixel_x-1] 
                pixel_E=image_array_labels[pixel_y,pixel_x+1] 
                pixel_NE=image_array_labels[pixel_y-1,pixel_x+1] 
                
            #condition for left bottom pixel 
            elif pixel_y==ver_pixs-1 and pixel_x==0: 
                pixel_S=None
                pixel_SW=None
                pixel_SE=None
                pixel_W=None
                pixel_NW=None
                pixel_N=image_array_labels[pixel_y-1,pixel_x] 
                pixel_E=image_array_labels[pixel_y,pixel_x+1] 
                pixel_NE=image_array_labels[pixel_y-1,pixel_x+1] 
            
            #condition for right bottom pixel
            elif pixel_y==ver_pixs-1 and pixel_x==hor_pixs-1: 
                pixel_S=None
                pixel_SW=None
                pixel_SE=None
                pixel_E=None
                pixel_NE=None
                pixel_N=image_array_labels[pixel_y-1,pixel_x] 
                pixel_NW=image_array_labels[pixel_y-1,pixel_x-1] 
                pixel_W=image_array_labels[pixel_y,pixel_x-1] 
                
            #condition for left limit pixels, not edge ones
            elif pixel_x==0 and pixel_y!=0 and pixel_y!=ver_pixs-1:
                pixel_W=None
                pixel_SW=None
                pixel_NW=None
                pixel_N=image_array_labels[pixel_y-1,pixel_x] 
                pixel_S=image_array_labels[pixel_y+1,pixel_x] 
                pixel_SE=image_array_labels[pixel_y+1,pixel_x+1] 
                pixel_E=image_array_labels[pixel_y,pixel_x+1] 
                pixel_NE=image_array_labels[pixel_y-1,pixel_x+1] 
                
            #condition for right limit pixels, not edge ones
            elif pixel_x==hor_pixs-1 and pixel_y!=0 and pixel_y!=ver_pixs-1:
                pixel_E=None
                pixel_NE=None
                pixel_SE=None
                pixel_N=image_array_labels[pixel_y-1,pixel_x] 
                pixel_NW=image_array_labels[pixel_y-1,pixel_x-1] 
                pixel_W=image_array_labels[pixel_y,pixel_x-1] 
                pixel_SW=image_array_labels[pixel_y+1,pixel_x-1] 
                pixel_S=image_array_labels[pixel_y+1,pixel_x] 
                
            else:
                #it is a pixel out from the edges and borders, typical one
                pixel_N=image_array_labels[pixel_y-1,pixel_x] 
                pixel_NW=image_array_labels[pixel_y-1,pixel_x-1] 
                pixel_W=image_array_labels[pixel_y,pixel_x-1] 
                pixel_SW=image_array_labels[pixel_y+1,pixel_x-1] 
                pixel_S=image_array_labels[pixel_y+1,pixel_x] 
                pixel_SE=image_array_labels[pixel_y+1,pixel_x+1] 
                pixel_E=image_array_labels[pixel_y,pixel_x+1] 
                pixel_NE=image_array_labels[pixel_y-1,pixel_x+1] 

            array_of_immediate_pixels=np.array([pixel_N,pixel_NW,pixel_W, pixel_SW, pixel_S, pixel_SE, pixel_E, pixel_NE])
            coords_of_imm_pixels=[(pixel_y-1,pixel_x),(pixel_y-1,pixel_x-1),(pixel_y,pixel_x-1),
                          (pixel_y+1,pixel_x-1),(pixel_y+1,pixel_x),(pixel_y+1,pixel_x+1),
                          (pixel_y,pixel_x+1),(pixel_y-1,pixel_x+1)]

            
            for pixel_i,coordinate in zip(array_of_immediate_pixels,coords_of_imm_pixels):

                if pixel_i==pixel.value:
                    already=0
                    for coordinate_old in all_coordinates:
                        
                        if coordinate[0]==coordinate_old[0] and coordinate[1]==coordinate_old[1]:
                            already=1
                    if already !=1:
                        all_coordinates.append(coordinate)
                        
                        pixel_classed=pixels_state(coordinate,pixel_i,state='New')
                    
                        array_containing_all_new_pixels.append(pixel_classed)        
            
          
                
            contiguous_cluster.extend(array_containing_all_new_pixels)
            
        pixel.state='Old'         
                    
                    
    return contiguous_cluster                    


#Find all the clusters present in an image labelled with n labels, and select them given a criteria
def Multiple_Consecutive_Clustered_regions(image_array_clusters, criteria, tolerance=0, n_clusters=0):
    '''
    For each big cluster, find the consecutive big regions, meaning that each big region should have an independent cluster.
    Watch out about reducing it to a cluster per value problem, as one value may have one or more associated clusters. The number
    of final regions to get should be what the scientist considers are distingishable regions (materials, materials domains, etc..)
    
    'Tolerance': Criteria can be either 'Tolerance', in which case all the clusters containing less than the specified number of pixels are removed.
    It should be specified as the a % of the total number of pixels (e.g tolerance=0.2 --> select the clusters that have more than a 
                                                                     20% of the total number of pixels)
    'Num_Clusters': On the other hand, criteria can be 'Num_Clusters', which means that only the specified number of clusters will be returned. These 
    are ordered from more to less pixels per cluster, and only the n_clusters with more pixels, independently on the pixel value, will 
    be returned.
    

    Returns
    -------
    final consecutive clusters from the image
    watch out as if the criteria is 'Tolerance', the output is a dictionary containing the clusters, in which each cluster is defined
    as dict [value of the cluster + _ + some index to avoid repetition]
    while if criteria is 'Num_Clusters', the output is a list with n_clusters clusters stored as ordered_clusters objects, and to extract 
    cluster itself and its pixels (stored as pixels_state objects) they should be taken as output[i].cluster, 
    where i goes from 0 to n_clusters

    '''
    
    class pixels_state:
        def __init__(self, coordinates, value, state):
            '''
            State can be 'Old', referring to pixels that were in the array before and therefore do not need to be considered 
            for checking again, or 'New', meaning that it has just been uploaded to the array and has to act as a point for getting new 
            conscutive pixels
            '''  
            self.coords=coordinates
            self.value=value
            self.state=state
    
    class order_clusters:
        def __init__(self, cluster_list):
            self.cluster=cluster_list
            self.lenght=len(cluster_list)
             
        
    image_array_clusters_copy=np.copy(image_array_clusters)
    
   
    
    ver_pixs=image_array_clusters.shape[0]
    hor_pixs=image_array_clusters.shape[1]
    
    radom_value_for_difference=2**14
    #whether it is tolerance or number of final regions
    
    
    # variables for tolerance criteria
    
    tolerance_pixels=int(round(tolerance*ver_pixs*hor_pixs))
    
    dictionary_of_clusters=dict()
    
    # variables for number of clusters criteria
    
    all_possible_clusters=[]

    
    for pixel_y in range(ver_pixs):
        
        for pixel_x in range(hor_pixs):
            
            
            if image_array_clusters_copy[pixel_y,pixel_x] != radom_value_for_difference:
                
                #then it means that this pixel does not belong to any cluster and can therefore form one, even if it is
                #just a single pixel cluster
                contiguous_cluster=Consecutive_pixels_particular_pixel(image_array_clusters_copy, (pixel_y,pixel_x))
                
                if criteria=='Tolerance':
                    
                    for pixel in contiguous_cluster:
                        
                        (pixel_change_y, pixel_change_x)=pixel.coords
                        image_array_clusters_copy[pixel_change_y, pixel_change_x]=radom_value_for_difference
                    
                    if len(contiguous_cluster) > tolerance_pixels:
                        
                        value_pixels_cluster=contiguous_cluster[0].value
                        
                        index_addition=len(dictionary_of_clusters)
                        
                        dictionary_of_clusters[str(value_pixels_cluster)+'_'+str(index_addition)]=contiguous_cluster
                        
                        #this way the dictionary is able to classify the clusters given its corresponding value, as
                        #the first character of the element is the value that the cluster has
                    
                    
                    output=dictionary_of_clusters    
                    
                elif criteria=='Num_Clusters':
                    
                    for pixel in contiguous_cluster:
                        
                        (pixel_change_y, pixel_change_x)=pixel.coords
                        image_array_clusters_copy[pixel_change_y, pixel_change_x]=radom_value_for_difference
                        
                    cluster_with_lenght_obj=order_clusters(contiguous_cluster)
                    
                    all_possible_clusters.append(cluster_with_lenght_obj)
                        
                    
                    
    if criteria=='Num_Clusters':
        
        #sort depending on the number of elements per cluster (descending order, first the larger clusters)
        #and then select only the first n_clusters, which will be the largest ones
        
        all_possible_clusters.sort(key=(lambda x: x.lenght), reverse=True)
        output=all_possible_clusters[:n_clusters]
        
    return  output



#Function that draws the contours (vertex and vectors) of a region given an initial pixel from its contour
def Contour_drawing_initial_pixel(image_array_labels, initial_pixel):
    
    
    class contour_vector:
        def __init__(self,m_in,m_out,init_coords,final_coords):
            self.m_in=m_in
            self.m_out=m_out
            self.init_coords=init_coords
            self.final_coords=final_coords
        
    #first define the rules for defining a contour in each of the 4 (N,S,E,W)
    
    def check_North(image_array,mat,initial_pixel_coords):
        #mat indicates the value of the pixels that are being contained inside the drawn contour
        (pixel_y,pixel_x)=initial_pixel_coords
        
        m_in=image_array[pixel_y-1,pixel_x-1]
        
        #specify conditions if the pixel is checked on the edge
        
        if pixel_x == hor_pixs:
            m_out=None
        else:            
            m_out=image_array[pixel_y-1,pixel_x]
        
        if m_in == mat and m_out != mat:
            next_vertex_coords=(pixel_y-1,pixel_x)
        else:
            next_vertex_coords=None
            
        return next_vertex_coords, m_out
    
    def check_South(image_array,mat,initial_pixel_coords):
        (pixel_y,pixel_x)=initial_pixel_coords
        
        m_in=image_array[pixel_y,pixel_x]
        
        #specify conditions if the pixel is checked on the edge
        
        if pixel_x==0:
            m_out=None
        else:
            m_out=image_array[pixel_y,pixel_x-1]
        
        if m_in == mat and m_out != mat:
            next_vertex_coords=(pixel_y+1,pixel_x)
        else:
            next_vertex_coords=None
            
        return next_vertex_coords, m_out
        
    
    def check_East(image_array,mat,initial_pixel_coords):
        (pixel_y,pixel_x)=initial_pixel_coords
        
        m_in=image_array[pixel_y-1,pixel_x]
        
        #specify conditions if the pixel is checked on the edge
        
        if pixel_y==ver_pixs:
            m_out=None
        else:
            m_out=image_array[pixel_y,pixel_x]
        
        if m_in == mat and m_out != mat:
            next_vertex_coords=(pixel_y,pixel_x+1)
        else:
            next_vertex_coords=None
            
        return next_vertex_coords, m_out
    
    def check_West(image_array,mat,initial_pixel_coords):
        (pixel_y,pixel_x)=initial_pixel_coords
        
        m_in=image_array[pixel_y,pixel_x-1]
        
        #specify conditions if the pixel is checked on the edge
        
        if pixel_y==0:
            m_out=None
        else:
            m_out=image_array[pixel_y-1,pixel_x-1]
        
        if m_in == mat and m_out != mat:
            next_vertex_coords=(pixel_y,pixel_x-1)
        else:
            next_vertex_coords=None
            
        return next_vertex_coords, m_out
    
    
    
    ver_pixs=image_array_labels.shape[0]
    hor_pixs=image_array_labels.shape[1]
    
    contour_vectors_list=[]
    
    image_array_labels_copy=np.copy(image_array_labels)
    
    initial_pixel_coords=initial_pixel
            
    # as the initial pixel coordinate is found as the bottom left edge of the pixel targeted, which is the actual coordinate,
    # (defined as top left corner of the pixel) of the pixel given as the initial, then the material inside for what is a 
    #contour EAST is the pixel_y-1 , pixel_x
    
    target_pixel=image_array_labels[initial_pixel_coords[0]-1,initial_pixel_coords[1]] 
    
    
    #and then define the exceptions depending if the pixel is in a contour or not 
    
    #add the padding to the image to make it ver_pixs+1 * hor_pixs+1
    
    # m_out material will be the following for the bottom and right edges, as the padding will have this value
    #so all the contours in which m_out = extra_pixels_value will be edge pixels from either the bottom or right edge
    #although maybe it is better to just set all the non existing m out as None rather than a value...
    extra_pixels_value=2**16
    
    padded_image=cv2.copyMakeBorder(image_array_labels_copy,0,1,0,1,cv2.BORDER_CONSTANT,value=extra_pixels_value)
    
    #list containing all the coordinates from all the vertex that are being analysed
    vertex_array=[]
    
    vertex_array.append(initial_pixel_coords)
    
    #list containing the vertexs that led to two contours (special case)
    list_double_vertexs=[]
    
    for vertex in vertex_array:
        
        #check if vertex is contained in an edge or corner to tailor the different checkings
        
        if vertex[0]==0 and vertex[1]==0:
            #top left corner
            #only check south
            next_vertex_coords, m_out=check_South(padded_image,target_pixel,vertex)
            
            if next_vertex_coords == initial_pixel_coords:
                new_contour=contour_vector(target_pixel,m_out,vertex,next_vertex_coords)
                contour_vectors_list.append(new_contour)
                #break  
            else:
                vertex_array.append(next_vertex_coords)
            
                #generate new contour vector and append it to the contour vectors list
                new_contour=contour_vector(target_pixel,m_out,vertex,next_vertex_coords)
                contour_vectors_list.append(new_contour)
            
        elif vertex[0]==0 and vertex[1]==hor_pixs:
            #top right corner
            #only check west
            next_vertex_coords, m_out=check_West(padded_image,target_pixel,vertex)
            
            if next_vertex_coords == initial_pixel_coords:
                new_contour=contour_vector(target_pixel,m_out,vertex,next_vertex_coords)
                contour_vectors_list.append(new_contour)
                #break  
            else:
                vertex_array.append(next_vertex_coords)
            
                #generate new contour vector and append it to the contour vectors list
                new_contour=contour_vector(target_pixel,m_out,vertex,next_vertex_coords)
                contour_vectors_list.append(new_contour)
        
        elif vertex[0]==ver_pixs and vertex[1]==0:
            #bottom left corner
            #only check east
            next_vertex_coords, m_out=check_East(padded_image,target_pixel,vertex)
            
            if next_vertex_coords == initial_pixel_coords:
                new_contour=contour_vector(target_pixel,m_out,vertex,next_vertex_coords)
                contour_vectors_list.append(new_contour)
                #break  
            else:
                vertex_array.append(next_vertex_coords)
            
                #generate new contour vector and append it to the contour vectors list
                new_contour=contour_vector(target_pixel,m_out,vertex,next_vertex_coords)
                contour_vectors_list.append(new_contour)
        elif vertex[0]==ver_pixs and vertex[1]==hor_pixs:
            #bottom right corner
            #only check north
            next_vertex_coords, m_out=check_North(padded_image,target_pixel,vertex)
            
            if next_vertex_coords == initial_pixel_coords:
                new_contour=contour_vector(target_pixel,m_out,vertex,next_vertex_coords)
                contour_vectors_list.append(new_contour)
                #break  
            else:
                vertex_array.append(next_vertex_coords)
            
                #generate new contour vector and append it to the contour vectors list
                new_contour=contour_vector(target_pixel,m_out,vertex,next_vertex_coords)
                contour_vectors_list.append(new_contour)
            
        elif vertex[0]==0 and vertex[1] !=0 and vertex[1] != hor_pixs:
            #north edge, excluding both corners
            #only check west and south
            
            possible_vertexs=[]
            m_outs=[]
            
            next_vertex_coords, m_out=check_West(padded_image,target_pixel,vertex)
            possible_vertexs.append(next_vertex_coords)
            m_outs.append(m_out)
            
            next_vertex_coords, m_out=check_South(padded_image,target_pixel,vertex)
            possible_vertexs.append(next_vertex_coords)
            m_outs.append(m_out)
            
            for possible_vertex, m_out in zip(possible_vertexs,m_outs):
                if possible_vertex != None:
                    #this should also handle the issue of one vertex diverging into two different contours, as this would
                    #append all the new possible vertices and make them new starting points for considering new vertices
                    #!!! nevertheless, repetition should be checked because on of the contours is going to led to the same vertexm
                    #and therefore it would be repeated if it is not considered
                    if possible_vertex == initial_pixel_coords:
                        new_contour=contour_vector(target_pixel,m_out,vertex,possible_vertex)
                        contour_vectors_list.append(new_contour)
                        #break  
                    else:
                        
                        vertex_array.append(possible_vertex)
                    
                        #generate new contour vector and append it to the contour vectors list
                        new_contour=contour_vector(target_pixel,m_out,vertex,possible_vertex)
                        contour_vectors_list.append(new_contour)
            
        elif vertex[0]==ver_pixs and vertex[1] !=0 and vertex[1] != hor_pixs:
            #south edge, excluding both corners
            #only check north and east
        
            possible_vertexs=[]
            m_outs=[]
            
            next_vertex_coords, m_out=check_North(padded_image,target_pixel,vertex)
            possible_vertexs.append(next_vertex_coords)
            m_outs.append(m_out)  
            
            next_vertex_coords, m_out=check_East(padded_image,target_pixel,vertex)
            possible_vertexs.append(next_vertex_coords)
            m_outs.append(m_out)
            
            for possible_vertex, m_out in zip(possible_vertexs,m_outs):
                if possible_vertex != None:
                    #this should also handle the issue of one vertex diverging into two different contours, as this would
                    #append all the new possible vertices and make them new starting points for considering new vertices
                    #!!! nevertheless, repetition should be checked because on of the contours is going to led to the same vertexm
                    #and therefore it would be repeated if it is not considered
                    if possible_vertex == initial_pixel_coords:
                        new_contour=contour_vector(target_pixel,m_out,vertex,possible_vertex)
                        contour_vectors_list.append(new_contour)
                        #break  
                    else:
                        
                        vertex_array.append(possible_vertex)
                    
                        #generate new contour vector and append it to the contour vectors list
                        new_contour=contour_vector(target_pixel,m_out,vertex,possible_vertex)
                        contour_vectors_list.append(new_contour)
            
            
            
        elif vertex[1]==0 and vertex[0] !=0 and vertex[0] != ver_pixs:
            #left edge, excluding both corners
            #only check south and east
            
            possible_vertexs=[]
            m_outs=[]            
            
            next_vertex_coords, m_out=check_South(padded_image,target_pixel,vertex)
            possible_vertexs.append(next_vertex_coords)
            m_outs.append(m_out)  
            
            next_vertex_coords, m_out=check_East(padded_image,target_pixel,vertex)
            possible_vertexs.append(next_vertex_coords)
            m_outs.append(m_out)
            
            for possible_vertex, m_out in zip(possible_vertexs,m_outs):
                if possible_vertex != None:
                    #this should also handle the issue of one vertex diverging into two different contours, as this would
                    #append all the new possible vertices and make them new starting points for considering new vertices
                    #!!! nevertheless, repetition should be checked because on of the contours is going to led to the same vertexm
                    #and therefore it would be repeated if it is not considered
                    if possible_vertex == initial_pixel_coords:
                        new_contour=contour_vector(target_pixel,m_out,vertex,possible_vertex)
                        contour_vectors_list.append(new_contour)
                        #break  
                    else:
                        
                        vertex_array.append(possible_vertex)
                    
                        #generate new contour vector and append it to the contour vectors list
                        new_contour=contour_vector(target_pixel,m_out,vertex,possible_vertex)
                        contour_vectors_list.append(new_contour)
            
        elif vertex[1]==hor_pixs and vertex[0] !=0 and vertex[0] != ver_pixs:
            #right edge, excluding both corners
            #only check north and west
            
            possible_vertexs=[]
            m_outs=[] 
            
            next_vertex_coords, m_out=check_North(padded_image,target_pixel,vertex)
            possible_vertexs.append(next_vertex_coords)
            m_outs.append(m_out)
            
            next_vertex_coords, m_out=check_West(padded_image,target_pixel,vertex)
            possible_vertexs.append(next_vertex_coords)
            m_outs.append(m_out)
            
            for possible_vertex, m_out in zip(possible_vertexs,m_outs):
                if possible_vertex != None:
                    #this should also handle the issue of one vertex diverging into two different contours, as this would
                    #append all the new possible vertices and make them new starting points for considering new vertices
                    #!!! nevertheless, repetition should be checked because on of the contours is going to led to the same vertexm
                    #and therefore it would be repeated if it is not considered
                    if possible_vertex == initial_pixel_coords:
                        new_contour=contour_vector(target_pixel,m_out,vertex,possible_vertex)
                        contour_vectors_list.append(new_contour)
                        #break 
                    else:
                        
                        vertex_array.append(possible_vertex)
                    
                        #generate new contour vector and append it to the contour vectors list
                        new_contour=contour_vector(target_pixel,m_out,vertex,possible_vertex)
                        contour_vectors_list.append(new_contour)
        
        else:
            #the majority, all the pixels that are not in an edge nor a corner
            
            possible_vertexs=[]
            m_outs=[] 
            
            next_vertex_coords, m_out=check_North(padded_image,target_pixel,vertex)
            possible_vertexs.append(next_vertex_coords)
            m_outs.append(m_out)
            
            next_vertex_coords, m_out=check_West(padded_image,target_pixel,vertex)
            possible_vertexs.append(next_vertex_coords)
            m_outs.append(m_out)
            
            next_vertex_coords, m_out=check_South(padded_image,target_pixel,vertex)
            possible_vertexs.append(next_vertex_coords)
            m_outs.append(m_out)
            
            next_vertex_coords, m_out=check_East(padded_image,target_pixel,vertex)
            possible_vertexs.append(next_vertex_coords)
            m_outs.append(m_out)
            
            
            number_of_found_vertexs=0
            for possible_vertex, m_out in zip(possible_vertexs,m_outs):
                if possible_vertex != None:
                    #this should also handle the issue of one vertex diverging into two different contours, as this would
                    #append all the new possible vertices and make them new starting points for considering new vertices
                    #!!! nevertheless, repetition should be checked because on of the contours is going to led to the same vertexm
                    #and therefore it would be repeated if it is not considered
                    if possible_vertex == initial_pixel_coords:
                        new_contour=contour_vector(target_pixel,m_out,vertex,possible_vertex)
                        contour_vectors_list.append(new_contour)
                        #break 
                    else:
                    
                        if len(list_double_vertexs)==0:
                            
                            vertex_array.append(possible_vertex)
                        
                            #generate new contour vector and append it to the contour vectors list
                            new_contour=contour_vector(target_pixel,m_out,vertex,possible_vertex)
                            contour_vectors_list.append(new_contour)
                        else:
                            
                            repeated_element=[repeated_vertex for repeated_vertex in list_double_vertexs if repeated_vertex==possible_vertex]
                            
                            if len(repeated_element)==0:
                                vertex_array.append(possible_vertex)
                            
                                #generate new contour vector and append it to the contour vectors list
                                new_contour=contour_vector(target_pixel,m_out,vertex,possible_vertex)
                                contour_vectors_list.append(new_contour)
                            else:
                                
                                #there is some element that is repeated
                                # only generate new contour vector and append it to the contour vectors list
                                # but do not append the coordinate of the new vertex as it already exists
                                new_contour=contour_vector(target_pixel,m_out,vertex,possible_vertex)
                                contour_vectors_list.append(new_contour)
                       
                    number_of_found_vertexs=number_of_found_vertexs+1       
                    
            if number_of_found_vertexs==2:
                list_double_vertexs.append(vertex)
            
   
    
    return contour_vectors_list, vertex_array


#Function that given a label, finds the initial pixel of the contour to draw the contour itself

def Find_Initial_Pixel_for_contour(image_array_labels, label_of_region):
    '''
    This function finds the inital pixel for defining the contour of a region in an image in which each region is labeled 
    differently. Then, the input of the function is the image, in which each pixel is labeled with a characteristic number,
    and the number corresponding to the label or region which has to be contoured
    
    !!! This function is sensitive to noise! If the first pixel found is in an edge, and this edge has an isolated pixel of noise 
    from another label, then this lonely (or a few of them together, but definitely not a region) one will be defined as
    region and analysed to be contoured. So the input must be a noiseless region definition

    Parameters
    ----------
    image_array_labels : TYPE
        DESCRIPTION.
    label_of_region: integer
        integer represing the regin we want to extract the inital pixel from

    Returns
    -------
    initial_pixel : TYPE
        DESCRIPTION.

    '''
    
    ver_pixs=image_array_labels.shape[0]
    hor_pixs=image_array_labels.shape[1]
    
    image_array_labels_copy=np.copy(image_array_labels)
    
    extra_pixels_value=2**16
    
    padded_image=cv2.copyMakeBorder(image_array_labels_copy,0,1,0,1,cv2.BORDER_CONSTANT,value=extra_pixels_value)
    
    
    #do the analysis from south to north, to avoid noisy regions that are typically on the top area of the microgrpahs, such as Pt
    
    for pixel_x in range(hor_pixs+1):
        for pixel_y in reversed(range(ver_pixs+1)):
            
            target_pixel=padded_image[pixel_y,pixel_x]
            
            if target_pixel==label_of_region:
                initial_pixel=(pixel_y+1,pixel_x)
                break
            
        if target_pixel==label_of_region:
            initial_pixel=(pixel_y+1,pixel_x)
            break   
        
        
    #!!! just make sure that when defining the material inside in the contour drawing function it is given by the pixel just 
    #above the given as initial one, as the iniital one is the one corresponding to the bottom of it and will have a different 
    #value, and of course, corresponds to a south interface
        
    return initial_pixel

#Draws the contour after computing the initial pixel given a label
def Contour_draw_computing_initial_pixel(image_array_labels, label_of_region):
    
    initial_pixel=Find_Initial_Pixel_for_contour(image_array_labels, label_of_region)
    contour_vectors_list, vertex_array=Contour_drawing_initial_pixel(image_array_labels, initial_pixel)
    
    return contour_vectors_list, vertex_array

   
#from all the labels in all the regions, computes the contour relative to each label, and therefore, its region (not 0)
def Contour_draw_All_regions(image_array_labels):
    '''
    The next step is to loop thought the different regions, each with a different label, and get as many contours arrays and 
    vertexs arrays as different regions or labels we have in the image (ideally, all the regions that have an important and
    characteristic feature that differentiates them from the rest, such as material, position, shape, and so on)


    Parameters
    ----------
    image_array_labels : array

    Returns
    -------
    dictionary containing the vertex and contours in each region labeled distinctly
    the vertexs and contours of each region can be extracted the following way:
    dictionary_name['label_vertexs'] or dictionary_name['label_contours']

    '''
   
    conts_vertx_per_region=dict()
    label_values=np.unique(image_array_labels)
    #label 0 is associated to pixels that do not have a label
    label_values=np.sort(label_values)[1:]
    
    for label in label_values:
        
        contour_vectors_list, vertex_array=Contour_draw_computing_initial_pixel(image_array_labels, label)
        conts_vertx_per_region[str(int(label))+'_'+'vertexs']=vertex_array
        conts_vertx_per_region[str(int(label))+'_'+'contours']=contour_vectors_list

  
    return  conts_vertx_per_region

#Make pixels considered as noise by the initial clustering be assigned to the cluster in which they should be,
#making the pixels inside a given border be assigned to this cluster
def Denoise_region(image_array,conts_vertx_per_region,region_label):
    '''
    

    Parameters
    ----------
    image_array : image with the labels obtained after the consecutive algorithm
    conts_vertx_per_region : dict, output from the function Contour_draw_All_regions to the previous image
    conregion_label : TYPE
        DESCRIPTION.

    Returns
    -------
    denoised_region : TYPE
        DESCRIPTION.

    '''
    
    def Check_vector_North(pixel_coords,contour_vectors):
        '''
        Check if given the pixel coordinates, if the FIRST vector on its north is following the contour direction,
        from right to left (only the first, as it may found a good one afterwards to misleading results)

        Parameters
        ----------
        pixel_coords : the coordinates of the evaluated pixel
        contour_vectors : list containing contour_vector objects
        limits_to_check: tuple, (min_top_vertex,max_bottom_vertex,min_left_vertex,max_right_vertex)


        Returns
        -------
        boolean True or False whether the pixel is or not inside the region

        '''
        
        possible_contours=[]
        for contour_vector in contour_vectors:
            if contour_vector.init_coords[0]==contour_vector.final_coords[0]:
                if contour_vector.final_coords[0]<=pixel_coords[0]:
                    if (contour_vector.init_coords[1]==pixel_coords[1] and contour_vector.final_coords[1]==pixel_coords[1]+1) or (contour_vector.init_coords[1]==pixel_coords[1]+1 and contour_vector.final_coords[1]==pixel_coords[1]):
                        possible_contours.append(contour_vector)
                    
        #now sort the possible contours from the pixel to the north and only check the first one just above it
        #sort depending on the value of y, putting first the bigger coordinate of y, which means 
        #that first it is the coordinate directly on top of the target pixel
        # sort in descending order, first the larger y
        #and then select only the first contour
        
        if len(possible_contours)==0:
            to_north=False
        else:
            possible_contours.sort(key=(lambda x: x.init_coords[0]), reverse=True)
        
            #check if the direction of the vector is from right to left
            if possible_contours[0].init_coords[1]==pixel_coords[1]+1 and possible_contours[0].final_coords[1]==pixel_coords[1]:
                to_north=True
            else:
                to_north=False
        
        return to_north
    
    
    def Check_vector_South(pixel_coords,contour_vectors):
        '''
        Check if given the pixel coordinates, if the FIRST vector on its south is following the contour direction,
        from left to right (only the first, as it may found a good one afterwards to misleading results)

        Parameters
        ----------
        pixel_coords : the coordinates of the evaluated pixel
        contour_vectors : list containing contour_vector objects


        Returns
        -------
        boolean True or False whether the pixel is or not inside the region

        '''
        
        possible_contours=[]
        for contour_vector in contour_vectors:
            if contour_vector.init_coords[0]==contour_vector.final_coords[0]:
                if contour_vector.final_coords[0]>pixel_coords[0]:
                    if (contour_vector.init_coords[1]==pixel_coords[1] and contour_vector.final_coords[1]==pixel_coords[1]+1) or (contour_vector.init_coords[1]==pixel_coords[1]+1 and contour_vector.final_coords[1]==pixel_coords[1]):
                        possible_contours.append(contour_vector)
                    
        #now sort the possible contours from the pixel to the south and only check the first one just below it
        #sort depending on the value of y, putting first the bigger coordinate of y, which means 
        #that first it is the coordinate directly below the target pixel
        # sort in ascending order, first the smaller y
        #and then select only the first contour
        
        if len(possible_contours)==0:
            to_south=False
        else:
            possible_contours.sort(key=(lambda x: x.init_coords[0]), reverse=False)
        
            #check if the direction of the vector is from left to right
            if possible_contours[0].init_coords[1]==pixel_coords[1] and possible_contours[0].final_coords[1]==pixel_coords[1]+1:
                to_south=True
            else:
                to_south=False
        
        return to_south
    
    
    def Check_vector_East(pixel_coords, contour_vectors):
        '''
        Check if given the pixel coordinates, if the FIRST vector on its east is following the contour direction,
        from bottom to top (only the first, as it may found a good one afterwards to misleading results)

        Parameters
        ----------
        pixel_coords : the coordinates of the evaluated pixel
        contour_vectors : list containing contour_vector objects


        Returns
        -------
        boolean True or False whether the pixel is or not inside the region

        '''
        
        possible_contours=[]
        for contour_vector in contour_vectors:
            if contour_vector.init_coords[1]==contour_vector.final_coords[1]:
                if contour_vector.final_coords[1]>pixel_coords[1]:
                    if (contour_vector.init_coords[0]==pixel_coords[0] and contour_vector.final_coords[0]==pixel_coords[0]+1) or (contour_vector.init_coords[0]==pixel_coords[0]+1 and contour_vector.final_coords[0]==pixel_coords[0]):
                        possible_contours.append(contour_vector)
                    
        #now sort the possible contours from the pixel to the east and only check the first one just at its right
        #sort depending on the value of x, putting first the bigger coordinate of x, which means 
        #that first it is the coordinate directly at right of the target pixel
        # sort in ascending order, first the smaller x
        #and then select only the first contour
        
        if len(possible_contours)==0:
            to_east=False
        else:
            possible_contours.sort(key=(lambda x: x.init_coords[1]), reverse=False)
        
            #check if the direction of the vector is from right to left
            if possible_contours[0].init_coords[0]==pixel_coords[0]+1 and possible_contours[0].final_coords[0]==pixel_coords[0]:
                to_east=True
            else:
                to_east=False
        
        return to_east
    
    
    def Check_vector_West(pixel_coords, contour_vectors):
        '''
        Check if given the pixel coordinates, if the FIRST vector on its west is following the contour direction,
        from top to bottom (only the first, as it may found a good one afterwards to misleading results)

        Parameters
        ----------
        pixel_coords : the coordinates of the evaluated pixel
        contour_vectors : list containing contour_vector objects


        Returns
        -------
        boolean True or False whether the pixel is or not inside the region

        '''
        
        possible_contours=[]
        for contour_vector in contour_vectors:
            if contour_vector.init_coords[1]==contour_vector.final_coords[1]:
                if contour_vector.final_coords[1]<=pixel_coords[1]:
                    if (contour_vector.init_coords[0]==pixel_coords[0] and contour_vector.final_coords[0]==pixel_coords[0]+1) or (contour_vector.init_coords[0]==pixel_coords[0]+1 and contour_vector.final_coords[0]==pixel_coords[0]):
                        possible_contours.append(contour_vector)
                    
        #now sort the possible contours from the pixel to the west and only check the first one just at its left
        #sort depending on the value of x, putting first the bigger coordinate of x, which means 
        #that first it is the coordinate directly at left of the target pixel
        # sort in descending order, first the larger x
        #and then select only the first contour
        
        if len(possible_contours)==0:
            to_west=False
        else:
            possible_contours.sort(key=(lambda x: x.init_coords[1]), reverse=True)
        
            #check if the direction of the vector is from right to left
            if possible_contours[0].init_coords[0]==pixel_coords[0] and possible_contours[0].final_coords[0]==pixel_coords[0]+1:
                to_west=True
            else:
                to_west=False
        
        return to_west
    
    label=region_label
    contour_vectors=conts_vertx_per_region[str(int(label))+'_contours']
    
    #limit the number of pixels to analise and loop through to reduce computing time
    #get maximum index for top bottom right and left and evaluate only within this range
    
    #these values can already be bigger than the original image, so the vertex indices can go from 0 to image size 
    #not image size -1, they are organised as [(y0,x1), (y1,x1),...,(yf,xf)]
    vertexs_for_label=conts_vertx_per_region[str(int(label))+'_vertexs']

    min_top_vertex=np.min(vertexs_for_label, axis=0)[0]
    max_bottom_vertex=np.max(vertexs_for_label, axis=0)[0]
    min_left_vertex=np.min(vertexs_for_label, axis=0)[1]
    max_right_vertex=np.max(vertexs_for_label, axis=0)[1]
    
    #these values define the range of values that can be contained inside the region
    #the max index (both bottom and right) does not need to be included in the interval, as these only 
    #define pixel borders and not real ones (the extra ones will never be inside the region)
    
    extra_pixels=[]
    
    for pixel_y in range(min_top_vertex,max_bottom_vertex):
        for pixel_x in range(min_left_vertex,max_right_vertex):
            
            if image_array[pixel_y,pixel_x] ==0:
                #only evaluate if the pixel is different from the label, as are the ones we want to change
                
                to_north=Check_vector_North((pixel_y,pixel_x),contour_vectors)
                to_south=Check_vector_South((pixel_y,pixel_x),contour_vectors)
                to_east=Check_vector_East((pixel_y,pixel_x),contour_vectors)
                to_west=Check_vector_West((pixel_y,pixel_x),contour_vectors)
                
                if to_north==True and to_south==True and to_east==True and to_west==True:
                    extra_pixels.append((pixel_y,pixel_x))

    return extra_pixels

#Denoise all the regions, looping through the labels that represent actual regions (not 0)
def Denoise_All_Regions(image_array,conts_vertx_per_region):
    '''
    Apply the denoising for all the regions and all the labels that have an associated region

    Parameters
    ----------
    image_array : TYPE
        DESCRIPTION.
    conts_vertx_per_region : TYPE
        DESCRIPTION.

    Returns
    -------
    dict_extra_pixels : TYPE
        DESCRIPTION.

    '''

    label_values=np.unique(image_array)
    #label 0 is associated to pixels that do not have a label
    label_values=np.sort(label_values)[1:]

    
    dict_extra_pixels=dict()
    for label in label_values:

        dict_extra_pixels[str(int(label))+'_extra_pixels']=Denoise_region(image_array,conts_vertx_per_region,label)
    
    return dict_extra_pixels

def Gaussian_pre_clustering(image_st,number_of_gaussians,variance_threshold):
    '''
    finds the bayesian GMM, and selects these pixels that belong to a sharp enough gaussian curve, as these features are
    difficult to be captured by K-means. Then, it redraws the original image by keeping all the original pixels except
    these that belong to these curves, which are assigned a label from 2 to 2+n where n is the thresholded gaussians -1
    
    Parameters
    ----------
    image : TYPE
        DESCRIPTION.
    number_of_gaussians : number of gaussians that fill the histogram
        DESCRIPTION.
    variance_threshold: defines these curves that will be considered sharp enough. It is the variance below which 
                    the curve is kept. The variances of the curves can be tracked by the covariances output of BGMM

    Returns
    -------
    None.

    '''   
    def BGMM(image_,number_clusters):
    
        weight_concentration_prior=1/number_clusters
        
        image_reshaped=np.reshape(image_, (image_.shape[0]*image_.shape[1],1))
        gmm = sklearn.mixture.BayesianGaussianMixture(n_components=number_clusters, n_init=5)
        gmm.fit(image_reshaped)
        
        #predictions from gmm
        labels = gmm.predict(image_reshaped)
        
        labels.shape=image_.shape
        means=gmm.means_
        covariances=gmm.covariances_
        
        return labels, means, covariances
    
    labels, means, covs=BGMM(image_st,number_clusters=number_of_gaussians)

    image_to_refine=np.copy(image_st)

    # ensure the labels have values that range from 2 to any number, to avoid comparing values with 0 and 1
    copy_labels_gauss=np.copy(labels)+2
    
    label_final_val=2
    for index,cov in enumerate(covs):
        if cov < variance_threshold:    
            print('yes')
            target_label=index+2
            image_to_refine=label_final_val*(copy_labels_gauss==target_label)+image_to_refine*(copy_labels_gauss!=target_label)
            
            label_final_val=label_final_val+1
    
    return image_to_refine


def Custom_Filter(micrograph, nm_size=5):
    x_pixels, y_pixels = micrograph.shape  # [pixels]
    x_edge_length = nm_size  # [nm]
    y_edge_length = nm_size # [nm]
    x_sampling = x_pixels / x_edge_length  # [pixels/nm]
    y_sampling = y_pixels / y_edge_length  # [pixels/nm]
    x_axis_vec = np.linspace(-x_edge_length / 2, x_edge_length / 2, x_pixels)  # vector of locations along x-axis
    y_axis_vec = np.linspace(-y_edge_length / 2, y_edge_length / 2, y_pixels)  # vector of locations along y-axis
    x_mat, y_mat = np.meshgrid(x_axis_vec, y_axis_vec)  # matrices of x-positions and y-positions
    
    u_max = x_sampling / 2
    v_max = y_sampling / 2
    u_axis_vec = np.linspace(-u_max / 2, u_max / 2, x_pixels)
    v_axis_vec = np.linspace(-v_max / 2, v_max / 2, y_pixels)
    u_mat, v_mat = np.meshgrid(u_axis_vec, v_axis_vec)  # matrices of u-positions and v-positions
    r = np.sqrt(u_mat**2+v_mat**2) # convert cartesian coordinates to polar radius
    
    # inverse width of gaussian, units same as real space axes
    filter_width = 3.5
    
    inverse_gauss_filter = 1-np.e**(-(r*filter_width)**2)
    
    # take the fft of the image
    fft_image_w_background = np.fft.fftshift(np.fft.fft2(micrograph))
    fft_abs_image_background = np.abs(fft_image_w_background)
    
    # apply the filter
    fft_image_corrected = fft_abs_image_background * inverse_gauss_filter
    
    # perform the inverse fourier transform on the filtered data
    image_corrected = np.real(np.fft.ifft2(np.fft.ifftshift(fft_image_corrected)))
    
    # find what was removed from the image by filtering
    filtered_background = micrograph - image_corrected
    
    image_corrected=(image_corrected-np.min(image_corrected))/np.max(image_corrected-np.min(image_corrected))
    filtered_background=(filtered_background-np.min(filtered_background))/np.max(filtered_background-np.min(filtered_background))

    return image_corrected

#Applied process

#Hyperparameters
gauss_blur_filter_size=(5,5)  #size of smoothing filter, go to line to change sigma
downscaling_factor=20 #for trials, n factor of downsampling size of image
n_gaussian_ref=3   #number of components in the gaussian refinement (like k means but if doubt overstimate)
k_means_clusters=3  #number of components in k means clustering
number_final_regions=4 #number of clearly distinct regions

imagedirectory=r'E:\Arxius varis\PhD\2nd_year\Code\trial_images\low_mag_images\nanowire1_tiff.tif'
imagedirectory=r'E:\Arxius varis\PhD\2nd_year\Code\trial_images\low_mag_images\nanowire3_tiff.tif'
image=plt.imread(imagedirectory)
plt.imshow(image, cmap=plt.cm.gray, vmin=image.min(), vmax=image.max())
plt.show()


#First standarisation of the image for filtering/blurring it with gaussian filter

image_st=(image-np.min(image))/np.max(image-np.min(image))
plt.hist(image_st.ravel(),256,[np.min(np.array([image_st])),np.max(np.array([image_st]))])
plt.show()

#Application of Gaussian filter for denoising


denoised_image=cv2.GaussianBlur(image_st, gauss_blur_filter_size, 1)
plt.imshow(denoised_image, cmap=plt.cm.gray, vmin=denoised_image.min(), vmax=denoised_image.max())
plt.show()
#Second standarisation of the image after filtering/blurring it with gaussian filter

image_st=(denoised_image-np.min(denoised_image))/np.max(denoised_image-np.min(denoised_image))

#Print histogram

plt.hist(image_st.ravel(),256,[np.min(np.array([image_st])),np.max(np.array([image_st]))])
plt.show()

#For sake of evaluation, better work with an image with less pixels, as only the consecutive pixel evaluation would take
#approximately 6 hours to run for a big region of 250.000 pixels in total.

#Then downsample the image and upsample it posteriorly 
#We select a max pooling method to keep track of the brighter elements and this way keep a higher contrast


ds_image=skimage.measure.block_reduce(image_st, block_size=tuple(np.int32(downscaling_factor*np.ones(len(np.shape(image_st))))), func=np.mean, cval=0)

#and standarise it again to ensure 0-1 values

ds_image_st=(ds_image-np.min(ds_image))/np.max(ds_image-np.min(ds_image))



#Gaussian refinement pre k means

ds_image_st=Gaussian_pre_clustering(ds_image_st,number_of_gaussians=n_gaussian_ref,variance_threshold=0.001)



#Clustering method
#Step that implies more creativity and most important one to prepeare the data for the following refinement algorithms

#Mean shift clustering
'''
labels_ms,clusters_ms=Mean_shift(ds_image_st)
#generate image from the labels and clusters
labels_ms_reshaped=np.reshape(labels_ms, np.shape(ds_image_st))
'''
#K-means clustering

values, labels, cost = best_km(ds_image_st, n_clusters =k_means_clusters)
labels_ms_reshaped = np.choose(labels, values)
labels_ms_reshaped.shape = ds_image_st.shape


plt.imshow(labels_ms_reshaped,cmap=plt.cm.gray )
plt.show()



#Next step is to take the pixels labels and create the consecutive pixel regions and
#gather only the ones that fulfill the tolerance/number of clusters conditions


ordered_clusters=Multiple_Consecutive_Clustered_regions(labels_ms_reshaped, criteria='Num_Clusters', tolerance=0, n_clusters=number_final_regions)


final_image=np.zeros(np.shape(labels_ms_reshaped))
value_image=1
for cluster in ordered_clusters:
    cluster_i=cluster.cluster
    for pixel in cluster_i:
        (pixel_y,pixel_x)=pixel.coords
        final_image[pixel_y,pixel_x]=value_image
        
    value_image=value_image+1


plt.imshow(final_image,cmap=plt.cm.gray )
plt.show()

#Next step is to extract the contours and its vertexs for all the n of clusters drawn in the image with the main clusters

#!!! Keep in mind that during the previous steps the image is full size but then it becomes the original size 
#plus one pixel in each direction, in this next step, where the contours are computed

 
conts_vertx_per_region=Contour_draw_All_regions(final_image)

#Each label, or integer from 0 to number of clusters - 1 has its own cluster (pixels), and vertex and contours array

#Denoise the obtained regions, making all the pixels located inside the defined contours to be reassigned to its supposed cluster

final_extra_pixs=Denoise_All_Regions(final_image,conts_vertx_per_region)

final_image_2=np.copy(final_image)

for label in final_extra_pixs:
    for pixel in final_extra_pixs[label]:  
        
        final_image_2[pixel[0],pixel[1]]=int(label[0])    


plt.imshow(final_image_2,cmap=plt.cm.gray )
plt.show()



'''
# End of script  Full_segmentation_routine_low_mags.py
# Start of the translation to profile
'''

#image calibration
imagedirectorydm3=r'E:\Arxius varis\PhD\2nd_year\Code\trial_images\low_mag_images\nanowire1_dm3.dm3'

imagedm3=hs.load(imagedirectorydm3)
meta1=imagedm3.metadata
meta2=imagedm3.original_metadata.export('parameters')

x_calibration=imagedm3.axes_manager['x'].scale
y_calibration=imagedm3.axes_manager['y'].scale

x_pixels_original=imagedm3.axes_manager['x'].size
y_pixels_original=imagedm3.axes_manager['y'].size

x_units=imagedm3.axes_manager['x'].units
y_units=imagedm3.axes_manager['y'].units

label_values=np.unique(final_image)
label_values=np.sort(label_values)[1:]

total_fov=x_calibration*x_pixels_original

downsampled_image_pixel_calibration=total_fov/(x_pixels_original/downscaling_factor)

nextnano_vertexs=dict()

for label in label_values:

    list_of_vertexs=conts_vertx_per_region[str(int(label))+'_'+'vertexs']
    
    list_of_nnvertexs_per_label=[]
    
    for vertex in list_of_vertexs:
        #very inefficient, watch out for the final version and how this is segmented finally
        #the vertex in the contour definitoin are y,x as needed for numpy imaging
        x=vertex[1]
        y=vertex[0]
        
        x=(x*downsampled_image_pixel_calibration)/10
        y=(y*downsampled_image_pixel_calibration)/10
        
        # !!! here we build an array of type y and x first, which is different from the nnvertex 
        # in the nextnano script watch out... take the calibrated relative contours in the final application
        # as the nextnano seems to require x and y format instead
        list_of_nnvertexs_per_label.append([y, x])
    
    nextnano_vertexs[str(int(label))+'_'+'nnvertexs']=list_of_nnvertexs_per_label







#%%

'''
Functions for atomistic modelling
'''

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
        is a list with [y,x] coordinates

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




        
    
'''
Define thickness and in-(ZA) plane rotation
'''



# the format the coordinates are is, in nm, 
'''
Here we define some custom geometries or morphologies by putting some 
vertex together, and then we construct some classes and objects 
that ressemble the classes that are outputted from the global workflow and
from the segmentation
This was just for testing the process so it could be directly coupled with
the full workflow
'''


# work with the substrate and NW only
labels_to_use = [1,2]


# generate the contours


shape_1 = nextnano_vertexs[str(int(1))+'_nnvertexs']
shape_2 = nextnano_vertexs[str(int(2))+'_nnvertexs']

plt.scatter(np.asarray(shape_1)[:,1],np.asarray(shape_1)[:,0])
plt.show()

# mode shape to have the propoer contours now as exception for this case
shape_1 = np.asarray(shape_1)
shape_1 = shape_1[shape_1[:,0]> 36.5]
print(shape_1)
plt.scatter(np.asarray(shape_1)[:,1],np.asarray(shape_1)[:,0])
plt.show()

shape_1 = list(shape_1)

# eltoremove = []
# for index,element in enumerate(shape_1):
#     if element[0]< 17.3 and element[1]>23.5:
#         eltoremove.append(index)

# shape_1 = np.delete(shape_1, eltoremove, axis = 0)



    
plt.scatter(np.asarray(shape_1)[:,1],np.asarray(shape_1)[:,0])
plt.show()   
plt.scatter(np.asarray(shape_2)[:,1],np.asarray(shape_2)[:,0])
plt.show()  
shape_1 = list(shape_1)




contour_vectors1 = []    
    
lenshape1_or = len(list(shape_1))

for index in range(lenshape1_or):
    
    # !!! this is just for the example used
  
    # to close the loop 
    
    shape_1_list = list(shape_1)+[list(shape_1)[0]]
    
    y_rel_init =  shape_1_list[index][0]
    x_rel_init =  shape_1_list[index][1]
    
    
    y_rel_final =  shape_1_list[index+1][0]
    x_rel_final =  shape_1_list[index+1][1]
    
    # !!! use this normally
    # y_rel_init, x_rel_init = rel_contour
    
    contour_vect = (y_rel_init, x_rel_init, y_rel_final, x_rel_final)
    
    contour_vectors1.append(contour_vect)
    
    
contour_vectors3 = []    
    
lenshape3_or = len(list(shape_2))

for index in range(lenshape3_or):
    
    # !!! this is just for the example used
  
    # to close the loop 
    
    shape_3_list = list(shape_2)+[list(shape_2)[0]]
    
    y_rel_init =  shape_3_list[index][0]
    x_rel_init =  shape_3_list[index][1]
    
    
    y_rel_final =  shape_3_list[index+1][0]
    x_rel_final =  shape_3_list[index+1][1]
    
    # !!! use this normally
    # y_rel_init, x_rel_init = rel_contour
    
    contour_vect = (y_rel_init, x_rel_init, y_rel_final, x_rel_final)
    
    contour_vectors3.append(contour_vect)    
        
    


class Contour_vector:
    def __init__(self, rel_init_coords, rel_final_coords):
        self.rel_init_coords=rel_init_coords
        self.rel_final_coords=rel_final_coords


contours_obj1 = []

for contour_el in contour_vectors1:
    contour_obj = Contour_vector(contour_el[:2], contour_el[2:])
    contours_obj1.append(contour_obj)
    
    
contours_obj3 = []

for contour_el in contour_vectors3:
    
    contour_obj = Contour_vector(contour_el[:2], contour_el[2:])
    contours_obj3.append(contour_obj)
    

conts_vert_of_segmented_image = dict()
conts_vert_of_segmented_image['1_rel_vertexs'] = shape_1
conts_vert_of_segmented_image['1_contours'] = contours_obj1
conts_vert_of_segmented_image['3_rel_vertexs'] = shape_2
conts_vert_of_segmented_image['3_contours'] = contours_obj3



    
images_segmented, conts_vertxs_per_region_segmented = SegmWrap.Segment_Images_ContourBased(
    images_to_segment, relative_positions_to_segment, pixel_sizes_to_segment)    
    
    
    
    
# Group clusters separated and denoise
# tolerance 0 cannot be as we need some pixels to be noise for the contour drawing
stacked_segmentation=Segment.Multiple_Consecutive_Clustered_regions(
    stacked_segmentation, criteria='Tolerance', tolerance=0.005)   

plt.figure(figsize=(48,48))
plt.imshow(stacked_segmentation)
plt.show()


# Keep in mind that during the previous steps the image is full size but after the following line then it becomes the original size 
# # plus one pixel in each direction, in this next step, where the contours are computed
conts_vertx_per_region = Segment.Contour_draw_All_regions(
    stacked_segmentation)

final_extra_pixs = Segment.Denoise_All_Regions(
    stacked_segmentation,conts_vertx_per_region)

for label in final_extra_pixs:
    for pixel in final_extra_pixs[label]:  
        stacked_segmentation[pixel[0],pixel[1]]=int(label[0])   

plt.figure(figsize=(48,48))
plt.imshow(stacked_segmentation)
plt.show()        

stacked_segmentation = SegmAlgs.remove_noise(stacked_segmentation)
plt.figure(figsize=(48,48))
plt.imshow(stacked_segmentation)
plt.show()







SegmWrap.Skip_n_contours_region_region_intercon(
    conts_vertx_per_region, segmented_image, skip_n = 2)





'''
Create and fill the supercell defined by that morphology with atoms
'''
# load the unit cell information into ase

temp_xyz_files_folder_directory = r'E:\Arxius varis\PhD\4rth_year\Code\Atomistic_model_builder\InP_InSb_NW_Diagonal\\'


z_thickness = 1   # nm

rotation_angle = 54.28  # degrees

zone_axis = [1,1,0]

cell_filepath=r'E:\Arxius varis\PhD\2nd_year\Code\unit_cells\inas_wz.cif'
cell_filepath=r'E:\Arxius varis\PhD\2nd_year\Code\unit_cells\moo2.cif'
cell_filepath=r'E:\Arxius varis\PhD\2nd_year\Code\unit_cells\ge_fd-3ms.cif'


# pick one of the example regions for testing that shape
label_of_seg_region = 1
cell_file_inp = r'E:\Arxius varis\PhD\2nd_year\Code\unit_cells\inp.cif'
 
# function that does the full shaping and orientation 
Build_shaped_atomistic(
    cell_file_inp, zone_axis, rotation_angle, z_thickness, 
    conts_vert_of_segmented_image, label_of_seg_region, 
    temp_xyz_files_folder_directory, adjust_y_bottomleft = True)

# pick one of the example regions for testing that shape
label_of_seg_region = 3
cell_file_insb = r'E:\Arxius varis\PhD\2nd_year\Code\unit_cells\insb_full.cif'

# function that does the full shaping and orientation 
Build_shaped_atomistic(
    cell_file_insb, zone_axis, rotation_angle, z_thickness, 
    conts_vert_of_segmented_image, label_of_seg_region, 
    temp_xyz_files_folder_directory, adjust_y_bottomleft = True)


# The global process would loop over the different labels and then combine cells

# combine the cells altoghether    

Combine_xyz_supercells(
    temp_xyz_files_folder_directory)


final_global_device_supcell = read(temp_xyz_files_folder_directory + 'global_device_supercell.xyz')
show_atoms(final_global_device_supcell, plane='xy')
show_atoms(final_global_device_supcell, plane='xz')
show_atoms(final_global_device_supcell, plane='yz')


#%%


def Cut_vertical_slab(
        global_supercell_folder_directory, horizontal_coords_slab):
    
    
    hor_i = horizontal_coords_slab[0]
    hor_f = horizontal_coords_slab[1]
    
    global_supercell_name = global_supercell_folder_directory +'\\' + 'global_device_supercell.xyz'
    
    with open(global_supercell_name, "r") as global_supercell_file:
        global_supercell_lines = global_supercell_file.readlines()[2:]
        global_supercell_file.close()
        
    # this list including the symbols of the elements
    global_supercell_words_tot = [np.asarray(line.split(), dtype=object) for line in global_supercell_lines]
    # this list is including the symbols of elements
    global_supercell_words = [np.asarray(line.split()[1:], dtype=np.float32) for line in global_supercell_lines]
    print(global_supercell_words)
    
    global_supercell_pos = np.asarray(global_supercell_words)
    print(global_supercell_pos)
    print(type(global_supercell_pos))
    
    # contains the elements information global_supercell_words_tot
    global_supercell_tot_array = np.asarray(global_supercell_words_tot)
    print(global_supercell_tot_array)
    print(type(global_supercell_tot_array))
    
    print(global_supercell_pos[0])
    
    global_supercell_tot_array = global_supercell_tot_array[global_supercell_pos[:,0]>=hor_i]
    global_supercell_pos = global_supercell_pos[global_supercell_pos[:,0]>=hor_i]
    
    global_supercell_tot_array = global_supercell_tot_array[global_supercell_pos[:,0]<=hor_f]
    global_supercell_pos = global_supercell_pos[global_supercell_pos[:,0]<=hor_f]
    
    slab_cut_lines = [str(len(global_supercell_pos)) + '\n', '\n']
    
    for element_name, position in zip(
            global_supercell_tot_array, global_supercell_pos):
        
        element = element_name[0]
        str_symb_coord = str(element)+'    '+str(position[0])+'    '+str(position[1])+'    '+str(position[2])+'\n'
        
        slab_cut_lines = slab_cut_lines + [str_symb_coord]
        
    # create the .xyz file
    slab_xyz_filename = global_supercell_folder_directory + '\\' + 'slab_cut_' + str(hor_i) + '_to_' + str(hor_f) + '.xyz'
    
    filename = Path(slab_xyz_filename)
    file_already_created = filename.is_file()
    
    if file_already_created == True:
        # overwrite the file
        with open(slab_xyz_filename, 'w+') as f:
            f.truncate(0)
            f.writelines(slab_cut_lines)
            f.close()
    else:
        # create new file
        with open(slab_xyz_filename, 'w+') as f:
            f.writelines(slab_cut_lines)
            f.close()
            
    return slab_cut_lines 


global_cell_file_directory = r'E:\Arxius varis\PhD\4rth_year\Code\Atomistic_model_builder\InP_InSb_NW_Diagonal_slab\\' 

horizontal_coords_slab = (700,1200) 

hor_i = horizontal_coords_slab[0] 
hor_f = horizontal_coords_slab[1] 

Cut_vertical_slab(
    global_cell_file_directory, horizontal_coords_slab)


slab_xyz_filename = global_cell_file_directory + '\\' + 'slab_cut_' + str(hor_i) + '_to_' + str(hor_f) + '.xyz'

slab_cut = read(slab_xyz_filename)
show_atoms(slab_cut, plane = 'xy')
show_atoms(slab_cut, plane = 'xz')
show_atoms(slab_cut, plane = 'yz')


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

    Returns
    -------
    direction_paralel : array uvw Miller indices of the direction 
                    paralel to the plane hkl inputted

    '''
    direction_paralel = np.dot(plane, reciprocal_metric_tensor)
    
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
    
    
    g1_g2_dot = np.dot(plane_1, np.dot(reciprocal_metric_tensor, plane_2))
    
    g1_norm = np.sqrt(np.dot(plane_1, np.dot(reciprocal_metric_tensor, plane_1)))
    g2_norm = np.sqrt(np.dot(plane_2, np.dot(reciprocal_metric_tensor, plane_2)))
    
    
    angle_between = (180/np.pi)*np.arccos(g1_g2_dot/(g1_norm*g2_norm))
    
    
    return angle_between

    
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
            2*np.array(plane_indices, dtype = np.int64), tolerance_diff = tolerance_diff)
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
    show_atoms(unit_cell, plane='xy')
    
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


    # angle_theo_from_plane_1_g_to_x, angle_theo_from_plane_2_g_to_x represent the theoretical 
    # angles from the planes found to the vector that will be aligned with the x axis
    # at the very end of all the surface rotations

    # then, to put the surface in the correct in plane orientatoin
    # we must undo the theoretical rotation between x and hkl1, which ends up aligning
    # hkl1 with the x axis, and then apply the rotation found experimentally between
    # the x axis and the hkl1


    # !!! define the rotationa angle from the plane that has less multiplicity in zone

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
        cell_filepath, scored_spot_pair, tolerance_diff = 0.5, suface_basis_choice = 'plane'):
    
    
    
    # load the cif file of the unit cell
    unit_cell = read(cell_filepath)
    # show atoms if wanted
    show_atoms(unit_cell, plane='xy')
    
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


    # angle_theo_from_plane_1_g_to_x, angle_theo_from_plane_2_g_to_x represent the theoretical 
    # angles from the planes found to the vector that will be aligned with the x axis
    # at the very end of all the surface rotations

    # then, to put the surface in the correct in plane orientatoin
    # we must undo the theoretical rotation between x and hkl1, which ends up aligning
    # hkl1 with the x axis, and then apply the rotation found experimentally between
    # the x axis and the hkl1


    # !!! define the rotationa angle from the plane that has less multiplicity in zone

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
    angles_exp = [angle_exp_plane_1_to_x, angle_exp_plane_2_to_x]
    angles_theo = [angle_theo_from_plane_1_g_to_x, angle_theo_from_plane_2_g_to_x]
    
    # define the plane and angles that need to be used for obtaining the final hkl
    plane_found_use = planes_for_inplane_rot[np.argmin(inplane_multipl)]
    angle_exp_use = angles_exp[np.argmin(inplane_multipl)]
    angle_theo_use = angles_theo[np.argmin(inplane_multipl)]
    print('plane_found_use')
    print(plane_found_use)
    
    # !!! IN-PLANE ROTATION ANGLEthis is the rotation we must rotate the surface 
    # object to align the inplane rotation we want to orient 
    # in plane as observed experimentally

    final_in_surf_plane_rotation = angles_exp[np.argmin(inplane_multipl)] - angles_theo[np.argmin(inplane_multipl)]


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


class Scored_spot_pair:
    
    def __init__(self, ZA, spot1_angle_to_x, spot2_angle_to_x,
                 hkl1_reference, hkl2_reference):
        self.ZA = ZA
        self.spot1_angle_to_x = spot1_angle_to_x
        self.spot2_angle_to_x = spot2_angle_to_x
        angle_between=np.abs(self.spot2_angle_to_x - self.spot1_angle_to_x)
        if angle_between>180:
            angle_between=360-angle_between

        self.angle_between = angle_between
        self.hkl1_reference = hkl1_reference
        self.hkl2_reference = hkl2_reference
  
cell_filepath=r'E:\Arxius varis\PhD\2nd_year\Code\unit_cells\moo2.cif'
cell_filepath=r'E:\Arxius varis\PhD\2nd_year\Code\unit_cells\ge_fd-3ms.cif'
cell_filepath=r'E:\Arxius varis\PhD\2nd_year\Code\unit_cells\inas_wz.cif'


zone_axis = [2,0,1] 
plane_1_found = [ -1, -1,  1]
plane_1_found = [-1,  1, 2]
plane_2_found = [-2,  0,  -2]
plane_2_found = [ -1, 0, 2]

angle_exp_plane_1_to_x = 19.119726329931154 
angle_exp_plane_2_to_x = 54.36019080191333
   


scored_spot_pair = Scored_spot_pair(
    zone_axis, angle_exp_plane_1_to_x, angle_exp_plane_2_to_x, 
    np.asarray(plane_1_found), np.asarray(plane_2_found))
   

final_in_surf_plane_rotation = Adjust_in_surface_plane_rotation(
    cell_filepath, scored_spot_pair, suface_basis_choice = 'plane')

print('angle to rotate the surface') 
print(final_in_surf_plane_rotation)


plane_final_cartesian_x, direction_final_cartesian_x = Find_plane_pointing_to_final_cartesian_x_axis(
    cell_filepath, scored_spot_pair, tolerance_diff = 0.4, suface_basis_choice = 'plane')

print('plane_final_cartesian_x')
print(plane_final_cartesian_x)
print('direction_final_cartesian_x')
print(direction_final_cartesian_x)


