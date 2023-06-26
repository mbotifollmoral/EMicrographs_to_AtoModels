# -*- coding: utf-8 -*-
"""
Created on Fri Dec 31 11:04:00 2021

@author: Marc
"""


'''
File intended to translate a complex geometry or contour  (of real structure)
into nextnano coordiantes
'''

import numpy as np
import matplotlib.pyplot as plt
import skimage
import skimage.measure
import sklearn.cluster
import sklearn.mixture
import cv2

import hyperspy.api as hs
import nextnanopy as nn
import gdspy
from nextnanopy.nnp.shapes import GdsPolygons

'''
# Script generating the segment, copy paste from  Full_segmentation_routine_low_mags.py
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
downscaling_factor=4  #for trials, n factor of downsampling size of image
n_gaussian_ref=3   #number of components in the gaussian refinement (like k means but if doubt overstimate)
k_means_clusters=3  #number of components in k means clustering
number_final_regions=4 #number of clearly distinct regions

imagedirectory=r'E:\Arxius varis\PhD\2nd_year\Code\trial_images\low_mag_images\nanowire1_tiff.tif'
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
# Start of the translation to nextnano profile
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
        y=(x_pixels_original/downscaling_factor) - vertex[0]
        
        x=round((x*downsampled_image_pixel_calibration)*1000)
        y=round((y*downsampled_image_pixel_calibration)*1000)
        
        list_of_nnvertexs_per_label.append([x, y])
    
    nextnano_vertexs[str(int(label))+'_'+'nnvertexs']=list_of_nnvertexs_per_label




# The GDSII file is called a library, which contains multiple cells.
lib = gdspy.GdsLibrary()

# Geometry must be placed in cells.
cell = lib.new_cell('FIRST')

#labelling not mandatory but can be useful, datatype is the label
#the labels are specially useful for the gds structure but not that much
#for the nextnano geometry generation, as the labels are inferred automatically

# Create a polygon from a list of vertices

for label in label_values:
    
    list_of_nnvertexs_per_polygon=nextnano_vertexs[str(int(label))+'_'+'nnvertexs']
    ld_label = {"layer": 0, "datatype": int(label)}

    polygonshape = gdspy.Polygon(list_of_nnvertexs_per_polygon, **ld_label)
    cell.add(polygonshape)
    
# Save the library in a file called 'first.gds'.
path_savegds=r'E:\Arxius varis\PhD\3rd_year\Code\FEM_modelling_code\gds_examples'
lib.write_gds(path_savegds+'\\first.gds')

# Optionally, save an image of the cell as SVG.
cell.write_svg(path_savegds+'\\first.svg')

# Display all cells using the internal viewer.
gdspy.LayoutViewer()


#create nextnano object from the generated gds

#No need to define labels to each shape as they are automatically assigned 

my_gds = GdsPolygons(r'E:\Arxius varis\PhD\3rd_year\Code\FEM_modelling_code\gds_examples\first.gds')


print(f"Number of polygons: {my_gds.nb_polygons}")
my_gds.show()

xy_s = my_gds.xy
#x and y hold the coordinates of the vertices
#x and y are arrays of dimension [number of polygons, number of vertices]

fig, ax = plt.subplots(1)
my_gds.show(ax=ax)

for xyi in xy_s:
    xi, yi = xyi
    ax.plot(xi,yi,'ro')
    

  

#!!! Generate polygonal prism from coordinates  
list_of_shapes = my_gds.get_polygonal_prisms(zi=0,zf=20) # initial and final z coordinates are needed for this method
print(list_of_shapes)
for shape in list_of_shapes:
    print(shape.text)  
    
  

 # The smoothing by adding a curve that adds points and smoothes down the
 # points added does not work as expected as generates strange curvse from 
 # strange patterns, already tried and not working
 # it is actually really not needed as there will be many points and the contour will 
 # allraedy be smooth enough
 
# c6 = gdspy.Curve(0, 3).i([(1, 0), (2, 0), (1, -1)], cycle=True)

# polygonshape = gdspy.Curve(0,0).i(list_of_nnvertexs_per_polygon, cycle=True)
# polygons_smooth = gdspy.Polygon([polygonshape.get_points()])
# print(polygonshape.get_points())
# cell.add(polygons_smooth)

# this does not smoooth but generate a curve that nothing has to do with real shape
