# -*- coding: utf-8 -*-
"""
Created on Tue Feb 15 12:06:24 2022

@author: Marc
"""

import numpy as np
import sklearn.cluster
import sklearn.mixture
import cv2
import networkx
from collections import defaultdict


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
    Changed sligthly from previous version, now it outputs directly the discretised image, with
    integers of the pixels of interest, and the remaining unclustered pixels having a 0 intensity
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
        
    # Once the clusters are ordered and selected according to the criteria, convert the image into 
    # a discretised array with every contigous regions with a specific number 
    
    output_discretised_image=np.zeros(np.shape(image_array_clusters))
    
    if criteria=='Tolerance':
    # Way of dealing with the 'Tolerance' criteria to get output discretised with 0,1,2....
        value_image=1
        for cluster in output:
            cluster_i=output[str(cluster)]
            for pixel in cluster_i:
                (pixel_y,pixel_x)=pixel.coords
                output_discretised_image[pixel_y,pixel_x]=value_image
            value_image=value_image+1
    
    if criteria=='Num_Clusters':
    # Way of dealing with the 'Num_clusters' criteria to get output discretised with 0,1,2....
        value_image=1
        for cluster in output:
            cluster_i=cluster.cluster
            for pixel in cluster_i:               
                (pixel_y,pixel_x)=pixel.coords
                output_discretised_image[pixel_y,pixel_x]=value_image
        
            value_image=value_image+1
      
    return  output_discretised_image


# DO NOT USE THIS FUNCTION
# KEEP IT TO USE SOME STUFF WITHIN
# Function that draws the contours (vertex and vectors) of a region given an initial pixel from its contour
def Contour_drawing_initial_pixel_BUGGED(image_array_labels, initial_pixel):
    
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
            
            # number_of_found_vertexs=0
            for possible_vertex, m_out in zip(possible_vertexs,m_outs):
                number_of_found_vertexs=0
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




class contour_vector:
    def __init__(self,m_in,m_out,init_coords,final_coords):
        self.m_in=m_in
        self.m_out=m_out
        self.init_coords=init_coords
        self.final_coords=final_coords
        
    def add_relative_contour(self,rel_init_coords,rel_final_coords):
        '''
        The coordinates that are intrinsec of the object contour_vector, init_coords, and final_coords,
        which define the contour in terms of the pixels within the image (pixel coordinates)
        are changed according to the relative coordinate system defined by the template matching
        algorithm, and after the proper conversion by dedicated function: Relative_Vertexs_Contours()
        they are added by the present method and define the relative coords:rel_init_coords,rel_final_coords 
        and can be directly accessed with the previous class
        
        Parameters
        ----------
        rel_init_coords : 
        rel_final_coords : 
        '''
        self.rel_init_coords=rel_init_coords
        self.rel_final_coords=rel_final_coords
        
        
    def graph_representation(self, intvertex_init, intvertex_final):
        '''
        !!! Internal references for a given label, not label interchangable
        Convert the contours in graphs formally, where each vertex
        is represented with an integer in an internal reference system
        to better track the edges
        '''
        self.intvertex_init = intvertex_init
        self.intvertex_final = intvertex_final
        
        



def Contour_drawing_initial_pixel(image_array_labels, initial_pixel):
    
            
    # first define the rules for defining a contour in each of the 4 (N,S,E,W)
    
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
    
    def append_to_vertex_array(possible_vertex,vertex_array):
        # change the append of possible vertex in the vertex array to ensure only new ones are added
        if not possible_vertex in vertex_array:
            vertex_array.append(possible_vertex)
        
    
    for vertex in vertex_array:
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
        
        
        vertex_None=0
        for possible_vertex in possible_vertexs:
            if possible_vertex ==None:
                vertex_None=vertex_None+1
        
        if vertex_None==3:
            # only one direction leads to another vertex
            # no vertex repetition, that vertex only leads to 1 possible vertex (1 contour)
            for possible_vertex, m_out in zip(possible_vertexs,m_outs):
                if possible_vertex !=None:
                    if possible_vertex == initial_pixel_coords:
                        new_contour=contour_vector(target_pixel,m_out,vertex,possible_vertex)
                        contour_vectors_list.append(new_contour)
                        #break 
                    elif possible_vertex in list_double_vertexs:
                        new_contour=contour_vector(target_pixel,m_out,vertex,possible_vertex)
                        contour_vectors_list.append(new_contour)
                        list_double_vertexs.remove(possible_vertex)                    
                    else:
                        append_to_vertex_array(possible_vertex,vertex_array)
                        
                    
                        #generate new contour vector and append it to the contour vectors list
                        new_contour=contour_vector(target_pixel,m_out,vertex,possible_vertex)
                        contour_vectors_list.append(new_contour)
                        #generate new contour vector and append it to the contour vectors list
                        new_contour=contour_vector(target_pixel,m_out,vertex,possible_vertex)
                        contour_vectors_list.append(new_contour)



        else:
            #which stands for vertex_None==2
            # this vertex leads to two vertexs and then account for the exception   
            
            for possible_vertex, m_out in zip(possible_vertexs,m_outs):
                if possible_vertex !=None:
                    if possible_vertex == initial_pixel_coords:
                        new_contour=contour_vector(target_pixel,m_out,vertex,possible_vertex)
                        contour_vectors_list.append(new_contour)
                        #break 
                    elif possible_vertex in list_double_vertexs:
                        new_contour=contour_vector(target_pixel,m_out,vertex,possible_vertex)
                        contour_vectors_list.append(new_contour)
                        list_double_vertexs.remove(possible_vertex)
                    else:
                        append_to_vertex_array(possible_vertex,vertex_array)
                    
                        #generate new contour vector and append it to the contour vectors list
                        new_contour=contour_vector(target_pixel,m_out,vertex,possible_vertex)
                        contour_vectors_list.append(new_contour)

            list_double_vertexs.append(vertex)
            
            
    # check for duplicates in the contour_vector_list
    
    def string_contour(contour):
        stringed_contour = str(contour.init_coords) + str(contour.final_coords) + str(contour.m_in) + str(contour.m_out)
        return stringed_contour
    
    stringed_contours = map(string_contour, contour_vectors_list)
    stringed_contours_list = list(stringed_contours)
    unique_vals, indexes_unique = np.unique(stringed_contours_list, return_index = True)
    indexes_unique_sorted  = sorted(indexes_unique)
    
    contour_vectors_array = np.asarray(contour_vectors_list)
    unique_contour_vect_array = contour_vectors_array[indexes_unique_sorted]
    # already with unique vals but list again 
    contour_vectors_list = list(unique_contour_vect_array)
                

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
    
    # if we want to keep 0 as noise (Segm1stAprox)
    #label 0 is associated to pixels that do not have a label
    # label_values=np.sort(label_values)[1:]
    
    # if we want to keep 0 as singal (SegmAlgs)
    label_values=np.sort(label_values)

    for label in label_values:
        contour_vectors_list, vertex_array=Contour_draw_computing_initial_pixel(
            image_array_labels, label)
        conts_vertx_per_region[str(int(label))+'_'+'vertexs']=np.asarray(vertex_array)
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
    label_values=np.sort(label_values)

    
    dict_extra_pixels=dict()
    for label in label_values:

        dict_extra_pixels[str(int(label))+'_extra_pixels']=Denoise_region(image_array,conts_vertx_per_region,label)
    
    return dict_extra_pixels









#Make pixels considered as noise by the initial clustering be assigned to the cluster in which they should be,
#making the pixels inside a given border be assigned to this cluster
def Denoise_region_other_labels(
        image_array,conts_vertx_per_region,region_label):
    '''
    Does the same as Denoise_region but takes out any pixel within contour that 
    is not of the same value as the label, even if it is not noise, 0, but just 
    a divergent path from the main one

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

            if image_array[pixel_y,pixel_x] != region_label:
                #only evaluate if the pixel is different from the label, as are the ones we want to change
                
                to_north=Check_vector_North((pixel_y,pixel_x),contour_vectors)
                to_south=Check_vector_South((pixel_y,pixel_x),contour_vectors)
                to_east=Check_vector_East((pixel_y,pixel_x),contour_vectors)
                to_west=Check_vector_West((pixel_y,pixel_x),contour_vectors)
                
                if to_north==True and to_south==True and to_east==True and to_west==True:
                    extra_pixels.append((pixel_y,pixel_x))
                    
    return extra_pixels




#Denoise all the regions, looping through the labels that represent actual regions (not 0)
def Denoise_All_Regions_other_labels(
        image_array,conts_vertx_per_region):
    '''
    Apply the denoising for all the regions and all the labels that have an associated region
    (not only for noise, 0, but for all other labels different than the main region label)
    
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
    label_values=np.sort(label_values)

    
    dict_extra_pixels=dict()
    for label in label_values:
        dict_extra_pixels[str(int(label))+'_extra_pixels']=Denoise_region_other_labels(
            image_array,conts_vertx_per_region,label)
    
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
        gmm = sklearn.mixture.BayesianGaussianMixture(n_components=number_clusters, n_init=5, max_iter=500)
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
            target_label=index+2
            image_to_refine=label_final_val*(copy_labels_gauss==target_label)+image_to_refine*(copy_labels_gauss!=target_label)
            
            label_final_val=label_final_val+1
    
    return image_to_refine




def Relative_Vertexs_Contours(
        array_to_segment, conts_vertx_per_region, relative_position, pixel_size):
    '''
    Convert the identified vertices and contours (vertices pair) into real coordinates in nm, depending on
    where the image is located according to the relative positions map found by template matching, in order to
    have for each segment the spatial localisation, in nm, of its contour and the region they hold
    The function only modifies the inputed conts_vertx_per_region dictionary by adding the relative pixels
    and modifying the contour_vector objects by adding the initial and final coords in relative terms

    Parameters
    ----------
    array_to_segment : must have a lavel 0 to account for the noise
    conts_vertx_per_region : dictionary
    relative_position : 
    pixel_size

    Returns
    -------
    conts_vertx_per_region : same dictionary

    '''
    
    
    (x_relative_position_start, y_relative_position_start, _ , _)=relative_position
        
    label_values=np.unique(array_to_segment)
    
    # if we want to keep 0 as noise (Segm1stAprox)
    #label 0 is associated to pixels that do not have a label
    # label_values=np.sort(label_values)[1:]
    
    # if we want to keep 0 as singal (SegmAlgs)
    label_values=np.sort(label_values)
    
    #the contour vector seems to work fine but the pixel ones seem to be capturing the integer only
    
    for label in label_values:
          
        rel_vertexs=np.float64(np.copy(conts_vertx_per_region[str(int(label))+'_'+'vertexs']))
        rel_vertexs[:,0]=y_relative_position_start+pixel_size*rel_vertexs[:,0]
        rel_vertexs[:,1]=x_relative_position_start+pixel_size*rel_vertexs[:,1]
        conts_vertx_per_region[str(int(label))+'_'+'rel_vertexs']=rel_vertexs
        
        for contour_vector in conts_vertx_per_region[str(int(label))+'_'+'contours']:
            init_rel_coords_y,init_rel_coords_x =contour_vector.init_coords
            final_rel_coords_y,final_rel_coords_x=contour_vector.final_coords
            
            init_rel_coords_y=y_relative_position_start+pixel_size*init_rel_coords_y
            init_rel_coords_x=x_relative_position_start+pixel_size*init_rel_coords_x
            final_rel_coords_y=y_relative_position_start+pixel_size*final_rel_coords_y
            final_rel_coords_x=x_relative_position_start+pixel_size*final_rel_coords_x
            
            rel_init_coords=(init_rel_coords_y,init_rel_coords_x)
            rel_final_coords=(final_rel_coords_y,final_rel_coords_x)
            
            contour_vector.add_relative_contour(rel_init_coords,rel_final_coords)
            

    return conts_vertx_per_region




  
# Python program to print all paths from a source to destination.
# This class represents a directed graph
# using adjacency list representation
class Graph:  
# modified from code contributed by Neelam Yadav
    def __init__(self, vertices):
        # No. of vertices
        self.V = vertices
         
        # default dictionary to store graph
        self.graph = defaultdict(list)
  
    # function to add an edge to graph
    def addEdge(self, u, v):
        self.graph[u].append(v)
  
    '''A recursive function to print all paths from 'u' to 'd'.
    visited[] keeps track of vertices in current path.
    path[] stores actual vertices and path_index is current
    index in path[]'''
    def printAllPathsUtil(self, u, d, visited, path):
 
        # Mark the current node as visited and store in path
        visited[u]= True
        path.append(u)
 
        # If current vertex is same as destination, then print
        # current path[]
        if u == d:
            print (path)
        else:
            # If current vertex is not destination
            # Recur for all the vertices adjacent to this vertex
            for i in self.graph[u]:
                if visited[i]== False:
                    self.printAllPathsUtil(i, d, visited, path)
                     
        # Remove current vertex from path[] and mark it as unvisited
        path.pop()
        visited[u]= False
  
  
    # Prints all paths from 's' to 'd'
    def printAllPaths(self, s, d):
 
        # Mark all the vertices as not visited
        visited =[False]*(self.V)
 
        # Create an array to store paths
        path = []
 
        # Call the recursive helper function to print all paths
        self.printAllPathsUtil(s, d, visited, path)
        
  



def Find_shortest_longest_closed_contour_simple_path(
        label, conts_vertx_per_region):
    '''
    Function to find, given a set of contours for a given label, the 
    shortest and longest simple path (path consisting of non repeated nodes)
    and returns these paths as list of contour objects, new vertexs and rel vertexs
    

    Parameters
    ----------
    label : TYPE
        DESCRIPTION.
    conts_vertx_per_region : TYPE
        DESCRIPTION.

    Returns
    -------
    contour_vertex_lists = (shortest_contours_list, shortest_vertexs_list,
                            shortest_rel_vertexs_list, longest_contours_list, 
                            longest_vertexs_list, longest_rel_vertexs_list)    

    '''
    
    
    # first we want to formalise the contours into graphs
    # make every edge to have an internal reference
    # to make comparisons easy
    # and keep track of this change in nomenclature
    
    vertexs_list = conts_vertx_per_region[str((label))+'_vertexs']
    contours_list = conts_vertx_per_region[str(int(label))+'_contours']  
    rel_vertexs_list = conts_vertx_per_region[str(int(label))+'_rel_vertexs'] 
    
    # vertex list is unique
    vertexs_tot_numb = len(vertexs_list)
    vertex_graph_intref = np.arange(0, vertexs_tot_numb, 1)
    
    for contour in contours_list:
        
        vertex_init = contour.init_coords
        vertex_final = contour.final_coords
        
        
        bool_init = vertexs_list==vertex_init 
        bool_init=bool_init[:,0]*bool_init[:,1]
        
        
        init_int_ref = vertex_graph_intref[bool_init][0]
        
        bool_final = vertexs_list==vertex_final 
        bool_final =bool_final[:,0]*bool_final[:,1]
        final_int_ref = vertex_graph_intref[bool_final][0]

        contour.graph_representation(init_int_ref, final_int_ref)
    
        
    # now the contours should have the internal reference of the
    # edge stored as well so me can easily map one back and forth
    
    
    # add the internal reference edges to the graph  
    list_int_edges = []
    
    for contour in contours_list:
        temp_listtotuple = [contour.intvertex_init, contour.intvertex_final]
        list_int_edges.append(tuple(temp_listtotuple))
    
    # now that all the edges are added in int reference format
    # find the initial and final vertices to compute the path
    # the vertices are vertex final n and initial n+1 where n+1 to n 
    # edge exists and they are unique in the contours list
    # although does not really matter just choose the first one
    
    initial_path_vertex = contours_list[0].intvertex_final
    final_path_vertex = contours_list[0].intvertex_init
        
    # now find all paths going from the initial vertex to last one
    # as the closed path is just adding the contour from    
    # final_path_vertex to initial_path_vertex
        
    nx_contour_graph = networkx.from_edgelist(list_int_edges, create_using=None)
    all_simple_paths = networkx.all_simple_paths(
        nx_contour_graph, initial_path_vertex, final_path_vertex, cutoff=None)
    
    all_simple_paths_list = list(all_simple_paths)
    
    
    # remove dummy paths or if small paths like noise pixels are closed as 
    # standalone units
    all_simple_paths_ref = [simple_path for simple_path in all_simple_paths_list if len(simple_path)> len(contours_list)/3]
    all_simple_paths_ref_lens = [len(simple_path) for simple_path in all_simple_paths_ref]
    
    shortest_simple_path = all_simple_paths_ref[np.argmin(np.asarray(all_simple_paths_ref_lens))]
    longest_simple_path = all_simple_paths_ref[np.argmax(np.asarray(all_simple_paths_ref_lens))]
    
    # other addditional  networkx methods all_shortest_paths, shortest_path
    
    shortest_path_edges = [(shortest_simple_path[i],shortest_simple_path[i+1]) for i, el in enumerate(shortest_simple_path[:-1])]
    shortest_path_edges = shortest_path_edges + [(shortest_simple_path[-1], shortest_simple_path[0])]
    
    longest_path_edges = [(longest_simple_path[i],longest_simple_path[i+1]) for i, el in enumerate(longest_simple_path[:-1])]
    longest_path_edges = longest_path_edges + [(longest_simple_path[-1], longest_simple_path[0])]

    # and add the refernce in the contour list as a new edge
    # as a new attribute of the contour vector class
    
    
    # find the m_in and m_out per contour again
    
    # create a new contour vector list based on the new path for both 
    # shortest and longest simple paths found
    
    shortest_contours_list = []
    
    for short_edge in shortest_path_edges:
        
        # map the edge in graph notation to pixel coords and keep same materials
        # m_in and m_out
        
        for original_contour in contours_list:
            
            if original_contour.intvertex_init == short_edge[0] and original_contour.intvertex_final == short_edge[1]:
                short_init_coords = original_contour.init_coords
                short_final_coords = original_contour.final_coords
                short_m_in = original_contour.m_in
                short_m_out = original_contour.m_out
                short_rel_init_coords = original_contour.rel_init_coords
                short_rel_final_coords = original_contour.rel_final_coords
                
                
        # fill the contour objects with the info        
        short_contour = contour_vector(
            short_m_in, short_m_out, short_init_coords, short_final_coords)
        short_contour.add_relative_contour(
            short_rel_init_coords, short_rel_final_coords)
        short_contour.graph_representation(
            short_edge[0], short_edge[1])
        
        shortest_contours_list.append(short_contour)
        
    longest_contours_list = []
    
    for long_edge in longest_path_edges:
        
        # map the edge in graph notation to pixel coords and keep same materials
        # m_in and m_out
        
        for original_contour in contours_list:
                
            if original_contour.intvertex_init == long_edge[0] and original_contour.intvertex_final == long_edge[1]:
                long_init_coords = original_contour.init_coords
                long_final_coords = original_contour.final_coords
                long_m_in = original_contour.m_in
                long_m_out = original_contour.m_out
                long_rel_init_coords = original_contour.rel_init_coords
                long_rel_final_coords = original_contour.rel_final_coords
                
                
        # fill the contour objects with the info        

        long_contour = contour_vector(
            long_m_in, long_m_out, long_init_coords, long_final_coords)
        long_contour.add_relative_contour(
            long_rel_init_coords, long_rel_final_coords)
        long_contour.graph_representation(
            long_edge[0], long_edge[1])
        
        longest_contours_list.append(long_contour)
    

    shortest_vertexs_list = []
    shortest_rel_vertexs_list = []
    
    for int_vertx in shortest_simple_path:
        
        vertex_corr = vertexs_list[vertex_graph_intref == int_vertx]
        rel_vertex_corr = rel_vertexs_list[vertex_graph_intref == int_vertx]
        
        shortest_vertexs_list.append(vertex_corr[0])
        shortest_rel_vertexs_list.append(rel_vertex_corr[0])
        
            
    longest_vertexs_list = []
    longest_rel_vertexs_list = []
    
    for int_vertx in longest_simple_path:
        
        vertex_corr = vertexs_list[vertex_graph_intref == int_vertx]
        rel_vertex_corr = rel_vertexs_list[vertex_graph_intref == int_vertx]
        
        longest_vertexs_list.append(vertex_corr[0])
        longest_rel_vertexs_list.append(rel_vertex_corr[0])
        

    shortest_vertexs_list = np.asarray(shortest_vertexs_list)
    shortest_rel_vertexs_list = np.asarray(shortest_rel_vertexs_list)
    longest_vertexs_list = np.asarray(longest_vertexs_list)
    longest_rel_vertexs_list = np.asarray(longest_rel_vertexs_list)
    
    contour_vertex_lists = (shortest_contours_list, shortest_vertexs_list, shortest_rel_vertexs_list, longest_contours_list, longest_vertexs_list, longest_rel_vertexs_list)    
    
    return contour_vertex_lists






def Conts_verts_per_region_shortest_longest_path(
        segmented_image, conts_vertx_per_region):
    '''
    Find all the shortest paths and longest paths for all the labels present
    in the segmented_image that is inputted
    and return two new  conts_vertx_per_region objects, dictionaries, with
    the shortest and longest contours for all the labels

    Parameters
    ----------
    segmented_image : TYPE
        DESCRIPTION.
    conts_vertx_per_region : TYPE
        DESCRIPTION.

    Returns
    -------
    conts_vertx_per_region_shortest : TYPE
        DESCRIPTION.
    conts_vertx_per_region_longest : TYPE
        DESCRIPTION.

    '''
    
    labels = np.unique(segmented_image)
    
    conts_vertx_per_region_shortest = dict()
    conts_vertx_per_region_longest = dict()
    
    for label in labels:
        
        contour_vertex_lists = Find_shortest_longest_closed_contour_simple_path(
            int(label), conts_vertx_per_region)
        
        (shortest_contours_list, shortest_vertexs_list, shortest_rel_vertexs_list, longest_contours_list, longest_vertexs_list, longest_rel_vertexs_list) = contour_vertex_lists 
    
        # fill the dictionary with the new data from labels
        conts_vertx_per_region_shortest[str(int(label))+'_contours'] = shortest_contours_list
        conts_vertx_per_region_shortest[str(int(label))+'_vertexs'] = shortest_vertexs_list
        conts_vertx_per_region_shortest[str(int(label))+'_rel_vertexs'] = shortest_rel_vertexs_list
    
        conts_vertx_per_region_longest[str(int(label))+'_contours'] = longest_contours_list
        conts_vertx_per_region_longest[str(int(label))+'_vertexs'] = longest_vertexs_list
        conts_vertx_per_region_longest[str(int(label))+'_rel_vertexs'] = longest_rel_vertexs_list
    
    
    return conts_vertx_per_region_shortest, conts_vertx_per_region_longest



upload to github only this file
def Unify_Contour_Vectors(
    conts_vertx_per_region, labels_equally_oriented_as_ref):
    '''
    Unify the cont_vertxs_data for all the labels within 
    labels_equally_oriented_as_ref 
    and make them one for a label 99
    this way cannot be confused with another region
    
    Parameters
    ----------
    conts_vertx_per_region : 
    labels_equally_oriented_as_ref : TYPE

    Returns
    -------
    conts_vertx_per_region_unified : dict

    '''
    
    conts_vertx_per_region_unified = dict()
    
    conts_vertx_per_region_unified['99_contours'] = []
    conts_vertx_per_region_unified['99_vertexs'] = []
    conts_vertx_per_region_unified['99_rel_vertexs'] = []
    
    for element_cont in conts_vertx_per_region:
        
        label_i = int(element_cont[:element_cont.find('_')])
        
        if label_i in labels_equally_oriented_as_ref:
            rest_of_dictentry = element_cont[element_cont.find('_'):]
            conts_vertx_per_region_unified['99' + rest_of_dictentry] = conts_vertx_per_region_unified['99' + rest_of_dictentry] + list(conts_vertx_per_region[element_cont])
    
    
    conts_vertx_per_region_unified['99_vertexs'] = np.asarray(conts_vertx_per_region_unified['99_vertexs'])
    conts_vertx_per_region_unified['99_rel_vertexs'] = np.asarray(conts_vertx_per_region_unified['99_rel_vertexs'])   
    
    
    # Remove the coincident contours that overlap and point at opposite directions
    conts_check = conts_vertx_per_region_unified['99_contours'].copy()
    
    order_pos1 = list(map(lambda x: [x.init_coords, x.final_coords], conts_check))
    order_pos2 = list(map(lambda x: [x.final_coords, x.init_coords], conts_check))
    
    final_conts = []
    
    for cont_check, order1, order2 in zip(
            conts_check, order_pos1, order_pos2):
        
        if order1 not in order_pos2:
            final_conts.append(cont_check)
            
            
    conts_vertx_per_region_unified['99_contours'] = final_conts
        
    
    return conts_vertx_per_region_unified

