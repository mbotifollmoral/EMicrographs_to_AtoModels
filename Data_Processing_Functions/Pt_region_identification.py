# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 14:14:11 2020

@author: Marc
"""


import numpy as np
import matplotlib.pyplot as plt
import scipy 
import skimage
import sklearn.cluster
import cv2
    

image=plt.imread(r'E:\Arxius varis\PhD\2nd_year\Code\trial_images\low_mag_images\nanowire2_tiff.tif')
img2=cv2.imread(r'C:/Users/Marc/Desktop/bunicuss.tif')
img2_gray = skimage.color.rgb2gray(img2)



#sliding windows method

def Sliding_Window(image,window_size, step):
    '''
    Parameters
    ----------
    image : TYPE
    window_size : TYPE
    step : TYPE

    Returns
    -------
    None.
    '''
    ver_pix=image.shape[0]
    hor_pix=image.shape[1]
    
    wind_image=[]
    
    for ver_elem in range(int(((ver_pix-window_size)/step)+1)):
        for hor_elem in range(int(((hor_pix-window_size)/step)+1)):
            #each window has to be labelled accordinigly to do not lose its track of where it comes from   
            window=image[ver_elem:(window_size+ver_elem),hor_elem:(window_size+hor_elem)]
            print()
            
            #array with all the generated windows, too memory expensive
            wind_image.append(window)
            
            #Do something in each windows
            
    return wind_image


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

#Spectral clustering

def spectral_clustering(image_,n_clusters):
    graph = sklearn.feature_extraction.image.img_to_graph(image_)
    print(graph)
    print(type(graph), graph.shape[0])
    print(graph.data)
    print(type(graph.data), len(graph.data))
    # Take a decreasing function of the gradient: we take it weakly
    # dependent from the gradient the segmentation is close to a voronoi
    beta = 1
    eps = 1e-6
    graph.data = np.exp(-beta * graph.data / graph.data.std()) + eps
    #graph.data = np.exp(-graph.data / graph.data.std())
    # Force the solver to be arpack, since amg is numerically
    # unstable on this example
    labels = sklearn.cluster.spectral_clustering(graph, n_clusters=n_clusters, eigen_solver='amg')
    labels = labels.reshape(image_.shape)
   
    return labels


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



blur=cv2.GaussianBlur(image, (5,5), 1)
image=blur
#Image preprocessing, standarisation
#Range from 0 to 1 the intensity values
image_st=(image-np.min(image))/np.max(image)


croped_image=image_st[0:511,:]
#windows=Sliding_Window(image_st, window_size=100, step=1)

# Group similar grey levels using 8 clusters
values, labels, cost = km_clust(image_st, n_clusters =3 )
print(values)
print(labels)
print(cost)


# Create the segmented array from labels and values
img_segm = np.choose(labels, values)
# Reshape the array as the original image
img_segm.shape = blur.shape
# Get the values of min and max intensity in the original image
vmin = image_st.min()
vmax = image_st.max()
# Plot the original image


plt.imshow(img_segm, cmap=plt.cm.gray, vmin=vmin, vmax=vmax)

plt.show()

# Group similar grey levels using 8 clusters
values, labels, cost = km_clust(croped_image, n_clusters =4)
print(values)
print(labels)
print(cost)


# Create the segmented array from labels and values
img_segm = np.choose(labels, values)
# Reshape the array as the original image
img_segm.shape = croped_image.shape
# Get the values of min and max intensity in the original image
vmin = image_st.min()
vmax = image_st.max()
# Plot the original image


plt.imshow(img_segm, cmap=plt.cm.gray, vmin=vmin, vmax=vmax)

plt.show()
#croped_image=skimage.measure.block_reduce(croped_image, block_size=tuple(np.int32(2*np.ones(len(np.shape(croped_image))))), func=np.max, cval=0)



   
scaling_factor=8

image_st=skimage.measure.block_reduce(blur, block_size=tuple(np.int32(scaling_factor*np.ones(len(np.shape(blur))))), func=np.mean, cval=0)

plt.imshow(blur,cmap=plt.cm.gray)
plt.show()
labels_ms,clusters_ms=Mean_shift(image_st)
print(labels_ms, len(labels_ms))
print(clusters_ms, len(clusters_ms))


labels_ms_reshaped=np.reshape(labels_ms, np.shape(image_st))


plt.imshow(labels_ms_reshaped,cmap=plt.cm.gray )
plt.show()

labels_ms_reshaped_f=scipy.ndimage.zoom(labels_ms_reshaped, scaling_factor, order=1)
plt.imshow(labels_ms_reshaped_f,cmap=plt.cm.gray )
plt.show()


def Region_outlier_identification(image_array, max_distance):
    '''

    Parameters
    ----------
    image: the image with the labels (each region has a different number associated)
    max_distance : number of pixels that are evaluated arround the target pixel to
                   assert whether it is an outlier of a region or not     

    Returns
    -------
    smoothed_image : image without the detected outliers

    '''
    changes=0
    
    ver_pixs=image_array.shape[0]
    hor_pixs=image_array.shape[1]
    
    smoothed_image=np.copy(image_array)
    
    for pixel_y in range(ver_pixs):
        for pixel_x in range(hor_pixs):

            target_pixel=image_array[pixel_y,pixel_x]
            
            
            
            
            #Define the surrounding pixels
            
            #condition for north
            if pixel_y-max_distance<0:
                north_surr_pixels=image_array[0:pixel_y,pixel_x]
                if pixel_y==0:
                    north_surr_pixels=np.array([])
            else:
                north_surr_pixels=image_array[pixel_y-max_distance:pixel_y,pixel_x]
             
            #condition for west
            if pixel_x-max_distance<0:
                west_surr_pixels=image_array[pixel_y,0:pixel_x]
                if pixel_x==0:
                    west_surr_pixels=np.array([])
            else:
                west_surr_pixels=image_array[pixel_y,pixel_x-max_distance:pixel_x]
            
            #condition for south
            if pixel_y+max_distance+1>ver_pixs:
                south_surr_pixels=image_array[pixel_y+1:ver_pixs,pixel_x]
                if pixel_y==ver_pixs-1:
                    south_surr_pixels=np.array([])
            else:
                south_surr_pixels=image_array[pixel_y+1:pixel_y+max_distance+1,pixel_x]
            
            #condition for east
            if pixel_x+max_distance+1>hor_pixs:
                east_surr_pixels=image_array[pixel_y,pixel_x+1:hor_pixs]
                if pixel_x==hor_pixs-1:
                   east_surr_pixels=np.array([])
            else:
                east_surr_pixels=image_array[pixel_y,pixel_x+1:pixel_x+max_distance+1]
               
            #Check that max_len(x_surr_pixels)==max_distance

            #rotate the pixel orders from the arrays to have them all in order
            #from closest to furthest and reshape the others
            
            #Check if the shapes and order of the arrays is the correct one
            north_surr_pixels=north_surr_pixels[::-1]
            north_surr_pixels.shape=(np.shape(north_surr_pixels)[0])
            west_surr_pixels=west_surr_pixels[::-1]
            #south_surr_pixels.shape=(np.shape(south_surr_pixels)[0])
            
            #Check if the array presents the same value or two
            #if it presents the same value it is uniform in that direction and if it is uniform in
            #the four directions, then it is not an outlier
            
            north_diff_vals=len(np.unique(north_surr_pixels))
            west_diff_vals=len(np.unique(west_surr_pixels))
            south_diff_vals=len(np.unique(south_surr_pixels))
            east_diff_vals=len(np.unique(east_surr_pixels))
            
            north_diff_vals_real=np.unique(north_surr_pixels)
            west_diff_vals_real=np.unique(west_surr_pixels)
            south_diff_vals_real=np.unique(south_surr_pixels)
            east_diff_vals_real=np.unique(east_surr_pixels)
            
            
            # north_surr_pixels=np.reshape(north_surr_pixels, newshape=(np.shape(north_surr_pixels)[0],1))
            # west_surr_pixels=np.reshape(west_surr_pixels, newshape=(np.shape(west_surr_pixels)[0],1))
            # south_surr_pixels=np.reshape(south_surr_pixels, newshape=(np.shape(south_surr_pixels)[0],1))
            # east_surr_pixels=np.reshape(east_surr_pixels, newshape=(np.shape(east_surr_pixels)[0],1))

            
            #retrieve second unique value of one of the 4 arrays (if any)
            
            
            
            list_arrays=[north_surr_pixels,west_surr_pixels,south_surr_pixels,east_surr_pixels]
            
            list_2_elems=[]
            
            second_elems=[]
            
            for array in list_arrays:
                if len(np.unique(array))==2:
                    outer_element=np.unique(array)[1]
                    list_2_elems.append(array)
                    second_elems.append(outer_element)
                    
            
            # point difference between edge pixels in an outlier and the ones inside
            
            for array in list_arrays:
    
                
                if np.shape(array)[0]==0:
                    continue
                    
                if array[0] != target_pixel:
                    edge=1
                    break
                else:
                    edge=0
            
       
            if edge==1:
                
                #the pixel is an edge pixel, identify if it is an outlier or not
                
                #first differentiate which directions have the same or similar pixels next to the target one
                #array containing the directions in which the pixel next to target changes its value
                directions_change=[]
                #array containing the directions in which the pixel next to target does not change its value
                directions_same=[]
                
                #count number of empty arrays, maximum of 2, meaning that the target pixel is a pixel from an edge/corner of the image
                empty_arrays=0
                #first surrounding pixels of the arrays that change the values
                first_surr_pixels_change=[]
                
                for array in list_arrays:
              
                    if np.shape(array)[0]==0:
                        empty_arrays=empty_arrays+1
                        continue
                    
                    if array[0]!=target_pixel:
                        directions_change.append(array)
                        first_surr_pixels_change.append(array[0])
                    else:
                        directions_same.append(array)
                        
                #first_surr_pixel_real=np.array([north_diff_vals_real[0],west_diff_vals_real[0],south_diff_vals_real[0],east_diff_vals_real[0]])
                
                directions_change=np.asarray(directions_change)
                directions_same=np.asarray(directions_same)
                first_surr_pixels_change=np.asarray(first_surr_pixels_change)
            
                if len(directions_change)+empty_arrays==4:
                    #it is a single pixel outlier, convert into the most common number
                    
                    (values,counts) = np.unique(first_surr_pixels_change,return_counts=True)
                    
                    smoothed_image[pixel_y,pixel_x]=values[np.argmax(counts)]
                    changes=changes+1
  
               
                    
                else:
                    #it can be still an outlier but not a single pixel outlier
                    
                    #we need to evaluate the arrays that havent changed the pixel value, to see if they change it or not
                    
                    #check if the array that keeps the same value, changes the pixel value along the max distance or not
                    
                        
                    #to identify as an outlier, the arrays that do not change the value, have to change it and these be the same between them
                    #and also compared to the value that the other arrays have changed being close to the edge
                    second_values=[]
                    for array in directions_same:
                        if len(np.unique(array))==1:
                            break
                        else:
                            second_values.append(np.unique(array)[1])
                    
                    second_values=np.asarray(second_values)
                    
                    first_values_change=[]
                    for array in directions_change:
                        first_values_change.append(array[0])
                    
                    first_values_change=np.asarray(first_values_change)
                    
                    if len(directions_same)==len(second_values) and len(np.unique(second_values))==1 and len(np.unique(first_values_change))==1 and np.unique(second_values)[0]==directions_change[0][0]:
                        #meaning that all the directions that do not change the pixel in the immediately next one, change it within the max distance
                        #but also the change is to the same value, as if it was not the same value it could be an interface
                        #but also that the values to which the edge pixel has changed directly (immediate pixel) are all the same
                        #but also this same value has to be the same as the one that changed directly in the arrays that changed directly
                        #then change the value because it is pixel at the edge of an outlier regior or accumulated pixels
                        smoothed_image[pixel_y,pixel_x]=np.unique(second_values)[0]
                        changes=changes+1
     
                
            else:
                #the pixel is not an edge pixel but is surrounded with pixels with the same value, identify if it is an outlier or not then
                    
                if north_diff_vals==1 and west_diff_vals==1 and south_diff_vals==1 and east_diff_vals==1:
                    #the surrounding is uniform, then do not change the values
                    break
                else:
                    if len(np.unique(np.array(second_elems)))==1 and len(second_elems)==4:
                    
                        #all the elements in the surrounding are the same in the max distance radius, then it is an outlier
                        #change the value of the pixel of the image to the one in the surroundings
                        smoothed_image[pixel_y,pixel_x]=outer_element
                        #if it does not coincide in all the four directions, it can be an inerface or something like that, do not change it then
                        changes=changes+1
            # extra condiitons should be added to consider the pixels in the edges of the outliers, by comparing the value of the pixels 
            # and the ones in their surroundings



    return smoothed_image, changes




distances=[]
changes_values=[]

# for i in range(2,20):
    
#     print(i)
#     smoothed_image,changes_i=Region_outlier_identification(labels_ms_reshaped, max_distance=int(i))
#     distances.append(i)
#     changes_values.append(changes_i)
#     plt.imshow(smoothed_image,cmap=plt.cm.gray )
#     plt.show()

    
    
    
  
smoothed_image,changes_i=Region_outlier_identification(labels_ms_reshaped, max_distance=15)  
plt.imshow(smoothed_image,cmap=plt.cm.gray )
plt.show()


distances=np.asarray(distances)
changes_values=np.asarray(changes_values)

plt.plot(distances,changes_values)
plt.show()


distances=[]
changes_values=[]


# for i in range(2,200):
    
 
#     smoothed_image,changes_i=Region_outlier_identification(labels_ms_reshaped_f, max_distance=int(i))
#     distances.append(i)
#     changes_values.append(changes_i)


#smoothed_image,changes_i=Region_outlier_identification(labels_ms_reshaped_f, max_distance=110)



# plt.plot(distances,changes_values)
# plt.show()








def Consecutive_pixels_new(image_array_labels):


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
    
            
    target_pixel=image_array_labels[0,0] 
    
    #process of converting into the class of pixels and appending them into the cluster's array
    
    
        
    #the list contains all the pixels that are contiguous to the first one, each element is a pixels_state class
    
    
    target_pixel_classed=pixels_state((0,0), target_pixel, state='New')
    contiguous_cluster.append(target_pixel_classed) 
    
    
    #a list contains all the new pixels found in each iteration
    
    all_coordinates=[(0,0)]
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




print('variables')

print(labels_ms_reshaped)

# array_pixel_class=Consecutive_pixels_new(labels_ms_reshaped)

# print(labels_ms_reshaped.min(),labels_ms_reshaped.max())

# print('contiguous clusters')

# print(len(array_pixel_class))



# new_image=np.ones(np.shape(labels_ms_reshaped))


# for pixel in array_pixel_class:
#     (pixel_y1,pixel_x1)=pixel.coords
#     new_image[pixel_y1,pixel_x1]=pixel.value

    
# plt.imshow(new_image,cmap=plt.cm.gray )
# plt.show()

plt.imshow(labels_ms_reshaped,cmap=plt.cm.gray )
plt.show()


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


    
ordered_clusters=Multiple_Consecutive_Clustered_regions(labels_ms_reshaped, criteria='Num_Clusters', tolerance=0, n_clusters=4)

final_image=np.zeros(np.shape(labels_ms_reshaped))
value_image=1
for cluster in ordered_clusters:
    cluster_i=cluster.cluster
    for pixel in cluster_i:
        (pixel_y,pixel_x)=pixel.coords
        final_image[pixel_y,pixel_x]=value_image
        
    value_image=value_image+1
        
    
    print(cluster.lenght)
  
    
plt.imshow(final_image,cmap=plt.cm.gray )
plt.show()


ordered_clusters_1=Multiple_Consecutive_Clustered_regions(labels_ms_reshaped, criteria='Tolerance', tolerance=0.1, n_clusters=0)
print(len(ordered_clusters_1))


final_image_1=np.zeros(np.shape(labels_ms_reshaped))

value_image=1
for cluster in ordered_clusters_1:
    cluster_i=ordered_clusters_1[cluster]
    for pixel in cluster_i:
        (pixel_y,pixel_x)=pixel.coords
        final_image_1[pixel_y,pixel_x]=value_image
        
    value_image=value_image+1
        
    
    print(len(cluster_i))
    
plt.imshow(final_image_1,cmap=plt.cm.gray )
plt.show()
