# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 18:19:03 2020

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


blur=cv2.GaussianBlur(image, (5,5), 1)
image=blur
#Image preprocessing, standarisation
#Range from 0 to 1 the intensity values
image_st=(image-np.min(image))/np.max(image)

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

scaling_factor=8

image_st=skimage.measure.block_reduce(blur, block_size=tuple(np.int32(scaling_factor*np.ones(len(np.shape(blur))))), func=np.mean, cval=0)

plt.imshow(blur,cmap=plt.cm.gray )
plt.show()
labels_ms,clusters_ms=Mean_shift(image_st)
print(labels_ms, len(labels_ms))
print(clusters_ms, len(clusters_ms))


labels_ms_reshaped=np.reshape(labels_ms, np.shape(image_st))
img_segm = np.choose(labels_ms, clusters_ms)

plt.imshow(labels_ms_reshaped,cmap=plt.cm.gray )
plt.show()

labels_ms_reshaped_f=scipy.ndimage.zoom(labels_ms_reshaped, scaling_factor, order=1)
plt.imshow(labels_ms_reshaped_f,cmap=plt.cm.gray )
plt.show()


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


array_pixel_class=Consecutive_pixels_new(labels_ms_reshaped)

print(labels_ms_reshaped.min(),labels_ms_reshaped.max())

print('contiguous clusters')

print(len(array_pixel_class))



new_image=np.ones(np.shape(labels_ms_reshaped))


for pixel in array_pixel_class:
    (pixel_y1,pixel_x1)=pixel.coords
    new_image[pixel_y1,pixel_x1]=pixel.value

    
plt.imshow(new_image,cmap=plt.cm.gray)
plt.show()



def Contour_drawing(image_array_labels):
    
    
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
    
    initial_pixel_coords=(0,0)
            
    target_pixel=image_array_labels[initial_pixel_coords[0],initial_pixel_coords[1]] 
    
    
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
    
    for vertex in vertex_array:
        
        #check if vertex is contained in an edge or corner to tailor the different checkings
        
        if vertex[0]==0 and vertex[1]==0:
            #top left corner
            #only check south
            next_vertex_coords, m_out=check_South(padded_image,target_pixel,vertex)
            
            vertex_array.append(next_vertex_coords)
            
            #generate new contour vector and append it to the contour vectors list
            new_contour=contour_vector(target_pixel,m_out,vertex,next_vertex_coords)
            contour_vectors_list.append(new_contour)
            
        elif vertex[0]==0 and vertex[1]==hor_pixs:
            #top right corner
            #only check west
            next_vertex_coords, m_out=check_West(padded_image,target_pixel,vertex)
            
            vertex_array.append(next_vertex_coords)
            
            #generate new contour vector and append it to the contour vectors list
            new_contour=contour_vector(target_pixel,m_out,vertex,next_vertex_coords)
            contour_vectors_list.append(new_contour)
        
        elif vertex[0]==ver_pixs and vertex[1]==0:
            #bottom left corner
            #only check east
            next_vertex_coords, m_out=check_East(padded_image,target_pixel,vertex)
            
            vertex_array.append(next_vertex_coords)
            
            #generate new contour vector and append it to the contour vectors list
            new_contour=contour_vector(target_pixel,m_out,vertex,next_vertex_coords)
            contour_vectors_list.append(new_contour)
            
        elif vertex[0]==ver_pixs and vertex[1]==hor_pixs:
            #bottom right corner
            #only check north
            
            next_vertex_coords, m_out=check_North(padded_image,target_pixel,vertex)
            
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
            
            for possible_vertex, m_out in zip(possible_vertexs,m_outs):
                if possible_vertex != None:
                    #this should also handle the issue of one vertex diverging into two different contours, as this would
                    #append all the new possible vertices and make them new starting points for considering new vertices
                    #!!! nevertheless, repetition should be checked because on of the contours is going to led to the same vertexm
                    #and therefore it would be repeated if it is not considered
                    vertex_array.append(possible_vertex)
                    
                    #generate new contour vector and append it to the contour vectors list
                    new_contour=contour_vector(target_pixel,m_out,vertex,possible_vertex)
                    contour_vectors_list.append(new_contour)
      

    
    return contour_vectors_list, vertex_array


    
    
    
def Contour_drawing_final(image_array_labels):
    
    
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
    
    initial_pixel_coords=(0,0)
            
    target_pixel=image_array_labels[initial_pixel_coords[0],initial_pixel_coords[1]] 
    
    
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





contour_vectors_list, vertex_array= Contour_drawing_final(new_image)
print(len(contour_vectors_list))
print(len(vertex_array))

contours_pixels=np.zeros((np.shape(new_image)[0]+1,np.shape(new_image)[0]+1))
for coord in vertex_array:
    contours_pixels[coord[0],coord[1]]=1

    
plt.imshow(contours_pixels,cmap=plt.cm.gray)
plt.show()

ys=max([ coord[0] for coord in vertex_array ])-np.asarray([ coord[0] for coord in vertex_array ])
xs=np.asarray([ coord[1] for coord in vertex_array ])


plt.scatter(xs,ys)
plt.show()


contour_vectors_list1, vertex_array1=Contour_drawing_initial_pixel(new_image, (34,4))
print(len(contour_vectors_list1))
print(len(vertex_array1))



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


def Contour_draw_computing_initial_pixel(image_array_labels, label_of_region):
    
    initial_pixel=Find_Initial_Pixel_for_contour(image_array_labels, label_of_region)
    contour_vectors_list, vertex_array=Contour_drawing_initial_pixel(image_array_labels, initial_pixel)
    
    return contour_vectors_list, vertex_array

    
    
contour_vectors_list, vertex_array= Contour_draw_computing_initial_pixel(new_image, label_of_region=0)
print(len(contour_vectors_list))
print(len(vertex_array))

contours_pixels=np.zeros((np.shape(new_image)[0]+1,np.shape(new_image)[0]+1))
for coord in vertex_array:
    contours_pixels[coord[0],coord[1]]=1

    
plt.imshow(contours_pixels,cmap=plt.cm.gray)
plt.show()


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



conts_vertx_per_region=Contour_draw_All_regions(new_image)






'''
UPDATED FUNCTION, DEBUGED TO WORK FINE IN ISOLATED PIXELS CLOSE TO THE EDGES OF THE IMAGE
USE THE ONE BELOW, THE FULL ONE

'''
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
                if vertex not in list_double_vertexs:
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
    if 0 in label_values:     
        label_values=np.sort(label_values)[1:]
    
    for label in label_values:
       
        contour_vectors_list, vertex_array=Contour_draw_computing_initial_pixel(image_array_labels, label)
        conts_vertx_per_region[str(int(label))+'_'+'vertexs']=vertex_array
        conts_vertx_per_region[str(int(label))+'_'+'contours']=contour_vectors_list

  
    return  conts_vertx_per_region








