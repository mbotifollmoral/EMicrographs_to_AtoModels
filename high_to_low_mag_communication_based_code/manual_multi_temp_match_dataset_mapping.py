# -*- coding: utf-8 -*-
"""
Created on Thu Aug  5 13:46:32 2021

@author: Marc
"""


import numpy as np
import os
import imutils
import cv2
import hyperspy.api as hs
import matplotlib.pyplot as plt
from PIL import Image

# !!! class modified in the functions module to account for Recalibration of the image and FFT
class image_in_dataset:
    def __init__(self, filename):
        hyperspy_2Dsignal=hs.load(filename)
        self.hyperspy_2Dsignal=hyperspy_2Dsignal
        image_arraynp=np.asarray(hyperspy_2Dsignal)
        self.image_arraynp_st=(image_arraynp-np.min(image_arraynp))/np.max(image_arraynp-np.min(image_arraynp))
        self.image_arraynp_st_int=np.uint8(255*self.image_arraynp_st)
        self.x_calibration=hyperspy_2Dsignal.axes_manager['x'].scale
        self.total_pixels=hyperspy_2Dsignal.axes_manager['x'].size
        self.units=hyperspy_2Dsignal.axes_manager['x'].units
        self.image_fulldirectory=filename
        name_end_index=-filename[::-1].find('.')
        name_start_index=-filename[::-1].find('\\')
        self.name=filename[name_start_index:name_end_index-1]
        
 # !!! function updated in the functions module to account for calibration correction before the pixel size gathering       
def Browse_Dataset_Images(dataset_system_path_name):
    
    images_in_dataset_list=[]
    pixel_sizes=[]
    
    for image_in_folder in os.listdir(dataset_system_path_name):
        image_name=dataset_system_path_name+'\\'+image_in_folder
        im_in_dataset=image_in_dataset(image_name)
        images_in_dataset_list.append(im_in_dataset)
        pixel_size=im_in_dataset.x_calibration
        pixel_sizes.append(pixel_size)
    pixel_sizes=np.sort(np.unique(np.asarray(pixel_sizes)))[::-1]
 
    return images_in_dataset_list, pixel_sizes
        

def Sort_Dataset_by_PixelSize(images_in_dataset_list, pixel_sizes):
    '''
    Create a list with lists of all the images with same 
    pixel size to be the same size as the previous array
    '''

    images_in_dataset_by_pixel_size=[]
    
    for i in range(len(pixel_sizes)):
        images_in_dataset_by_pixel_size.append([])

    for index, pixel_size in enumerate(pixel_sizes):
        for image in images_in_dataset_list:
            if image.x_calibration == pixel_size:
                images_in_dataset_by_pixel_size[index].append(image)
    
    return images_in_dataset_by_pixel_size
    



def Multiscale_Template_Matching(query_image_st, template_image_st, scale_factor):
    '''
    Manual multiscale template matching from image 
    Numpy arrays in the inputs, normalized in 255 
    The array must go through np.uint8(255*image) to change type
    
    '''

    # resize the image according to the scale, and keep track
    # of the ratio of the resizing
    #resized is the template not the query as in the example
    resized_tem = imutils.resize(template_image_st, width = int(template_image_st.shape[1] * scale_factor))
    
    (tH, tW) = resized_tem.shape[:2]
    
     # detect edges in the resized template
     # matching to find the template in the query image which is being progressively downscaled
    result = cv2.matchTemplate(query_image_st, resized_tem, cv2.TM_CCORR_NORMED)

    (_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)
    print('Coeff non edged: ', maxVal)
    # check to see if the iteration should be visualized

    # unpack the bookkeeping variable and compute the (x, y) coordinates
    # of the bounding box based on the resized ratio
    (startX, startY) = (int(maxLoc[0]), int(maxLoc[1]))
    (endX, endY) = (int((maxLoc[0] + tW)), int((maxLoc[1] + tH)))
    # draw a bounding box around the detected result and display the image
    image_to_show_rectangle = np.dstack([query_image_st, query_image_st, query_image_st])
    rectnonedg=cv2.rectangle(image_to_show_rectangle, (startX, startY), (endX, endY), (0, 0, 255), 2)
    plt.imshow(image_to_show_rectangle)
    plt.imshow(rectnonedg)
    plt.show()   
    
    #How the edges look like in the query image in the found area

    cropimagequerynonedge=query_image_st[startY:endY, startX:endX]
    plt.imshow(cropimagequerynonedge)
    plt.show()  

    #!!! Check carefully the way to extract the coordinates (as they are relative to the
    # query image when doing the template matching process)
    #The coordinates are extracted as the pixel numbers of the query image with the format
    # being coordinates=(startX,startY, endX, endY)   
    coordinates_template=(startX,startY, endX, endY)
    
    return coordinates_template, maxVal



def Get_Ideal_Scale(hyperspy_2Dsignal_query_image, hyperspy_2Dsignal_template_image):
    '''
    Hyperspy signals in the inputs
    '''
     
    #QUERY IMAGE (image at low mag, image in which the other image (template) has to be fitted)
    x_calibration_query=hyperspy_2Dsignal_query_image.axes_manager['x'].scale
    #TEMPLATE IMAGE (image at low mag, image in which the other image (template) has to be fitted)
    x_calibration_tem=hyperspy_2Dsignal_template_image.axes_manager['x'].scale
    
    ideal_scale=(x_calibration_tem)/(x_calibration_query)

    return ideal_scale
    

#functions and classes for the coordinate system and mapping generation


class query_template_match:
    '''
    This class is quite inefficient in terms of memory, as stores two images already stored in the
    core list of images. Maybe we can just reference back to this list instead of restacking 
    the full information into the class
    '''
    def __init__(self, query,template, coeff, coords):
        #keep in mind that the lowest mag does not have query and 
        #highest mag does not have template!

        #both are image_in_dataset objects
        self.query=query
        self.template=template
        self.coeff=coeff
        #in pixels within the query, as defined in the multiscale template matching function
        self.coordinates=coords
        


def Make_Sure_1_Lowest_Mag_Image(images_in_dataset_by_pixel_size):
    '''
    Ensures that regarding the lowest magnification avaiable (highest pixel size)
    there is only a single image of it

    Parameters
    ----------
    images_in_dataset_by_pixel_size : TYPE
        DESCRIPTION.

    Returns
    -------
    images_in_dataset_by_pixel_size : TYPE
        DESCRIPTION.

    '''
    
    if len(images_in_dataset_by_pixel_size[0]) > 1:
        images_in_dataset_by_pixel_size[0]=[images_in_dataset_by_pixel_size[0][0]]
  
    return images_in_dataset_by_pixel_size



def Find_best_Query_Template_pairs(images_in_dataset_by_pixel_size):
    #Be sure to check if there is only one single loweset mag image
    
    #keep in mind that the lowest mag does not have query and highest mag does not have template!
  
    flat_images_in_dataset_by_pixel_size=[item for sublist in images_in_dataset_by_pixel_size for item in sublist][::-1]
    
    list_of_best_query_template_pairs=[]
 
    #loop through all possible templates (all images except the lowest mag one)
    for image_as_template in flat_images_in_dataset_by_pixel_size[:len(flat_images_in_dataset_by_pixel_size)-1]:
        flat_images_in_dataset_by_pixel_size_copy=flat_images_in_dataset_by_pixel_size.copy()
        index_del=flat_images_in_dataset_by_pixel_size_copy.index(image_as_template)
        flat_images_in_dataset_by_pixel_size_copy=flat_images_in_dataset_by_pixel_size_copy[index_del+1:]
        
        query_template_possible_pairs=[]
        query_template_possible_pairs_coeffs=[]
        
        for possible_query in flat_images_in_dataset_by_pixel_size_copy:
            
            scale_factor=Get_Ideal_Scale(possible_query.hyperspy_2Dsignal, image_as_template.hyperspy_2Dsignal)
            # if the image is very high mag, just allow very high mag querys to be the possible
            # candidates as only those with atomic reoslution present the features that should match
            # pixel size bigger than 0.08 does not present atomic resolution (titan, 2k detector...)
            if image_as_template.x_calibration <= 0.01 and possible_query.x_calibration <= 0.08:

                coordinates_template, maxVal=Multiscale_Template_Matching(possible_query.image_arraynp_st_int, image_as_template.image_arraynp_st_int, scale_factor)
            
                query_template_possible_pair=query_template_match(possible_query,image_as_template, maxVal, coordinates_template)
                query_template_possible_pairs.append(query_template_possible_pair)
                query_template_possible_pairs_coeffs.append(maxVal)
                    

            # Threhsold the scaling to only scalings that are more than 0.01 to avoid false positives
            # due to the a few pixels template matching situation (prompt to give always very high coefs)
            #!!! The scale factor threhold, hyperparameter, should be ok being 0.01 (i.e. allowing
            # scalings less stressful than this) but the closer the images are between each other 
            # the better (matches with high scale factors, meaning a more linear correlation of the iamges)
            if scale_factor >= 0.02 and image_as_template.x_calibration > 0.01:
            
                coordinates_template, maxVal=Multiscale_Template_Matching(possible_query.image_arraynp_st_int, image_as_template.image_arraynp_st_int, scale_factor)
                
                query_template_possible_pair=query_template_match(possible_query,image_as_template, maxVal, coordinates_template)
                query_template_possible_pairs.append(query_template_possible_pair)
                query_template_possible_pairs_coeffs.append(maxVal)
            
            
        best_query_template_match=query_template_possible_pairs[np.argmax(np.asarray(query_template_possible_pairs_coeffs))]
        list_of_best_query_template_pairs.append(best_query_template_match)
            

    return list_of_best_query_template_pairs




def Find_best_Query_Template_pairs_Restrictive(images_in_dataset_by_pixel_size, N_ps_higher):
    '''
    Only allow to take as querys N_ps_higher steps up in pixel size. Then, the template image pixel size
    checks only N pixel sizes higher. This limits the templates from being placed in a too low image
    This, however, can be detrimental in some scenarios where a certain patch of the device
    is not caught by high magnification images but only in low ones. N_ps_higher should be 
    typically 3 or 4

    Parameters
    ----------
    images_in_dataset_by_pixel_size : TYPE
        DESCRIPTION.

    Returns
    -------
    list_of_best_query_template_pairs : TYPE
        DESCRIPTION.

    '''
    
    #keep in mind that the lowest mag does not have query and highest mag does not have template!
  
    flat_images_in_dataset_by_pixel_size=[item for sublist in images_in_dataset_by_pixel_size for item in sublist][::-1]
    
    list_of_best_query_template_pairs=[]
 
    #loop through all possible templates (all images except the lowest mag one)
    for image_as_template in flat_images_in_dataset_by_pixel_size[:len(flat_images_in_dataset_by_pixel_size)-1]:
        flat_images_in_dataset_by_pixel_size_copy=flat_images_in_dataset_by_pixel_size.copy()
        index_del=flat_images_in_dataset_by_pixel_size_copy.index(image_as_template)
        flat_images_in_dataset_by_pixel_size_copy=flat_images_in_dataset_by_pixel_size_copy[index_del+1:]
        
        #limit by the N possible pixels sizes higher than those of image_as_template
        pixel_sizes=np.unique(np.asarray([elem.x_calibration for elem in flat_images_in_dataset_by_pixel_size]))
        pixelsize_template_index=np.argmin(np.abs(pixel_sizes-image_as_template.x_calibration))
        
        if pixelsize_template_index+N_ps_higher<=len(pixel_sizes)-1:
            pixel_low_bound=pixel_sizes[pixelsize_template_index+1]
            pixel_high_bound=pixel_sizes[pixelsize_template_index+N_ps_higher]
        else:
            pixel_low_bound=pixel_sizes[pixelsize_template_index+1]
            pixel_high_bound=pixel_sizes[len(pixel_sizes)-1]
        
        
        query_template_possible_pairs=[]
        query_template_possible_pairs_coeffs=[]
        
        for possible_query in flat_images_in_dataset_by_pixel_size_copy:
            
            if possible_query.x_calibration>=pixel_low_bound and possible_query.x_calibration <=pixel_high_bound:
                
            
                scale_factor=Get_Ideal_Scale(possible_query.hyperspy_2Dsignal, image_as_template.hyperspy_2Dsignal)
                if scale_factor >= 0.02:
                    coordinates_template, maxVal=Multiscale_Template_Matching(possible_query.image_arraynp_st_int, image_as_template.image_arraynp_st_int, scale_factor)
                
                    query_template_possible_pair=query_template_match(possible_query,image_as_template, maxVal, coordinates_template)
                    query_template_possible_pairs.append(query_template_possible_pair)
                    query_template_possible_pairs_coeffs.append(maxVal)
                    
        best_query_template_match=query_template_possible_pairs[np.argmax(np.asarray(query_template_possible_pairs_coeffs))]
        list_of_best_query_template_pairs.append(best_query_template_match)
            

    return list_of_best_query_template_pairs


def Coordinate_system_initializer_1_low_mag_image(lowest_mag_image):
    '''
    Use this function in case we only have ONE single image in our dataset in the lowest
    available magnification: thus in the list of lists of images per pixel size, the first one
    corresponding to the lowest magnification just has one image, then use this image to
    define the coordinate system
    
    This function can be used if TWO OR MORE lowest mag images are present, thus, if the first
    list of the lowest mag images (in the list of lists of sorted pixel sizes) has more than
    one element, BUT you manually select which image you want, therefore discarding the others
    and assuming the templates or higher mag images will be all embeedded in this single image
    and removing the others does not make us lose information
    For instance, the image input being image=list_of_lists[0] if image.name="name of choice"

    input is an image_in_datset object referring to the lowest mag image
    -------
    coordinate_system : TYPE
        DESCRIPTION.

    '''
    
    image_pixel_size=lowest_mag_image.x_calibration
    image_total_pixels=lowest_mag_image.total_pixels
    image_fov=image_pixel_size*image_total_pixels
    
    #the coordinate system is initialized in the form of (startX, startY, endX, endY)
    #but this initialisation is in units of nm, not in pixels endX=endY=image_fov
    coordinate_system=(0,0,image_fov,image_fov)
    
    return coordinate_system


def Coordinate_system_initializer_2ormore_low_mag_image(lowest_mag_image):
    '''
    The function when 2 or more images of the lowest mag are available is extremely harder
    both to code and get the coordinate systems and then to address the coordinates of the
    templates and querys of higher mags
    
    The way to find the meshed coordinate systems is to perform template matching of the 
    images between each other (of course without rescaling as they are all of the lowest mag)
    but with successive croppings of each of them, to find where they overlap or where they do not
    and compare the results of the croppings and overlappings.
    When the cropped area of one image to another matches well, just mix these images with the
    actual restored image and there it is the global system, which is now definied in funcation of 
    n images and their respective starting and ending position within the global coordiante system
    for latter comparison of the higher mag querys and templates
    
    Given than this process is much more exceptional than the previous one and only occurs 
    occasionally, we will leave the programming of this function for further steps of refinement
    given the extra complexity of the process (and given the fact that even if there are 2 or more
    low mag images, one can just be manually selected by deleting the others in the anaysed dataset)
    

    '''
    coordinate_system=None
    
    return coordinate_system


def Coordinate_system_Calculation(images_in_dataset_by_pixel_size,list_of_best_query_template_pairs):
    
    flat_images_in_dataset_by_pixel_size=[item for sublist in images_in_dataset_by_pixel_size for item in sublist]
    
    #the coordinate system is initialized in the form of (startX, startY, endX, endY)
    #no need now as it starts from 0 and the distance is the distance+0=distance
    #just initialise to add them in the relative position list
    
    relative_positions=[]  
    
    coordinate_system=Coordinate_system_initializer_1_low_mag_image(flat_images_in_dataset_by_pixel_size[0])
    relative_positions.append(coordinate_system)
    
    # loop from the lowest to the highest magnifications
    # !!! This assumes just ONE SINGLE LOWEST MAG IMAGE
    for image_in_dataset in flat_images_in_dataset_by_pixel_size[1:]:
        #for computing the lenght of the image
        fov_image_in_dataset=image_in_dataset.x_calibration*image_in_dataset.total_pixels
        
        template_to_locate=image_in_dataset
        
        x_distances=[]
        y_distances=[]
        
        while template_to_locate != flat_images_in_dataset_by_pixel_size[0]:
            
            for template_query_pair in list_of_best_query_template_pairs:
            
                if template_query_pair.template==template_to_locate:
                    # get the calibration of the query and the coordinates of the template
                    # with respecte the query, just the startX and startY
                    calib_query=template_query_pair.query.x_calibration
                    #the next template is the query of the previous template to locate
                    next_template=template_query_pair.query
                    (startX, startY, _, _)=template_query_pair.coordinates #important, in pixels
                    x_dist=calib_query*startX
                    y_dist=calib_query*startY
                    x_distances.append(x_dist)
                    y_distances.append(y_dist)
                    break
                    
            template_to_locate=next_template
       
        
       
        x_relative_position_start=sum(x_distances)
        y_relative_position_start=sum(y_distances)
        
        x_relative_position_end=x_relative_position_start+fov_image_in_dataset
        y_relative_position_end=y_relative_position_start+fov_image_in_dataset
       
        relative_position=(x_relative_position_start, y_relative_position_start, x_relative_position_end, y_relative_position_end)
       
        relative_positions.append(relative_position)

    
    return relative_positions


#functions to output the results of the images on top of each other

def Plot_Images_Stack_Scatter(images_in_dataset_by_pixel_size,relative_positions):
    # creates a scatter plot with squres instead of points where the images should be placed
    
    flat_images_in_dataset_by_pixel_size=[item for sublist in images_in_dataset_by_pixel_size for item in sublist]
    
    range_total=relative_positions[0][2]-relative_positions[0][0]
    step=flat_images_in_dataset_by_pixel_size[len(flat_images_in_dataset_by_pixel_size)-1].x_calibration
    
    coordinates_x=[]
    coordinates_y=[]
    sizes=[]
   
    #correct the positions to print the top left vertex in the coordinate instead of the center
    for relative_position in relative_positions[1:]:
        coordinates_x.append(relative_position[0]+(relative_position[2]-relative_position[0])/2)
        coordinates_y.append(range_total-relative_position[3]+(relative_position[2]-relative_position[0])/2)
        sizes.append(relative_position[2]-relative_position[0])
    
    coordinates_x=np.asarray(coordinates_x)
    coordinates_y=np.asarray(coordinates_y)
    sizes=np.asarray(sizes)
    
    c = np.linspace(0,len(sizes),len(sizes))

    plt.figure(figsize=(6,6))
    plt.xlim([0, range_total])
    plt.ylim([0, range_total])
    plt.scatter(coordinates_x,coordinates_y, s=sizes*1.5, c=c, marker='s', cmap='summer')
    plt.show()
    # multiply sizes by 1.5 if fig size is 6,6 gives a more or less accurate idea of image sizes
    return



def Plot_Images_Stack_Scaling(images_in_dataset_by_pixel_size,relative_positions):
    '''
    Creates a super high res low mag image and inserts the other images in the positions
    where the template matching stated they fit the best, with their corresponding rescaling.
    The rescaling and super upsampling has to be done very limitedly as doing it raw makes
    the memory to explode. e.g. 16TB required to allocate the upscaling of a single very low 
    mag image with a very high mag pixel size

    Parameters
    ----------
    images_in_dataset_by_pixel_size : TYPE
        DESCRIPTION.
    relative_positions : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    
    flat_images_in_dataset_by_pixel_size=[item for sublist in images_in_dataset_by_pixel_size for item in sublist]
    
    #Delete the so big lowest mag one to try
    # to limit the images in the sq20 device to be allocated in memory, we do not consider the first 2 images
    flat_images_in_dataset_by_pixel_size=flat_images_in_dataset_by_pixel_size[2:]
    relative_positions_copy=relative_positions.copy()[2:]
    
    # flat_images_in_dataset_by_pixel_size=flat_images_in_dataset_by_pixel_size
    # relative_positions_copy=relative_positions.copy()
    
    new_xref=relative_positions_copy[0][0]
    new_yref=relative_positions_copy[0][1]
    
    coordinates_mod=np.zeros((len(relative_positions_copy),4))
    for coordinate,relative_position in zip(coordinates_mod,relative_positions_copy):
        coordinate[0]=relative_position[0]-new_xref
        coordinate[1]=relative_position[1]-new_yref
        coordinate[2]=relative_position[2]-new_xref
        coordinate[3]=relative_position[3]-new_yref
        
    range_total=coordinates_mod[0][2]-coordinates_mod[0][0]
    step=flat_images_in_dataset_by_pixel_size[len(flat_images_in_dataset_by_pixel_size)-1].x_calibration
   
    lowmag_total_upsampled_pixels=int(np.ceil(range_total/step))
    
    grid_side = np.linspace(0,range_total,lowmag_total_upsampled_pixels)
    
    #downscale images to fit in memory
    fixed_downscale_factor=10
    dimside=int(np.ceil(len(grid_side)/fixed_downscale_factor))
    dim = (dimside, dimside)
    
    # resize image
    resized_global = cv2.resize(flat_images_in_dataset_by_pixel_size[0].image_arraynp_st_int, dim, interpolation = cv2.INTER_LINEAR)
    plt.figure(figsize=(100,100))
    plt.imshow(resized_global)
    plt.show()
    
    # loop through all the images and scale them up to fit the pixels numbers and position given by
    # relative positions array
    
    for coordinate,image_i in zip(coordinates_mod[1:],flat_images_in_dataset_by_pixel_size[1:]):
        step_i=step*fixed_downscale_factor
        range_total=coordinate[2]-coordinate[0]
        
        total_pixels_im=int(np.ceil(range_total/step_i))
        dim_i= (total_pixels_im, total_pixels_im)

        resized_im = cv2.resize(image_i.image_arraynp_st_int, dim_i, interpolation = cv2.INTER_LINEAR)
        
        startX_p=int(np.ceil(coordinate[0]/step_i))
        startY_p=int(np.ceil(coordinate[1]/step_i))
        endX_p=startX_p+total_pixels_im
        endY_p=startY_p+total_pixels_im
        
        
        resized_global[startY_p:endY_p,startX_p:endX_p]=resized_im
    
    plt.figure(figsize=(100,100))
    plt.imshow(resized_global)
    plt.show()
   
    return


#Script start

dataset_system_path_name=r'E:\Arxius varis\PhD\2nd_year\Code\trial_images\full_device_STEM_datasets\SQ20-250-2\\'
images_in_dataset_list, pixel_sizes=Browse_Dataset_Images(dataset_system_path_name)

images_in_dataset_by_pixel_size=Sort_Dataset_by_PixelSize(images_in_dataset_list, pixel_sizes)

#list_of_best_query_template_pairs=Find_best_Query_Template_pairs(images_in_dataset_by_pixel_size)

images_in_dataset_by_pixel_size=Make_Sure_1_Lowest_Mag_Image(images_in_dataset_by_pixel_size)

N_ps_higher=3
list_of_best_query_template_pairs=Find_best_Query_Template_pairs_Restrictive(images_in_dataset_by_pixel_size, N_ps_higher)

#list_of_best_query_template_pairs=Find_best_Query_Template_pairs(images_in_dataset_by_pixel_size)

relative_positions=Coordinate_system_Calculation(images_in_dataset_by_pixel_size,list_of_best_query_template_pairs)

print(relative_positions)
    

#plot the images one on top of each other

   
Plot_Images_Stack_Scaling(images_in_dataset_by_pixel_size,relative_positions)



