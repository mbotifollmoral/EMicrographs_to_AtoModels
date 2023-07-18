# -*- coding: utf-8 -*-
"""
Created on Thu Feb 17 17:27:45 2022

@author: Marc
"""


import numpy as np
import matplotlib.pyplot as plt
import cv2
import skimage.measure
from numpy import fft
import ctypes
from matplotlib.patches import Circle, Rectangle
import sys
import os
from pathlib import Path

# !!! NEED to set the path to 
# Alg_Comb_Single_Image_Strain.py
# as the console working directory
Project_main_path = os.getcwd()
if 'EMicrographs_to_AtoModels' in Project_main_path:
    Project_main_path = Project_main_path[:Project_main_path.find('EMicrographs_to_AtoModels')-1]
# Project_main_path has the EMicrographs_to_AtoModels folder
sys.path.append(Project_main_path)

from EMicrographs_to_AtoModels.Functions.General_functions import Segmentation_1stAprox as Segment
from EMicrographs_to_AtoModels.Functions.General_functions import ImageCalibTransf as ImCalTrans




# GPA_wrapper

plt.rcParams["figure.figsize"] = (12,12)

def FT(img):
    return fft.ifftshift(fft.fft2(fft.fftshift(img)))

def IFT(img):
    return fft.fftshift(fft.ifft2(fft.ifftshift(img)))


GPA_base_dll_path = Project_main_path + '\\EMicrographs_to_AtoModels\Functions\General_functions\GPA_dll'
# os.chdir(GPA_base_dll_path)

lib = ctypes.CDLL(GPA_base_dll_path + '\\' + 'GPA.dll')

Handle = ctypes.POINTER(ctypes.c_char)
c_float_array = np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags='C_CONTIGUOUS')
c_int_array = np.ctypeslib.ndpointer(dtype=int, ndim=1, flags='C_CONTIGUOUS')

lib.createGPA.argtypes = None
lib.createGPA.restype = Handle

lib.deleteGPA.argtypes = [Handle]
lib.deleteGPA.restype = None

lib.load_img.argtypes = [Handle, c_float_array, c_int_array, ctypes.c_float]
lib.load_img.restypes = None

lib.mark_spot1.argtypes = [Handle,c_int_array, ctypes.c_int, c_float_array]
lib.mark_spot1.restypes = None

lib.mark_spot2.argtypes = [Handle,c_int_array, ctypes.c_int ,c_float_array]
lib.mark_spot2.restypes = None

lib.select_ref_area.argtypes = [Handle,c_int_array, c_int_array]
lib.select_ref_area.restypes = None

lib.calc_GPA.argtypes = [Handle]
lib.calc_GPA.restypes = None

lib.apply_rotation.argtypes = [Handle, ctypes.c_float] 
lib.apply_rotation.restypes = None

lib.get.argtypes = [Handle,c_float_array,c_float_array,c_float_array,c_float_array,c_float_array,c_float_array]
lib.get.restypes = None

class GPA:    
    def __init__(self):
        self.instance = lib.createGPA()
        self.ref_load = False
        self.Sp1_marked = False
        self.Sp2_marked = False
        self.get_ready = False

    def __del__(self):
        lib.deleteGPA(self.instance)   
        
    def load_image(self,img, calibration):
        #image = np.empty(self.Npix2, dtype=np.double)
        self.dim = img.shape
        size = np.asarray(self.dim,dtype = int)
        image = np.asarray(img.flatten(), dtype = np.float32)
        lib.load_img(self.instance, image, size, calibration*size[0])
        self.ref_load = True
    
    def mark_spot1(self, coordinates, win_size):
        if(self.ref_load != True):
            print("Load an image first")
            return np.empty((2,2), dtype=np.float32)
        amp = np.empty(self.dim[0]*self.dim[1], dtype=np.float32)
        coo = np.asarray(coordinates,dtype = int)
        lib.mark_spot1(self.instance,coo,win_size,np.asarray(amp, dtype=np.float32))
        self.Sp1_marked = True
        return amp.reshape(self.dim[0],self.dim[1])
    
    def mark_spot2(self, coordinates, win_size):
        if(self.ref_load != True):
            print("Load an image first")
            return np.empty((2,2), dtype=np.float32)
        amp = np.empty(self.dim[0]*self.dim[1], dtype=np.float32)
        coo = np.asarray(coordinates,dtype = int)
        lib.mark_spot2(self.instance,coo,win_size,np.asarray(amp, dtype=np.float32))
        self.Sp2_marked = True
        return amp.reshape(self.dim[0],self.dim[1])
    
    def select_ref_area(self, TL, BR):
        if(self.ref_load != True):
            print("Load an image first")
        tl = np.asarray(TL,dtype = int)    
        br = np.asarray(BR,dtype = int)
        lib.select_ref_area(self.instance,tl,br)
        
    def calc_GPA(self):
        if(self.ref_load and self.Sp1_marked and self.Sp2_marked == True):
            lib.calc_GPA(self.instance)
            self.get_ready = True
        else:
            print("Mark two diffraction spots first!")
            
    def apply_rotation(self, angle):
        lib.apply_rotation(self.instance, angle)
        
    def get(self):
        if(self.get_ready != True):
            x = np.empty((2,2), dtype=np.float32)
            print("run a calculation first!!")
            return x,x,x,x,x,x
        dxx = np.empty(self.dim[0]*self.dim[1], dtype=np.float32)
        dyy = np.empty(self.dim[0]*self.dim[1], dtype=np.float32)
        dxy = np.empty(self.dim[0]*self.dim[1], dtype=np.float32)
        dyx = np.empty(self.dim[0]*self.dim[1], dtype=np.float32)
        rot = np.empty(self.dim[0]*self.dim[1], dtype=np.float32)
        shear = np.empty(self.dim[0]*self.dim[1], dtype=np.float32)
        lib.get(self.instance, dxx, dyy, dxy, dyx, rot, shear)
        dxx = dxx.reshape(self.dim[0],self.dim[1])
        dyy = dyy.reshape(self.dim[0],self.dim[1])
        dxy = dxy.reshape(self.dim[0],self.dim[1])
        dyx = dyx.reshape(self.dim[0],self.dim[1])
        rot = rot.reshape(self.dim[0],self.dim[1])
        shear = shear.reshape(self.dim[0],self.dim[1])
        return dxx, dyy, dxy, dyx, rot, shear
 
    
#Adjust reference to the calculated images and phases function??    

    
def GPA_full(image_array,calibration,spot1, spot2, mask_size, reference, rotation_angle,display=True):
    '''
    Full GPA process, to call the full wrapped dll at once with all the necessary inputs.
    spot1 and spot2 must be tuples or lists with two elements, first the y coord (row) and then 
    x coord (col), in pixels, referring to the IMAGES reference system and not cartesian
    Mask size must be in pixels as well
    reference, coords in pixels, of the top-left corner and bottom-right corner of the rectangle/square
    referring to the reference, in format: row_i (yi) , col_i(xi), row_f (yf) , col_f (xf)
    Display =True to show the images of the process

    Parameters
    ----------
    image_array : 
    calibration : 
    spot1 : 
    spot2 : 
    mask_size : 
    reference:
    rotation_angle : angle to the x axis to the first g vector or spot, in degrees

    Returns
    -------
    dxx, dyy, dxy, dyx, rot, shear

    '''
    #image system
    # row1,col1=spot1
    # row2,col2=spot2
    
    #cartesian system
    Sp1Y,Sp1X = spot1
    Sp2Y,Sp2X = spot2
    
    row_i , row_f, col_i , col_f = reference
    
    Analysis = GPA()
    
    Analysis.load_image(image_array,calibration)
    
    #---
    # amp1 = Analysis.mark_spot1((Sp1X,Sp1Y),mask_size)
    
    # amp2 = Analysis.mark_spot2((Sp2X,Sp2Y),mask_size)
    
    # Analysis.calc_GPA()
    
    # Analysis.apply_rotation(rotation_angle)
        
    # exx, eyy, exy, eyx, rot, shear = Analysis.get()
    # #---
    
    
    # reference, watch out about the order of the parameters as the function demands top left and 
    # bottom right but in col (x), and row (y) order, and no need to add 1 
    Analysis.select_ref_area((col_i,row_i),(col_f,row_f))
    
    amp1 = Analysis.mark_spot1((Sp1X,Sp1Y),mask_size)
    
    amp2 = Analysis.mark_spot2((Sp2X,Sp2Y),mask_size)
    
    Analysis.calc_GPA()
    
    Analysis.apply_rotation(rotation_angle)
        
    exx, eyy, exy, eyx, rot, shear = Analysis.get()
    
    #Adjust reference to the calculated images and phases function?? 

    if display==True:
        #show the spots positions in the FFT and the strain components
        
        #FFT and chosen spot positions
        fft_img = np.log(abs(FT(image_array)))
        fig,ax = plt.subplots(1,constrained_layout=True)
        fig.set_size_inches(12, 12)
        fig.suptitle('FFT and chosen g vectors',fontsize=18)
        ax.imshow(fft_img,interpolation='nearest', cmap='gray')
        ax.scatter(Sp1X, Sp1Y, c="red", marker="x")
        circle = Circle((Sp1X, Sp1Y), mask_size, facecolor='none',edgecolor="red", linewidth=1, alpha=1)
        ax.add_patch(circle)
        ax.scatter(Sp2X, Sp2Y, c="blue", marker="x")
        circle = Circle((Sp2X, Sp2Y), mask_size, facecolor='none',edgecolor="blue", linewidth=1, alpha=1)
        ax.add_patch(circle)
        
        #Calculated phase
        fig,ax=plt.subplots(1,2, constrained_layout=True)
        fig.set_size_inches(12, 6)
        fig.suptitle('Spot intensity maps',fontsize=18)
        ax[0].imshow(amp1,interpolation='nearest', cmap='gray')
        ax[0].set_title('Spot 1 (g1)')
        ax[1].imshow(amp2,interpolation='nearest', cmap='gray')
        ax[1].set_title('Spot 2 (g2)')
        
        #Calculated components

        c = 2
        clims_fix=(-0.05,0.05)
        clims_fix_rot=(-5,5)
        cm = 'jet'
        colorbarshrink=1
        fig,ax=plt.subplots(3,2, constrained_layout=True)
        fig.set_size_inches(16, 18)
        fig.suptitle('Calculated components', fontsize=18)
        M = exx[row_i:row_f+1,col_i:col_f+1].mean()
        S = exx[row_i:row_f+1,col_i:col_f+1].std()
        clims_ranged=(M-c*S,M+c*S)
        im=ax[0][0].imshow(exx,interpolation='nearest', cmap=cm, clim=clims_fix)
        ax[0][0].set_title('e_xx')
        fig.colorbar(im, ax=ax[0][0], shrink=colorbarshrink)
        M = eyy[row_i:row_f+1,col_i:col_f+1].mean()
        S = eyy[row_i:row_f+1,col_i:col_f+1].std()
        clims_ranged=(M-c*S,M+c*S)
        im=ax[1][1].imshow(eyy,interpolation='nearest', cmap=cm, clim=clims_fix)
        ax[1][1].set_title('e_yy')
        fig.colorbar(im, ax=ax[1][1], shrink=colorbarshrink)
        M = exy[row_i:row_f+1,col_i:col_f+1].mean()
        S = exy[row_i:row_f+1,col_i:col_f+1].std()
        clims_ranged=(M-c*S,M+c*S)
        im=ax[0][1].imshow(exy,interpolation='nearest', cmap=cm, clim=clims_fix)
        ax[0][1].set_title('e_xy')
        fig.colorbar(im, ax=ax[0][1], shrink=colorbarshrink)
        M = eyx[row_i:row_f+1,col_i:col_f+1].mean()
        S = eyx[row_i:row_f+1,col_i:col_f+1].std()
        clims_ranged=(M-c*S,M+c*S)
        im=ax[1][0].imshow(eyx,interpolation='nearest', cmap=cm, clim=clims_fix)
        ax[1][0].set_title('e_yx')
        fig.colorbar(im, ax=ax[1][0], shrink=colorbarshrink)
        M = rot[row_i:row_f+1,col_i:col_f+1].mean()
        S = rot[row_i:row_f+1,col_i:col_f+1].std()
        clims_ranged=(M-c*S,M+c*S)
        im=ax[2][0].imshow(rot,interpolation='nearest', cmap=cm, clim=clims_fix_rot)
        ax[2][0].set_title('Rotation') 
        fig.colorbar(im, ax=ax[2][0], shrink=colorbarshrink)
        M = shear[row_i:row_f+1,col_i:col_f+1].mean()
        S = shear[row_i:row_f+1,col_i:col_f+1].std()
        clims_ranged=(M-c*S,M+c*S)
        im=ax[2][1].imshow(shear,interpolation='nearest', cmap=cm, clim=clims_fix)
        ax[2][1].set_title('Shear')  
        fig.colorbar(im, ax=ax[2][1], shrink=colorbarshrink)
        plt.show()
        
    return exx, eyy, exy, eyx, rot, shear 


#GPA specific functions

def Choose_g_vectors(refined_distances, refined_angles_to_x, refined_pixels, g2='perpendicular'):
    '''
    Define the vectors or pixels positions to be used for the GPA calculation
    
    g1 is computed based on the closest spot to x, as typically the interfaces will
    be almost paralel to x. If it was not the case in some scenarions, another 
    setting just having as input the interface orientation would be used to choose 
    the spot closer to this interface orientation
    
    Two possible setups for computing g2, based on g1 to ensure orthogonality
    
    if g2=='closest':
    # choose the largest distance (closest spot to the center) that is not g1, in
    # the right half of the FFT
        
    if g2=='perpendicular':
    # choose the most perpendicular spot to the choosen g1, also from the right half of FFT
        
    Parameters
    ----------
    refined_distances : distances, of physically meaningful spots, no central spot, 
                        in angstroms (although it does not matter here)
    refined_angles_to_x : angles to x axis, of physically meaningful spots, 
                        no central spot, in degrees, from 180 to -180
    refined_pixels : pixels positions, of physically meaningful spots, no central spot, y and x

    Returns
    -------
    The positions, in pixels, of the spots to be used as g vectors for the GPA calculation
    in an array of shape=(2,2), first array is g1 with y1,x1 and second array g2 with y2,x2
    
    We output as well a 2 elements list with the angles to x of the given spots g1, g2
    As the angle counts from x positive axis, the smallest angle is at the right of
    the FFT and we discard the spots on the left (in case we use just one half of the FFT later)
    
    The 0,0 of the pixels reference is top left corner 
    '''
    
    #if the vertical planes are just the closest to the horizontal plane, then:
    # as the distance array is sorted from largest distances to 
    
    #only account for the spots of the right half of the FFT
    refined_distances=np.asarray(refined_distances)
    refined_angles_to_x=np.asarray(refined_angles_to_x)
    refined_pixels=np.asarray(refined_pixels)
    
    # both 90 an d-90 angles included so it goes from 90 to -90 -> [-90,90]
    newdists=refined_distances[refined_angles_to_x>=-90]
    newangls=refined_angles_to_x[refined_angles_to_x>=-90]
    newpixels=refined_pixels[refined_angles_to_x>=-90]
    
    newdists=newdists[newangls<=90]
    newpixels=newpixels[newangls<=90]
    newangls=newangls[newangls<=90]
    
    g_vectors_pixels=[]
    angles_to_x_of_g=[]  
    
    if len(newdists)==0:
        print('! G vectors selection warning:')
        print('No spots to select: Either amorphous or not enough resolution')
        return g_vectors_pixels, angles_to_x_of_g
    
    if len(newdists)==1:
        print('G vectors selection warning:')
        print('Only 1 spot identified, GPA is not possible')
        return g_vectors_pixels, angles_to_x_of_g
    
    if len(newdists)==2:
        g_vectors_pixels=newpixels
        angles_to_x_of_g=newangls
        return g_vectors_pixels, angles_to_x_of_g

    g1_vertical_plane=newpixels[np.argmin(np.abs(newangls))]
    g_vectors_pixels.append(g1_vertical_plane)
    g1_angle=newangls[np.argmin(np.abs(newangls))]
    angles_to_x_of_g.append(g1_angle)
    
    #to find the second vector, two options are given:
    if g2=='closest':
        # choose the largest distance (closest spot to the center) that is not g1, in
        # the right half of the FFT

        newpixels=newpixels[newangls !=g1_angle]
        newangls=newangls[newangls !=g1_angle]
        # as they are sorted the first element is the largest distance
        g2_horizontal_plane=newpixels[0]
        g2_angle=newangls[0] 
                           
        g_vectors_pixels.append(g2_horizontal_plane)
        angles_to_x_of_g.append(g2_angle)
        
    if g2=='perpendicular':
        # choose the most perpendicular spot to the choosen g1

        newdists=newdists[newangls !=g1_angle]
        newpixels=newpixels[newangls !=g1_angle]
        newangls=newangls[newangls !=g1_angle]
        
        #the most perpendicular one to g1
        modenewangls=np.abs(np.abs(newangls-g1_angle)-90)
        
        newdists1=[dist for _,dist in sorted(zip(modenewangls,newdists), key=lambda pair: pair[0])]
        newpixels1=[pixels for _,pixels in sorted(zip(modenewangls,newpixels), key=lambda pair: pair[0])]
        newangls1=[angls for _,angls in sorted(zip(modenewangls,newangls), key=lambda pair: pair[0])]
          
        g2_horizontal_plane= newpixels1[0]
        g2_angle=newangls1[0]             
        # this chooses the angle closest to the perpendicular, but make sure it is
        # the smallest distance with that angle, as the pixelwise precision definition
        # of the central spot may make the angle vary slighly and make the final spot be 
        # the one farthest from the center
        
        for dist,angle,pixel in zip(newdists1,newangls1,newpixels1):

            if abs(angle)<abs(newangls1[0])+0.5 and abs(angle)>abs(newangls1[0])-0.5 and dist>=newdists1[0]:
            
                g2_horizontal_plane= pixel
                g2_angle=angle
            
        g_vectors_pixels.append(g2_horizontal_plane)
        angles_to_x_of_g.append(g2_angle)
        
    return g_vectors_pixels, angles_to_x_of_g


#Functions also important for the general algorithm of low to high mag final communication


# Provisional function for a quick segmentation to give the reference automatically
def Quick_Segmentation_for_Reference(image_array, k_clusters):
    #input the image ready for doing the segmentation already, if denoised or standarised or downscaled...
    # image_st=(image_array-np.min(image_array))/np.max(image_array-np.min(image_array))    
    
    # denoised_image=Adaptative_Gaussian_Blur(image_st, x_calibration)
    #denoised_image=cv2.GaussianBlur(image_st, gauss_blur_filter_size, 1)
    
    #Second standarisation of the image after filtering/blurring it with gaussian filter
    
    image_st=(image_array-np.min(image_array))/np.max(image_array-np.min(image_array))
    
    downscaling_factor=int(image_st.shape[0]/64)
    ds_image=skimage.measure.block_reduce(image_st, block_size=tuple(np.int32(downscaling_factor*np.ones(len(np.shape(image_st))))), func=np.mean, cval=0)
    
    values, labels, cost = Segment.best_km(ds_image, n_clusters =k_clusters)
    labels_ms_reshaped = np.choose(labels, values)
    labels_ms_reshaped.shape = ds_image.shape
  
    labels_ms_reshaped=Segment.Multiple_Consecutive_Clustered_regions(labels_ms_reshaped, criteria='Tolerance', tolerance=0.10)

    plt.imshow(labels_ms_reshaped,cmap=plt.cm.gray)
    plt.show()
    
    substrate_segment=np.copy(labels_ms_reshaped)
    last_row_sgmented=labels_ms_reshaped[-1,:]
    substrate_val_inter=np.unique(last_row_sgmented)
    counted=[]
    for val in substrate_val_inter:
        counts=len(last_row_sgmented[last_row_sgmented==substrate_val_inter])
        counted.append(counts)       
    substrate_val=substrate_val_inter[np.argmax(np.asarray(counted))]
    
    for ini, i in enumerate(substrate_segment):
        for inij, j in enumerate(i):
            if j!=substrate_val:
                substrate_segment[ini,inij]=0
            else:
                substrate_segment[ini,inij]=1
    
    conts_vertx_per_region=Segment.Contour_draw_All_regions(substrate_segment)

    final_extra_pixs=Segment.Denoise_All_Regions(substrate_segment,conts_vertx_per_region)
      
    for label in final_extra_pixs:
        for pixel in final_extra_pixs[label]:   
            substrate_segment[pixel[0],pixel[1]]=int(label[0])    
    
    size_down=(labels_ms_reshaped.shape[0]*downscaling_factor,labels_ms_reshaped.shape[0]*downscaling_factor)
    segmented_image= cv2.resize(substrate_segment, size_down, interpolation = cv2.INTER_AREA)
    plt.imshow(segmented_image,cmap=plt.cm.gray)
    plt.show()
     
    # this line tailors the process to the images used to test, but should be removed as soon as the 
    #independent region finding for each label is calculated  
          
    #segmented_image[0:int(np.floor(size_down[0]/2)),:]=0
    
    return segmented_image
          


def Segmented_Region_Conversion_into_Matrix(segmented_image, label_of_region):
    '''
    Turn a segmented region identified by the label=label_of_region into 
    the smallest possible rectangular matrix that contains all the pixels with a value
    equal to label_of_region
    The segmented image only needs that the specified region to consider has the value
    label_of_region and the rest of pixels can be whatever
    
    IMPORTANT: It extracts the coordinates of the cropped region to be stored for later
    steps to know where the region came from in the original image
    The tuple format is coordinates_of_region=(first_row,last_row,first_col,last_col)
    So to crop the region from the original image it can be done like the following:
    region=image[first_row:last_row+1,first_col:last_col+1]

    Parameters
    ----------
    segmented_image : matrix of the segmented image
    label_of_region : int

    Returns
    -------
    matrix_of_region 

    '''
    
    image_of_region=np.copy(segmented_image)
    
    image_of_region[image_of_region != label_of_region] = 0
    image_of_region[image_of_region == label_of_region] = 1
    
    positions_1=np.where(image_of_region==1)
    first_row=min(positions_1[0])
    last_row=max(positions_1[0])
    
    # to discard the correct case in which the lowest index is 0, where second index must be 1
    first_col=min(positions_1[1])
    last_col=max(positions_1[1])
    
    matrix_of_region=image_of_region[first_row:last_row+1,first_col:last_col+1]
    #coordinates, with row i, row final, col i, col final
    coords_of_region=(first_row,last_row,first_col,last_col)
    return matrix_of_region, coords_of_region



def Find_Biggest_Squares_per_Segmented_Region(matrix_of_region):
    
    #first turn the result matrix a copy of the original and put everything to 0
    #except the border pixels of the top and left edges
    
    matrix_of_squares=np.copy(matrix_of_region)
    
    matrix_of_squares[1:matrix_of_squares.shape[0],1:matrix_of_squares.shape[1]]=0
    
    for y in range(1,matrix_of_squares.shape[0]):
        for x in range(1,matrix_of_squares.shape[1]):
            if matrix_of_region[y,x]==1:
                matrix_of_squares[y,x]=min(matrix_of_squares[y-1,x],matrix_of_squares[y,x-1],matrix_of_squares[y-1,x-1])+1
                  
    return matrix_of_squares
    
 
    
def Find_Biggest_Square_Closest_to_Region_Center(matrix_of_region, matrix_of_squares):
    '''
    Returns the coordinates of the square, in format,
    coords_of_square= [first_row,last_row,first_col,last_col], 
    [closest_square_top_left_corner[0],closest_square_bottom_right_corner[0],closest_square_top_left_corner[1],closest_square_bottom_right_corner[1]]
    to crop the image use directly the coords without adding 1

    Parameters
    ----------
    matrix_of_region : TYPE
        DESCRIPTION.
    matrix_of_squares : TYPE
        DESCRIPTION.

    Returns
    -------
    coords_of_square_to_crop : TYPE
        DESCRIPTION.

    '''
    
    #in regular regions, just the central position:
    # probably the weighted to the positions of the region would be better but if better just very slightly
    # and only in counted occasions so not really worth programming it
    
    central_coords=(matrix_of_region.shape[0]//2, matrix_of_region.shape[1]//2) 
    
    size_of_biggest_square=int(np.max(matrix_of_squares))
    pos_bottom_right_corner_bigst_sqrs=np.where(matrix_of_squares==size_of_biggest_square)
    
    pos_center_bigst_sqrs=np.array([pos_bottom_right_corner_bigst_sqrs[0],pos_bottom_right_corner_bigst_sqrs[1]])
    pos_center_bigst_sqrs[0]=pos_bottom_right_corner_bigst_sqrs[0]-np.floor(size_of_biggest_square/2)
    pos_center_bigst_sqrs[1]=pos_bottom_right_corner_bigst_sqrs[1]-np.floor(size_of_biggest_square/2)
    
    distance_to_center=np.sqrt((pos_center_bigst_sqrs[0]-central_coords[0])**2+(pos_center_bigst_sqrs[1]-central_coords[1])**2)
    
    closest_square_bottom_right_corner=(pos_bottom_right_corner_bigst_sqrs[0][np.argmin(distance_to_center)],pos_bottom_right_corner_bigst_sqrs[1][np.argmin(distance_to_center)])
    closest_square_top_left_corner=[closest_square_bottom_right_corner[0]-(size_of_biggest_square-1),closest_square_bottom_right_corner[1]-(size_of_biggest_square-1)]
    
    coords_of_square=[closest_square_top_left_corner[0],closest_square_bottom_right_corner[0],closest_square_top_left_corner[1],closest_square_bottom_right_corner[1]]
   
    return coords_of_square




def Locate_Reference_in_Original_Image(coords_of_region, coords_of_reference):
    '''
    Converts the coordiantes of the square of the reference, in terms of the cropped segmented region
    into the corresponding coordinates of the image where the region is taken from, to be given to GPA for 
    the final reference (they can be slightly reduced to avoid the effects of the image border)
    
    The coordiantes must be in the index format, to crop a 1 must be added: [first_row,last_row,first_col,last_col] 

    output, as usual, of indices to add 1 to crop coords_of_square= [first_row,last_row,first_col,last_col]
    
    Parameters
    ----------
    coordinates_of_region : 
    coords_of_reference : 

    Returns
    -------
    scaled_reference_coordinates : 

    '''
      
    (row_reg_i,row_reg_f,col_reg_i,col_reg_f)=coords_of_region
    (row_ref_i,row_ref_f,col_ref_i,col_ref_f)=coords_of_reference
    
    square_size=col_ref_f-col_ref_i
    
    #coords initial and final, with (y (row), x (col)) format
    scal_ref_cord_i=(row_reg_i+row_ref_i,col_reg_i+col_ref_i)
    scal_ref_cord_f=(scal_ref_cord_i[0]+square_size,scal_ref_cord_i[1]+square_size)
   
    scaled_reference_coords=(scal_ref_cord_i[0],scal_ref_cord_f[0],scal_ref_cord_i[1],scal_ref_cord_f[1])

    return scaled_reference_coords
    

def Reduce_Reference_if_Border(image_st_reference, scaled_reference_coords, reduction=0.1):
    '''
    Reduce a % = reduction of the size of the square of the reference in case it contacts the edges of
    the image
    '''
    image_size=image_st_reference.shape[0]
    row_i,row_f,col_i,col_f=scaled_reference_coords
    square_size=row_f+1-row_i
    
    # if the square constacts the right, left or bottom or top (top for other cases than reference)
    if row_f+1==image_size or col_f+1==image_size or row_i==0 or col_i==0:
        
        pxls_delete_side=int(np.floor(square_size*reduction)/2)
        row_i,row_f,col_i,col_f=row_i+pxls_delete_side,row_f-pxls_delete_side,col_i+pxls_delete_side,col_f-pxls_delete_side
    
    reduced_reference_coords=row_i,row_f,col_i,col_f
    return reduced_reference_coords
    
#compute mask size

def Define_GPA_Mask_Size(image, FFT_calibration, smallest_feature=0):
    # FFT calibration in nm^-1/pixel
    # As GPA is statistical, it should be applied mostly to low magnification/high FOV images
    # we do not want too strong local variations within the images to get good averages to extract
    #and use them to translate to the final model: we do not require high resolution, then something
    # 9 - 6nm should be good enough for most cases.
    
    # make mask of a given number of pixels coinciding with some nm, to make it sample dependent adjunst the
    # resolution to the features observed in the image
    
    #the smallest_feature argument stands for the value computed, in nm ,of the smalles segmented feature in
    # the device we are analysing: if we have a QW between two layers, then the QW is smallest and is 20nm thick
    
    if smallest_feature !=0:
        # define hyperparameter feature resolution, of the variation within the smallest element identified:
        # if the smallest feature is a QW of 20nm, then the resolution we get is 20/2.5. If more resolution is
        # wanted, then increase this value but it will be much noisier for resolutions < 5nm
        feature_resolution=1
        resolution=smallest_feature/feature_resolution
        mask_size=int(np.floor(1/(resolution*FFT_calibration)))
        GPA_eff_resolution=1/(mask_size*FFT_calibration)
        # Put a security check in case the feature is too large and then the resolution is too low:
        if GPA_eff_resolution > 10:
            resolution=10
            mask_size=int(np.floor(1/(resolution*FFT_calibration)))
            GPA_eff_resolution=1/(mask_size*FFT_calibration)
    else:
        resolution=7  #nm
        mask_size=int(np.floor(1/(resolution*FFT_calibration)))
        GPA_eff_resolution=1/(mask_size*FFT_calibration)
        
    return mask_size, GPA_eff_resolution

#compute the angle based on the peaks identified or on the interface

def Compute_GPA_Rotation_X_Axis(
        list_of_ALL_spots_angles, substrate_segment_image, substrate_segment_contour):
    '''
    Computes the angle to be used for GPA
    It firstly copmutes if there is a certain spot identified to have an angle close to 0 or 90 and in the 
    latter case it is substracted by 90
    In case there is not, the angle of the interface is computed. It can be modified as if "angle"
    or "interface" are the different criterias possible to choose

    Parameters
    ----------
    list_of_ALL_spots_angles : all the angles from -180 to 180 of the identified spots
    substrate_segment_image : image with only the substrate as a segment, 1, and the rest 0 intenstiy
    substrate_segment_contour : contours computed for this substrate with intensity 1

    Returns
    -------
    angle_of_horizontal_planes : TYPE
        DESCRIPTION.

    '''
    # compute the angle between the theoretical x axis, perfectly horizontal, with the horizontal planes
    # actually observed in the image to correct it to GPA
    
    # First compute the angles by taking into account the spots that were found by the peak finding
    # Adjust the finding so it only computes angle if it is closer to 0, or to +-90 and then substract 90
       
    for angle in list_of_ALL_spots_angles:
    # as the angles are sorted the first will appear being the spots closest to the cener   
        if angle<=0+5 and angle>=0-5:
            angle_of_horizontal_planes=angle
            return angle_of_horizontal_planes
        elif angle<=90+5 and angle>=90-5:
            angle_of_horizontal_planes=angle-90
            return angle_of_horizontal_planes
        elif angle<=-90+5 and angle>=-90-5:
            angle_of_horizontal_planes=90 - abs(angle)
            return angle_of_horizontal_planes
    # Otherwise, work with finding out the interface orientation
    #work with segmented profile where 0 intensity is not the substrate and 1 intensity is the substrate pixels
    # then only substrate_segment_contour[str(int(1))+'_contours'] contains contours
    coords_same_y=[]
    
    contour_vectors=substrate_segment_contour[str(int(1))+'_contours']
    
    for contour in contour_vectors:
        #format of this coords is (y,x)
        init_c=contour.init_coords
        final_c=contour.final_coords
        #contours coordinates are not pixel coordinates as each pixel has 4 coordinates, and the
        # last pixel in an image with last index=image.shape[0]-1, has the 1st coordinate=image.shape[0]-1
        # and the last coordiante= image.shape[0]
        
        #pixels that share the same y (height) and do not belong to the edges of the image
        if init_c[0] != substrate_segment_image.shape[0] and init_c[0] !=0 and init_c[0]==final_c[0]  :
            coords_same_y.append([init_c[0],init_c[1]])
            
    coords_same_y=np.asarray(coords_same_y)   
    #unify the coordinates with the same y
    y_values=np.unique(coords_same_y[:,0])
    
    coord_values_compressed=[]
    
    # way of computing the stacked segments in the interface with same y, done by substracting the
    # values that coincide with the x cooridnates and finding the 0s, much more efficient than loop

    for y_val in y_values:
        y_val_coords=coords_same_y[coords_same_y[:,0]==y_val]
        y_val_coords=np.asarray(sorted(y_val_coords, key=lambda x: x[1])[::-1])
        next_index_to_start=0
        while next_index_to_start<y_val_coords.shape[0]:       
            first_x_val=y_val_coords[next_index_to_start,1]
            y_coords_shape=y_val_coords.shape[0]
            x_vals_array=np.arange(first_x_val-(y_coords_shape-next_index_to_start)+1, first_x_val+next_index_to_start+1, 1)[::-1]
            x_vals_array.shape=y_coords_shape
            y_val_coords_to_eval=np.subtract(y_val_coords[:,1],x_vals_array)
            y_eval_0s=np.where(y_val_coords_to_eval==0)
            starting_y_val_sgmnt=y_val_coords[y_eval_0s[0][0]]
            ending_y_val_sgmnt=y_val_coords[y_eval_0s[0][-1]]
            coord_values_compressed.append(starting_y_val_sgmnt)  
            coord_values_compressed.append(ending_y_val_sgmnt)  
            
            next_index_to_start=y_eval_0s[0][-1]+1
     
    coord_values_compressed=np.asarray(coord_values_compressed)
    
    #plot x y graph
    #sort from lower x to higher x
    coord_values_compressed=np.asarray(sorted(coord_values_compressed, key=lambda x: x[1]))
    
    #define segments in the format (y , x_start, y_final)
    
    segment_definition=[]
    index=0
    while index<coord_values_compressed.shape[0]:
        segment=[coord_values_compressed[index,0],coord_values_compressed[index,1],coord_values_compressed[index+1,1]]
        segment_definition.append(segment)
        index=index+2
    
    segment_definition=np.asarray(segment_definition)
    #find largest segment
    largest_segment=segment_definition[np.argmax(np.subtract(segment_definition[:,2],segment_definition[:,1]))]
    central_position=np.floor((largest_segment[2]-largest_segment[1])/2)
    
    min_index_down=np.min(segment_definition[:,1])
    max_index_up=np.max(segment_definition[:,2])
    
    angles_array=[]

    down_index=central_position
    up_index=central_position+1
    
    # doing the increase of the segment lenght first on one side and then to the other keeping the same
    # central position is less sensitive to noise in case there is a bad segmentation at some point 
    while down_index>min_index_down:
        
        index_vals=[]
        for index in [down_index, up_index]:
            init_x_vals=segment_definition[:,1]
            init_x_vals=init_x_vals-index
            init_x_vals[init_x_vals>0]=2**10
            index_val=segment_definition[np.argmin(abs(init_x_vals)),0]
            index_vals.append(index_val)
            
        yi,yf=index_vals
       
        angle=np.arctan2((yf-yi),(up_index-down_index))*(180/np.pi)
        
        angles_array.append(angle)
        down_index=max(down_index-1, min_index_down)
        
    down_index=central_position
    up_index=central_position+1  
    
    while up_index<max_index_up:
         
        index_vals=[]
        for index in [down_index, up_index]:
            init_x_vals=segment_definition[:,1]
            init_x_vals=init_x_vals-index
            init_x_vals[init_x_vals>0]=2**10
            index_val=segment_definition[np.argmin(abs(init_x_vals)),0]
            index_vals.append(index_val)
            
        yi,yf=index_vals
       
        angle=np.arctan2((yf-yi),(up_index-down_index))*(180/np.pi)
        
        angles_array.append(angle)
        
        up_index=min(up_index+1,max_index_up)
    # opposite angle as the quadrants are vertically inverted because the y are smaller at the top of the image
    angles_array=-np.asarray(angles_array)    
    
    # delete the outliers that are caused by noise generating big angles that are not possible
    # for an interface, at least a "growth" one
    # we assume that difficult to have a real rotation of the interface of >2.5 degrees, so allow noise 
    # rotation of up to +-5º in case they are centered arround 2.5
    angles_array=angles_array[abs(angles_array)<5]
    
    # plt.hist(angles_array,30,[np.min(angles_array),np.max(angles_array)])
    # plt.show()
    
    angle_of_horizontal_planes=np.mean(angles_array)
    return angle_of_horizontal_planes


def Plot_Image_with_GPA_Reference(image_array, scaled_reference_coords):
    # Plot the image and the square where the reference is computed 
    fig,ax = plt.subplots(1)
    ax.imshow(image_array,cmap=plt.cm.gray, vmin=image_array.min(), vmax=image_array.max())
    reference_position = Rectangle((scaled_reference_coords[2],scaled_reference_coords[0]), scaled_reference_coords[3]-scaled_reference_coords[2], scaled_reference_coords[1]-scaled_reference_coords[0], angle=0.0, facecolor='none',edgecolor="blue", linewidth=1, alpha=1)
    ax.add_patch(reference_position)
    plt.show(block=False)
    
    
    
#!!! To be completed as soon as segmentation is robust

def Compute_Smallest_Feature_Thickness(segmented_image):
    #To be done once the segmentation routine is strong enough
    smallest_feature_thikness=1
    return smallest_feature_thikness


def Define_Region_as_GPA_Reference(
        analysed_image, segmented_image):
    '''
    # The information from GPA is gonna come from the biggest region that is segmented 
    # in the downer half of the segmented region
    # search for the segment that occupies more pixels in the downer half of 
    # the segmented image
    '''
    
    
    # CHECK 1: Most common label in downer half of the image
    half_image = segmented_image[int(np.floor(np.shape(segmented_image)[0]/2)):,int(np.floor(np.shape(segmented_image)[1]/2)):]
    
    labels_in_half, label_counts = np.unique(half_image, return_counts = True)
    
    label_of_GPA_ref = int(labels_in_half[np.argmax(label_counts)])
    
    
    
    # Retrieve data extracted from the reference 
    crop_outputs_dict = analysed_image.Crop_outputs

    # Check if there is a crystal there
    crop_list_refined_cryst_spots_GPA_pos = crop_outputs_dict[str(int(label_of_GPA_ref)) + '_list_refined_cryst_spots']

    # If at least one crystal was found in that label, then return it as GPA one
    if len(crop_list_refined_cryst_spots_GPA_pos) != 0:
        
        return label_of_GPA_ref
    
    
    # If that label has no crystal in it, find the most common label in the 
    # whole segmented image
    # CHECK 2: Most common label in whole image
    
    labels_in_whole, label_counts_whole = np.unique(
        segmented_image, return_counts = True)
    
    label_of_GPA_ref = int(labels_in_whole[np.argmax(label_counts_whole)])
    
    # Check if there is a crystal there
    crop_list_refined_cryst_spots_GPA_pos = crop_outputs_dict[str(int(label_of_GPA_ref)) + '_list_refined_cryst_spots']
    
    # If at least one crystal was found in that label, then return it as GPA one
    if len(crop_list_refined_cryst_spots_GPA_pos) != 0:
        
        return label_of_GPA_ref
    
    
    # If that label has no crystal in it, then sort the crystals by 
    # fit quality, score, and use the best scored one as GPA reference
    # CHECK 3: Best scored crystal as direct GPA reference
    
    # store All the crystals, their lables, from All the segmented regions shuffled 
    # and then sort them from lowest (best) score, to highest score
    crysts_scores = []
    labels_scored = []
    
    # these two lists correlate 1 to 1 so we loop through the labels
    # depending on the order of the best crysts found for each label
    
    for compl_label in labels_in_whole:
        
        # Pixels within the whole image in which the crop of the reference is taken, 
        # so the box of the reference itself [first_row,last_row,first_col,last_col]
        scaled_reference_coords_compl_label = crop_outputs_dict[str(int(compl_label)) + '_pixel_ref_cords']
    
        # 
        image_crop_hs_signal_compl_label = crop_outputs_dict[str(int(compl_label)) + '_hs_signal']
        FFT_image_array_compl_label, _ =  ImCalTrans.Compute_FFT_ImageArray(np.asarray(image_crop_hs_signal_compl_label))
    
        crop_list_refined_cryst_spots_compl_label = crop_outputs_dict[str(int(compl_label)) + '_list_refined_cryst_spots']
        refined_pixels_compl_label = crop_outputs_dict[str(int(compl_label)) + '_refined_pixels']
        spots_int_reference_compl_label = crop_outputs_dict[str(int(compl_label)) + '_spots_int_reference']

        # if at least one crystal was found in that label
        if len(crop_list_refined_cryst_spots_compl_label) != 0:
            # sum the scores as done in the phase identificator, to check the 
            # best phase
            for cryst in crop_list_refined_cryst_spots_compl_label:
                cryst_score = 0
                for spot_pair in cryst.spot_pairs_obj:
                    spot_pair_score = spot_pair.score
                    cryst_score = cryst_score + spot_pair_score
                total_spot_pairs = len(cryst.spot_pairs_obj)
                
            
                crysts_scores.append(cryst_score/total_spot_pairs**2)
                labels_scored.append(int(compl_label))
                
        
    # If absolutely no crystal is found, no crystal identified, then
    # raise the error that the process cannot be done            
    if len(crysts_scores) == 0:
        raise Exception('No crystalline phase could be identified in the present micrograph/sample/device')
    else:
        
        # Sort the labels list based on the score for the crystals found for each
        # label being the lowest label the best crystal
        labels_scored_sorted = [label for _ , label in sorted(zip(crysts_scores, labels_scored), key=lambda x: x[0])]
    
        # Get the label from the best crystal scored as it will be the one 
        # used if that label is considered
        label_of_GPA_ref = labels_scored_sorted[0]
    
        return label_of_GPA_ref
    




def Compute_Feature_defining_GPA_Res(
        segmented_image, conts_vertxs_per_region_segmented, pixel_size):
    '''
    Compute feature lenghts that allow to get a reference on the size of the 
    elements that have been segmented and represent the device, to have a
    measure of how the resolution f the GPA mask should be
    
    Parameters
    ----------
    segmented_image : 
    conts_vertxs_per_region_segmented :
    pixel_size : 

    Returns
    -------
    averaged_features = [aver_of_aver, aver_of_max, aver_of_min, aver_of_med, aver_of_desvest]

    '''
    
    labels = np.unique(segmented_image)
    
    # metrics to include: mean, median, desvest, max, min differences
    feature_metrics = []
    
    
    for label in labels:
        
        # work directly with the relative vertexs in nm to get directly a
        # value in nm (distance)
        contours_label = conts_vertxs_per_region_segmented[str(int(label)) + '_contours'] 
        
        distances_computed = []
        
        for contour_ref in contours_label:
            
            rel_init_coords_ref = contour_ref.rel_init_coords
            rel_final_coords_ref = contour_ref.rel_final_coords
        
        
            (ref_init_rel_coords_y, ref_init_rel_coords_x) = rel_init_coords_ref
            (ref_final_rel_coords_y, ref_final_rel_coords_x) = rel_final_coords_ref
            
            # for those contours with same value in y
            if ref_init_rel_coords_y == ref_final_rel_coords_y:
            
                for contour_check in contours_label:
                    
                    rel_init_coords_check = contour_check.rel_init_coords
                    rel_final_coords_check = contour_check.rel_final_coords
                
                    (check_init_rel_coords_y, check_init_rel_coords_x) = rel_init_coords_check
                    (check_final_rel_coords_y, check_final_rel_coords_x) = rel_final_coords_check
            
                    if ref_init_rel_coords_x == check_final_rel_coords_x and ref_final_rel_coords_x == check_init_rel_coords_x:
                        
                        # compute the y difference in absolute value and append it
                        y_diff = np.abs(check_final_rel_coords_y - ref_init_rel_coords_y)
                        
                        # ensure the added valus are bigger than the pixel size to 
                        # avoid adding noise to the computed values
                        if y_diff > 3*pixel_size:
                            distances_computed.append(y_diff)
        
        # just a safet check if no distance was added as they are all < 3*pixel_size
        if len(distances_computed) == 0:
            distances_computed = [pixel_size, pixel_size]
             
        # distances per metric stored in distances_computed
        distances_computed = np.asarray(distances_computed)
        
        average_label = np.mean(distances_computed)
        max_label = np.max(distances_computed)
        min_label = np.min(distances_computed) 
        median_label = np.median(distances_computed)
        desvest_label = np.std(distances_computed)
        
        metrics_per_label = [average_label, max_label, min_label, median_label, desvest_label]
        feature_metrics.append(metrics_per_label)
        
        
    # feature_metrics is a list of lists that correlates 1 to 1 with the labels list
    feature_metrics = np.asarray(feature_metrics)
    
    # compute the average of all the metrics analysed
    aver_of_aver = np.mean(feature_metrics[:,0])
    aver_of_max = np.mean(feature_metrics[:,1])
    aver_of_min = np.mean(feature_metrics[:,2])
    aver_of_med = np.mean(feature_metrics[:,3])
    aver_of_desvest = np.mean(feature_metrics[:,4])
    
    averaged_features = [aver_of_aver, aver_of_max, aver_of_min, aver_of_med, aver_of_desvest]
    
    return averaged_features



def Mod_GPA_RefRectangle(
        scaled_reference_coords_GPA_ref, factor):
    '''
    Reduce the dimensions of the GPA reference square (function
    generalised to rectange) a factor, so factor = 0.1 is increasing it 10%
    while factor = -0.2 is decreasing it 20%
    keeping its central position

    Parameters
    ----------
    scaled_reference_coords_GPA_ref : TYPE
        DESCRIPTION.
    factor : TYPE
        DESCRIPTION.

    Returns
    -------
    new_scaled_reference_coords_GPA_ref : TYPE
        DESCRIPTION.

    '''
    
    i_y_i, i_y_f, i_x_i, i_x_f = scaled_reference_coords_GPA_ref
    
    x_lenght = i_x_f - i_x_i
    y_lenght = i_y_f - i_y_i
    
    new_x_lenght = x_lenght - x_lenght*factor
    new_y_lenght = y_lenght - y_lenght*factor
    
    diff_x = x_lenght*factor
    diff_y = y_lenght*factor
    
    f_y_i = i_y_i - int(np.floor(diff_y/2))
    f_y_f = i_y_f + int(np.floor(diff_y/2))
    f_x_i = i_x_i - int(np.floor(diff_x/2))
    f_x_f = i_x_f + int(np.floor(diff_x/2))
    
    new_scaled_reference_coords_GPA_ref = [f_y_i, f_y_f, f_x_i, f_x_f]
    
    return new_scaled_reference_coords_GPA_ref



def Get_GPA_best_g_vects_pair(
        analysed_image, label_of_GPA_ref, image_segmented):
    '''
    This function finds the best possible g vector pair, not only considering the
    score of the spot pairs obtained (which can cause the spots found be too
    too close to the FFT center, or too far away and too close to the edge 
    of the image, making that the mask remains too little to capture the
    heterostructure separated spots in the FFT), but also its position within
    the FFT, giving preference to spots that are closer to the averaged distance 
    found from all the spots, which places the g vectors where most of the spots
    where found, so more likelihood that the spots are well identified and
    well placed pixelwise (i.e. proper peak finding)
    The function checks for every condition that is checked which is the 
    spot pair in the reference region that is more epitaxial to any pair of spots
    from any of the other crystals found

    Parameters
    ----------
    analysed_image : analysed_image object for the image we are analysing
    label_of_GPA_ref : label of the segment that corresponds to the region 
                    we want to consider as reference for the GPA 

    Returns
    -------
    best_GPA_ref_spot_pair : scored_spot_pair representing the best g vects found

    '''
    
    # Retrieve data extracted from the reference 
    crop_outputs_dict = analysed_image.Crop_outputs
    
    # Pixels within the whole image in which the crop of the reference is taken, 
    # so the box of the reference itself [first_row,last_row,first_col,last_col]
    scaled_reference_coords_GPA_ref = crop_outputs_dict[str(label_of_GPA_ref) + '_pixel_ref_cords']
    
    # 
    image_crop_hs_signal_GPA_ref = crop_outputs_dict[str(label_of_GPA_ref) + '_hs_signal']
    FFT_image_array_GPA_ref, _ =  ImCalTrans.Compute_FFT_ImageArray(np.asarray(image_crop_hs_signal_GPA_ref))
    
    crop_list_refined_cryst_spots_GPA_ref = crop_outputs_dict[str(label_of_GPA_ref) + '_list_refined_cryst_spots']
    refined_pixels_GPA_ref = crop_outputs_dict[str(label_of_GPA_ref) + '_refined_pixels']
    spots_int_reference_GPA_ref = crop_outputs_dict[str(label_of_GPA_ref) + '_spots_int_reference']
    
    # Get the best cryst spot to get all distances of this found phase
    
    best_cryst_spot_GPA_ref = crop_list_refined_cryst_spots_GPA_ref[0]
    
    all_distances_found = []

    for spot_pair in best_cryst_spot_GPA_ref.spot_pairs_obj:
        
        spot1_dist_GPA_ref = spot_pair.spot1_dist
        spot2_dist_GPA_ref = spot_pair.spot2_dist
        
        all_distances_found.append(spot1_dist_GPA_ref)
        all_distances_found.append(spot2_dist_GPA_ref)
        
    # average the unique distances found
    mean_dist_found = np.mean(np.unique(np.asarray(all_distances_found)))
    
    # Get the scores of the crystals and their spot pairs to
    # loop through all of them to know the epitaxial nature with the reference
    # region found to be the GPA reference
    
    labels_unique = np.unique(image_segmented)
    labels_unique_no_ref = labels_unique[labels_unique != label_of_GPA_ref]
        
    labels_scored = []
    
    # store All the crystals from All the segmented regions shuffled 
    # and then sort them from lowest (best) score, to highest score
    cryst_per_label_scored = []
    crysts_scores = []
    
    # these two lists correlate 1 to 1 so we loop through the labels
    # depending on the order of the best crysts found for each label
    
    for compl_label in labels_unique_no_ref:
        
        # Pixels within the whole image in which the crop of the reference is taken, 
        # so the box of the reference itself [first_row,last_row,first_col,last_col]
        scaled_reference_coords_compl_label = crop_outputs_dict[str(int(compl_label)) + '_pixel_ref_cords']
    
        # 
        image_crop_hs_signal_compl_label = crop_outputs_dict[str(int(compl_label)) + '_hs_signal']
        FFT_image_array_compl_label, _ =  ImCalTrans.Compute_FFT_ImageArray(np.asarray(image_crop_hs_signal_compl_label))
    
        crop_list_refined_cryst_spots_compl_label = crop_outputs_dict[str(int(compl_label)) + '_list_refined_cryst_spots']
        refined_pixels_compl_label = crop_outputs_dict[str(int(compl_label)) + '_refined_pixels']
        spots_int_reference_compl_label = crop_outputs_dict[str(int(compl_label)) + '_spots_int_reference']

        # if at least one crystal was found in that label
        if len(crop_list_refined_cryst_spots_compl_label) != 0:
            # sum the scores as done in the phase identificator, to check the 
            # best phase
            for cryst in crop_list_refined_cryst_spots_compl_label:
                cryst_score = 0
                for spot_pair in cryst.spot_pairs_obj:
                    spot_pair_score = spot_pair.score
                    cryst_score = cryst_score + spot_pair_score
                total_spot_pairs = len(cryst.spot_pairs_obj)
                
            
                crysts_scores.append(cryst_score/total_spot_pairs**2)
                cryst_per_label_scored.append(cryst)
    
    
    # If no crystal was found in the other labels or regions that are not 
    # the GPA one, then just do the process finding the best pairs with no
    # reference on the other phase spots, and just the positioning of the
    # spots in the FFT range of distances
    if len(cryst_per_label_scored) == 0:
        
        best_spot_pair_found = 0
        # try to find a spot where both spots are closer to the center of the FFT
        # than the average of inerplanar distances found
        # CHECK 1: Both peaks below average distance found
        for spot_pair in best_cryst_spot_GPA_ref.spot_pairs_obj:
            
            spot1_dist_GPA_ref = spot_pair.spot1_dist
            spot2_dist_GPA_ref = spot_pair.spot2_dist
            
            if spot1_dist_GPA_ref >= mean_dist_found and spot2_dist_GPA_ref >= mean_dist_found:
                    
                best_GPA_ref_spot_pair = spot_pair
                best_spot_pair_found = 1
                
                if best_spot_pair_found == 1:
                    break
            
        # CHECK 2: Both peaks in the progressively opening mask
   
        # if not found, find the planes that are within a progressively increasing
        # circular sector that has its central circular arc in the very middle of
        # this circular sector                          
        if best_spot_pair_found == 0:
            
            for incr in range(1,11):
            
                for spot_pair in best_cryst_spot_GPA_ref.spot_pairs_obj:
                    
                    spot1_dist_GPA_ref = spot_pair.spot1_dist
                    spot2_dist_GPA_ref = spot_pair.spot2_dist
                    # let us increase 0.25 angs per side, so 0.5 in total per iter
                
                    if spot1_dist_GPA_ref <= mean_dist_found + 0.25*incr and spot1_dist_GPA_ref >= mean_dist_found - 0.25*incr:
                        if spot2_dist_GPA_ref <= mean_dist_found + 0.25*incr and spot2_dist_GPA_ref >= mean_dist_found - 0.25*incr:
                            
                            best_GPA_ref_spot_pair = spot_pair
                            best_spot_pair_found = 1
                                
                            if best_spot_pair_found == 1:
                                break
                        
                if best_spot_pair_found == 1:
                    break
                
        # CHECK 3: 1 spot below the average only
        
        # last resort, find the spot where just one of the distances is higher than threshold
        if best_spot_pair_found == 0:
            
            for spot_pair in best_cryst_spot_GPA_ref.spot_pairs_obj:
                
                spot1_dist_GPA_ref = spot_pair.spot1_dist
                spot2_dist_GPA_ref = spot_pair.spot2_dist
                
                if spot1_dist_GPA_ref >= mean_dist_found or spot1_dist_GPA_ref >= mean_dist_found:
                    
                    best_GPA_ref_spot_pair = spot_pair
                    
                    best_spot_pair_found = 1
                    
                if best_spot_pair_found == 1:
                    break
           
        # CHECK 4: Last resort, just get the best score
        if best_spot_pair_found == 0:
            best_GPA_ref_spot_pair = best_cryst_spot_GPA_ref.spot_pairs_obj[0]
            best_spot_pair_found = 1
            
        return best_GPA_ref_spot_pair        
            
    
    else:
    # if len(cryst_per_label_scored) != 0, so at least some planes were found
        
    
        # Sort the labels list based on the score for the crystals found for each
        # label being the lowest label the best crystal
        all_crystals_scored_sort = [crsyt for _ , crsyt in sorted(zip(crysts_scores, cryst_per_label_scored), key=lambda x: x[0])]
    
        # Actual spot pair searching process starts
    
        # CHECK 1: Both peaks below average distance found
    
        best_spot_pair_found = 0
        # try to find a spot where both spots are closer to the center of the FFT
        # than the average of inerplanar distances found
        
        # store the spot pairs from reference that could be the GPA one
        spot_pairs_ref_possible = []
        
        for spot_pair in best_cryst_spot_GPA_ref.spot_pairs_obj:
            
            spot1_dist_GPA_ref = spot_pair.spot1_dist
            spot2_dist_GPA_ref = spot_pair.spot2_dist
            
            if spot1_dist_GPA_ref >= mean_dist_found and spot2_dist_GPA_ref >= mean_dist_found:
                
                spot_pairs_ref_possible.append(spot_pair)
                best_spot_pair_found = 1
        
        
        # If at least 1 possible spot pair is found
        if best_spot_pair_found == 1:
    
            # this list contains, for every spot pair in reference region to be checked
            # if it is the one to be used for GPA, so paired 1 to 1 with the list of spot
            # pairs spot_pairs_ref_possible, the metric explaining how good and close
            # the pair of spot pairs is correlating with each other, meaining that 
            # the more epitaxial the spots are, the best score (lowest number) will
            # be provided and will be used and that spot pair with the lowest
            # score will be the chosen spot pair for the GPA calculation
            best_spot_pair_pair_metrics = [] 
            
            # chekc for all possible spot pairs to be used as GPA spot pairs g vects     
            for pos_ref_spot_pair in spot_pairs_ref_possible:    
                
                # the ref spots info
                dist_1_posref = pos_ref_spot_pair.spot1_dist
                dist_2_posref = pos_ref_spot_pair.spot2_dist
                
                angle_x_1_posref = pos_ref_spot_pair.spot1_angle_to_x
                angle_x_2_posref = pos_ref_spot_pair.spot2_angle_to_x
        
                # here gather the metric for all the spots in all the crystals,
                # we do not want to separate by crystal but to get from all
                # possible crystals which is the score or metric implying a best
                # epitaxy with the reference spot considered, so every reference 
                # spot considered has a best score, from 1 spot pair from 1 of the 
                # crystals found
                
                best_diff_metrics_all_spot_pairs = []
                    
                # analyse the planes from other labels          
                for cryst in all_crystals_scored_sort:
                                  
                    for spot_pair in cryst.spot_pairs_obj:
                        
                        dist_1_check = spot_pair.spot1_dist
                        dist_2_check = spot_pair.spot2_dist
                        
                        angle_x_1_check = spot_pair.spot1_angle_to_x
                        angle_x_2_check = spot_pair.spot2_angle_to_x
                        
                        
                        # compute the differences between the combinations of spots1 and 2
                        # from each spot checked, so 4 combinations
                        distances_diff_1 = np.abs(dist_1_posref - dist_1_check)
                        distances_diff_2 = np.abs(dist_1_posref - dist_2_check)
                        distances_diff_3 = np.abs(dist_2_posref - dist_1_check)
                        distances_diff_4 = np.abs(dist_2_posref - dist_2_check)
                        
                        angles_diff_1 = np.abs(angle_x_1_posref - angle_x_1_check)
                        angles_diff_2 = np.abs(angle_x_1_posref - angle_x_2_check)
                        angles_diff_3 = np.abs(angle_x_2_posref - angle_x_1_check)
                        angles_diff_4 = np.abs(angle_x_2_posref - angle_x_2_check)
                        
                        # then the lowest sum of defs expresses the actual best
                        # 1 to 1 spot correlation from reference to the other crystal 
                        # chceked and therefore is its defining value
                        # This is angle in degrees plus distance in nm, so completely
                        # unphyiscal and just a quick figure of merit
                        sum_diffs_1 = distances_diff_1 + angles_diff_1
                        sum_diffs_2 = distances_diff_2 + angles_diff_2
                        sum_diffs_3 = distances_diff_3 + angles_diff_3
                        sum_diffs_4 = distances_diff_4 + angles_diff_4
                        
                        # best_sum_diffs is the score of the pairing of the
                        # possible spot in reference and the one checked in the
                        # crystal from other region identified
                        best_sum_diffs = np.min([sum_diffs_1, sum_diffs_2, 
                                                 sum_diffs_3, sum_diffs_4])
                        
                        # store it
                        best_diff_metrics_all_spot_pairs.append(best_sum_diffs)
                    
                    
                    
                # Get the best metric, so the best spot pair positioning to the
                # reference one that is being checked
                best_spot_pair_cryst = np.min(best_diff_metrics_all_spot_pairs)
        
                best_spot_pair_pair_metrics.append(best_spot_pair_cryst)
                
            # best_spot_pair_pair_metrics and spot_pairs_ref_possible correlate
            # 1 to 1 and we can get from the scores, the spot pair in reference
            # which correlates the best with another one from another crystal found
            # so the lowest score, the best, and is the GPA spot pair that will 
            # easily allows us to get the best 
            # !!! It is a measure of the quality of the epitaxy
            best_GPA_ref_spot_pair = spot_pairs_ref_possible[np.argmin(best_spot_pair_pair_metrics)]
            
            return best_GPA_ref_spot_pair
               
        
        # CHECK 2: Both peaks between the progressive opening of a circular sector
    
        # if no spot pair has been found yet, use the mask opening check   
        if best_spot_pair_found == 0: 
           
            for incr in range(1, 11):
                
                # store the spot pairs from reference that could be the GPA one
                spot_pairs_ref_possible = []
                
                for spot_pair in best_cryst_spot_GPA_ref.spot_pairs_obj:
                    
                    spot1_dist_GPA_ref = spot_pair.spot1_dist
                    spot2_dist_GPA_ref = spot_pair.spot2_dist
                    # let us increase 0.25 angs per side, so 0.5 in total per iter
                
                    if spot1_dist_GPA_ref <= mean_dist_found + 0.25*incr and spot1_dist_GPA_ref >= mean_dist_found - 0.25*incr:
                        if spot2_dist_GPA_ref <= mean_dist_found + 0.25*incr and spot2_dist_GPA_ref >= mean_dist_found - 0.25*incr:
                            
                            spot_pairs_ref_possible.append(spot_pair)
                            # If at least 1 spot pair is found within the 
                            # mask ranges, then at least one will be returned
                            best_spot_pair_found = 1
                            
                # If no spot pairs were found, go to next iteration to make 
                # the mask bigger 
                if best_spot_pair_found == 0:
                    continue
                
                # If it does not go to the next iteration, it will go through 
                # the checking for the best spot pair that correlates the best with
                # the given spot ref in a mor epitaxial way, so next perform this check
                
                            
                # this list contains, for every spot pair in reference region to be checked
                # if it is the one to be used for GPA, so paired 1 to 1 with the list of spot
                # pairs spot_pairs_ref_possible, the metric explaining how good and close
                # the pair of spot pairs is correlating with each other, meaining that 
                # the more epitaxial the spots are, the best score (lowest number) will
                # be provided and will be used and that spot pair with the lowest
                # score will be the chosen spot pair for the GPA calculation
                best_spot_pair_pair_metrics = [] 
                
                # chekc for all possible spot pairs to be used as GPA spot pairs g vects     
                for pos_ref_spot_pair in spot_pairs_ref_possible:    
                    
                    # the ref spots info
                    dist_1_posref = pos_ref_spot_pair.spot1_dist
                    dist_2_posref = pos_ref_spot_pair.spot2_dist
                    
                    angle_x_1_posref = pos_ref_spot_pair.spot1_angle_to_x
                    angle_x_2_posref = pos_ref_spot_pair.spot2_angle_to_x
                
                    # here gather the metric for all the spots in all the crystals,
                    # we do not want to separate by crystal but to get from all
                    # possible crystals which is the score or metric implying a best
                    # epitaxy with the reference spot considered, so every reference 
                    # spot considered has a best score, from 1 spot pair from 1 of the 
                    # crystals found
                    
                    best_diff_metrics_all_spot_pairs = []
                        
                    # analyse the planes from other labels          
                    for cryst in all_crystals_scored_sort:
                                      
                        for spot_pair in cryst.spot_pairs_obj:
                            
                            dist_1_check = spot_pair.spot1_dist
                            dist_2_check = spot_pair.spot2_dist
                            
                            angle_x_1_check = spot_pair.spot1_angle_to_x
                            angle_x_2_check = spot_pair.spot2_angle_to_x
                            
                            
                            # compute the differences between the combinations of spots1 and 2
                            # from each spot checked, so 4 combinations
                            distances_diff_1 = np.abs(dist_1_posref - dist_1_check)
                            distances_diff_2 = np.abs(dist_1_posref - dist_2_check)
                            distances_diff_3 = np.abs(dist_2_posref - dist_1_check)
                            distances_diff_4 = np.abs(dist_2_posref - dist_2_check)
                            
                            angles_diff_1 = np.abs(angle_x_1_posref - angle_x_1_check)
                            angles_diff_2 = np.abs(angle_x_1_posref - angle_x_2_check)
                            angles_diff_3 = np.abs(angle_x_2_posref - angle_x_1_check)
                            angles_diff_4 = np.abs(angle_x_2_posref - angle_x_2_check)
                            
                            # then the lowest sum of defs expresses the actual best
                            # 1 to 1 spot correlation from reference to the other crystal 
                            # chceked and therefore is its defining value
                            # This is angle in degrees plus distance in nm, so completely
                            # unphyiscal and just a quick figure of merit
                            sum_diffs_1 = distances_diff_1 + angles_diff_1
                            sum_diffs_2 = distances_diff_2 + angles_diff_2
                            sum_diffs_3 = distances_diff_3 + angles_diff_3
                            sum_diffs_4 = distances_diff_4 + angles_diff_4
                            
                            # best_sum_diffs is the score of the pairing of the
                            # possible spot in reference and the one checked in the
                            # crystal from other region identified
                            best_sum_diffs = np.min([sum_diffs_1, sum_diffs_2, 
                                                     sum_diffs_3, sum_diffs_4])
                            
                            # store it
                            best_diff_metrics_all_spot_pairs.append(best_sum_diffs)
                        
                        
                        
                    # Get the best metric, so the best spot pair positioning to the
                    # reference one that is being checked
                    best_spot_pair_cryst = np.min(best_diff_metrics_all_spot_pairs)
                
                    best_spot_pair_pair_metrics.append(best_spot_pair_cryst)
        
                # best_spot_pair_pair_metrics and spot_pairs_ref_possible correlate
                # 1 to 1 and we can get from the scores, the spot pair in reference
                # which correlates the best with another one from another crystal found
                # so the lowest score, the best, and is the GPA spot pair that will 
                # easily allows us to get the best 
                # !!! It is a measure of the quality of the epitaxy
                best_GPA_ref_spot_pair = spot_pairs_ref_possible[np.argmin(best_spot_pair_pair_metrics)]
                
                return best_GPA_ref_spot_pair                        
                   
             
        # CHECK 3: Just one of the peaks below average distance found
             
        # if still no spot pair has been found yet, use the only 1 peak below average  
        if best_spot_pair_found == 0: 
            
            # store the spot pairs from reference that could be the GPA one
            spot_pairs_ref_possible = []
            
            for spot_pair in best_cryst_spot_GPA_ref.spot_pairs_obj:
                
                spot1_dist_GPA_ref = spot_pair.spot1_dist
                spot2_dist_GPA_ref = spot_pair.spot2_dist
                
                if spot1_dist_GPA_ref >= mean_dist_found or spot2_dist_GPA_ref >= mean_dist_found:
                    
                    spot_pairs_ref_possible.append(spot_pair)
                    # If at least 1 spot pair is found within the 
                    # mask ranges, then at least one will be returned
                    best_spot_pair_found = 1
    
            # if at least one spot pair is found meeting this condition, check
            # how good they are in terms of epitaxy with the other 
            if best_spot_pair_found == 1: 
                
                # this list contains, for every spot pair in reference region to be checked
                # if it is the one to be used for GPA, so paired 1 to 1 with the list of spot
                # pairs spot_pairs_ref_possible, the metric explaining how good and close
                # the pair of spot pairs is correlating with each other, meaining that 
                # the more epitaxial the spots are, the best score (lowest number) will
                # be provided and will be used and that spot pair with the lowest
                # score will be the chosen spot pair for the GPA calculation
                best_spot_pair_pair_metrics = [] 
                
                # chekc for all possible spot pairs to be used as GPA spot pairs g vects     
                for pos_ref_spot_pair in spot_pairs_ref_possible:    
                    
                    # the ref spots info
                    dist_1_posref = pos_ref_spot_pair.spot1_dist
                    dist_2_posref = pos_ref_spot_pair.spot2_dist
                    
                    angle_x_1_posref = pos_ref_spot_pair.spot1_angle_to_x
                    angle_x_2_posref = pos_ref_spot_pair.spot2_angle_to_x
            
                    # here gather the metric for all the spots in all the crystals,
                    # we do not want to separate by crystal but to get from all
                    # possible crystals which is the score or metric implying a best
                    # epitaxy with the reference spot considered, so every reference 
                    # spot considered has a best score, from 1 spot pair from 1 of the 
                    # crystals found
                    
                    best_diff_metrics_all_spot_pairs = []
                        
                    # analyse the planes from other labels          
                    for cryst in all_crystals_scored_sort:
                                      
                        for spot_pair in cryst.spot_pairs_obj:
                            
                            dist_1_check = spot_pair.spot1_dist
                            dist_2_check = spot_pair.spot2_dist
                            
                            angle_x_1_check = spot_pair.spot1_angle_to_x
                            angle_x_2_check = spot_pair.spot2_angle_to_x
                            
                            
                            # compute the differences between the combinations of spots1 and 2
                            # from each spot checked, so 4 combinations
                            distances_diff_1 = np.abs(dist_1_posref - dist_1_check)
                            distances_diff_2 = np.abs(dist_1_posref - dist_2_check)
                            distances_diff_3 = np.abs(dist_2_posref - dist_1_check)
                            distances_diff_4 = np.abs(dist_2_posref - dist_2_check)
                            
                            angles_diff_1 = np.abs(angle_x_1_posref - angle_x_1_check)
                            angles_diff_2 = np.abs(angle_x_1_posref - angle_x_2_check)
                            angles_diff_3 = np.abs(angle_x_2_posref - angle_x_1_check)
                            angles_diff_4 = np.abs(angle_x_2_posref - angle_x_2_check)
                            
                            # then the lowest sum of defs expresses the actual best
                            # 1 to 1 spot correlation from reference to the other crystal 
                            # chceked and therefore is its defining value
                            # This is angle in degrees plus distance in nm, so completely
                            # unphyiscal and just a quick figure of merit
                            sum_diffs_1 = distances_diff_1 + angles_diff_1
                            sum_diffs_2 = distances_diff_2 + angles_diff_2
                            sum_diffs_3 = distances_diff_3 + angles_diff_3
                            sum_diffs_4 = distances_diff_4 + angles_diff_4
                            
                            # best_sum_diffs is the score of the pairing of the
                            # possible spot in reference and the one checked in the
                            # crystal from other region identified
                            best_sum_diffs = np.min([sum_diffs_1, sum_diffs_2, 
                                                     sum_diffs_3, sum_diffs_4])
                            
                            # store it
                            best_diff_metrics_all_spot_pairs.append(best_sum_diffs)
                        
                        
                    # Get the best metric, so the best spot pair positioning to the
                    # reference one that is being checked
                    best_spot_pair_cryst = np.min(best_diff_metrics_all_spot_pairs)
            
                    best_spot_pair_pair_metrics.append(best_spot_pair_cryst)
                    
                # best_spot_pair_pair_metrics and spot_pairs_ref_possible correlate
                # 1 to 1 and we can get from the scores, the spot pair in reference
                # which correlates the best with another one from another crystal found
                # so the lowest score, the best, and is the GPA spot pair that will 
                # easily allows us to get the best 
                # !!! It is a measure of the quality of the epitaxy
                best_GPA_ref_spot_pair = spot_pairs_ref_possible[np.argmin(best_spot_pair_pair_metrics)]
                
                return best_GPA_ref_spot_pair
                                
    
        # CHECK 4: Final, add the best scored spot pair with no further checks
        # as the very last resort, desperate check
        if best_spot_pair_found == 0: 
            
            best_GPA_ref_spot_pair = best_cryst_spot_GPA_ref.spot_pairs_obj[0]
            
            return best_GPA_ref_spot_pair
                        
                        
                        




def Get_GPA_Res_HeteroStruct_Separated_Spots(
        analysed_image, image_in_dataset_whole, 
        best_GPA_ref_spot_pair,
        image_segmented, label_of_GPA_ref,
        GPA_spot_max_dist = 2):
    '''
    Compute the GPA resolution to later find the circular mask which has to 
    contain the analysis.
    It searcher for spots from the other crystals forming the htereostructure
    and adjust the radius based on their position. If no hetero is found, then
    it assumes a single spot is present so not a highly mimsmatched structure
    and then define a small mask to provide a smooth map in most of the cases

    Parameters
    ----------
    analysed_image : object containing the crystals found for that given image
    image_in_dataset_whole : info of the main image
    best_GPA_ref_spot_pair: best scored spot pair with the g vects
    image_segmented : segmented image of the image anlaysed
    label_of_GPA_ref : label within image_segmented where the GPA reference 
                    lattice is taken from
    GPA_spot_max_dist : float expressing the maximum size of the mask 
                within which the spots forming the heterostructure are
                seeked. This means a circle of radius 2nm is drawn around
                the spot used as a reference from the reference region, and 
                peaks from the other structures are searched to define a mask
                that captures both of them and therefore captures the idea of
                heterostructure we want to obtain with the GPA
                
                Be aware that increasing this value so more nm, means making a 
                smaller radius as we are in the reciprocal space, which means being
                more conservative and only tackle heterostructures with a smaller
                mismatch.
                If we want to analyse structures with more mismatch, then this value 
                needs to be decreased to draw a larger circle that captures both
                of them despite the mismatch (i.e. if the spots are more separated
                just decrease this value)
                
                If no spot is found within this radius then the resolution is 
                automatically set to 4nm, which is a quite small radius that 
                should always lead to a generally smooth map
         The default is 2 nm

    Returns
    -------
    GPA_resolution : TYPE
        DESCRIPTION.

    '''
    
    # Retrieve data extracted from the reference 
    crop_outputs_dict = analysed_image.Crop_outputs

    # Pixels within the whole image in which the crop of the reference is taken, 
    # so the box of the reference itself [first_row,last_row,first_col,last_col]
    scaled_reference_coords_GPA_ref = crop_outputs_dict[str(int(label_of_GPA_ref)) + '_pixel_ref_cords']

    # 
    image_crop_hs_signal_GPA_ref = crop_outputs_dict[str(int(label_of_GPA_ref)) + '_hs_signal']
    FFT_image_array_GPA_ref, _ =  ImCalTrans.Compute_FFT_ImageArray(np.asarray(image_crop_hs_signal_GPA_ref))

    crop_list_refined_cryst_spots_GPA_ref = crop_outputs_dict[str(int(label_of_GPA_ref)) + '_list_refined_cryst_spots']
    refined_pixels_GPA_ref = crop_outputs_dict[str(int(label_of_GPA_ref)) + '_refined_pixels']
    spots_int_reference_GPA_ref = crop_outputs_dict[str(int(label_of_GPA_ref)) + '_spots_int_reference']

    # Get the best cryst spot
    best_cryst_spot_GPA_ref = crop_list_refined_cryst_spots_GPA_ref[0]

    # Retrieve the best spot pair to be considered the GPA g vectors
    
    # Find the best spots considered in that crystal phase, which should 
    # be the best ones to constitute the GPA g vectors
    # and its information to help build the virutal crystal, although they
    #  need to be updated after the refining of the g vectors with the ref
    spot1_int_ref_GPA_ref = best_GPA_ref_spot_pair.spot1_int_ref
    spot2_int_ref_GPA_ref = best_GPA_ref_spot_pair.spot2_int_ref
    
    
    # Retrieve the info of the spots acting as g vectors
    hkl1_reference_GPA_ref = best_GPA_ref_spot_pair.hkl1_reference
    hkl2_reference_GPA_ref = best_GPA_ref_spot_pair.hkl2_reference
    spot1_dist_GPA_ref = best_GPA_ref_spot_pair.spot1_dist
    spot2_dist_GPA_ref = best_GPA_ref_spot_pair.spot2_dist
    angle_between_GPA_ref = best_GPA_ref_spot_pair.angle_between
    spot1_angle_x_GPA_ref = best_GPA_ref_spot_pair.spot1_angle_to_x
    spot2_angle_x_GPA_ref = best_GPA_ref_spot_pair.spot2_angle_to_x
    found_phase_name_GPA_ref = best_cryst_spot_GPA_ref.phase_name
    
    
    # coords of the best peaks to use as GPA g vectors in coordinartes of the crop
    # NOT of the whole image
    cord_spot_1_GPA_ref = refined_pixels_GPA_ref[int(spot1_int_ref_GPA_ref)]
    cord_spot_2_GPA_ref = refined_pixels_GPA_ref[int(spot2_int_ref_GPA_ref)]
    

    # image in dataset base
    image_array_whole = image_in_dataset_whole.image_arraynp_st
    total_pixels_whole = image_in_dataset_whole.total_pixels
    pixel_size_whole = image_in_dataset_whole.x_calibration
    FFT_calibration_whole = image_in_dataset_whole.FFT_calibration
    FFT_whole, _ =  ImCalTrans.Compute_FFT_ImageArray(np.asarray(image_in_dataset_whole.hyperspy_2Dsignal))


    # from spot 1 and 2 of the coords expressed relative to the crop size
    # get the spot that is further from the center, closer to the edges of
    # the image, as this is the one which will find the spot further and will
    # require a bigger mask
    closer_spot_ind = np.argmin(np.array([spot1_dist_GPA_ref, spot2_dist_GPA_ref]))
    closer_spot_dists = np.min(np.array([spot1_dist_GPA_ref, spot2_dist_GPA_ref]))
    closer_spot_coords = [cord_spot_1_GPA_ref, cord_spot_2_GPA_ref][closer_spot_ind]
    
    # generate a 9x9 pixel square arround the best coordinate scaled to the whole image
    # so from all this pixels we can find the best one that represents the maximum of
    # the peak
    
    scaled_cord_spot_1_GPA_y = int(np.round((cord_spot_1_GPA_ref[0]/FFT_image_array_GPA_ref.shape[0])*total_pixels_whole))
    scaled_cord_spot_1_GPA_x = int(np.round((cord_spot_1_GPA_ref[1]/FFT_image_array_GPA_ref.shape[1])*total_pixels_whole))
    scaled_cord_spot_2_GPA_y = int(np.round((cord_spot_2_GPA_ref[0]/FFT_image_array_GPA_ref.shape[0])*total_pixels_whole))
    scaled_cord_spot_2_GPA_x = int(np.round((cord_spot_2_GPA_ref[1]/FFT_image_array_GPA_ref.shape[1])*total_pixels_whole))
    
    if scaled_cord_spot_1_GPA_y-1 >= 0 and scaled_cord_spot_1_GPA_y+2 <= total_pixels_whole and scaled_cord_spot_1_GPA_x-1 >= 0 and scaled_cord_spot_1_GPA_x+2 <= total_pixels_whole:
        
        FFT_crop_spot_1 = FFT_whole[scaled_cord_spot_1_GPA_y-1:scaled_cord_spot_1_GPA_y+2,scaled_cord_spot_1_GPA_x-1:scaled_cord_spot_1_GPA_x+2]
        cords_max1 = np.where(FFT_crop_spot_1 == np.max(FFT_crop_spot_1))
    
        scaled_cord_spot_1_GPA_y = scaled_cord_spot_1_GPA_y - 1 + cords_max1[0][0]
        scaled_cord_spot_1_GPA_x = scaled_cord_spot_1_GPA_x - 1 + cords_max1[1][0]
        
    
    if scaled_cord_spot_2_GPA_y-1 >= 0 and scaled_cord_spot_2_GPA_y+2 <= total_pixels_whole and scaled_cord_spot_2_GPA_x-1 >= 0 and scaled_cord_spot_2_GPA_x+2 <= total_pixels_whole:
        
        FFT_crop_spot_2 = FFT_whole[scaled_cord_spot_2_GPA_y-1:scaled_cord_spot_2_GPA_y+2,scaled_cord_spot_2_GPA_x-1:scaled_cord_spot_2_GPA_x+2]
        cords_max2 = np.where(FFT_crop_spot_2 == np.max(FFT_crop_spot_2))
    
        scaled_cord_spot_2_GPA_y = scaled_cord_spot_2_GPA_y - 1 + cords_max2[0][0]
        scaled_cord_spot_2_GPA_x = scaled_cord_spot_2_GPA_x - 1 + cords_max2[1][0]
    # else:
    #     the coordinates are the same as they were
    
    spot_1_coords = np.array([scaled_cord_spot_1_GPA_y, scaled_cord_spot_1_GPA_x])
    spot_2_coords = np.array([scaled_cord_spot_2_GPA_y, scaled_cord_spot_2_GPA_x])    
        
    
    closer_spot_whole_coords = [spot_1_coords, spot_2_coords][closer_spot_ind]
    
    labels_unique = np.unique(image_segmented)
    labels_unique_no_ref = labels_unique[labels_unique != label_of_GPA_ref]
    
    
    
    # Sort the best crystals from each label containing crystals found
    # to analyse first the best crystals found and loop through the crystals 
    # from first the ones with lowest (best) scores to the highests
    
    labels_scored = []
    best_cryst_per_label_scored = []
    
    # these two lists correlate 1 to 1 so we loop through the labels
    # depending on the order of the best crysts found for each label
    
    for compl_label in labels_unique_no_ref:
        
        # Pixels within the whole image in which the crop of the reference is taken, 
        # so the box of the reference itself [first_row,last_row,first_col,last_col]
        scaled_reference_coords_compl_label = crop_outputs_dict[str(int(compl_label)) + '_pixel_ref_cords']
    
        # 
        image_crop_hs_signal_compl_label = crop_outputs_dict[str(int(compl_label)) + '_hs_signal']
        FFT_image_array_compl_label, _ =  ImCalTrans.Compute_FFT_ImageArray(np.asarray(image_crop_hs_signal_compl_label))
    
        crop_list_refined_cryst_spots_compl_label = crop_outputs_dict[str(int(compl_label)) + '_list_refined_cryst_spots']
        refined_pixels_compl_label = crop_outputs_dict[str(int(compl_label)) + '_refined_pixels']
        spots_int_reference_compl_label = crop_outputs_dict[str(int(compl_label)) + '_spots_int_reference']

        # if at least one crystal was found in that label
        if len(crop_list_refined_cryst_spots_compl_label) != 0:
            # sum the scores as done in the phase identificator, to check the 
            # best phase
            cryst_score = 0
            for spot_pair in crop_list_refined_cryst_spots_compl_label[0].spot_pairs_obj:
                spot_pair_score = spot_pair.score
                cryst_score = cryst_score + spot_pair_score
            total_spot_pairs = len(crop_list_refined_cryst_spots_compl_label[0].spot_pairs_obj)
                
            best_cryst_per_label_scored.append(cryst_score/total_spot_pairs**2)
            labels_scored.append(compl_label)
    
    
    # if no other phase was found in the process, so just one crystal the refrence one
    if len(best_cryst_per_label_scored) == 0:
        GPA_resolution = 4   # nm
        return GPA_resolution

    # else, if at least one phase was found, do all the checks  
    
    # Sort the labels list based on the score for the crystals found for each
    # label being the lowest label the best crystal
    labels_scored_sorted = [label for _ , label in sorted(zip(best_cryst_per_label_scored, labels_scored), key=lambda x: x[0])]


    # loop throuhg the labels sorted by their best crystal score
    for compl_label in labels_scored_sorted:
        
        # We want to store the coordinates found for the labels which have the
        # best crystals in terms of its score
        # we append all the coordinates found for that crop, not only the best 
        # crystal but hopefully the best coordinates that are found within
        # the mask rage are from the best crystal, 
        # this would also tackle the case in which another crystal in different 
        # orietnation is not the main one found and produces a better epitaxy
        # so in this case it would be found as the one defining the GPA information
        all_possible_coords_to_check = []

        # Pixels within the whole image in which the crop of the reference is taken, 
        # so the box of the reference itself [first_row,last_row,first_col,last_col]
        scaled_reference_coords_compl_label = crop_outputs_dict[str(int(compl_label)) + '_pixel_ref_cords']
    
        # 
        image_crop_hs_signal_compl_label = crop_outputs_dict[str(int(compl_label)) + '_hs_signal']
        FFT_image_array_compl_label, _ =  ImCalTrans.Compute_FFT_ImageArray(np.asarray(image_crop_hs_signal_compl_label))
    
        crop_list_refined_cryst_spots_compl_label = crop_outputs_dict[str(int(compl_label)) + '_list_refined_cryst_spots']
        refined_pixels_compl_label = crop_outputs_dict[str(int(compl_label)) + '_refined_pixels']
        spots_int_reference_compl_label = crop_outputs_dict[str(int(compl_label)) + '_spots_int_reference']
    

        # if at least one crystal was found in that label
        if len(crop_list_refined_cryst_spots_compl_label) != 0:
            # store the coordinates of spots
            # convert them relative to the whole image
            scaled_refined_pixels_compl_label = (refined_pixels_compl_label/FFT_image_array_compl_label.shape[0])*total_pixels_whole
            all_possible_coords_to_check = all_possible_coords_to_check + list(scaled_refined_pixels_compl_label)
            

        # now check que coordinate that is closer to closer_spot_whole_coords        
        all_possible_coords_to_check = np.asarray(all_possible_coords_to_check) 
       
        distances_y = np.abs(all_possible_coords_to_check[:,0] - closer_spot_whole_coords[0])*FFT_calibration_whole
        distances_x = np.abs(all_possible_coords_to_check[:,1] - closer_spot_whole_coords[1])*FFT_calibration_whole
    
        distances_in_nminv = np.sqrt(distances_y**2 + distances_x**2)
        distances_in_nm = 1/distances_in_nminv
    
        # turn these pixel distances into nm
    
        # define the threshold in nm of maximum separation between the spots 
        # to be considered a possible heterostructure GPA_spot_max_dist
        # as we center the calculation f the distances to the spot used as reference
        # then the spots closer to this spot are the ones with higher distance in nm
        
        distances_close_enough = distances_in_nm[distances_in_nm >= GPA_spot_max_dist]
        
        
        
        if len(distances_close_enough) != 0:
            # get the minimum one and define res bsed on this
            distances_close_enough = np.sort(distances_close_enough)[::-1]
            #Ü divide by 2 as it is the radius not the diameter
            GPA_resolution = (distances_close_enough[0] + distances_close_enough[0]*0.2)/2
            
            # Exception if the pixels found for the rest of found phases are 
            # exactly the same and then the resoluion is infiinite , limit 
            # this to the 4nm
            if GPA_resolution > 4:
                GPA_resolution = 4
            
            return GPA_resolution
            
        else:
            
            # try the same with the other spot that is not the closest one
            # and if not just define it manually
            closer_spot_whole_coords = [spot_2_coords, spot_1_coords][closer_spot_ind]
            
            distances_y = np.abs(all_possible_coords_to_check[:,0] - closer_spot_whole_coords[0])*FFT_calibration_whole
            distances_x = np.abs(all_possible_coords_to_check[:,1] - closer_spot_whole_coords[1])*FFT_calibration_whole
    
            distances_in_nminv = np.sqrt(distances_y**2 + distances_x**2)
            distances_in_nm = 1/distances_in_nminv
    
            # turn these pixel distances into nm
    
            # define the threshold in nm of maximum separation between the spots 
            # to be considered a possible heterostructure
            
            # as we center the calculation f the distances to the spot used as reference
            # then the spots closer to this spot are the ones with higher distance in nm
            
            distances_close_enough = distances_in_nm[distances_in_nm >= GPA_spot_max_dist]
            
            if len(distances_close_enough) != 0:
                # get the minimum one and define res bsed on this
                distances_close_enough = np.sort(distances_close_enough)[::-1]
                #Ü divide by 2 as it is the radius not the diameter
                GPA_resolution = (distances_close_enough[0] + distances_close_enough[0]*0.2)/2
                            
                # Exception if the pixels found for the rest of found phases are 
                # exactly the same and then the resoluion is infiinite , limit 
                # this to the 4nm
                if GPA_resolution > 4:
                    GPA_resolution = 4

                return GPA_resolution
                
            else:
    
                # then not heterostructure, as a result define a smaller mask
                GPA_resolution = 4   # nm
                
            return GPA_resolution
    
    


