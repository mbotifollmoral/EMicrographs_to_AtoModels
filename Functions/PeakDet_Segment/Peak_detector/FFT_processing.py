#Import packages

import numpy as np
import matplotlib.pyplot as plt
import hyperspy.api as hs
import os
import time
import cv2
import ImageTreatment_and_interfaces as ii

''' Experimental treatment '''

#Calculate experimental FFT
def experimental_fft(sample, window_size = 2048, zone = [0,0]):
    
 #Charge dm3
    if type(sample) == str:
        imagedm3=hs.load(sample)
        meta1=imagedm3.metadata
        meta2=imagedm3.original_metadata.export('parameters')

        #FFT calibration
        x_calibration=imagedm3.axes_manager['x'].scale
        y_calibration=imagedm3.axes_manager['y'].scale

        x_pixels_original=imagedm3.axes_manager['x'].size
        y_pixels_original=imagedm3.axes_manager['y'].size

        x_units=imagedm3.axes_manager['x'].units
        y_units=imagedm3.axes_manager['y'].units

        #FFT_calibration,FFT_pixels,FFT_units=FFT_calibration(imagedm3)

        #Samples images treatment
        imagearray=np.asarray(imagedm3)
        total_image = imagearray
    else: total_image = sample
    #image_square = square(total_image, 256, zone)

    np.random.seed(int(time.time()))

    #Crop the image if wanted
    init_y = zone[1]
    init_x = zone[0]

    if init_x > len(total_image) - window_size : init_x = len(total_image) - window_size  - 1 
    if init_y > len(total_image) - window_size : init_y = len(total_image) - window_size  - 1 
    if init_x < 0: init_x = 0
    if init_y < 0: init_y = 0
    
    x = len(total_image) - init_y 
    y = init_x

    #image=image[init_y:init_y+window_size,init_x:init_x+window_size]
    hs_image_cropping = total_image[x - window_size : x, y: y + window_size]


    #Correct calibration in case it is necessary, if no correction is needed, just us 1
    real_calibration_factor=1

    image=hs_image_cropping

    #First standarisation of the image for filtering/blurring it with gaussian filter
    image_st=(image-np.min(image))/np.max(image-np.min(image))

    #Show samples images
    

    #Compute FFT for image and respective prediction

    denoised_image=image_st
    #denoised_image=cv2.GaussianBlur(image_st, gauss_blur_filter_size, 1)

    #Second standarisation of the image after filtering/blurring it with gaussian filter

    image_st=(denoised_image-np.min(denoised_image))/np.max(denoised_image-np.min(denoised_image))

    #Then downsample the image and upsample it posteriorly
    #We select a max pooling method to keep track of the brighter elements and this way keep a 
    #higher contrast

    ds_image=image_st

    #and standarise it again to ensure 0-1 values

    ds_image_st=(ds_image-np.min(ds_image))/np.max(ds_image-np.min(ds_image))


    # take the fft of the image
    #fft_image_w_background = np.fft.fftshift(np.log(np.fft.fft2(ds_image_st)))
    fft_image_w_background = np.fft.fftshift((np.fft.fft2(ds_image_st)))
    fft_abs_image_background = np.abs(fft_image_w_background)

    # apply the filter
    fft_abs_image_backgroundc=np.copy(fft_abs_image_background)
    fft_abs_image_background2=np.copy(fft_abs_image_background)

    fft_abs_image_backgroundc=(fft_abs_image_backgroundc-np.min(fft_abs_image_backgroundc))/np.max(fft_abs_image_backgroundc-np.min(fft_abs_image_backgroundc))
    fft_abs_image_background2=(fft_abs_image_background2-np.min(fft_abs_image_background2))/np.max(fft_abs_image_background2-np.min(fft_abs_image_background2))
    gauss_blur_filter_size=(5,5)
    #fft_abs_image_background2=cv2.GaussianBlur(fft_abs_image_background2, gauss_blur_filter_size, 1)
    fft_abs_image_background2=(fft_abs_image_background2-np.min(fft_abs_image_background2))/np.max(fft_abs_image_background2-np.min(fft_abs_image_background2))


    #fft_abs_image_background2_c= cv2.resize(fft_abs_image_background2, dsize=(2048, 2048), interpolation=cv2.INTER_CUBIC)
    #fft_abs_image_background2_n= cv2.resize(fft_abs_image_background2, dsize=(2048, 2048), interpolation=cv2.INTER_NEAREST)
    #fft = cv2.resize(fft_abs_image_background2, dsize=(2048, 2048), interpolation=cv2.INTER_LINEAR) 
    fft = fft_abs_image_background2
    
    return fft

def experimental_fft_with_log(sample, window_size, zone):

    #Charge dm3
    if type(sample) == str:
        imagedm3=hs.load(sample)
        meta1=imagedm3.metadata
        meta2=imagedm3.original_metadata.export('parameters')

        #FFT calibration
        x_calibration=imagedm3.axes_manager['x'].scale
        y_calibration=imagedm3.axes_manager['y'].scale

        x_pixels_original=imagedm3.axes_manager['x'].size
        y_pixels_original=imagedm3.axes_manager['y'].size

        x_units=imagedm3.axes_manager['x'].units
        y_units=imagedm3.axes_manager['y'].units

        #FFT_calibration,FFT_pixels,FFT_units=FFT_calibration(imagedm3)

        #Samples images treatment
        imagearray=np.asarray(imagedm3)
        total_image = imagearray
    else: total_image = sample
    image_square = ii.square(total_image, window_size, zone)

    np.random.seed(int(time.time()))

    #Crop the image if wanted
    init_y = zone[1]
    init_x = zone[0]

    if init_x > len(total_image) - window_size : init_x = len(total_image) - window_size  - 1 
    if init_y > len(total_image) - window_size : init_y = len(total_image) - window_size  - 1 
    if init_x < 0: init_x = 0
    if init_y < 0: init_y = 0
    
    x = len(total_image) - init_y 
    y = init_x

    #image=image[init_y:init_y+window_size,init_x:init_x+window_size]
    hs_image_cropping = total_image[x - window_size : x, y: y + window_size]


    #Correct calibration in case it is necessary, if no correction is needed, just us 1
    real_calibration_factor=1

    image=np.asarray(hs_image_cropping)

    #First standarisation of the image for filtering/blurring it with gaussian filter
    image_st=(image-np.min(image))/np.max(image-np.min(image))

    #Show samples images
    

    #Compute FFT for image and respective prediction

    denoised_image=image_st
    #denoised_image=cv2.GaussianBlur(image_st, gauss_blur_filter_size, 1)

    #Second standarisation of the image after filtering/blurring it with gaussian filter

    image_st=(denoised_image-np.min(denoised_image))/np.max(denoised_image-np.min(denoised_image))

    #Then downsample the image and upsample it posteriorly
    #We select a max pooling method to keep track of the brighter elements and this way keep a 
    #higher contrast

    ds_image=image_st

    #and standarise it again to ensure 0-1 values

    ds_image_st=(ds_image-np.min(ds_image))/np.max(ds_image-np.min(ds_image))


    # take the fft of the image
    fft_image_w_background = np.fft.fftshift(np.log(np.fft.fft2(ds_image_st)))
    fft_abs_image_background = np.abs(fft_image_w_background)

    # apply the filter
    fft_abs_image_backgroundc=np.copy(fft_abs_image_background)
    fft_abs_image_background2=np.copy(fft_abs_image_background)

    fft_abs_image_backgroundc=ii.normalize_image(fft_abs_image_backgroundc)
    fft_abs_image_background2=ii.normalize_image(fft_abs_image_background2)
    
    gauss_blur_filter_size=(5,5)
    #fft_abs_image_background2=cv2.GaussianBlur(fft_abs_image_background2, gauss_blur_filter_size, 1)
    fft_abs_image_background2=ii.normalize_image(fft_abs_image_background2)


    #fft_abs_image_background2_c= cv2.resize(fft_abs_image_background2, dsize=(2048, 2048), interpolation=cv2.INTER_CUBIC)
    #fft_abs_image_background2_n= cv2.resize(fft_abs_image_background2, dsize=(2048, 2048), interpolation=cv2.INTER_NEAREST)
    fft = cv2.resize(fft_abs_image_background2, dsize=(2048, 2048), interpolation=cv2.INTER_LINEAR)
    
 
    return image_square, image_st, fft

def standarize_fft(signal, fft):
    opt_scale = 0.025
    scale = signal.axes_manager['x'].scale
    
    ratio = opt_scale/scale
    exp = int(np.log2(ratio))
    real_size = len(fft)
    estandarize_size = int(real_size/(2**exp))
    # print(real_size,estandarize_size, scale)
    
    lim_l = (real_size - estandarize_size)//2
    lim_r = (real_size + estandarize_size)//2
    
    standarize_fft = fft[lim_l : lim_r, lim_l : lim_r]
    
    return  standarize_fft


''' FFT noise model functions '''

#Drift to a FFT simulated spectre

def drift(s,theta,matrix):
    
    total_pixels = len(matrix)
    m = np.zeros((total_pixels,total_pixels))
    centre = total_pixels/2
    cos , sin = np.cos(theta) , np.sin(theta)
    index = np.where(matrix == 1)
    
    new_x = index[1] - centre
    new_y = centre - index[0] 

    drift_x = new_x*cos-new_y*(s*cos-sin)
    drift_y = -new_x*sin+new_y*(s*sin+cos)

    x = np.uint16(centre - drift_y)
    y = np.uint16(centre + drift_x)

    coordinates = x, y
    m[coordinates] = 1
    
    return m


#Transform puntual atoms to Gaussian

def puntual_to_gaussian(sigma,matrix):
    
    total_pixels = len(matrix)
    index = np.where(matrix != 0)
    
    if sigma == 0:
        m = matrix
        
    else:
        
        m = np.zeros((total_pixels,total_pixels))
        s = 2*sigma*sigma
        mux = index[0]
        muy = index[1]

   
        mj = np.arange(total_pixels) 
        mj = np.asarray([mj]*total_pixels)
        mi = mj.T  
    
        for k in range(len(mux)): 
            m = m + np.exp(-((mi-mux[k])**2+(mj-muy[k])**2)/s) * matrix[mux[k],muy[k]]
    
    return m, index


#Distorsion Functions to each single atom. Number 2 is a improvement of the first.

def total_distorsion(distorsion_x, distorsion_y,m):

    size_x = m[0].size
    size_y = m.size // size_x 
    matrix = m*0
    
    for i in range(size_x):
        random_values_x = np.random.normal(0,distorsion_x,size_x)
        random_values_y = np.random.normal(0,distorsion_y,size_y)
     
        for j in range(size_y):
            new_i = i + int(random_values_x[j])
            new_j = j + int(random_values_y[j])
            
            if(new_i >= size_x): 
                new_i = size_x - 1
     
            elif(new_i < 0):
                new_i= 0
              
            if(new_j >= size_y):
                new_j = size_y - 1
   
            elif(new_j < 0):
                new_j = 0
                
            matrix[j][i] = m[new_j][new_i]
            
    return matrix
                
    
def total_distorsion2(distorsion_x, distorsion_y,m):

        
    size = m[0].size
    matrix = m*0
    #matrix = np.zeros((size,size))

    for j in range(size):
        random_values_x = np.random.normal(0,distorsion_x)
        random_values_y = np.random.normal(0,distorsion_y,size)

        for i in range(size):
            new_i = i + int(random_values_x)
            new_j = j + int(random_values_y[i])

            if(new_i >= size-1):
                new_i = size - 1

            elif(new_i < 0):
                new_i= 0

            if(new_j >= size-1):
                new_j = size- 1

            elif(new_j < 0):
                new_j = 0

            matrix[j][i] = m[new_j][new_i]

  
        
    return matrix
                    
#Function to recut atoms

def cut_points(sigma,index,matrix):
    
    min_intensity = 10
    k = int(sigma*np.sqrt(2*np.log(255/min_intensity)))+1
    zoom_matrix = []
    
    for i,j in zip(index[0],index[1]):
        zoom_matrix.append(matrix[i-k:i+k+1,j-k:j+k+1])
 
    return zoom_matrix


#Fuction to apply distorsions to all atoms in the image

def points_distorsion(distorsion_x,distorsion_y,index,sigma,total_matrix):
    
    matrix  = total_matrix + 0
    distorsion_point  = []
    zoom_matrix = cut_points(sigma,index, total_matrix)
    
    for i in range(len(zoom_matrix)):
        if zoom_matrix[i].size > 0:
            size = zoom_matrix[i][0].size
            break
      
    for i, point in enumerate(zoom_matrix):
        distorsion_point.append(total_distorsion2(distorsion_x,distorsion_y, point))
        new_i = index[0][i] - size//2
        new_j = index[1][i] - size//2
        matrix[new_i : new_i + size, new_j : new_j + size] = distorsion_point[i]
                                              
    distorsion_point.append(matrix)  
    
    return distorsion_point


#Functions to find centers of the atoms.Number 2 is a improvement of the first.

def find_picks(intensity,resolution,image_matrix):

    total_pixels = len(image_matrix)
    matrix_treshold = np.zeros((total_pixels,total_pixels))
    resolution = int(resolution)
    
    picks = np.where(image_matrix > intensity)
    matrix_treshold[picks] = 1
    matrix_treshold = matrix_treshold * image_matrix
    
    if resolution != 0 :
        sin_marcos = np.zeros((total_pixels + 2*resolution,total_pixels + 2*resolution))
        sin_marcos[resolution : total_pixels + resolution, 
                   resolution : total_pixels + resolution ] = matrix_treshold + 0
        picks = np.where(sin_marcos > intensity)
                                   
        for i,j in zip(picks[0],picks[1]):
            
            recuadro = sin_marcos[i-resolution : i + resolution, 
                                  j - resolution : j + resolution]
            
            if (sin_marcos[i][j] < np.amax(recuadro)): sin_marcos[i][j] = 0
                    
        matrix_treshold = sin_marcos[resolution : total_pixels + resolution, 
                                     resolution : total_pixels + resolution ]   
        
        noise_matrix =  image_matrix- matrix_treshold       
                
    return matrix_treshold, noise_matrix


def find_picks2(intensity,resolution,image_matrix):
    
    total_pixels = len(image_matrix)
    matrix_treshold = np.zeros((total_pixels,total_pixels))
    noise_matrix = np.zeros((total_pixels,total_pixels))
    resolution = int(resolution)
    
    picks = np.where(image_matrix > intensity)
    matrix_treshold[picks] = 1
    matrix_treshold = matrix_treshold * image_matrix

    if resolution != 0 :
        sin_marcos = np.zeros((total_pixels,total_pixels))
        sin_marcos[resolution : total_pixels - resolution, 
                   resolution : total_pixels - resolution ] = 1
        matrix_treshold = matrix_treshold * sin_marcos
                                   
        
        for i,j in zip(picks[0],picks[1]):
            
            if (i-resolution >= 0) & (j-resolution >= 0) & (j+resolution < total_pixels) & (i+resolution < total_pixels):
            
                recuadro = matrix_treshold[i-resolution : i + resolution, 
                                   j - resolution : j + resolution]
            
                if (matrix_treshold[i][j] < np.amax(recuadro)):
                    matrix_treshold[i][j] = 0
                    
        noise_matrix =  image_matrix- matrix_treshold 
        
    return matrix_treshold, noise_matrix


#Function to apply all noise filters in a image

def noise_filters(fft, s, t, sigma, l, hd, vd):
    
    total_pixels = len(fft)

    poisson_matrix = np.random.poisson(l, size=(total_pixels, total_pixels))/255
    drift_matrix = drift(s,np.radians(t),fft)
    noise_matrix, index = puntual_to_gaussian(sigma, drift_matrix)
    total_matrix = noise_matrix + poisson_matrix
    
    all_matrix = points_distorsion(hd, vd, index, sigma, total_matrix)
    #all_matrix = total_matrix, total_matrix, total_matrix, total_matrix[0:10,0:10], total_matrix
    
    return all_matrix         




''' FFT simulation '''

#Compute distances and angles
def distances_angles(name_cell,eje,min_d,res,res_value):
    
    Silicon = tl.Crystal(bytes('Cells/'+name_cell, 'utf-8'))
    #Silicon = Crystal(b'Cells/Ge.uce')
    ZA = np.array(eje, dtype=np.int32)
    DPsize = Silicon.calcKineDP(ZA[0], ZA[1], ZA[2], min_d)
    total_DP=[]
    
    for ref in range(DPsize):
        hkls=Silicon.getIndexes(ref)
        total_DP.append(hkls)

    hkl1=total_DP[0]
    angles=[]
    distances=[]
    Fs=[]

    for index_hkl2, hkl2 in enumerate(total_DP):
        angle=Silicon.kineDP_angles(hkl1, hkl2)
        distance=Silicon.getDistances(index_hkl2)
        F=Silicon.getF(index_hkl2)
        distances.append(distance)
        angles.append(angle)
        Fs.append(F)

    angles=np.asarray(angles)
    distances=np.asarray(distances)
    Fs=np.asarray(Fs)
    angles.shape=(np.shape(angles)[0],1)
    distances.shape=(np.shape(distances)[0],1)
    Fs.shape=(np.shape(Fs)[0],1)

    distances_angles=np.hstack((distances, angles))
    distances_angles_Fs=np.hstack((distances_angles, Fs))

    total_DP=np.asarray(total_DP)
    #delete these reflections that are repeated twice or more
    temp_array_reflects_s = np.ascontiguousarray(total_DP).view(np.dtype((np.void, total_DP.dtype.itemsize * total_DP.shape[1])))
    _, idx,counts = np.unique(temp_array_reflects_s, return_index=True, return_counts=True)
    total_DP_reduced=total_DP[idx]

    distances_angles_Fs_reduced=[]
    for red_reflection in total_DP_reduced:
        for index_tot, tot_reflection in enumerate(total_DP):
            if red_reflection[0]==tot_reflection[0] and red_reflection[1]==tot_reflection[1] and red_reflection[2]==tot_reflection[2]:
                distances_angles_Fs_reduced.append(distances_angles_Fs[index_tot])
                break

    total_DP=list(total_DP_reduced)
    distances_angles_Fs=distances_angles_Fs_reduced
    distances_angles_Fs_r=[distances_angles_Fs_el for distances_angles_Fs_el in sorted(distances_angles_Fs, key=lambda x: x[0])][::-1]
    total_DP=[reflection_el for reflection_el,_ in sorted(zip(total_DP,distances_angles_Fs), key=lambda x: x[1][0])][::-1]

    distances_angles_Fs=np.asarray(distances_angles_Fs_r) 

    if res:
        index_res=len(distances_angles_Fs[:,0][distances_angles_Fs[:,0]>res_value])
        distances_angles_Fs=distances_angles_Fs[:index_res,:]

        
    return distances_angles_Fs




#Calibrations and distance functions

def Calib_By_FOV(FOV, total_pixels):

    hyperspy_2D_signal=hs.signals.Signal2D(np.random.random((total_pixels,total_pixels)))
    hyperspy_2D_signal.axes_manager.set_signal_dimension(2)
    hyperspy_2D_signal.axes_manager[0].name='x'
    hyperspy_2D_signal.axes_manager[1].name='y'
    pixel_size=FOV/total_pixels
    hyperspy_2D_signal.axes_manager['x'].scale=pixel_size
    hyperspy_2D_signal.axes_manager['y'].scale=pixel_size
    hyperspy_2D_signal.axes_manager['x'].units='nm'
    hyperspy_2D_signal.axes_manager['y'].units='nm'
    fft_shifted = hyperspy_2D_signal.fft(shift=True)
    FFT_calibration=fft_shifted.axes_manager['x'].scale
  
    return FFT_calibration

def Calib_By_pixel_size(pixel_size, total_pixels):
    
    hyperspy_2D_signal=hs.signals.Signal2D(np.random.random((total_pixels,total_pixels)))
    hyperspy_2D_signal.axes_manager.set_signal_dimension(2)
    hyperspy_2D_signal.axes_manager[0].name='x'
    hyperspy_2D_signal.axes_manager[1].name='y'
    hyperspy_2D_signal.axes_manager['x'].scale=pixel_size
    hyperspy_2D_signal.axes_manager['y'].scale=pixel_size
    hyperspy_2D_signal.axes_manager['x'].units='nm'
    hyperspy_2D_signal.axes_manager['y'].units='nm'
    fft_shifted = hyperspy_2D_signal.fft(shift=True)
    FFT_calibration=fft_shifted.axes_manager['x'].scale
    
    return FFT_calibration


# Compute coordinates of fft

def Distances_angles_to_pixels(distances_angles_Fs, angle_offset, FFT_calib, total_pixels):
    
    #by convention, let us define that positive angles mean that plane is set clockwise with respect the first reflexion
    
    pixels_angles_Fs=np.copy(distances_angles_Fs)
    
    pixels_angles_Fs[:,1]=(angle_offset-pixels_angles_Fs[:,1])*(np.pi/180)
    
    #distances in nm
    pixels_angles_Fs[:,0]=pixels_angles_Fs[:,0]/10
    reciprocal_distances=1/(pixels_angles_Fs[:,0])
    reciprocal_x=reciprocal_distances*np.cos(pixels_angles_Fs[:,1])
    reciprocal_y=reciprocal_distances*np.sin(pixels_angles_Fs[:,1])
    xs_pixels=np.floor(reciprocal_x/(FFT_calib))
    ys_pixels=np.floor(reciprocal_y/(FFT_calib))
    
    x_diff_center=int(total_pixels/2)
    y_diff_center=int(total_pixels/2)
    
    final_x_coords=xs_pixels+x_diff_center
    final_y_coords=y_diff_center-ys_pixels
    xy_cords_Fs=np.vstack((final_x_coords,final_y_coords))
    xy_cords_Fs=np.vstack((xy_cords_Fs, pixels_angles_Fs[:,2])).T
    #check if the pixels are too far away from FOV, if the coords are bigger than total pixels
 
    indexes=[]
    for index, coord in enumerate(xy_cords_Fs):
        if coord[0]> total_pixels-1 or coord[1]> total_pixels-1 or coord[0]<0 or coord[1]<0:
            indexes.append(index)
            
    xy_cords_Fs=np.delete(xy_cords_Fs, indexes, 0)
   
    return xy_cords_Fs


#Compute FFT and Real Image

def fft_and_crystal_images(distances_angles_Fs,pixel_size, total_pixels,angle_mainrefl_to_x):
    
    fft = np.zeros((total_pixels,total_pixels))
    FFT_calib = Calib_By_pixel_size(pixel_size, total_pixels)
    #FFT_calib = Calib_By_FOV(fov, total_pixels)
    
    xy_cords_Fs=Distances_angles_to_pixels(distances_angles_Fs, angle_mainrefl_to_x, FFT_calib, total_pixels)
    xys_cords=np.asarray(xy_cords_Fs[:,0:2], dtype=np.int32)

    for coord in xys_cords:
        x_cord=int(coord[0])
        y_cord=int(coord[1])

        fft[y_cord, x_cord]=1


    real_image = np.abs(np.fft.fftshift(np.fft.ifft2(fft)))
    
    return fft, real_image


#Generate model

def noise_model(fft_image,real_image,s,theta,sigma,lamda,hd,vd,scale, total_pixels,resolution = 6,intensity = 50):

    picks1_matrix = np.zeros((total_pixels, total_pixels))
    picks2_matrix = np.zeros((total_pixels, total_pixels))
    filtro_gaussiano = np.zeros((total_pixels, total_pixels))
    filtro_gaussiano[total_pixels//2][total_pixels//2] = 1


    '''Drift en la FFT'''

    drift_matrix = drift(s,np.radians(theta),fft_image)

    '''Crear imagen real'''

    real_image = np.fft.fftshift(np.fft.ifft2(drift_matrix))
    real_image_normalized = ivi.normalize_image(np.abs(real_image))

    '''Modificar imagen real'''

    real_image_treshold, real_image_noise = find_picks2(0,resolution,real_image_normalized)
    picks1 = np.where(real_image_treshold != 0)
    picks1_matrix[picks1] = 1
    picks2 = np.where(real_image_noise != 0)
    picks2_matrix[picks2] = 1

    real_image_treshold = real_image * picks1_matrix
    real_image_noise = real_image * picks2_matrix

    gaussian_matrix, index = puntual_to_gaussian(sigma, real_image_treshold)
    poisson_matrix = np.random.poisson(lamda, size=(total_pixels, total_pixels))/255*(1+1j)
    total_matrix = gaussian_matrix 
    distorsion_matrix = points_distorsion(hd, vd, index, sigma, total_matrix)
    final_matrix = distorsion_matrix[-1] + real_image_noise

    '''Volver a la FFT'''

    final_fft = np.fft.fft2(final_matrix)
    
    final_fft = final_fft * puntual_to_gaussian(intensity, filtro_gaussiano)[0]
    final_matrix1  = final_matrix/np.amax(np.abs(final_matrix)) + poisson_matrix
    final_matrix = np.fft.fftshift(np.fft.ifft2(final_fft)) 
    final_matrix = final_matrix/np.amax(np.abs(final_matrix)) + poisson_matrix
    final_fft = np.fft.fft2(final_matrix)
    final_fft = np.abs(final_fft)
    final_fft = np.log(final_fft)
    final_fft = ivi.normalize_image(final_fft)**scale
    final_fft = ivi.normalize_image(final_fft)
    drift_matrix[total_pixels//2][total_pixels//2] = 1
    final_fft[total_pixels//2][total_pixels//2] = np.max(final_fft)

    return drift_matrix, final_fft, final_matrix1,gaussian_matrix
                    

def save_simulated_spectra(name_file,size,total_pixels,scale,min_d,res,res_value,angle_mainrefl_to_x):
    
    t0 = time.time()
    labels = []
    noise = []
    j = 0
    
    '''Cristales y direcciones'''
    
    cells = os.listdir('Cells')
    
    ejes = [[0,0,1], [0,1,0], [1,0,0], [0,1,1], [1,0,1], [1,1,0], [1,1,1]]
    
    size_pixel = [0.02, 0.03, 0.04]
    
    '''Cálculo FFT'''
    
    for cell in cells:
        
        for eje in ejes:
           
            distances_angles_Fs = distances_angles(cell,eje,min_d,res,res_value)
            
            for pixel_size in size_pixel:
    
                fft_image, real_image = fft_and_crystal_images(distances_angles_Fs,pixel_size, total_pixels,angle_mainrefl_to_x)


                '''Variables del modelo'''

                theta_value = 0
                theta_dispersion = 0.91
                theta = np.random.normal(theta_value,theta_dispersion,size)

                s_value = 0
                s_dispersion = 0.06
                s = np.random.normal(s_value,s_dispersion,size)

                sigma_value = 1.5
                sigma_dispersion = 0.15
                sigma = np.random.normal(sigma_value,sigma_dispersion,size)

                hd_value = 0.8
                hd_dispersion = 0.06
                hd = np.random.normal(hd_value,hd_dispersion,size)

                vd_value = 0.4
                vd_dispersion = 0.06
                vd = np.random.normal(vd_value,vd_dispersion,size)

                lambda_value = 30
                lambda_dispersion = 6.06
                lambda_ = np.random.normal(lambda_value,lambda_dispersion,size)


                '''Creación imágenes y labels'''

                for i in range(size):

                    drift_matrix, final_fft, real_image_final,gaussian_matrix= noise_model(fft_image, real_image,
                                                                            s[i],theta[i],sigma[i],lambda_[i],
                                                                            hd[i],vd[i],scale,total_pixels)

                    '''Añadir a la lista '''

                    labels.append(drift_matrix) 
                    noise.append(final_fft)
                    
                    t = int(time.time() - t0)
                    total = size*len(ejes)*len(size_pixel)*len(cells)
                    tiempo = str(t//3600)+':'+str((t//60)%60) +':'+ str(t%60)
                    ivi.barra_de_carga(j,total,tiempo)
                    j += 1

                    #print(cell, eje, i,pixel_size, tiempo)

            np.savez('Models/'+name_file+'.npz', x = noise, y = labels)
