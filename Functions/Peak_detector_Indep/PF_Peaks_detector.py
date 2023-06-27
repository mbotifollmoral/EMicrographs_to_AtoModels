import numpy as np
from scipy.signal import find_peaks
import cv2
import matplotlib.pyplot as plt
import os
import sys

Project_main_path = os.getcwd()
if 'EMicrographs_to_AtoModels' in Project_main_path:
    Project_main_path = Project_main_path[:Project_main_path.find('EMicrographs_to_AtoModels')-1]
# Project_main_path has the EMicrographs_to_AtoModels folder
sys.path.append(Project_main_path)

from EMicrographs_to_AtoModels.Functions.Peak_detector_Indep import PF_ImageTreatment_and_interfaces as ii


#Global variables
treshold = 20
delta = 10

'''
Functions for the 1D scanning peak finding
'''

# Function to convert puntual signal in circular
def cercles(sigma, matrix):
   
    index = np.where(matrix > 0)
    total_pixels = len(matrix)
    image = np.zeros((total_pixels,total_pixels))

    if (sigma == 0) or (len(index[0])>100):
        image = matrix
        
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
            
        index = np.where(m>0.038)
        image[index] = 1
        
    return image

    
    
#Algorithm to detect peaks in a FFT spectrum
def peaks_detector(img, delta = delta, treshold = treshold):
    
    #Peak detector v1.0
    
    fft = img.copy()
    m = int(len(fft)/2)
    
    # !!! Marc's modification
    #size = 2*m
    size = len(fft)

    fft[m - delta : m + delta, m - delta : m + delta] = 0
    image = fft.copy()
    
    peaks_image_v = np.zeros((size,size))
    peaks_image_h = np.zeros((size,size))

    th = treshold + 10*np.log2(size/256)
    
    for i in range(size):
        zone = i

        y = image[:,zone : zone + 2]
        y = y[:,0]

        if np.max(image)/th > np.max(y)/4:
            limit = np.max(image)/th
        else:
            limit = np.max(y)/4

        peaks, _ = find_peaks(y, height = limit, distance = 5)

        for peak in peaks:
            peaks_image_v[peak, zone] = 1


        y = image[zone : zone + 2]
        y = y[0]

        if np.max(image)/th > np.max(y)/4:
            limit = np.max(image)/th
        else:
            limit = np.max(y)/4

        peaks, _ = find_peaks(y, height = limit, distance = 10)

        for peak in peaks:
            peaks_image_h[zone,peak] = 1
            
    peaks_image = peaks_image_v * peaks_image_h
    remove_center(peaks_image)
    
    return  peaks_image


# Extract the pixel coordinates from the binary matrix
def peaks_image_to_coords(
        peaks_image):
    
    pixels_of_peaks_y = np.where(peaks_image == 1)[0]
    pixels_of_peaks_x = np.where(peaks_image == 1)[1]
    
    
    peaks_pixel_coords = np.vstack((pixels_of_peaks_y, pixels_of_peaks_x)).T
    
    return peaks_pixel_coords
    
# Get coordinates after the peak finding in standarised FFT
def standarised_peaks_image_to_coords(
        peaks_image, FFT_no_log, lim_l, lim_r):
    
    standarised_FFT_size = peaks_image.shape[0]
    original_FFT_size = FFT_no_log.shape[0]
    
    # compare both sizes to address the enlarging or smalling of the FFTs
    if standarised_FFT_size == original_FFT_size:
        
        pixels_of_peaks_y = np.where(peaks_image == 1)[0]
        pixels_of_peaks_x = np.where(peaks_image == 1)[1]
        
        peaks_pixel_coords = np.vstack((pixels_of_peaks_y, pixels_of_peaks_x)).T
    
    elif standarised_FFT_size > original_FFT_size:
        # the original image is smaller than the standarised one
        
        complete_peaks_image = peaks_image[lim_l : lim_r, lim_l : lim_r]
        
        pixels_of_peaks_y = np.where(complete_peaks_image == 1)[0]
        pixels_of_peaks_x = np.where(complete_peaks_image == 1)[1]
        
        peaks_pixel_coords = np.vstack((pixels_of_peaks_y, pixels_of_peaks_x)).T
    else:
        # the original image is bigger than the standarised one
            
        complete_peaks_image = np.zeros(FFT_no_log.shape)
        complete_peaks_image[lim_l : lim_r, lim_l : lim_r] = peaks_image
        
        pixels_of_peaks_y = np.where(complete_peaks_image == 1)[0]
        pixels_of_peaks_x = np.where(complete_peaks_image == 1)[1]
        
        peaks_pixel_coords = np.vstack((pixels_of_peaks_y, pixels_of_peaks_x)).T
    
    return peaks_pixel_coords


# Get coordinates after the peak finding in standarised FFT and psize and FOV
def standarised_FOV_peaks_image_to_coords(
        peaks_image, standarized_FFT_NoFOV, FFT_no_log, lim_l, lim_r):
    
    standarised_FFT_withFOV_size = peaks_image.shape[0]
    standarized_FFT_NoFOV_size = standarized_FFT_NoFOV.shape[0]
    original_FFT_size = FFT_no_log.shape[0]
    
    # turn the standarise with FOV bakc to its orignal size and become the
    # standarised image without the FOV standarised just the pixel size
    
    if standarised_FFT_withFOV_size == standarized_FFT_NoFOV_size:
        peaks_image = peaks_image
    elif standarised_FFT_withFOV_size > standarized_FFT_NoFOV_size:
        # we need to downscale the peaks image
        new_peaks_image = np.zeros((standarized_FFT_NoFOV_size, standarized_FFT_NoFOV_size))
        
        
        pixels_of_peaks_y = np.where(peaks_image == 1)[0]
        pixels_of_peaks_x = np.where(peaks_image == 1)[1]
        
        relative_pos_pixelspeaks_y = pixels_of_peaks_y/(standarised_FFT_withFOV_size-1)
        relative_pos_pixelspeaks_x = pixels_of_peaks_x/(standarised_FFT_withFOV_size-1)
        
        new_peaks_y = relative_pos_pixelspeaks_y*(standarized_FFT_NoFOV_size-1)
        new_peaks_x = relative_pos_pixelspeaks_x*(standarized_FFT_NoFOV_size-1)
        
        for new_peak_y, new_peak_x in zip(
                new_peaks_y, new_peaks_x):
            
            new_peaks_image[int(new_peak_y), int(new_peak_x)] = 1
        
        peaks_image = new_peaks_image
        
    else:
        # we need to upscale the peaks image
        new_peaks_image = np.zeros((standarized_FFT_NoFOV_size, standarized_FFT_NoFOV_size))
        
        pixels_of_peaks_y = np.where(peaks_image == 1)[0]
        pixels_of_peaks_x = np.where(peaks_image == 1)[1]
        
        relative_pos_pixelspeaks_y_init = pixels_of_peaks_y/(standarised_FFT_withFOV_size-1)
        relative_pos_pixelspeaks_x_init = pixels_of_peaks_x/(standarised_FFT_withFOV_size-1)
        
        relative_pos_pixelspeaks_y_end = (pixels_of_peaks_y+1)/(standarised_FFT_withFOV_size-1)
        relative_pos_pixelspeaks_x_end = (pixels_of_peaks_x+1)/(standarised_FFT_withFOV_size-1)
        for index, (rel_y_end, rel_x_end) in enumerate(zip(
                relative_pos_pixelspeaks_y_end, relative_pos_pixelspeaks_x_end)):
            
            if rel_y_end > 1:
                relative_pos_pixelspeaks_y_end[index] = 1
            if rel_x_end > 1:
                relative_pos_pixelspeaks_x_end[index] = 1
        print('relative_pos_pixelspeaks_y_init')
        print(relative_pos_pixelspeaks_y_init)
        
        new_peaks_y_init = relative_pos_pixelspeaks_y_init*(standarized_FFT_NoFOV_size-1)
        new_peaks_x_init = relative_pos_pixelspeaks_x_init*(standarized_FFT_NoFOV_size-1)
        
        new_peaks_y_end = relative_pos_pixelspeaks_y_end*(standarized_FFT_NoFOV_size-1)
        new_peaks_x_end = relative_pos_pixelspeaks_x_end*(standarized_FFT_NoFOV_size-1)
        
        for y_i, x_i, y_f, x_f in zip(
                new_peaks_y_init, new_peaks_x_init, new_peaks_y_end, new_peaks_x_end):
            
            possible_pixels_to_check = standarized_FFT_NoFOV[int(y_i):int(y_f), int(x_i):int(x_f)]
            
            # the pixel that is identified as the peak is where the maximum in
            # the standarised image is present
            new_y_i = y_i + np.where(possible_pixels_to_check == np.max(possible_pixels_to_check))[0][0]
            new_x_i = x_i + np.where(possible_pixels_to_check == np.max(possible_pixels_to_check))[1][0]
            
            new_peaks_image[int(new_y_i), int(new_x_i)] = 1
            
        peaks_image = new_peaks_image
        
    
    # make the standarised without fov, just pixel size, go back to its original
    
    # compare both sizes to address the enlarging or smalling of the FFTs
    if standarized_FFT_NoFOV_size == original_FFT_size:
        
        pixels_of_peaks_y = np.where(peaks_image == 1)[0]
        pixels_of_peaks_x = np.where(peaks_image == 1)[1]
        
        peaks_pixel_coords = np.vstack((pixels_of_peaks_y, pixels_of_peaks_x)).T
    
    elif standarized_FFT_NoFOV_size > original_FFT_size:
        # the original image is smaller than the standarised one
        
        complete_peaks_image = peaks_image[lim_l : lim_r, lim_l : lim_r]
        
        pixels_of_peaks_y = np.where(complete_peaks_image == 1)[0]
        pixels_of_peaks_x = np.where(complete_peaks_image == 1)[1]
        
        peaks_pixel_coords = np.vstack((pixels_of_peaks_y, pixels_of_peaks_x)).T
    else:
        # the original image is bigger than the standarised one
            
        complete_peaks_image = np.zeros(FFT_no_log.shape)
        complete_peaks_image[lim_l : lim_r, lim_l : lim_r] = peaks_image
        
        pixels_of_peaks_y = np.where(complete_peaks_image == 1)[0]
        pixels_of_peaks_x = np.where(complete_peaks_image == 1)[1]
        
        peaks_pixel_coords = np.vstack((pixels_of_peaks_y, pixels_of_peaks_x)).T
    
    return peaks_pixel_coords




#Remove central part of spectrum
def remove_center(img):
        m = int(len(img)/2)
        size = 2*m
        delta = 10*size//256 # generalizar mejor
        
        img[m - delta : m + delta, m - delta : m + delta] = 0
              
# Function to show the peaks detected and combine the spectrum with n¡the signal in a image    
def show_peaks_detected(fft, peaks, ratio = 1, plot = False):
    
    m = len(fft)
    
    fft_color = cv2.cvtColor(np.uint8(255*ii.normalize_image(np.log(fft+ 0.00001))),cv2.COLOR_GRAY2RGB)
    peaks[m//2,m//2] = 1
    peaks = cercles(len(peaks)/512, peaks)
    peaks_color = cv2.cvtColor(np.uint8(255*peaks),cv2.COLOR_GRAY2RGB)
    peaks_color[:, :, 1] = 0
    peaks_color[:, :, 2] = 0
    
    #size = 256
    #edges = size//8
    #index = np.asarray(range(edges , size - edges))
    
    #ratio = 0.3 + np.log2(m/256)/10

    added_image = cv2.addWeighted(peaks_color ,2-ratio,fft_color,ratio,0)
    # added_image = added_image[:,:,0]*1/3 + added_image[:,:,1]*1/3 + added_image[:,:,2]*1/3
    
    
    if plot:
        plt.figure(figsize=(10, 10)) 
        plt.imshow(added_image, cmap = 'viridis')
        #plt.imshow(peaks_color)
    
    return added_image, fft_color, peaks_color


'''
Functions for the CNN peak finding
'''
def centering(lado, img):
    
    index = np.where(img)
  
    n = len(img)
    final_img = np.zeros((n,n))+img
    sin_marcos = np.zeros((n,n))
    
    sin_marcos[lado : n - lado, lado : n - lado ] = 1
    
    final_img = final_img * sin_marcos
    
    for i,j in zip(index[0],index[1]):
        
        if final_img[i,j] > 0:
            new_i = 0
            new_j = 0
            k = 0
            
            while i != new_i or j != new_j:
                
                if k > 0:
                    i = new_i
                    j = new_j
                
                
                recuadro = final_img[i-lado//2 : i + lado//2+1,
                               j - lado//2 : j + lado//2+1]
                
                # points = np.where(recuadro>0.5)
                # new_i = int(np.mean(points[0])) + i - lado//2
                # new_j = int(np.mean(points[1])) + j - lado//2
                
                new_index = np.where(recuadro == np.max(recuadro))
                new_i = new_index[0][0] + i - lado//2
                new_j = new_index[1][0] + j - lado//2
               
                k += 1
                
            final_img[i-lado//2 : i + lado//2+1,  j - lado//2 : j + lado//2+1] = np.zeros((lado,lado))
            final_img[new_i,new_j] = 1
            
            #See progress
            
            #plt.imshow(final_img, cmap=plt.cm.gray)
            #plt.show()
    
                     
    return final_img

def symetrize(img):
    
    
    index = np.where(img)
    #treshold = 9
    n = len(img)
    symetrize_img = img + 0
    
    for i,j in zip(index[0],index[1]):
        i_center  = n - i - 1
        j_center  = n - j - 1
        treshold = 3 + np.abs((i_center-n//2))//20
    
        
        if not np.any(img[i_center-treshold:i_center+treshold,
                      j_center-treshold:j_center+treshold]):
            
            symetrize_img[i,j] = 0
    
    symetrize_img[n//2,n//2] = 1
    
    return symetrize_img



def peaks_recognition_veryold(side,treshold, prediction, FFT_size=2048):
    '''
    Deprecated
    Does not find the very center of the spots
    '''

    
    # only downsample if FFT is bigger than indicated
    if FFT_size < 256:
        size= FFT_size
    else:
        size = 256

    edges = size//8
    
    index = np.asarray(range(edges , size - edges))
        
    zoomed_prediction = cv2.resize(prediction[0], dsize=(size, size), interpolation=cv2.INTER_LINEAR)

    prediction_without_backgraund = zoomed_prediction[index][:,index] > treshold

    peaks_image = centering(side, prediction_without_backgraund*zoomed_prediction[index][:,index])
    # peaks_image = centering(side, prediction_without_backgraund)
    
    print('peaks_image centering')
    print(np.unique(peaks_image))
    
    peaks_image_s = symetrize(peaks_image)
    
    print('peaks_image symetrize')
    print(np.unique(peaks_image))
    
    #go back to original downscaled size 256
    new_peaks_image = np.zeros((size, size))
    
    new_peaks_image[edges:size-edges, edges:size-edges] = peaks_image_s
    
    peaks_image_s = new_peaks_image
    
    

    # rescale back to the real size
    
    if FFT_size > 256:
        
        peaks_image_s = cv2.resize(peaks_image_s, dsize=(FFT_size, FFT_size), interpolation=cv2.INTER_LINEAR)
        
        peaks_image_s = centering(side,peaks_image_s)
        
    
    return peaks_image_s,prediction_without_backgraund



def peaks_recognition_old(side,treshold, prediction, fft, FFT_size=2048):
    '''
    Deprecated
    Does not find the very center of the spots
    '''
    
    # only downsample if FFT is bigger than indicated
    if FFT_size < 256:
        size= FFT_size
    else:
        size = 256

    edges = size//8
    
    
    index = np.asarray(range(edges , size - edges))
        

    zoomed_prediction = cv2.resize(
        prediction[0], dsize=(size, size), interpolation=cv2.INTER_LINEAR)
    
    prediction_without_backgraund = zoomed_prediction[index][:,index] > treshold
    
    #peaks_image = centering(side,prediction_without_backgraund*zoomed_prediction[index][:,index])
    peaks_image = centering(side,prediction_without_backgraund*fft[index][:,index])
    
    peaks_image_s = symetrize(peaks_image)    
    
    #go back to original downscaled size 256
    new_peaks_image = np.zeros((size, size))
    
    new_peaks_image[edges:size-edges, edges:size-edges] = peaks_image_s
    
    peaks_image_s = new_peaks_image
        

    # rescale back to the real size
    
    if FFT_size > 256:
        
        peaks_image_s = cv2.resize(
            peaks_image_s, dsize=(FFT_size, FFT_size), interpolation=cv2.INTER_LINEAR)
        
        peaks_image_s = centering(side,peaks_image_s)
        
    
    return peaks_image_s,prediction_without_backgraund




def peaks_recognition(
        side, treshold, prediction, fft, FFT_size=2048):
    '''
    Proper function with now finding the very center of the spots that are 
    identified by the CNN, being this center the maximum of the intensities
    distribution

    Parameters
    ----------
    side : 
    treshold : 
    prediction : 
    fft : 
    FFT_size : 

    Returns
    -------
    peaks_image_s : TYPE
        DESCRIPTION.
    prediction_without_backgraund : TYPE
        DESCRIPTION.

    '''
    
    # only downsample if FFT is bigger than indicated
    if FFT_size < 256:
        size= FFT_size
    else:
        size = 256

    edges = size//8
    
    index = np.asarray(range(edges, size-edges))
    
    zoomed_prediction = cv2.resize(prediction[0], dsize = (size,size), interpolation = cv2.INTER_LINEAR)
    prediction_without_backgraund = zoomed_prediction[index][:, index] > treshold
    
    zoomed_fft = cv2.resize(fft, dsize=(size, size), interpolation=cv2.INTER_LINEAR)
    peaks_image = centering(side,prediction_without_backgraund*zoomed_fft[index][:,index])
    
    peaks_image_s = symetrize(peaks_image)

    #go back to original downscaled size 256
    new_peaks_image = np.zeros((size, size))
    
    new_peaks_image[edges:size-edges, edges:size-edges] = peaks_image_s
    
    peaks_image_s = new_peaks_image
    # rescale back to the real size
    
    if FFT_size > 256:
        
        peaks_image_s = cv2.resize(
            peaks_image_s, dsize=(FFT_size, FFT_size), interpolation=cv2.INTER_NEAREST)
        
        
        rescale = FFT_size//128 #
        
        peaks_image_s = centering(2*side*rescale-1,(peaks_image_s > 0 )*fft) #    
            
        
    
    return peaks_image_s, prediction_without_backgraund


