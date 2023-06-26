# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 13:06:18 2020

@author: Marc
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy 
import skimage
import cv2
    



image=plt.imread(r'E:\Arxius varis\PhD\2nd_year\Code\trial_images\low_mag_images\nanowire2_tiff.tif')
image2=cv2.imread(r'C:/Users/Marc/Desktop/bunicuss.tif')
print(image2)
print(image2.shape)
image2_gray = skimage.color.rgb2gray(image2)



print(np.array(image))
print(np.shape(np.array(image)))
print(np.min(np.array([image])), np.max(image))
plt.imshow(image)
plt.show()
plt.hist(image.ravel(),256,[np.min(np.array([image]))-20,np.max(np.array([image]))+20])
plt.show()

blur=cv2.GaussianBlur(image, (5,5), 1)
image=blur
#Image preprocessing, standarisation
#Range from 0 to 1 the intensity values
image_st=(image-np.min(image))/np.max(image)
print(image_st)
#image_st=(image2_gray-np.min(image2_gray))/np.max(image2_gray)


#Edge detectioin process/function

#filters_definition:

sobel_horizontal=np.array([[1, 2, 1],[0, 0, 0],[-1, -2, -1]])
sobel_vertical = np.array([[-1, 0, 1],[-2, 0, 2],[-1, 0, 1]])
laplacian_filter=np.array([[1, 1, 1],[1, -8, 1],[1, 1, 1]])



#Canny Edge detection function use cv2.Canny() built in function


#LoG edge detaction funcion
def LoG_Edge_Detection(image,Gauss_kern_size,Gauss_sigma, Laplace_kern_size):
    '''
    Parameters
    ----------
    image : numpy array describing the image

    Returns
    -------
    image_edges : the result from the operator to the input image
    '''
    blur=cv2.GaussianBlur(image, (Gauss_kern_size,Gauss_kern_size), Gauss_sigma)
    plt.imshow(blur, cmap='gray')
    plt.show()
    lapl=cv2.Laplacian(blur, cv2.CV_64F, ksize=Laplace_kern_size)
    plt.imshow(lapl, cmap='gray')
    plt.show()
    #Zero crossing detection implementation
    z_c_image = np.zeros(lapl.shape)
    
    # For each pixel, count the number of positive
    # and negative pixels in the neighborhood
    
    for i in range(1, lapl.shape[0] - 1):
        for j in range(1, lapl.shape[1] - 1):
            negative_count = 0
            positive_count = 0
            neighbour = [lapl[i+1, j-1],lapl[i+1, j],lapl[i+1, j+1],lapl[i, j-1],lapl[i, j+1],lapl[i-1, j-1],lapl[i-1, j],lapl[i-1, j+1]]
            d = max(neighbour)
            e = min(neighbour)
            for h in neighbour:
                if h>0:
                    positive_count += 1
                elif h<0:
                    negative_count += 1
 
 
            # If both negative and positive values exist in 
            # the pixel neighborhood, then that pixel is a 
            # potential zero crossing
            
            z_c = ((negative_count > 0) and (positive_count > 0))
            
            # Change the pixel value with the maximum neighborhood
            # difference with the pixel
 
            if z_c:
                if lapl[i,j]>0:
                    z_c_image[i, j] = lapl[i,j] + np.abs(e)
                elif lapl[i,j]<0:
                    z_c_image[i, j] = np.abs(lapl[i,j]) + d
                
    # Normalize and change datatype to 'uint8' (optional)
    image_edges = (z_c_image-z_c_image.min())/z_c_image.max()
    
    #thresholding and median blurring
    
    return image_edges

#kirsh edge detection filter
def Kirsch_Edge_Detection(image):
    '''
    Applies the Kirsch Edge detection operator to a given image

    Parameters
    ----------
    image : numpy array describing the image
    
    Returns
    -------
    kirsch : the result from the operator to the input image
    '''
    
    E=np.array([[-3, -3, 5],[-3, 0, 5],[-3, -3, 5]])
    NE=np.array([[-3, 5, 5],[-3, 0, 5],[-3, -3, -3]])
    N=np.array([[5, 5, 5],[-3, 0, -3],[-3, -3, -3]])
    NW=np.array([[5, 5, -3],[5, 0, -3],[-3, -3, -3]])
    W=np.array([[5, -3, -3],[5, 0, -3],[5, -3, -3]])
    SW=np.array([[-3, -3, -3],[5, 0, -3],[5, 5, -3]])
    S=np.array([[-3, -3, -3],[-3, 0, -3],[5, 5, 5]])
    SE=np.array([[-3, -3, -3],[-3, 0, 5],[-3, 5, 5]])
    
    conv_E=scipy.ndimage.convolve(image, E, mode='reflect')
    conv_NE=scipy.ndimage.convolve(image, NE, mode='reflect')
    conv_N=scipy.ndimage.convolve(image, N, mode='reflect')
    conv_NW=scipy.ndimage.convolve(image, NW, mode='reflect')
    conv_W=scipy.ndimage.convolve(image, W, mode='reflect')
    conv_SW=scipy.ndimage.convolve(image, SW, mode='reflect')
    conv_S=scipy.ndimage.convolve(image, S, mode='reflect')
    conv_SE=scipy.ndimage.convolve(image, SE, mode='reflect')

    totalmatrix=np.stack((conv_E,conv_NE, conv_N,conv_NW,conv_W,conv_SW,conv_S,conv_SE),axis=-1)
    kirsch=np.max(totalmatrix, axis=-1)
    return kirsch

def Extended_Kirsch_Edge_Detection(image):
    '''
    Applies an extended Kirsch Edge detection operator (5x5 matrices) to a given image

    Parameters
    ----------
    image : numpy array describing the image

    Returns
    -------
    kirsch : the result from the operator to the input image

    '''
    E=np.array([[-7, -7, -7, 9, 9],
                [-7, -3, -3, 5, 9],
                [-7, -3, 0, 5, 9],
                [-7, -3, -3, 5, 9],
                [-7, -7, -7, 9, 9]])
    
    NE=np.array([[-7, 9, 9, 9, 9],
                [-7, -3, 5, 5, 9],
                [-7, -3, 0, 5, 9],
                [-7, -3, -3, -3, 9],
                [-7, -7, -7, -7, -7]])
    
    N=np.array([[9, 9, 9, 9, 9],
                [9, 5, 5, 5, 9],
                [-7, -3, 0, -3, -7],
                [-7, -3, -3, -3, -7],
                [-7, -7, -7, -7, -7]])
    
    NW=np.array([[9, 9, 9, 9, -7],
                [9, 5, 5, -3, -7],
                [9, 5, 0, -3, -7],
                [9, -3, -3, -3, -7],
                [-7, -7, -7, -7, -7]])
    
    W=np.array([[9, 9, -7, -7, -7],
                [9, 5, -3, -3, -7],
                [9, 5, 0, -3, -7],
                [9, 5, -3, -3, -7],
                [9, 9, -7, -7, -7]])
    
    SW=np.array([[-7, -7, -7, -7, -7],
                [9, -3, -3, -3, -7],
                [9, 5, 0, -3, -7],
                [9, 5, 5, -3, -7],
                [9, 9, 9, 9, -7]])
    
    S=np.array([[-7, -7, -7, -7, -7],
                [-7, -3, -3, -3, -7],
                [-7, -3, 0, -3, -7],
                [9, 5, 5, 5, 9],
                [9, 9, 9, 9, 9]])
    
    SE=np.array([[-7, -7, -7, -7, -7],
                [-7, -3, -3, -3, 9],
                [-7, -3, 0, 5, 9],
                [-7, -3, 5, 5, 9],
                [-7, 9, 9, 9, 9]])
    
    conv_E=scipy.ndimage.convolve(image, E, mode='reflect')
    conv_NE=scipy.ndimage.convolve(image, NE, mode='reflect')
    conv_N=scipy.ndimage.convolve(image, N, mode='reflect')
    conv_NW=scipy.ndimage.convolve(image, NW, mode='reflect')
    conv_W=scipy.ndimage.convolve(image, W, mode='reflect')
    conv_SW=scipy.ndimage.convolve(image, SW, mode='reflect')
    conv_S=scipy.ndimage.convolve(image, S, mode='reflect')
    conv_SE=scipy.ndimage.convolve(image, SE, mode='reflect')

    totalmatrix=np.stack((conv_E,conv_NE, conv_N,conv_NW,conv_W,conv_SW,conv_S,conv_SE),axis=-1)
    kirsch=np.max(totalmatrix, axis=-1)
    return kirsch

#Apply the convolution to the image, that returns the same dimensions as the original image


out_1 = scipy.ndimage.convolve(image_st, sobel_horizontal, mode='reflect')
plt.imshow(out_1, cmap='gray')
plt.show()
out_2 = scipy.ndimage.convolve(image_st, sobel_vertical, mode='reflect')
plt.imshow(out_2, cmap='gray')
plt.show()
out_3 = scipy.ndimage.convolve(image_st, laplacian_filter, mode='reflect')
plt.imshow(out_3, cmap='gray')
plt.show()


final_im=Kirsch_Edge_Detection(image_st)
plt.imshow(final_im, cmap='gray')
plt.show()


#Extended krisch process
final_im_ext=Extended_Kirsch_Edge_Detection(image_st)
plt.imshow(final_im_ext, cmap='gray')
plt.show()

#LoG
print('LoG')
logim=LoG_Edge_Detection(image_st,Gauss_kern_size=9,Gauss_sigma=1, Laplace_kern_size=3)
plt.imshow(logim, cmap='gray')
plt.show()

#Canny edge detection
print('Canny ED')
image_cv2=cv2.imread(r'E:\Arxius varis\PhD\2nd_year\Code\trial_images\nanowire1_tiff.tif')
print(image_cv2)
image_st_cv2=(image_cv2)/np.max(image_cv2)
print(image_st_cv2)

canny_im=cv2.Canny(image_cv2, 40, 50,7)
plt.imshow(canny_im, cmap='gray')
plt.show()

#Let us try downsampling the input image
a=tuple(np.int32(2*np.ones(len(np.shape(image_st)))))
b=tuple(np.int32(4*np.ones(len(np.shape(image_st)))))
c=tuple(np.int32(8*np.ones(len(np.shape(image_st)))))
d=tuple(np.int32(16*np.ones(len(np.shape(image_st)))))

print(a)
print(b)
downim_max=skimage.measure.block_reduce(image_st, block_size=tuple(np.int32(2*np.ones(len(np.shape(image_st))))), func=np.max, cval=0)

plt.imshow(downim_max, cmap='gray')
plt.show()

downim_mean=skimage.measure.block_reduce(image_st, block_size=tuple(np.int32(2*np.ones(len(np.shape(image_st))))), func=np.mean, cval=0)

plt.imshow(downim_mean, cmap='gray')
plt.show()

final_down_max=Kirsch_Edge_Detection(downim_max)
plt.imshow(final_down_max, cmap='gray')
plt.show()

final_down_mean=Kirsch_Edge_Detection(downim_mean)
plt.imshow(final_down_mean, cmap='gray')
plt.show()

#downsampling factor of 4
downim_max4=skimage.measure.block_reduce(image_st, block_size=b, func=np.max, cval=0)

plt.imshow(downim_max4, cmap='gray')
plt.show()

downim_mean4=skimage.measure.block_reduce(image_st, block_size=b, func=np.mean, cval=0)

plt.imshow(downim_mean4, cmap='gray')
plt.show()

final_down_max4=Kirsch_Edge_Detection(downim_max4)
plt.imshow(final_down_max4, cmap='gray')
plt.show()

final_down_mean4=Kirsch_Edge_Detection(downim_mean4)
plt.imshow(final_down_mean4, cmap='gray')
plt.show()


#downsampling factor of 8
downim_max8=skimage.measure.block_reduce(image_st, block_size=c, func=np.max, cval=0)

plt.imshow(downim_max8, cmap='gray')
plt.show()

downim_mean8=skimage.measure.block_reduce(image_st, block_size=c, func=np.mean, cval=0)

plt.imshow(downim_mean8, cmap='gray')
plt.show()

final_down_max8=Kirsch_Edge_Detection(downim_max8)
plt.imshow(final_down_max8, cmap='gray')
plt.show()

final_down_mean8=Kirsch_Edge_Detection(downim_mean8)
plt.imshow(final_down_mean8, cmap='gray')
plt.show()

#downsampling factor of 16
downim_max6=skimage.measure.block_reduce(image_st, block_size=d, func=np.max, cval=0)

plt.imshow(downim_max6, cmap='gray')
plt.show()

downim_mean6=skimage.measure.block_reduce(image_st, block_size=d, func=np.mean, cval=0)

plt.imshow(downim_mean6, cmap='gray')
plt.show()

final_down_max6=Kirsch_Edge_Detection(downim_max6)
plt.imshow(final_down_max6, cmap='gray')
plt.show()

final_down_mean6=Kirsch_Edge_Detection(downim_mean6)
plt.imshow(final_down_mean6, cmap='gray')
plt.show()


#resampling 16

upsam_cont6=scipy.ndimage.zoom(final_down_mean6, 16, order=1)
plt.imshow(upsam_cont6, cmap='gray')
plt.show()

#resampling 8

upsam_cont8=scipy.ndimage.zoom(final_down_mean8, 8, order=1)
plt.imshow(upsam_cont8, cmap='gray')
plt.show()


upsam_cont8=(upsam_cont8-np.min(upsam_cont8))/np.max(upsam_cont8)
plt.hist(upsam_cont8.ravel(),256,[np.min(np.array([upsam_cont8])),np.max(np.array([upsam_cont8]))])
plt.show()

#threshold the borders
reshaped_cont8 = upsam_cont8.reshape(upsam_cont8.shape[0]*upsam_cont8.shape[1])
for i in range(reshaped_cont8.shape[0]):
    if reshaped_cont8[i] > 0.22:
        reshaped_cont8[i] = 1
    else:
        reshaped_cont8[i] = 0
upsam_cont8_thres = reshaped_cont8.reshape(upsam_cont8.shape[0],upsam_cont8.shape[1])
plt.imshow(upsam_cont8_thres, cmap='gray')
plt.show()

#LoG Edge detection


