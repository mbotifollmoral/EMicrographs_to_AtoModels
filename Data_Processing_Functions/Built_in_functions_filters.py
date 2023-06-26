# -*- coding: utf-8 -*-
"""
Created on Mon Dec 21 13:09:55 2020

@author: Marc
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy 
import skimage.measure, skimage.filters
import sklearn.cluster
import cv2
import sklearn.feature_extraction
import sklearn.mixture


image=plt.imread(r'E:\Arxius varis\PhD\2nd_year\Code\trial_images\nanowire2_tiff.tif')
plt.imshow(image, cmap=plt.cm.gray, vmin=image.min(), vmax=image.max())
plt.show()


#First standarisation of the image for filtering/blurring it with gaussian filter

image_st=(image-np.min(image))/np.max(image)

#Application of Gaussian filter for denoising

filter_size=(5,5)
denoised_image=cv2.GaussianBlur(image_st, filter_size, 1)

#Second standarisation of the image after filtering/blurring it with gaussian filter

image_st=(denoised_image-np.min(denoised_image))/np.max(denoised_image)

#Print histogram


#For sake of evaluation, better work with an image with less pixels, as only the consecutive pixel evaluation would take
#approximately 6 hours to run for a big region of 250.000 pixels in total.

#Then downsample the image and upsample it posteriorly 
#We select a max pooling method to keep track of the brighter elements and this way keep a higher contrast

scaling_factor=1
ds_image=skimage.measure.block_reduce(image_st, block_size=tuple(np.int32(scaling_factor*np.ones(len(np.shape(image_st))))), func=np.mean, cval=0)


#and standarise it again to ensure 0-1 values

ds_image_st=(ds_image-np.min(ds_image))/np.max(ds_image-np.min(ds_image))


#We have three histograms to fit with gaussians. Let us see methods do to so

plt.imshow(ds_image_st, cmap=plt.cm.gray, vmin=ds_image_st.min(), vmax=ds_image_st.max())


#Same study with edge detection filters but already implemented with skimage
#Same results, though, not useful as an standalone element but maybe important for refining

fig, axes = plt.subplots(figsize=(8, 8), nrows=2, ncols=2, 
                         sharex=True, sharey=True)

for axis, filt_alg, title in zip(axes.flat, 
                                 [skimage.filters.sobel, skimage.filters.scharr,
                                  skimage.filters.prewitt, skimage.filters.roberts],
                                 ['Sobel', 'Scharr', 'Prewitt', 'Roberts']):
    new_im=filt_alg(ds_image_st)
    axis.imshow(new_im, cmap='gray')
    axis.set_title(title, fontsize=16)

fig.tight_layout()


prewitt_bin = skimage.filters.prewitt(ds_image_st)
prewitt_bin=(prewitt_bin-np.min(prewitt_bin))/np.max(prewitt_bin-np.min(prewitt_bin))


threshold = 0.3
prewitt_bin[prewitt_bin > threshold] = 1
prewitt_bin[prewitt_bin <= threshold] = 0

fig, axis = plt.subplots()
axis.set_title('Binary image')
axis.imshow(prewitt_bin, cmap='gray')


#background or gradient dealing


section_x = 400

fig, axes = plt.subplots(ncols=2, figsize=(8, 4))
im_handle = axes[0].imshow(ds_image_st, cmap='jet')
# Add a colorbar
cbar = plt.colorbar(im_handle, ax=axes[0], fraction=0.046, pad=0.04)
axis.set_title('Normalized imge');
axes[0].axvline(x=section_x, color='k')
axes[0].set_title('Raw image', fontsize=14)
axes[0].set_xlabel('X', fontsize=14)
axes[0].set_ylabel('Y', fontsize=14)

axes[1].plot(np.squeeze(ds_image_st[:, section_x]));
axes[1].set_title('Line profile with x={}'.format(section_x), fontsize=14)
axes[1].set_xlabel('Y', fontsize=14)
axes[1].set_ylabel('Intensity', fontsize=14)
fig.tight_layout()

x_pixels, y_pixels = ds_image_st.shape  # [pixels]
x_edge_length = 37.0  # [nm]
y_edge_length = 37.0  # [nm]
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

fig, axis = plt.subplots()
axis.imshow(inverse_gauss_filter, cmap=plt.cm.OrRd)
axis.set_title('background removal filter')

# take the fft of the image
fft_image_w_background = np.fft.fftshift(np.fft.fft2(ds_image_st))
fft_abs_image_background = np.abs(fft_image_w_background)

# apply the filter
fft_image_corrected = fft_image_w_background * inverse_gauss_filter

# perform the inverse fourier transform on the filtered data
image_corrected = np.real(np.fft.ifft2(np.fft.ifftshift(fft_image_corrected)))

# find what was removed from the image by filtering
filtered_background = ds_image_st - image_corrected

fig, axes = plt.subplots(ncols=3, figsize=(12, 4), sharey=True)
for axis,img, title in zip(axes.flat, 
                           [ds_image_st, image_corrected, filtered_background],
                           ['Original', 'Corrected', 'Removed background']):
    axis.imshow(img, cmap='jet', vmin=ds_image_st.min(), vmax=ds_image_st.max())
    axis.set_title(title)




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
    
    fig, axis = plt.subplots()
    axis.imshow(inverse_gauss_filter, cmap=plt.cm.OrRd)
    axis.set_title('background removal filter')
    
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

    
    


