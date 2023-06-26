# -*- coding: utf-8 -*-
"""
Created on Thu Feb 17 17:39:18 2022

@author: Marc
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy import signal as scisignal

# Functions to compute the SNR and noise of the images, not finally used!

def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    r"""Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
    The Savitzky-Golay filter removes high frequency noise from data.
    It has the advantage of preserving the original shape and
    features of the signal better than other types of filtering
    approaches, such as moving averages techniques.
    Parameters
    ----------
    y : array_like, shape (N,)
        the values of the time history of the signal.
    window_size : int
        the length of the window. Must be an odd integer number.
    order : int
        the order of the polynomial used in the filtering.
        Must be less then `window_size` - 1.
    deriv: int
        the order of the derivative to compute (default = 0 means only smoothing)
    Returns
    -------
    ys : ndarray, shape (N)
        the smoothed signal (or it's n-th derivative).
    Notes
    -----
    The Savitzky-Golay is a type of low-pass filter, particularly
    suited for smoothing noisy data. The main idea behind this
    approach is to make for each point a least-square fit with a
    polynomial of high order over a odd-sized window centered at
    the point.
    Examples
    --------
    t = np.linspace(-4, 4, 500)
    y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
    ysg = savitzky_golay(y, window_size=31, order=4)
    import matplotlib.pyplot as plt
    plt.plot(t, y, label='Noisy signal')
    plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
    plt.plot(t, ysg, 'r', label='Filtered signal')
    plt.legend()
    plt.show()
    References
    ----------
    .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
       Data by Simplified Least Squares Procedures. Analytical
       Chemistry, 1964, 36 (8), pp 1627-1639.
    .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
       W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
       Cambridge University Press ISBN-13: 9780521880688
    """
    import numpy as np
    from math import factorial
    
    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
    except ValueError: 
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order+1)
    half_window = (window_size -1) // 2
    # precompute coefficients
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve( m[::-1], y, mode='valid')


def Wiener_filter_reciprocal(raw_image, FFT_calibration):   
    # NOT COMPLETE!!!
    #Filter like Vincezo's one in STEM_CELL, just deleting the high freq noise, converted into a constant
    #this approach will not be suitable for images in which the resolution is at the edges of high frequency
    # and then we do not have high frequency component on freqs > 2*semiconv_angle/e-_wavelenght
    def FT(img):
        return np.fft.ifftshift(np.fft.fft2(np.fft.fftshift(img)))
    
    def IFT(img):
        return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(img)))
    
    #define a profile in the FFT
    
    FFT_image=np.log(np.absolute(FT(raw_image)))
    img_size=raw_image.shape[0]
    
    profile_pixels=FFT_image[int(img_size/2)-int(np.floor(img_size*0.125)):int(img_size/2)+int(np.floor(img_size*0.125)),int(img_size/2)::]
    integrated_int=np.sum(profile_pixels, axis=0)/profile_pixels.shape[0]
    spatial_freqs=np.arange(0,FFT_calibration*(img_size/2), step=FFT_calibration)
    
    plt.plot(spatial_freqs,integrated_int)
    plt.show()
    
    smoothed_array=savitzky_golay(integrated_int, 51, order=1, deriv=0, rate=1)
    plt.plot(spatial_freqs,smoothed_array)
    plt.show()
    
    dydx = np.diff(smoothed_array)/np.diff(spatial_freqs)
    
    plt.plot(spatial_freqs[:-1],dydx)
    plt.show()
    
    smoothed_deriv=savitzky_golay(dydx, 51, order=1, deriv=0, rate=1)
    plt.plot(spatial_freqs[:-1],smoothed_deriv)
    plt.show()

    dydx2 = np.diff(smoothed_deriv)/np.diff(spatial_freqs[:-1])
    
    plt.plot(spatial_freqs[:-2],dydx2)
    plt.show()
    
    smoothed_deriv2=savitzky_golay(dydx2, 51, order=1, deriv=0, rate=1)
    plt.plot(spatial_freqs[:-2],smoothed_deriv2)
    plt.show()

    #Stopped here because it would only work in high magnificaiton ones, where the noise
    # component is easily extractable    
    
    # TO get the function that selects the noise threshold  FFT pixels, averaged, to get the FFT threshold
    # a + bi to be substracted to the whole FFT and then iFFT, some kind of function of the 2nd* derivative
    # to select the regions that are flat (they have almost no variance) and are close to 0 
    # and also located, preferibly, at the edges of the FFT frequency domains
    # * The 1st derivative should also work generally speaking, except in lower mag images where 2nd better
    Wiener_filtered_image=1

    return Wiener_filtered_image

def Compute_SNR_per_Segmented(image, image_calibration):
    '''
    Original image, and calibration of the image in nm/pixels
    Lewis Jones et al approach for computing SNR
    It does not work very well in the full image

    Parameters
    ----------
    image : 
    image_calibration :
    Returns
    -------
    TYPE
        DESCRIPTION.

    '''
    
    def Background_Definer_LowPass(image, image_calibraton):
        
        def LowPass_Gaussian_Blur(image_array, image_scale):
            '''
            Denoises the image based on the image size and scale to properly and tailoredly fit the features displayed
            This only depends on the scale of the image, not on the FOV, as the FWHM depends only on the feature size,
            e.g. interplanar distances, and this is defined by the magnification
            image scale must be in nm/pixels
            Eventually we should not need any filtering as the DL model should do all the work
            '''
            #average feature size in a crystal, arround 2-3 angrstoms = 0.2-0.3 nm, larger the more secure
            avg_feature_size=0.5  #nm
            FWHM_gaussian=2*avg_feature_size
            desvest_nm=FWHM_gaussian/(2*np.sqrt(2*np.log(2)))
            desvest_pixels=int(np.ceil(desvest_nm/image_scale))  # in number of pixels
            kernel_size=2*desvest_pixels+1  #too agressive
            gaussian_blurred_image=cv2.GaussianBlur(image_array, (kernel_size, kernel_size), desvest_pixels)
            return gaussian_blurred_image
   
        background_image=LowPass_Gaussian_Blur(image, image_calibraton)
        return background_image
    
    def Signal_Definer_BandPass(image, background_image):
        
        def BandPass_Gaussian_Blur(image_array):
            '''
            Denoises the image based on the image size and scale to properly and tailoredly fit the features displayed
            This only depends on the scale of the image, not on the FOV, as the FWHM depends only on the feature size,
            e.g. interplanar distances, and this is defined by the magnification
            image scale must be in nm/pixels
            Eventually we should not need any filtering as the DL model should do all the work
            '''
            #FWHM of 3 pixels should be good to extract signal, Lewis J. et al.
            FWHM_gaussian=3 #pixels
            desvest_pixels=int(np.floor(FWHM_gaussian/(2*np.sqrt(2*np.log(2)))))
            kernel_size=2*desvest_pixels+1  #too agressive
            gaussian_blurred_image=cv2.GaussianBlur(image_array, (kernel_size, kernel_size), desvest_pixels)
            return gaussian_blurred_image
        
        image_no_back=image - background_image
        signal_image=BandPass_Gaussian_Blur(image_no_back)
        return signal_image
    
    def Noise_Definer(image, signal_image):
        noise_image=image-signal_image 
        return noise_image
    
    background_image=Background_Definer_LowPass(image, image_calibration)
    signal_image=Signal_Definer_BandPass(image, background_image)
    noise_image=Noise_Definer(image, signal_image)
    
    print('Background image')
    plt.imshow(background_image,cmap=plt.cm.gray, vmin=background_image.min(), vmax=background_image.max())
    plt.show()
    print('Signal image')
    plt.imshow(signal_image,cmap=plt.cm.gray, vmin=signal_image.min(), vmax=signal_image.max())
    plt.show()
    print('Noise image')
    plt.imshow(noise_image,cmap=plt.cm.gray, vmin=noise_image.min(), vmax=noise_image.max())
    plt.show()
    
    #in case we want to segment or get the SNR from some regions, just do the array boolean to keep the 
    #pixels of interest of that specific label
    #for label of segmented region
    
    # background_image=background_image[segmented_image==label]
    # signal_image=signal_image[segmented_image==label]
    # noise_image=noise_image[segmented_image==label]
    
    signal_desvest=np.std(signal_image.flatten())
    noise_desvest=np.std(noise_image.flatten())
    
    SNR=signal_desvest/noise_desvest
     
    return SNR

def SNR_func(raw_image, denoised_image):
    #Best SNR function, just denoise and substract
    noise_image=denoised_image - raw_image
    # print('Noise image')
    # plt.imshow(noise_image,cmap=plt.cm.gray, vmin=noise_image.min(), vmax=noise_image.max())
    # plt.show()
    def FT(img):
        return np.fft.ifftshift(np.fft.fft2(np.fft.fftshift(img)))    
    noisefft=np.log(abs(FT(noise_image)))
    # plt.imshow(noisefft,cmap=plt.cm.gray, vmin=noisefft.min(), vmax=noisefft.max())
    # plt.show()
    
    desvest_signal=np.std(denoised_image)
    desvest_noise=np.std(noise_image)
    SNR=desvest_signal/desvest_noise
    return SNR
    


def Compute_SNR_fourier(image, FFT_calibration, top_interplanar_distance):
    # Delete the low frequencies directly from the fourier transform, does not work at all!
    # does not work well because the FT and iFT relations are not good, work with the FFT and IFFT functions
    # in the ImageCalibTransf file
    def FT(img):
        return np.fft.ifftshift(np.fft.fft2(np.fft.fftshift(img)))
    
    def IFT(img):
        return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(img)))
    
    fft_image=np.log(abs(FT(image)))
    print(fft_image)
    random=IFT(fft_image)
    print(np.absolute(random))
    lowpassed_FFT=np.copy(fft_image)
    
    distance_nm=top_interplanar_distance/10
    shape_im=fft_image.shape[0]
    for row in range(shape_im):
        for col in range(shape_im):
            recipr_distance=np.sqrt((abs(shape_im/2-row)*FFT_calibration)**2+((abs(shape_im/2-col)*FFT_calibration)**2))
            if recipr_distance==0:
                recipr_distance=0.0001
            dist=1/recipr_distance
            if dist < distance_nm:
                lowpassed_FFT[row,col]=0
    
    print(lowpassed_FFT)
    lowfre_noise=np.absolute(IFT(lowpassed_FFT))
    print('Low freq noise image')
    plt.imshow(lowfre_noise,cmap=plt.cm.gray, vmin=lowfre_noise.min(), vmax=lowfre_noise.max())
    plt.show()
      
    signal_desvest=np.std(image.flatten())
    noise_desvest=np.std(lowfre_noise.flatten())    
    
    SNR=signal_desvest/noise_desvest
    
    return SNR

def Wiener_Noise_Ratio(im, mysize):
    
    lMean = scisignal.correlate(im, np.ones(mysize), 'same') / np.prod(mysize, axis=0)
    lVar = (scisignal.correlate(im ** 2, np.ones(mysize), 'same') /np.prod(mysize, axis=0) - lMean ** 2)
    noise = np.mean(np.ravel(lVar), axis=0)
    print(noise)
    lNoise=lMean-lVar
    noise2=np.mean(np.ravel(lNoise))
    means=np.mean(np.ravel(lMean))
    
    SNR1=means/noise
    SNR2=means/noise2
    print('SNR1', SNR1)
    print('SNR2', SNR2)

    return SNR1, SNR2



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


def Adaptative_Gaussian_Blur(image_array, image_scale):
    '''
    Denoises the image based on the image size and scale to properly and tailoredly fit the features displayed
    This only depends on the scale of the image, not on the FOV, as the FWHM depends only on the feature size,
    e.g. interplanar distances, and this is defined by the magnification
    image scale must be in nm/pixels
    Eventually we should not need any filtering as the DL model should do all the work
    '''
    #average feature size in a crystal, arround 2-3 angrstoms = 0.2-0.3 nm, larger the more secure
    avg_feature_size=0.3  #nm
    FWHM_gaussian=2*avg_feature_size
    desvest_nm=FWHM_gaussian/(2*np.sqrt(2*np.log(2)))
    desvest_pixels=int(np.ceil(desvest_nm/image_scale))  # in number of pixels
    kernel_size=2*desvest_pixels+1  #too agressive
    kernel_size=3
    gaussian_blurred_image=cv2.GaussianBlur(image_array, (kernel_size, kernel_size), desvest_pixels)
    return gaussian_blurred_image