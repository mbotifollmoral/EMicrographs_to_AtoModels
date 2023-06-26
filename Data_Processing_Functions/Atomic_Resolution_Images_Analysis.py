# -*- coding: utf-8 -*-
"""
Created on Tue Dec 22 17:46:44 2020

@author: Marc
"""
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import scipy 
import skimage
import skimage.measure as measure
import sklearn.cluster
import cv2
import pycroscopy
import torch
import torch.utils.data
import numba
import atomai as aoi
import stemtool
import pyUSID as usid
import pyNSID
import sidpy as sid
import h5py
import pyfftw
import time
import hyperspy.api as hs

#Functions

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




def Modification_of_training_set(dataset_npy, dataset_val, path_to_save_dataset):

    
    images_2 = dataset_npy['X_train']
    labels_2 = dataset_npy['y_train']
    images_test_2 = dataset_npy['X_test']
    labels_test_2 = dataset_npy['y_test']

    #Convert training labels
    
    new_labels_2=np.copy(labels_2)
    new_labels_2[new_labels_2==0]=2
    new_labels_2 -=1
    new_labels_2=np.abs(new_labels_2-1)
    new_labels_2.shape=images_2.shape  #in case shape changes must be done
    
    #Convert test labels
    
    new_labels_test_2=np.copy(labels_test_2)
    new_labels_test_2[new_labels_test_2==0]=2
    new_labels_test_2 -=1
    new_labels_test_2=np.abs(new_labels_test_2-1)
    new_labels_test_2.shape=images_test_2.shape  #in case shape changes must be done
        
    #Create new dataset, dictionary to convert into npy file
    new_dataset_2=dict()
    
    new_dataset_2['X_train']=images_2
    new_dataset_2['y_train']=new_labels_2
    new_dataset_2['X_test']=images_test_2
    new_dataset_2['y_test']=new_labels_test_2
    new_dataset_2['validation_img']=dataset_val['validation_img']
    
    np.savez(path_to_save_dataset,new_dataset_2)
    
        
    
    
def Find_percentage_of_thresholded(FFT_image, threshold):
    FFT_image[FFT_image<=threshold]=0
    percentage_of_tresholded=np.count_nonzero(FFT_image.ravel())/len(FFT_image.ravel())
    return percentage_of_tresholded

def Threshold_given_percentage(FFT_image, percentage):
    y_pixs,x_pixs=np.shape(FFT_image)
    n_int_pixs=int(round(percentage*y_pixs*x_pixs))
    FFT_ravel=np.sort(np.ravel(FFT_image))[::-1]
    threshold=FFT_ravel[n_int_pixs]
    return threshold

def mpfit_Distance(FFT_image,FOV):
    mpfit_model=[[-2.87175127e-11],
                     [ 8.11320079e-09],
                     [-8.18658056e-07],
                     [ 3.33222163e-05],
                     [-2.02745223e-04],
                     [-2.26140649e-02],
                     [ 5.95346985e-01],
                     [-7.69005862e-01]]
    # without the averages
    mpfit_model_c=[[-3.46636981e-11],
                   [ 1.00423053e-08],
                   [-1.06223267e-06],
                   [ 4.84860471e-05],
                   [-6.82330526e-04],
                   [-1.58450088e-02],
                   [ 5.79540436e-01],
                   [-1.10510783e+00]]
    #set the working limits of the model
    if FOV >=30:
        mpfit_dist=np.array([40])
    else:
        
        fov_vals=np.array([FOV**7,FOV**6,FOV**5,FOV**4,FOV**3,FOV**2,FOV**1,1])
        mpfit_dist=np.e**np.dot(fov_vals,mpfit_model)
        mpfit_dist=np.e**np.dot(fov_vals,mpfit_model_c)
     
    #Adjustments depending on the sizze of the image
    if np.shape(FFT_image)[0]==2048:
        mpfit_dist=mpfit_dist*1.30
    elif np.shape(FFT_image)[0]<256:
        mpfit_dist=mpfit_dist*1.55     
    elif np.shape(FFT_image)[0]==256:
        mpfit_dist=mpfit_dist*1.55     
    elif np.shape(FFT_image)[0]==1024:
        mpfit_dist=mpfit_dist*1.05
    elif np.shape(FFT_image)[0]==512:
        mpfit_dist=mpfit_dist*1.15
    else:
        mpfit_dist=mpfit_dist*1.15
        
    return mpfit_dist[0]

def FFT_threshold(FOV):
    FFT_thresh_model=[[-1.01291174e-11],
                          [ 2.88297492e-09],
                          [-3.01778444e-07],
                          [ 1.44327587e-05],
                          [-3.23378617e-04],
                          [ 3.61163733e-03],
                          [-3.72515413e-02],
                          [-1.96361805e-01]]
    # without the averages
    FFT_thresh_model_c=[[ 1.54099057e-12],
                        [-6.56354380e-10],
                        [ 1.05878669e-07],
                        [-8.09680716e-06],
                        [ 2.96148198e-04],
                        [-4.30807411e-03],
                        [ 1.81389577e-03],
                        [-2.45698182e-01]]
    #set the working limits of the model
    if FOV >=80:
        FFT_thresh=np.array([0.6])
    else:
                
        fov_vals=np.array([FOV**7,FOV**6,FOV**5,FOV**4,FOV**3,FOV**2,FOV**1,1])
        FFT_thresh=np.e**np.dot(fov_vals,FFT_thresh_model)
        FFT_thresh=np.e**np.dot(fov_vals,FFT_thresh_model_c)
      
    
    return FFT_thresh[0]

def FFT_percentage(FFT_image,FOV):
    FFT_perc_model=[[-3.00411834e-11],
                        [ 1.17313244e-08],
                        [-1.81232383e-06],
                        [ 1.40635117e-04],
                        [-5.76020214e-03],
                        [ 1.20704617e-01],
                        [-1.20113823e+00],
                        [-2.14024711e+00]]
    # without the averages
    FFT_perc_model_c=[[ 1.38602821e-11],
                      [-2.46874956e-09],
                      [-1.63526870e-08],
                      [ 2.67725990e-05],
                      [-1.91230990e-03],
                      [ 5.28789844e-02],
                      [-6.40863899e-01],
                      [-3.71037505e+00]]
    #set the working limits of the model
    if FOV >=110:
        FFT_perc=np.array([0.00025])  #In case it is too much for higher FOVs, just delete this and keep the FFT_perc_model for all ranges
    # elif FOV <3:
    #     FFT_perc=np.array([0.01])
    else:
        
        
        fov_vals=np.array([FOV**7,FOV**6,FOV**5,FOV**4,FOV**3,FOV**2,FOV**1,1])
        FFT_perc=np.e**np.dot(fov_vals,FFT_perc_model)
        FFT_perc=np.e**np.dot(fov_vals,FFT_perc_model_c)        
        
        if FOV <4.5:
            FFT_perc=FFT_perc*(10**(np.log(128/np.shape(FFT_image)[0])/np.log(4)))
        elif FOV >=4.5 and FOV <=20 :
            FFT_perc=FFT_perc*(10**(np.log(512/np.shape(FFT_image)[0])/np.log(4)))
        else:
            FFT_perc=FFT_perc*(10**(np.log(2048/np.shape(FFT_image)[0])/np.log(4)))
    
    #Adjustments depending on the sizze of the image
    if np.shape(FFT_image)[0]<256:
        FFT_perc=FFT_perc*0.25
    elif np.shape(FFT_image)[0]==256:
        FFT_perc=FFT_perc*0.45  
    elif np.shape(FFT_image)[0]==512:
        FFT_perc=FFT_perc*0.55
    elif np.shape(FFT_image)[0]==1024:
        FFT_perc=FFT_perc*0.80    
    else:
        FFT_perc=FFT_perc*0.80
        
    return FFT_perc[0]

def FFT_hyperparams(FFT_image,FOV):
    #Return, in order, the mpfit dist, the threshold, and the percentage
    
    mpfit_dist=mpfit_Distance(FFT_image,FOV)
    FFT_thresh=FFT_threshold(FOV)
    FFT_perc=FFT_percentage(FFT_image,FOV)
    
    return mpfit_dist,FFT_thresh,FFT_perc



#Hyperparameters
gauss_blur_filter_size=(5,5)  #size of smoothing filter, go to line to change sigma
downscaling_factor=1 #for trials, n factor of downsampling size of image
FFT_thresholding=0.5  #value above which the pixels are kept
st_distance=30 #distance parameter in the Stem tool method
FFT_thresholdingG=0.6 #value above which the pixels are kept, in the gaussian filtered FFT
window_size=256  #window size of the sliding windows


np.random.seed(int(np.round(time.time())))
#image=plt.imread(r'E:\Arxius varis\PhD\2nd_year\Code\trial_images\atomic_resolution_images\InP-InSb6.tif')


#dm3 loading, and calibration extraction
imagedm3=hs.load(r'E:\Arxius varis\PhD\2nd_year\Code\trial_images\dm3_atomic_resolution\GeQW2.dm3')
meta1=imagedm3.metadata
meta2=imagedm3.original_metadata.export('parameters')


imagedm3.plot()
plt.show()

x_calibration=imagedm3.axes_manager['x'].scale
y_calibration=imagedm3.axes_manager['y'].scale

x_pixels=imagedm3.axes_manager['x'].size
y_pixels=imagedm3.axes_manager['y'].size

x_units=imagedm3.axes_manager['x'].units
y_units=imagedm3.axes_manager['y'].units

imagearray=np.asarray(imagedm3)
image=imagearray

plt.imshow(image, cmap=plt.cm.gray, vmin=image.min(), vmax=image.max())
plt.show()
init_y=np.random.choice(np.arange(0,image.shape[0]-window_size,1)) 
init_x=np.random.choice(np.arange(0,image.shape[1]-window_size,1)) 
#image=image[init_y:init_y+window_size,init_x:init_x+window_size]
#First standarisation of the image for filtering/blurring it with gaussian filter
image_st=(image-np.min(image))/np.max(image-np.min(image))
plt.hist(image_st.ravel(),256,[np.min(np.array([image_st])),np.max(np.array([image_st]))])
plt.show()


#Application of Gaussian filter for denoising


denoised_image=cv2.GaussianBlur(image_st, gauss_blur_filter_size, 1)


plt.imshow(denoised_image, cmap=plt.cm.gray, vmin=denoised_image.min(), vmax=denoised_image.max())
plt.show()
#Second standarisation of the image after filtering/blurring it with gaussian filter

image_st=(denoised_image-np.min(denoised_image))/np.max(denoised_image-np.min(denoised_image))

#Print histogram

plt.hist(image_st.ravel(),256,[np.min(np.array([image_st])),np.max(np.array([image_st]))])
plt.show()

#For sake of evaluation, better work with an image with less pixels, as only the consecutive pixel evaluation would take
#approximately 6 hours to run for a big region of 250.000 pixels in total.

#Then downsample the image and upsample it posteriorly 
#We select a max pooling method to keep track of the brighter elements and this way keep a higher contrast


ds_image=measure.block_reduce(image_st, block_size=tuple(np.int32(downscaling_factor*np.ones(len(np.shape(image_st))))), func=np.mean, cval=0)

#and standarise it again to ensure 0-1 values

ds_image_st=(ds_image-np.min(ds_image))/np.max(ds_image-np.min(ds_image))


print('FFT computing and refinement')

# take the fft of the image
fft_image_w_background = np.fft.fftshift(np.log(np.fft.fft2(ds_image_st)))
fft_abs_image_background = np.abs(fft_image_w_background)

print('Original FFT')
plt.hist(fft_abs_image_background.ravel(),256,[np.min(np.array([fft_abs_image_background])),np.max(np.array([fft_abs_image_background]))])
plt.show()
plt.imshow(fft_abs_image_background)
plt.show()
# apply the filter
fft_abs_image_background2=np.copy(fft_abs_image_background)
fft_abs_image_backgroundc=np.copy(fft_abs_image_background)


fft_abs_image_backgroundc=(fft_abs_image_backgroundc-np.min(fft_abs_image_backgroundc))/np.max(fft_abs_image_backgroundc-np.min(fft_abs_image_backgroundc))


print('Original FFT')
fft_abs_image_background2=(fft_abs_image_background2-np.min(fft_abs_image_background2))/np.max(fft_abs_image_background2-np.min(fft_abs_image_background2))
plt.hist(fft_abs_image_background2.ravel(),256,[np.min(np.array([fft_abs_image_background2])),np.max(np.array([fft_abs_image_background2]))])
plt.show()

plt.imshow(fft_abs_image_background2)
plt.show()
fft_abs_image_background2=cv2.GaussianBlur(fft_abs_image_background2, gauss_blur_filter_size, 1)
fft_abs_image_background2=(fft_abs_image_background2-np.min(fft_abs_image_background2))/np.max(fft_abs_image_background2-np.min(fft_abs_image_background2))

print('Gaussian Filtered FFT')
plt.imshow(fft_abs_image_background2)
plt.show()

plt.hist(fft_abs_image_background2.ravel(),256,[np.min(np.array([fft_abs_image_background2])),np.max(np.array([fft_abs_image_background2]))])
plt.show()

#trial with original FFT
#fft_abs_image_background2=fft_abs_image_backgroundc

#Automatic hyperparameter finding
fov=np.shape(fft_abs_image_background2)[0]*y_calibration
print('fov',fov)
st_distance,_,FFT_perc=FFT_hyperparams(fft_abs_image_background2,fov)
print('mpfit',st_distance,'perc',FFT_perc )
FFT_thresholdingG=Threshold_given_percentage(fft_abs_image_background2, FFT_perc)
print('fft_threhs',FFT_thresholdingG)



#Threshold given the percentage /1

print('Thresholded Gaussian FFT: '+str(FFT_thresholdingG))
fft_abs_image_GFT=np.copy(fft_abs_image_background2)
fft_abs_image_GFT[fft_abs_image_GFT>FFT_thresholdingG]=1
fft_abs_image_GFT[fft_abs_image_GFT<=FFT_thresholdingG]=0

plt.imshow(fft_abs_image_GFT)
plt.show()

print('Thresholded Gaussian FFT keeping original intensity: '+str(FFT_thresholdingG))
fft_abs_image_GFT_o=np.copy(fft_abs_image_background2)
fft_abs_image_GFT_o[fft_abs_image_GFT_o<=0.5]=0

plt.imshow(fft_abs_image_GFT_o)
plt.show()


print('Thresholded Gaussian FFT keeping original intensity noise: '+str(FFT_thresholdingG))
fft_abs_image_GFT_o=np.copy(fft_abs_image_background2)
fft_abs_image_GFT_o[fft_abs_image_GFT_o>0.5]=0

plt.imshow(fft_abs_image_GFT_o)
plt.show()



print('STEM Tool based method (2D Gaussians)')

center_difractogram=stemtool.util.fourier_reg.find_max_index(fft_abs_image_background2)

print('ST gaussian FFT')
twodfit_blur=stemtool.afit.peaks_vis(fft_abs_image_background2, dist=st_distance, thresh=FFT_thresholdingG, imsize=(15, 15))
plt.show()

print('ST gauss. thres. FFT')
twodfit_blur_thres=stemtool.afit.peaks_vis(fft_abs_image_GFT, dist=st_distance, thresh=FFT_thresholdingG, imsize=(15, 15))
plt.show()

print('ST gauss. thres. or. FFT')
twodfit_blur_thres_o=stemtool.afit.peaks_vis(fft_abs_image_GFT_o, dist=st_distance, thresh=FFT_thresholdingG, imsize=(15, 15))
plt.show()

print('AtomAI method, DL')
#atom ai fit
savepath=r'E:\Arxius varis\PhD\2nd_year\Code\datasets\atomai_nn_trainingset_ferroic\trained_model_bright_spots.pt'
savepath2=r'E:\Arxius varis\PhD\2nd_year\Code\datasets\atomai_nn_trainingset_ferroic\trained_model.pt'


#Train the model

# dataset_1 = np.load(r'E:\Arxius varis\PhD\2nd_year\Code\datasets\atomai_nn_trainingset_ferroic\ferroics-custom-bright-only.npy', allow_pickle=True)


# images_1 = dataset_1[()]['X_train']
# labels_1 = dataset_1[()]['y_train']
# images_test_1 = dataset_1[()]['X_test']
# labels_test_1 = dataset_1[()]['y_test']
# print(images_1.shape, labels_1.shape)

'''
# Initialize model
model = aoi.models.Segmentor(nb_classes=1)

# Train
model.fit(images_1, labels_1, images_test_1, labels_test_1, training_cycles=500, compute_accuracy=True, swa=True) 
#model.load_weights(r'E:\Arxius varis\PhD\1st_year\Built-in models - Codes - Weights\FerroicBlocks - DCNN\FerroicBlocks-master\saved_models\atomfinder-BFO-ts0-512-3-best_weights.pt')

torch.save(model, savepath)

#Directly load the model

'''
print('DL model bright spots')

model1 = torch.load(savepath)


# Get NN output with atomic coordinates

nn_output, coordinates_1 =model1.predict(fft_abs_image_background2)

aoi.utils.plot_coord(fft_abs_image_background2, coordinates_1[0], fsize=12)

del model1

print('DL model all spots')

model2 = torch.load(savepath2)

# Get NN output with atomic coordinates
print('DL gaussian FFT')
nn_output, coordinates_1 =model2.predict(fft_abs_image_background2)

aoi.utils.plot_coord(fft_abs_image_background2, coordinates_1[0], fsize=12)

print('DL gauss. thres. FFT')
nn_output, coordinates_1 =model2.predict(fft_abs_image_GFT)

aoi.utils.plot_coord(fft_abs_image_GFT, coordinates_1[0], fsize=12)

print('DL gauss. thres. or. FFT')
nn_output, coordinates_1 =model2.predict(fft_abs_image_GFT_o)

aoi.utils.plot_coord(fft_abs_image_GFT_o, coordinates_1[0], fsize=12)


del model2



percentage=Find_percentage_of_thresholded(fft_abs_image_background2, FFT_thresholdingG)

print(percentage)
print(FFT_thresholdingG)


del image


#%%

#Model computing for finding out the hyperparameters

fov_vals=np.array([107.569297,76.06298,53.78464889,38.03149032,26.892324,19.015745,13.44616,9.50787,6.72307,4.7539362,3.3615,2.3769675,1.6807675])
mpfit_vals=np.array([40,40,40,45,40,40,25,16.25,17.5,5.33333333,1,1.75,1.75])
thr_vals=np.array([0.65,0.56666,0.60,0.6375,0.5875,0.516666666,0.675,0.675,0.675,0.73333,0.725,0.775,0.7875])
perc_vals=np.array([0.000289678573608,0.00057307879124,0.00042366981,0.00038313865,0.00052165985,0.00384648640,0.00082397460,0.001163482665,0.0011081695,0.00718178,0.01037596445,0.010665,0.015289302])

# mpfit_vals=np.log(mpfit_vals)
# thr_vals=np.log(thr_vals)
# perc_vals=np.log(perc_vals)

#Models with all the data, not directly the averages
fov_vals_c=np.array([76.06298,38.031490,76.0629806,26.892324,38.03149,107.569297,53.78464889,26.89232444,38.03149032,26.89232,38.0314903,53.784648,76.06298,26.892324,
19.015745,9.50787,19.015745,6.72307,9.50787,26.892324,13.44616,6.72307,9.50787,6.72307,9.50787,13.44616,19.015745,6.72307,
4.7539362,2.3769675,4.7539362,1.6807675,2.3769675,6.7230,3.3615,1.6807675,2.3769675,1.6807675,2.3769675,3.3615,4.7539362,1.6807675])
mpfit_vals_c=np.array([40,40,40,40,40,20,40,40,60,40,40,40,40,40,
40,20,40,40,15,30,10,10,15,10,15,40,40,10,                 
10,5,5,5,1,1,1,0.5,0.5,0.5,0.5,1,1,1])
thr_vals_c=np.array([0.55,0.65,0.60,0.60,0.60,0.60,0.65,0.60,0.70,0.60,0.60,0.55,0.55,0.55,
0.50,0.65,0.55,0.60,0.60,0.70,0.75,0.70,0.70,0.75,0.75,0.60,0.50,0.65,
0.70,0.75,0.75,0.8,0.75,0.75,0.75,0.8,0.8,0.8,0.8,0.7,0.75,0.75])
perc_vals_c=np.array([0.00066637992858,0.00036549568176,0.00046610832214,0.000313520431,0.000396013259,0.001085042953,0.000289678573608,0.000620126724,0.00032925605773,0.00061392784,0.00052428245,0.00055766105651,0.000586748123,0.00053906440734,
0.0037498474121,0.00153732299804,0.002475738525390,0.001338958740234,0.000919342041015,0.002529144287,0.000812530517578,0.00139999389648,0.0014915466308,0.000728607177,0.0007057189941,0.0008354187011,0.0053138732910,0.0009651184082,
0.01275634,0.0055541992,0.0065307,0.0012817382,0.0015258789,0.0151367187,0.0054321289,0.04693603,0.010131,0.007141113,0.025451660,0.0153198,0.002258300,0.00579833])



xs=np.arange(min(fov_vals),80,2)
xs=np.reshape(xs, (len(xs),1))

x_matrix=np.array([xs**4,xs**3,xs**2,xs**1,np.ones(np.shape(xs))]).T[0]

x_matrix_2=np.array([xs**7,xs**6,xs**5,xs**4,xs**3,xs**2,xs**1,np.ones(np.shape(xs))]).T[0]



plt.plot(fov_vals,mpfit_vals)
plt.title('mpfit distance paramter (FOV)')
plt.xlabel('FOV (nm)')
plt.ylabel('mpfit distance')


mpfit_model=np.polyfit(fov_vals_c,np.log(mpfit_vals_c),7)
mpfit_model=np.reshape(mpfit_model,(len(mpfit_model),1))
ys_mpfit=np.dot(x_matrix_2,mpfit_model)
plt.plot(xs,np.e**ys_mpfit)


plt.show()

plt.plot(fov_vals,thr_vals)
plt.title('Threhsold values (FOV)')
plt.xlabel('FOV (nm)')
plt.ylabel('Threshold')

thr_model=np.polyfit(fov_vals_c,np.log(thr_vals_c),7)
thr_model=np.reshape(thr_model,(len(thr_model),1))
ys_thr=np.dot(x_matrix_2,thr_model)
plt.plot(xs,np.e**ys_thr)

plt.show()

plt.plot(fov_vals,perc_vals)
plt.title('%/1 pixels (FOV)')
plt.xlabel('FOV (nm)')
plt.ylabel('%/1')

perc_model=np.polyfit(fov_vals_c,np.log(perc_vals_c),7)
perc_model=np.reshape(perc_model,(len(perc_model),1))
ys_perc=np.dot(x_matrix_2,perc_model)
plt.plot(xs,np.e**ys_perc)

plt.show()

#%%




    
    