# -*- coding: utf-8 -*-
"""
Created on Thu Jul 15 10:56:27 2021

@author: Marc
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.utils.data
import atomai as aoi
import hyperspy.api as hs
import cv2



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
    

#Define the model to be stored in the GPU not CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)



#Train the model

#dataset_1 = np.load(r'E:\Arxius varis\PhD\2nd_year\Code\datasets\atomai_nn_trainingset_ferroic\ferroics-custom-bright-only.npy', allow_pickle=True)

trainingset=np.load(r'E:\Arxius varis\PhD\2nd_year\Code\datasets\FFT_noise_model_datasets\NoiseModel1.npz', allow_pickle=True)
testset=np.load(r'E:\Arxius varis\PhD\2nd_year\Code\datasets\FFT_noise_model_datasets\NoiseModel_test1.npz', allow_pickle=True)

'''

print(dataset_1)
#Training data
images_1 = dataset_1[()]['X_train']
labels_1 = dataset_1[()]['y_train']
#Test data
images_test_1 = dataset_1[()]['X_test']
labels_test_1 = dataset_1[()]['y_test']
print(images_1.shape, labels_1.shape)

print(labels_1[0][0])
print(labels_1[0][0].shape)
image1_1=images_1[0][0]
img_label1=labels_1[0][0]


'''
#Training data
images_1 = trainingset['x']
labels_1 = trainingset['y']
newshapetraining=(images_1.shape[0],1, images_1.shape[1],images_1.shape[2])
images_1=np.reshape(images_1, newshapetraining)
labels_1=np.reshape(labels_1, newshapetraining)


print(images_1.shape, labels_1.shape)



image1_1=images_1[390][0]
img_label1=labels_1[0][0]


plt.imshow(image1_1)
plt.show()
plt.imshow(img_label1)
plt.show()



#Test data
imagestest_1 = testset['x']
labelstest_1 = testset['y']
newshapetest=(imagestest_1.shape[0],1, imagestest_1.shape[1],imagestest_1.shape[2])
imagestest_1=np.reshape(imagestest_1, newshapetest)
labelstest_1=np.reshape(labelstest_1, newshapetest)

print(imagestest_1.shape, labelstest_1.shape)

'''
#Path to save the model weights, the weights in torch, must be of extension .pt
savepath=r'E:\Arxius varis\PhD\2nd_year\Code\datasets\atomai_nn_trainingset_ferroic\modelFFTpeaks.pt'

# Initialize model
model = aoi.models.Segmentor(nb_classes=1)

# Train
model.fit(images_1, labels_1, imagestest_1, labelstest_1, training_cycles=500, compute_accuracy=True, swa=True) 

#If you already have the trained model and weights saved in a savepath, just load the model and weights
#this is for a given model, so the wieghts must match this model
#model.load_weights(r'E:\Arxius varis\PhD\1st_year\Built-in models - Codes - Weights\FerroicBlocks - DCNN\FerroicBlocks-master\saved_models\atomfinder-BFO-ts0-512-3-best_weights.pt')

#To save the model in a certain savepath, this means saving the weights and model CNN architecture
torch.save(model, savepath)

#to set the memory free again, delete the model, it is important if many models want to be
#loaded sequentially
#del model

'''
 
#Directly load the model that has been saved

#this model loading is general for any model architecture and weights
savepath=r'E:\Arxius varis\PhD\2nd_year\Code\datasets\atomai_nn_trainingset_ferroic\modelFFTpeaks.pt'

model1 = torch.load(savepath)

#Evaluate the model performance on a specific experimental image or simulated


nn_output, coordinates_1 =model1.predict(image1_1)
print(nn_output)
print(np.shape(nn_output))
plt.imshow(nn_output[0])
plt.show()
print(coordinates_1[0])
print(np.shape(coordinates_1[0]))

aoi.utils.plot_coord(image1_1, coordinates_1[0], fsize=12)

#del model1


#experimetnal image
imagedm3=hs.load(r'E:\Arxius varis\PhD\2nd_year\Code\trial_images\dm3_atomic_resolution\GeQW2.dm3')
meta1=imagedm3.metadata
meta2=imagedm3.original_metadata.export('parameters')



x_calibration=imagedm3.axes_manager['x'].scale
y_calibration=imagedm3.axes_manager['y'].scale

x_pixels_original=imagedm3.axes_manager['x'].size
y_pixels_original=imagedm3.axes_manager['y'].size

x_units=imagedm3.axes_manager['x'].units
y_units=imagedm3.axes_manager['y'].units

#%%
#FFT calibration

#FFT_calibration,FFT_pixels,FFT_units=FFT_calibration(imagedm3)



imagearray=np.asarray(imagedm3)
image=imagearray

plt.imshow(image, cmap=plt.cm.gray, vmin=image.min(), vmax=image.max())
plt.show()

window_size=1024*2 #window size of the sliding windows

#Crop the image if wanted
if window_size==x_pixels_original:
    init_x=0
    init_y=0
else:
    init_y=np.random.choice(np.arange(0,image.shape[0]-window_size,1)) 
    init_x=np.random.choice(np.arange(0,image.shape[1]-window_size,1)) 

#image=image[init_y:init_y+window_size,init_x:init_x+window_size]
hs_image_cropping=imagedm3.isig[init_y:init_y+window_size,init_x:init_x+window_size]


#Correct calibration in case it is necessary, if no correction is needed, just us 1
real_calibration_factor=1

print(x_calibration)
image=np.asarray(hs_image_cropping)
#State that the FFT has the same pixels as the windows created

#First standarisation of the image for filtering/blurring it with gaussian filter
image_st=(image-np.min(image))/np.max(image-np.min(image))

plt.imshow(image_st, cmap=plt.cm.gray, vmin=image_st.min(), vmax=image_st.max())
plt.show()


#Application of Gaussian filter for denoising


denoised_image=image_st
#denoised_image=cv2.GaussianBlur(image_st, gauss_blur_filter_size, 1)

#Second standarisation of the image after filtering/blurring it with gaussian filter

image_st=(denoised_image-np.min(denoised_image))/np.max(denoised_image-np.min(denoised_image))

#Print histogram



#For sake of evaluation, better work with an image with less pixels, as only the consecutive pixel evaluation would take
#approximately 6 hours to run for a big region of 250.000 pixels in total.

#Then downsample the image and upsample it posteriorly 
#We select a max pooling method to keep track of the brighter elements and this way keep a higher contrast


ds_image=image_st

#and standarise it again to ensure 0-1 values

ds_image_st=(ds_image-np.min(ds_image))/np.max(ds_image-np.min(ds_image))


# take the fft of the image
fft_image_w_background = np.fft.fftshift(np.log(np.fft.fft2(ds_image_st)))
fft_abs_image_background = np.abs(fft_image_w_background)

# apply the filter
fft_abs_image_backgroundc=np.copy(fft_abs_image_background)
fft_abs_image_background2=np.copy(fft_abs_image_background)


fft_abs_image_backgroundc=(fft_abs_image_backgroundc-np.min(fft_abs_image_backgroundc))/np.max(fft_abs_image_backgroundc-np.min(fft_abs_image_backgroundc))
fft_abs_image_background2=(fft_abs_image_background2-np.min(fft_abs_image_background2))/np.max(fft_abs_image_background2-np.min(fft_abs_image_background2))
gauss_blur_filter_size=(5,5)
fft_abs_image_background2=cv2.GaussianBlur(fft_abs_image_background2, gauss_blur_filter_size, 1)
fft_abs_image_background2=(fft_abs_image_background2-np.min(fft_abs_image_background2))/np.max(fft_abs_image_background2-np.min(fft_abs_image_background2))


nn_output, coordinates_1 =model1.predict(fft_abs_image_background2)
print(nn_output)
print(np.shape(nn_output))
plt.imshow(nn_output[0])
plt.show()
print(coordinates_1[0])
print(np.shape(coordinates_1[0]))

aoi.utils.plot_coord(fft_abs_image_background2, coordinates_1[0], fsize=12)

#del model1
