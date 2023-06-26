# -*- coding: utf-8 -*-
"""
Created on Fri Dec 18 10:04:15 2020

@author: Marc
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import scipy 
import skimage, skimage.measure
import sklearn.cluster, sklearn.mixture, sklearn.decomposition
import cv2
from keras_unet.models import vanilla_unet, custom_unet


#Applied process

image=plt.imread(r'E:\Arxius varis\PhD\2nd_year\Code\trial_images\nanowire1_tiff.tif')
plt.imshow(image, cmap=plt.cm.gray, vmin=image.min(), vmax=image.max())
plt.show()


#First standarisation of the image for filtering/blurring it with gaussian filter

image_st=(image-np.min(image))/np.max(image-np.min(image))
plt.hist(image_st.ravel(),256,[np.min(np.array([image_st])),np.max(np.array([image_st]))])
plt.show()

#Application of Gaussian filter for denoising

filter_size=(5,5)
denoised_image=cv2.GaussianBlur(image_st, filter_size, 1)
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

scaling_factor=8
ds_image=skimage.measure.block_reduce(image_st, block_size=tuple(np.int32(scaling_factor*np.ones(len(np.shape(image_st))))), func=np.mean, cval=0)

#and standarise it again to ensure 0-1 values

ds_image_st=(ds_image-np.min(ds_image))/np.max(ds_image-np.min(ds_image))


def PCA(image_array):
    #image_vect=image_array.ravel()
    image_array = image_array[:, ::-1]
    plt.imshow(image_array, cmap=plt.cm.gray, vmin=image_array.min(), vmax=image_array.max())
    plt.show()

    pca = sklearn.decomposition.PCA()
    pca.fit(image_array)
    plt.plot(pca.explained_variance_ratio_[0:50], '-o', linewidth=2, c='black')
    plt.xlabel('Number of components', fontsize = 16)
    plt.ylabel('Explained variance', fontsize = 16)
    plt.tick_params(labelsize = 14)
    plt.yscale("log")
    
    # Select number of components
    nc = 7 
    # Select decomposition type ('PCA' or 'NMF')
    decomposition_type = 'NMF'
    
    # Run decomposition
    if decomposition_type == 'NMF':
        clf = sklearn.decomposition.NMF(n_components=nc, random_state=42)
    elif decomposition_type == 'PCA':
        clf = sklearn.decomposition.PCA(n_components=nc, random_state=42)
    else:
        raise NotImplementedError('Available methods: "PCA", "NMF"')
    X_vec_t = clf.fit_transform(image_array)
    components = clf.components_
    components = components.reshape(nc, np.shape(image_array)[1])
    print(components.shape, X_vec_t.shape)
    
    rows = int(np.ceil(float(nc)/5))
    cols = int(np.ceil(float(nc)/rows))
    
    print('NUMBER OF COMPONENTS: ' + str(nc))
    print('Components...')
    gs1 = gridspec.GridSpec(rows, cols)
    fig1 = plt.figure(figsize = (4*cols, 3.5*(1+rows)//1.5))   
    for i in range(nc):
        ax1 = fig1.add_subplot(gs1[i])
        j = 0
        ax1.plot(components[i])
        ax1.set_title('Component {}'.format(i + 1))
    plt.show()
    
    print('Abundance maps...')
    gs2 = gridspec.GridSpec(rows, cols)
    fig2 = plt.figure(figsize = (4*cols, 4*(1+rows//1.5)))   
    for i in range(nc):
        ax2 = fig2.add_subplot(gs2[i])
        ax2.imshow(X_vec_t[:, i].reshape(np.shape(image_array)[1], 1), cmap = 'jet')
        ax2.set_title('Component {}'.format(i + 1))
    plt.show()


PCA(ds_image_st.T)
image2=cv2.imread(r'C:/Users/Marc/Desktop/bunicuss.tif')
image3=np.array([ds_image_st.reshape(128,128,1)])

print(np.shape(image3))
model = custom_unet(input_shape=(128,128,1))
pred=model.predict(image3)

print(pred)
print(np.shape(pred[0]))

plt.imshow(pred[0], cmap=plt.cm.gray, vmin=pred[0].min(), vmax=pred[0].max())