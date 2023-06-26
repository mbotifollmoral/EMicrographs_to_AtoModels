# -*- coding: utf-8 -*-
"""
Created on Fri Sep 11 12:40:15 2020

@author: Marc
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy 
import skimage
import sklearn.cluster
import cv2
import sklearn.feature_extraction
import sklearn.mixture



image=plt.imread(r'E:\Arxius varis\PhD\2nd_year\Code\trial_images\low_mag_images\nanowire2_tiff.tif')
img2=cv2.imread(r'C:/Users/Marc/Desktop/bunicuss.tif')
img2_gray = skimage.color.rgb2gray(img2)



#K-Means clustering

#Source: Instruments & Data Tools
#Inspired from the Vector Quantization Example

#use multiple random initializations and therefore get the minimum cost for the clusters and set this 
#k means as the good one fot he final clustering
#The number of clusters should be the number of different materials we want to get (maybe Pt requires two clusters for the identification 
#as it is a mixture of black and white) 


def km_clust(array, n_clusters):
    # Create a line array, the lazy way
    X = array.reshape((-1, 1))
    # Define the k-means clustering problem
    k_m = sklearn.cluster.KMeans(n_clusters=n_clusters, init='random',n_init=4,random_state=2**13)
    # Solve the k-means clustering problem
    k_m.fit(X)
    # Get the coordinates of the clusters centres as a 1D array
    values = k_m.cluster_centers_.squeeze()
    # Get the label of each point
    labels = k_m.labels_
    cost=k_m.inertia_
    return(values, labels, cost)

def best_km(array, n_clusters):
    iterations=100
    kmeans_values=[]
    kmeans_labels=[]
    kmeans_cost=[]
    for index,i in enumerate(range(iterations)):
        values, labels, cost= km_clust(array, n_clusters = n_clusters)
        kmeans_values.append(values)
        kmeans_labels.append(labels)
        kmeans_cost.append(cost)
    kmeans_cost=np.array(kmeans_cost)
    best_cost=kmeans_cost.min()    
    best_values=kmeans_values[kmeans_cost.argmin()]
    best_labels=kmeans_labels[kmeans_cost.argmin()]
    return (best_values, best_labels,best_cost)

#Spectral clustering

def spectral_clustering(image_,n_clusters):
    graph = sklearn.feature_extraction.image.img_to_graph(image_)
    print(graph)
    print(type(graph), graph.shape[0])
    print(graph.data)
    print(type(graph.data), len(graph.data))
    # Take a decreasing function of the gradient: we take it weakly
    # dependent from the gradient the segmentation is close to a voronoi
    beta = 1
    eps = 1e-6
    graph.data = np.exp(-beta * graph.data / graph.data.std()) + eps
    #graph.data = np.exp(-graph.data / graph.data.std())
    # Force the solver to be arpack, since amg is numerically
    # unstable on this example
    labels = sklearn.cluster.spectral_clustering(graph, n_clusters=n_clusters, eigen_solver='amg')
    labels = labels.reshape(image_.shape)
   
    return labels


#Mean shift clustering algorithm

def Mean_shift(image):
    image_reshaped=np.reshape(image, (image.shape[0]**2,1))
    # The following bandwidth can be automatically detected using
    #bandwidth = sklearn.cluster.estimate_bandwidth(image, quantile=0.2, n_samples=500)
    
    ms = sklearn.cluster.MeanShift( bin_seeding=True)
    ms.fit(image_reshaped)
    labels = ms.labels_
    cluster_centers = ms.cluster_centers_
    
    return labels, cluster_centers


#try guassian mixture modelling to cluster
def GMM(image_,number_clusters):
    
    image_reshaped=np.reshape(image_, (image_.shape[0]*image_.shape[1],1))
    gmm = sklearn.mixture.GaussianMixture(n_components=number_clusters)
    gmm.fit(image_reshaped)
    
    #predictions from gmm
    labels = gmm.predict(image_reshaped)
    
    labels.shape=image_.shape
    
    return labels



# Read the data as greyscale 


img_norm=(img-np.min(img))/np.max(img)

img=img_norm
   
blur=cv2.GaussianBlur(img, (5,5), 1)

# blur1=skimage.measure.block_reduce(blur, block_size=tuple(np.int32(4*np.ones(len(np.shape(blur))))), func=np.mean, cval=0)
# plt.imshow(blur,cmap=plt.cm.gray )
# plt.show()
# labels_ms,clusters_ms=Mean_shift(blur1)
# print(labels_ms, len(labels_ms))
# print(clusters_ms, len(clusters_ms))


# labels_ms_reshaped=np.reshape(labels_ms, np.shape(blur1))
# img_segm = np.choose(labels_ms, clusters_ms)

# plt.imshow(labels_ms_reshaped,cmap=plt.cm.gray )
# plt.show()

# labels_ms_reshaped=scipy.ndimage.zoom(labels_ms_reshaped, 4, order=1)
# plt.imshow(labels_ms_reshaped,cmap=plt.cm.gray )
# plt.show()

#GMM
gmm_clustering=GMM(blur,2)
print(gmm_clustering.shape)
plt.imshow(gmm_clustering,cmap=plt.cm.gray)
plt.show()


#GMM
gmm_clustering=GMM(blur,4)
print(gmm_clustering.shape)
plt.imshow(gmm_clustering,cmap=plt.cm.gray)
plt.show()

#GMM
gmm_clustering=GMM(blur,5)
print(gmm_clustering.shape)
plt.imshow(gmm_clustering,cmap=plt.cm.gray)
plt.show()


#Let us identify where in the image each cluster comes from, to start relating the cluster with a position
fig3 = plt.figure(3)


labelx=0
labels_position=labels_ms==labelx
labels_position_neg=labels_ms!=labelx
label_final_pos=labels_position*clusters_ms[labelx]+labels_position_neg*0
print(label_final_pos, np.shape(labels_position_neg))

labels_position_count=labels_ms[labels_ms==labelx]
print(labels_position_count.shape)
print(100*len(labels_position_count)/len(labels_position),'%')
label_final_pos.shape = blur1.shape

ax1 = fig3.add_subplot(2,2,1)
ax1.imshow(label_final_pos, cmap=plt.cm.gray)
ax1.set_title('Label positions')


labelx=1
labels_position=labels_ms==labelx
labels_position_neg=labels_ms!=labelx
label_final_pos=labels_position*clusters_ms[labelx]+labels_position_neg*0
print(label_final_pos, np.shape(labels_position_neg))

labels_position_count=labels_ms[labels_ms==labelx]
print(labels_position_count.shape)
print(100*len(labels_position_count)/len(labels_position),'%')
label_final_pos.shape = blur1.shape

ax1 = fig3.add_subplot(2,2,2)
ax1.imshow(label_final_pos, cmap=plt.cm.gray)
ax1.set_title('Label positions')


labelx=2
labels_position=labels_ms==labelx
labels_position_neg=labels_ms!=labelx
label_final_pos=labels_position*clusters_ms[labelx]+labels_position_neg*0
print(label_final_pos, np.shape(labels_position_neg))

labels_position_count=labels_ms[labels_ms==labelx]
print(labels_position_count.shape)
print(100*len(labels_position_count)/len(labels_position),'%')
label_final_pos.shape = blur1.shape

ax1 = fig3.add_subplot(2,2,3)
ax1.imshow(label_final_pos, cmap=plt.cm.gray)
ax1.set_title('Label positions')

plt.show()











print(blur.shape, type(blur))
labels_spec=spectral_clustering(blur1,n_clusters=3)
print(len(labels_spec), labels_spec)
labels_spec=labels_spec/labels_spec.max()
plt.imshow(labels_spec,cmap=plt.cm.gray )
plt.show()

# Group similar grey levels using 8 clusters
values, labels, cost = km_clust(blur, n_clusters =3 )
print(values)
print(labels)
print(cost)

#Choose the best K-means clustering regarding the cost each of the kmeans has

# values1, labels1, cost1=best_km(img, n_clusters = 3)
# print(values1)
# print(labels1)
# print(cost1)

# Create the segmented array from labels and values
img_segm = np.choose(labels, values)
# Reshape the array as the original image
img_segm.shape = img.shape
# Get the values of min and max intensity in the original image
vmin = img.min()
vmax = img.max()
fig = plt.figure(1)
# Plot the original image
ax1 = fig.add_subplot(1,2,1)
ax1.imshow(blur,cmap=plt.cm.gray, vmin=vmin, vmax=vmax)
ax1.set_title('Original image')

ax2 = fig.add_subplot(1,2,2)
ax2.imshow(img_segm, cmap=plt.cm.gray, vmin=vmin, vmax=vmax)
ax2.set_title('Simplified levels')
#Denoising the image with a Gaussian filter (does it improve the segmentation?)

plt.show()




#Let us identify where in the image each cluster comes from, to start relating the cluster with a position

fig1 = plt.figure(2)
print(np.shape(labels))
print(labels.min(), labels.max())
print(type(labels))
labelx=0
labels_position=labels==labelx
labels_position_neg=labels!=labelx
print(labels_position, np.shape(labels_position))
print(labels_position_neg, np.shape(labels_position_neg))
label_final_pos=labels_position*values[labelx]+labels_position_neg*1
print(label_final_pos, np.shape(labels_position_neg))

labels_position_count=labels[labels==labelx]
print(labels_position_count.shape)
print(100*len(labels_position_count)/len(labels_position),'%')
label_final_pos.shape = img.shape

ax1 = fig1.add_subplot(2,2,1)
ax1.imshow(label_final_pos, cmap=plt.cm.gray, vmin=vmin, vmax=vmax)
ax1.set_title('Label positions')


labelx=1
labels_position=labels==labelx
labels_position_neg=labels!=labelx
label_final_pos=labels_position*values[labelx]+labels_position_neg*1

labels_position_count=labels[labels==labelx]
print(labels_position_count.shape)
print(100*len(labels_position_count)/len(labels_position),'%')
label_final_pos.shape = img.shape

ax2 = fig1.add_subplot(2,2,2)
ax2.imshow(label_final_pos, cmap=plt.cm.gray, vmin=vmin, vmax=vmax)
ax2.set_title('Label positions')

labelx=2
labels_position=labels==labelx
labels_position_neg=labels!=labelx
label_final_pos=labels_position*values[labelx]+labels_position_neg*1

labels_position_count=labels[labels==labelx]
print(labels_position_count.shape)
print(100*len(labels_position_count)/len(labels_position),'%')
label_final_pos.shape = img.shape

ax3 = fig1.add_subplot(2,2,3)
ax3.imshow(label_final_pos, cmap=plt.cm.gray, vmin=vmin, vmax=vmax)
ax3.set_title('Label positions')

labelx=3
labels_position=labels==labelx
labels_position_neg=labels!=labelx
label_final_pos=labels_position*values[labelx]+labels_position_neg*1

labels_position_count=labels[labels==labelx]
print(labels_position_count.shape)
print(100*len(labels_position_count)/len(labels_position),'%')
label_final_pos.shape = img.shape

ax4 = fig1.add_subplot(2,2,4)
ax4.imshow(label_final_pos, cmap=plt.cm.gray, vmin=vmin, vmax=vmax)
ax4.set_title('Label positions')

plt.show()