import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from collections import Counter
import scipy.stats as stats
import cv2
import sys

sys.path.append(r'E:\Arxius varis\PhD\4rth_year\Code\Functions\Peak_detector_Final')
import PF_ImageTreatment_and_interfaces as PF_II




def reduce_image(n_image, size, plot = False):
    
    n = size
    reduce_image = np.zeros((n,n))
    total_pixels = len(n_image)
    ratio = int(total_pixels/n)

    for i in range(n):
        for j in range(n):
            reduce_image[i,j] = np.mean(n_image[i*ratio:i*ratio+ratio, j*ratio:j*ratio+ratio])
    if plot:     
        plt.figure(figsize=(9, 6))
        plt.subplot(121),plt.imshow(n_image, cmap=plt.cm.gray)
        plt.subplot(122),plt.imshow(reduce_image, cmap=plt.cm.gray)
        plt.show()
    
    return reduce_image



def reduce_image_by_steps(n_image, size, plot = False):
    
    
    # in steps
    # with open cv:
    # single step process     
    # reduce_image = cv2.resize(n_image, (size,size), interpolation = cv2.INTER_NEAREST)
     
        
        
        
    orig_image_size = n_image.shape[0]
    

    # size will typically be around 256
    # let us do the resizing in steps of 2
    resize_step = 2
    
    if orig_image_size == size:
        return n_image
    elif orig_image_size > size:
        old_over_new = orig_image_size/size
        steps = int(old_over_new//resize_step)
        
        reduce_image = np.copy(n_image)
        
        for step in range(1, steps+1):
            
            partial_size = int(np.floor((old_over_new/step)*size))
            
            reduce_image = cv2.resize(reduce_image, (partial_size,partial_size), interpolation = cv2.INTER_LINEAR)
            plt.imshow(reduce_image)
            plt.show()
            
            plt.imshow(np.abs(np.log(np.fft.fftshift(np.fft.fft2(reduce_image)))))
            plt.show()
            
            
        reduce_image = cv2.resize(reduce_image, (size,size), interpolation = cv2.INTER_LINEAR)
        plt.imshow(reduce_image)
        plt.show()
        
        plt.imshow(np.abs(np.log(np.fft.fftshift(np.fft.fft2(reduce_image)))))
        plt.show()

        
    else:
        # orig_image_size < size
        
        old_over_new = 1/(orig_image_size/size)
        steps = int(old_over_new//resize_step)
        
        reduce_image = np.copy(n_image)
        
        
        for step in range(1, steps + 1):
            
            partial_size = orig_image_size*step
            
            reduce_image = cv2.resize(reduce_image, (partial_size, partial_size), interpolation = cv2.INTER_LINEAR)
            
        reduce_image = cv2.resize(reduce_image, (size, size), interpolation = cv2.INTER_LINEAR)
        
        
    reduce_image_f = cv2.resize(n_image, (size,size), interpolation = cv2.INTER_LINEAR)
    plt.imshow(reduce_image_f)
    plt.show()
    
    plt.imshow(np.abs(np.log(np.fft.fftshift(np.fft.fft2(reduce_image_f)))))
    plt.show()
    
    plt.imshow(reduce_image_f - reduce_image)
    plt.show()
    
    plt.imshow(np.abs(np.log(np.fft.fftshift(np.fft.fft2(reduce_image_f)))) - np.abs(np.log(np.fft.fftshift(np.fft.fft2(reduce_image)))))
    plt.show()
    
    
    
   
    # in steps
    # with pil image
    # turn array into Pillow image object
    
    # from PIL import Image
    # image_PIL = Image.from array function (n_image)
    # image_PIL.resize((size, size), Image.NEAREST)
    # reduce_image = image_PIL.to array function ()
    
    if plot:     
        plt.figure(figsize=(9, 6))
        plt.subplot(121),plt.imshow(n_image, cmap=plt.cm.gray)
        plt.subplot(122),plt.imshow(reduce_image, cmap=plt.cm.gray)
        plt.show()
    
    return reduce_image






def dm3_to_data(reduce_image, scale,rescale = 1, partial_img = False):

    n = len(reduce_image)
    
    if partial_img:
        index = np.where(reduce_image > 0)
        vector_i = index[0]
        vector_j = index[1]
        vector_image = []

        for i,j in zip(index[0], index[1]):
            vector_image.append(reduce_image[i,j])

        vector_image = np.asarray(vector_image)
        
    else:    
        mj = np.arange(n) 
        mj = np.asarray([mj]*n)
        mi = mj.T

        vector_i = PF_II.normalize_image(mi.reshape(n**2))
        vector_j = PF_II.normalize_image(mj.reshape(n**2))
        vector_image = PF_II.normalize_image(reduce_image.reshape(n**2))

    X=np.array([vector_i/rescale,vector_j/rescale,vector_image*scale])
    X = X.T
    
    return X


def k_means(X, n_clusters,index = [0,0], n = 0, partial_img = False):
    kmeans = KMeans(n_clusters = n_clusters).fit(X)
    labels = kmeans.predict(X)
    
    if partial_img:
        l = -np.ones((n,n))
        k = 0

        for i,j in zip(index[0], index[1]):
            l[i,j] = labels[k] + 1
            k += 1
    else:    
        n = int(np.sqrt(len(X)))
        l = labels.reshape(n,n) + 1

    return l


def dbscan(X, epsilon , n_samples):
    db = DBSCAN(eps = epsilon , min_samples = n_samples).fit(X)
    labels = db.labels_
    n = int(np.sqrt(len(X)))
    l = labels.reshape(n,n) + 1
    
    return l


def agglomerative(X, n_clusters):
    ag = AgglomerativeClustering(n_clusters = n_clusters).fit(X)
    labels = ag.labels_
    n = int(np.sqrt(len(X)))
    l = labels.reshape(n,n) + 1
    
    return l


def border(reduce_image, sigma, border_size, vmax, vmin = 0):
    
    gauss = cv2.GaussianBlur(reduce_image, (sigma,sigma), 0)
    gauss_256 = np.uint8(256*(gauss))
    
    canny = cv2.Canny(gauss_256, vmin , vmax)
    
    (contornos,_) = cv2.findContours(canny.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    r_image = reduce_image.copy()

    cv2.drawContours(r_image,contornos,-1,0, border_size)
    
    n_z, matrix = agrupar(r_image)
    n_z = int(np.max(matrix))
    
    for i in range(n_z + 1):
        if np.sum(matrix == i) < 100:
            matrix[np.where(matrix == i)] = -1
        
    l = matrix + 1
    #l = remove_noise(matrix+1)
    
    return l     


def automate_segmentation(image, number_of_pixels = 256, plot = True):
    scale = image.axes_manager['x'].scale * len(image.data)
    n_img = PF_II.normalize_image(image.data)
    r_image = reduce_image(n_img, number_of_pixels)
    
    if scale > 500:
        sigma, border_size, vmax= 11, 2, 50 
    elif scale > 200: 
        sigma, border_size, vmax= 13, 3, 50 
    elif scale > 50:
        sigma, border_size, vmax= 11, 5, 20
    else:
        sigma, border_size, vmax= 13, 5, 10
        
    labels_canny = border(r_image, sigma, border_size,  vmax)
    labels_wout_noise_canny = reagrupar(labels_canny) 

    if plot:
        plt.figure(figsize=(12, 6))
        plt.subplot(121),plt.imshow(labels_canny)
        plt.subplot(122),plt.imshow(labels_wout_noise_canny) 
        plt.plot()
        
    return labels_wout_noise_canny


def remove_noise(seg_matrix):
    
    index = np.where(seg_matrix == 0)
    n = 256
    
    while(len(index[0]) > 0):
        for i,j in zip(index[0],index[1]):
            
            xl = i-1
            xr = i+2
    
            yl = j-1
            yr = j+2
            
            if i == 0:
                xl = 0 
    
            if i == (n-1):
                xr = n
    
            if j == 0:
                yl = 0 
    
            if j == (n-1):
                yr = n
    
            recuadro = seg_matrix[xl:xr, yl:yr]
            seg_matrix[i,j] = np.max(recuadro)
        index = np.where(seg_matrix == 0)
    
    return seg_matrix
    

def agrupar(img):
    
    index = np.where(img)
    n = len(img)
    
    matrix_zones = -np.ones((n,n))
    final_matrix = -np.ones((n,n))
    
    n_zonas = 0
    final_zonas = 0
    l = 0  

    for i,j in zip(index[0],index[1]):
            
        xl = i-1
        xr = i+2

        yl = j-1
        yr = j+2

        if i == 0:
            xl = 0 
        if i == (n-1):
            xr = n
        if j == 0:
            yl = 0 
        if j == (n-1):
            yr = n
            
        recuadro = matrix_zones[xl:xr, yl:yr]
        adyacencia = recuadro > -1    
 
        if np.any(adyacencia):

            zonas_col = Counter(recuadro[np.where(recuadro>-1)]).keys()
            n_zonas_col= len(zonas_col)

            if n_zonas_col == 1:

                k = np.where(recuadro>-1)
                zona = int(recuadro[k][0])
                matrix_zones[i][j]  = zona

            else:
                minim = min(zonas_col)
  
                for k in zonas_col:
                    matrix_zones[np.where(matrix_zones==k)] = minim

        else: 
            n_zonas += 1
            matrix_zones[i][j] = n_zonas-1

        l += 1    
        
    #Remove empty zones
    for i in range(n_zonas):
        if np.any(matrix_zones == i):
            final_matrix[np.where(matrix_zones == i)] = final_zonas
            final_zonas += 1
            

    return final_zonas, final_matrix
    
    
def reagrupar(img):
    n = len(img)
    img_r = np.zeros((n,n))
    m = int(np.max(img)+1)
    
    # Remove separeted regions and keep the biggest
    for k in range(1,m):
        zone = img*(img==k)
        n_zones, matrix_zones = agrupar(zone)
      
        matrix_zones_wb = matrix_zones[np.where(matrix_zones!=-1)]
        mode = stats.mode(matrix_zones_wb,axis=None)[0]
      
        new_zone  = zone*(matrix_zones==mode)
        
        img_r += new_zone
    
    # Reordenate the different regions
    index = np.where(img_r==0)
    
    for i,j in zip(index[0],index[1]):
        
        if j<1: 
            #img_r[i,j] = np.max(img_r[i-1:i+2,j:j+2])
            img_r[i,j] = img_r[i,1]
            
        elif j>n-2: 
            img_r[i,j] = img_r[i,j-1] 
            
        elif i<1: 
            #img_r[i,j] = np.max(img_r[i-1:i+2,j:j+2])
            img_r[i,j] = img_r[1, j]
            
        elif i>n-2: 
            img_r[i,j] = img_r[i-1,j] 
            
        else:
            recuadro = img_r[i-1:i+2][:,j-1:j+2]
            img_r[i,j] = np.max(recuadro)

    
    return img_r 
    
    
def split_separated_regions(l, pixels, clusters):
    
    img_seg = np.zeros((pixels,pixels))
    zone = 1
    
    for i in range(clusters):
        
        img_zone = (l == i + 1)
        
        n, seg_image = agrupar(img_zone)

        for j in range(n):

            if len(np.where(seg_image == j)[0]) < pixels:
                img_seg[np.where(seg_image == j)] = -2

            else:
                img_seg[np.where(seg_image == j)] = zone
                zone += 1

    return zone, img_seg   
    
    
    
    
    
    
    