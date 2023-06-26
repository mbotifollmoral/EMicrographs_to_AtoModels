import numpy as np
from scipy.signal import find_peaks
import cv2
import ImageTreatment_and_interfaces as ii
import matplotlib.pyplot as plt

#Global variables
treshold = 20
delta = 10



# Function to convert puntual signal in circular
def cercles(sigma,matrix):
   
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
    size = 2*m
    

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
    if plot:
        plt.figure(figsize=(10, 10)) 
        plt.imshow(added_image)
        #plt.imshow(peaks_color)
    
    return added_image, fft_color, peaks_color

def centering(lado, img):
    
    index = np.where(img)
  
    n = len(img)
    final_img = np.zeros((n,n))+img
    
    sin_marcos = np.zeros((n,n))
    sin_marcos[lado : n - lado, lado : n - lado ] = 1
    final_img = final_img * sin_marcos
                                   
    
    for i,j in zip(index[0],index[1]):
        
        if final_img[i,j] > 0.5:
            new_i = 0
            new_j = 0
            k = 0
            
            while i != new_i or j != new_j:
                
                if k > 0:
                    i = new_i
                    j = new_j
                
                
                recuadro = final_img[i-lado//2 : i + lado//2+1, 
                               j - lado//2 : j + lado//2+1]
                
                points = np.where(recuadro>0.5)
                new_i = int(np.mean(points[0])) + i - lado//2
                new_j = int(np.mean(points[1])) + j - lado//2
               
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



def peaks_recognition(side,treshold, prediction):
    
    size = 256
    edges = size//8
    index = np.asarray(range(edges , size - edges))
        
    zoomed_prediction = cv2.resize(prediction[0], dsize=(size, size), interpolation=cv2.INTER_LINEAR)
    prediction_without_backgraund = zoomed_prediction[index][:,index] > treshold
    peaks_image = centering(side,prediction_without_backgraund)
    
    #peaks_image = cv2.resize(peaks_image, dsize=(128, 128), interpolation=cv2.INTER_LINEAR)
    
    peaks_image_s = symetrize(peaks_image)
    
    
    return peaks_image_s,prediction_without_backgraund