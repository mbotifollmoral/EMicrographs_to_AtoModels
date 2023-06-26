# -*- coding: utf-8 -*-
"""
Created on Thu Feb  3 18:25:02 2022

@author: Marc
"""

'''
Wrapper connecting Python to the GPA dll from Enzo/Vincezo's code

'''


import numpy as np
from numpy import fft
import matplotlib.pyplot as plt
import hyperspy.api as hs
import ctypes
from matplotlib.patches import Circle

plt.rcParams["figure.figsize"] = (12,12)

def FT(img):
    return fft.ifftshift(fft.fft2(fft.fftshift(img)))

def IFT(img):
    return fft.fftshift(fft.ifft2(fft.ifftshift(img)))

lib = ctypes.CDLL(r"E:\Arxius varis\PhD\3rd_year\Code\GPA_automation\GPA_dll\GPA_dll_v2\GPA.dll")

Handle = ctypes.POINTER(ctypes.c_char)
c_float_array = np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags='C_CONTIGUOUS')
c_int_array = np.ctypeslib.ndpointer(dtype=int, ndim=1, flags='C_CONTIGUOUS')

lib.createGPA.argtypes = None
lib.createGPA.restype = Handle

lib.deleteGPA.argtypes = [Handle]
lib.deleteGPA.restype = None

lib.load_img.argtypes = [Handle, c_float_array, c_int_array, ctypes.c_float]
lib.load_img.restypes = None

lib.mark_spot1.argtypes = [Handle,c_int_array, ctypes.c_int, c_float_array]
lib.mark_spot1.restypes = None

lib.mark_spot2.argtypes = [Handle,c_int_array, ctypes.c_int ,c_float_array]
lib.mark_spot2.restypes = None

lib.select_ref_area.argtypes = [Handle,c_int_array, c_int_array]
lib.select_ref_area.restypes = None

lib.calc_GPA.argtypes = [Handle]
lib.calc_GPA.restypes = None

lib.apply_rotation.argtypes = [Handle, ctypes.c_float] 
lib.apply_rotation.restypes = None

lib.get.argtypes = [Handle,c_float_array,c_float_array,c_float_array,c_float_array,c_float_array,c_float_array]
lib.get.restypes = None

class GPA:    
    def __init__(self):
        self.instance = lib.createGPA()
        self.ref_load = False
        self.Sp1_marked = False
        self.Sp2_marked = False
        self.get_ready = False

    def __del__(self):
        lib.deleteGPA(self.instance)   
        
    def load_image(self,img, calibration):
        #image = np.empty(self.Npix2, dtype=np.double)
        self.dim = img.shape
        size = np.asarray(self.dim,dtype = int)
        image = np.asarray(img.flatten(), dtype = np.float32)
        lib.load_img(self.instance, image, size, calibration*size[0])
        self.ref_load = True
    
    def mark_spot1(self, coordinates, win_size):
        if(self.ref_load != True):
            print("Load an image first")
            return np.empty((2,2), dtype=np.float32)
        amp = np.empty(self.dim[0]*self.dim[1], dtype=np.float32)
        coo = np.asarray(coordinates,dtype = int)
        lib.mark_spot1(self.instance,coo,win_size,np.asarray(amp, dtype=np.float32))
        self.Sp1_marked = True
        return amp.reshape(self.dim[0],self.dim[1])
    
    def mark_spot2(self, coordinates, win_size):
        if(self.ref_load != True):
            print("Load an image first")
            return np.empty((2,2), dtype=np.float32)
        amp = np.empty(self.dim[0]*self.dim[1], dtype=np.float32)
        coo = np.asarray(coordinates,dtype = int)
        lib.mark_spot2(self.instance,coo,win_size,np.asarray(amp, dtype=np.float32))
        self.Sp2_marked = True
        return amp.reshape(self.dim[0],self.dim[1])
    
    def select_ref_area(self, TL, BR):
        if(self.ref_load != True):
            print("Load an image first")
        tl = np.asarray(TL,dtype = int)    
        br = np.asarray(BR,dtype = int)
        lib.select_ref_area(self.instance,tl,br)
        
    def calc_GPA(self):
        if(self.ref_load and self.Sp1_marked and self.Sp2_marked == True):
            lib.calc_GPA(self.instance)
            self.get_ready = True
        else:
            print("Mark two diffraction spots first!")
            
    def apply_rotation(self, angle):
        lib.apply_rotation(self.instance, angle)
        
    def get(self):
        if(self.get_ready != True):
            x = np.empty((2,2), dtype=np.float32)
            print("run a calculation first!!")
            return x,x,x,x,x,x
        dxx = np.empty(self.dim[0]*self.dim[1], dtype=np.float32)
        dyy = np.empty(self.dim[0]*self.dim[1], dtype=np.float32)
        dxy = np.empty(self.dim[0]*self.dim[1], dtype=np.float32)
        dyx = np.empty(self.dim[0]*self.dim[1], dtype=np.float32)
        rot = np.empty(self.dim[0]*self.dim[1], dtype=np.float32)
        shear = np.empty(self.dim[0]*self.dim[1], dtype=np.float32)
        lib.get(self.instance, dxx, dyy, dxy, dyx, rot, shear)
        dxx = dxx.reshape(self.dim[0],self.dim[1])
        dyy = dyy.reshape(self.dim[0],self.dim[1])
        dxy = dxy.reshape(self.dim[0],self.dim[1])
        dyx = dyx.reshape(self.dim[0],self.dim[1])
        rot = rot.reshape(self.dim[0],self.dim[1])
        shear = shear.reshape(self.dim[0],self.dim[1])
        return dxx, dyy, dxy, dyx, rot, shear
 
    
#Adjust reference to the calculated images and phases function??    

    
def GPA_full(image_array,calibration,spot1, spot2, mask_size, reference, rotation_angle,display=True):
    '''
    Full GPA process, to call the full wrapped dll at once with all the necessary inputs.
    spot1 and spot2 must be tuples or lists with two elements, first the y coord (row) and then 
    x coord (col), in pixels, referring to the IMAGES reference system and not cartesian
    Mask size must be in pixels as well
    reference, coords in pixels, of the top-left corner and bottom-right corner of the rectangle/square
    referring to the reference, in format: row_i (yi) , col_i(xi), row_f (yf) , col_f (xf)
    Display =True to show the images of the process

    Parameters
    ----------
    image_array : 
    calibration : 
    spot1 : 
    spot2 : 
    mask_size : 
    reference:
    rotation_angle : angle to the x axis to the first g vector or spot, in degrees

    Returns
    -------
    dxx, dyy, dxy, dyx, rot, shear

    '''
    #image system
    # row1,col1=spot1
    # row2,col2=spot2
    
    #cartesian system
    Sp1Y,Sp1X = spot1
    Sp2Y,Sp2X = spot2
    
    row_i , row_f, col_i , col_f = reference
    
    Analysis = GPA()
    
    Analysis.load_image(image_array,calibration)
    
    #---
    # amp1 = Analysis.mark_spot1((Sp1X,Sp1Y),mask_size)
    
    # amp2 = Analysis.mark_spot2((Sp2X,Sp2Y),mask_size)
    
    # Analysis.calc_GPA()
    
    # Analysis.apply_rotation(rotation_angle)
        
    # exx, eyy, exy, eyx, rot, shear = Analysis.get()
    # #---
    
    
    # reference, watch out about the order of the parameters as the function demands top left and 
    # bottom right but in col (x), and row (y) order, and no need to add 1 as these are coordinates
    # not axes to be cropped (in the plotting we do crop)
    Analysis.select_ref_area((col_i,row_i),(col_f,row_f))
    
    amp1 = Analysis.mark_spot1((Sp1X,Sp1Y),mask_size)
    
    amp2 = Analysis.mark_spot2((Sp2X,Sp2Y),mask_size)
    
    Analysis.calc_GPA()
    
    Analysis.apply_rotation(rotation_angle)
        
    exx, eyy, exy, eyx, rot, shear = Analysis.get()
    
    #Adjust reference to the calculated images and phases function?? 

    if display==True:
        #show the spots positions in the FFT and the strain components
        
        #FFT and chosen spot positions
        fft_img = np.log(abs(FT(image_array)))
        fig,ax = plt.subplots(1,constrained_layout=True)
        fig.set_size_inches(12, 12)
        fig.suptitle('FFT and chosen g vectors',fontsize=18)
        ax.imshow(fft_img,interpolation='nearest', cmap='gray')
        ax.scatter(Sp1X, Sp1Y, c="red", marker="x")
        circle = Circle((Sp1X, Sp1Y), mask_size, facecolor='none',edgecolor="red", linewidth=1, alpha=1)
        ax.add_patch(circle)
        ax.scatter(Sp2X, Sp2Y, c="blue", marker="x")
        circle = Circle((Sp2X, Sp2Y), mask_size, facecolor='none',edgecolor="blue", linewidth=1, alpha=1)
        ax.add_patch(circle)
        
        #Calculated phase
        fig,ax=plt.subplots(1,2, constrained_layout=True)
        fig.set_size_inches(12, 6)
        fig.suptitle('Spot intensity maps',fontsize=18)
        ax[0].imshow(amp1,interpolation='nearest', cmap='gray')
        ax[0].set_title('Spot 1 (g1)')
        ax[1].imshow(amp2,interpolation='nearest', cmap='gray')
        ax[1].set_title('Spot 2 (g2)')
        
        #Calculated components

        c = 2
        clims_fix=(-0.05,0.05)
        clims_fix_rot=(-5,5)
        cm = 'jet'
        colorbarshrink=1
        fig,ax=plt.subplots(3,2, constrained_layout=True)
        fig.set_size_inches(16, 18)
        fig.suptitle('Calculated components', fontsize=18)
        M = exx[row_i:row_f+1,col_i:col_f+1].mean()
        S = exx[row_i:row_f+1,col_i:col_f+1].std()
        clims_ranged=(M-c*S,M+c*S)
        im=ax[0][0].imshow(exx,interpolation='nearest', cmap=cm, clim=clims_fix)
        ax[0][0].set_title('e_xx')
        fig.colorbar(im, ax=ax[0][0], shrink=colorbarshrink)
        M = eyy[row_i:row_f+1,col_i:col_f+1].mean()
        S = eyy[row_i:row_f+1,col_i:col_f+1].std()
        clims_ranged=(M-c*S,M+c*S)
        im=ax[1][1].imshow(eyy,interpolation='nearest', cmap=cm, clim=clims_fix)
        ax[1][1].set_title('e_yy')
        fig.colorbar(im, ax=ax[1][1], shrink=colorbarshrink)
        M = exy[row_i:row_f+1,col_i:col_f+1].mean()
        S = exy[row_i:row_f+1,col_i:col_f+1].std()
        clims_ranged=(M-c*S,M+c*S)
        im=ax[0][1].imshow(exy,interpolation='nearest', cmap=cm, clim=clims_fix)
        ax[0][1].set_title('e_xy')
        fig.colorbar(im, ax=ax[0][1], shrink=colorbarshrink)
        M = eyx[row_i:row_f+1,col_i:col_f+1].mean()
        S = eyx[row_i:row_f+1,col_i:col_f+1].std()
        clims_ranged=(M-c*S,M+c*S)
        im=ax[1][0].imshow(eyx,interpolation='nearest', cmap=cm, clim=clims_fix)
        ax[1][0].set_title('e_yx')
        fig.colorbar(im, ax=ax[1][0], shrink=colorbarshrink)
        M = rot[row_i:row_f+1,col_i:col_f+1].mean()
        S = rot[row_i:row_f+1,col_i:col_f+1].std()
        clims_ranged=(M-c*S,M+c*S)
        im=ax[2][0].imshow(rot,interpolation='nearest', cmap=cm, clim=clims_fix_rot)
        ax[2][0].set_title('Rotation') 
        fig.colorbar(im, ax=ax[2][0], shrink=colorbarshrink)
        M = shear[row_i:row_f+1,col_i:col_f+1].mean()
        S = shear[row_i:row_f+1,col_i:col_f+1].std()
        clims_ranged=(M-c*S,M+c*S)
        im=ax[2][1].imshow(shear,interpolation='nearest', cmap=cm, clim=clims_fix)
        ax[2][1].set_title('Shear')  
        fig.colorbar(im, ax=ax[2][1], shrink=colorbarshrink)
        plt.show()
        
    return exx, eyy, exy, eyx, rot, shear 
        
        
'''

# Example for calling the function GPA_full     

    
#Load trial image

img = hs.load(r'E:\Arxius varis\PhD\3rd_year\Code\GPA_automation\GPA_dll\GeQW2-wien-rebin.dm3')
img = np.array(img)

plt.imshow(img,interpolation='nearest', cmap='gray') 

print(type(img))
print(type(img[0,0]))

#Needed hyperparameters

mask_size = 15



Sp1X = 694
Sp1Y = 514

spot1=[Sp1Y,Sp1X]

Sp2X = 420
Sp2Y = 448

spot2=[Sp2Y,Sp2X]
#GPA calculation


calibration=1
reference=[256,256,768,768]
rotation_angle=-0.3

exx, eyy, exy, eyx, rot, shear =GPA_full(img, calibration,spot1, spot2, mask_size, reference, rotation_angle,display=True)


'''   