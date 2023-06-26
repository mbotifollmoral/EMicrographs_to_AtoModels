# -*- coding: utf-8 -*-
"""
Created on Sun Feb 12 17:55:07 2023

@author: Marc
"""

import numpy as np
import re
import matplotlib.pyplot as plt
from PIL import Image
from time import time
import pyfftw 
import os
import sys
import ctypes
from matplotlib.patches import Circle, Rectangle



def FT(img):
    return np.fft.ifftshift(np.fft.fft2(np.fft.fftshift(img)))

def IFT(img):
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(img)))



lib = ctypes.CDLL(r"E:\Arxius varis\PhD\4rth_year\Code\Strain_to_atomistic\GPA_to_atom_dll\GPA_to_atom.dll")
Handle = ctypes.POINTER(ctypes.c_char)
c_float_array = np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags='C_CONTIGUOUS')
c_int_array = np.ctypeslib.ndpointer(dtype=int, ndim=1, flags='C_CONTIGUOUS')

lib.createGPA.argtypes = None
lib.createGPA.restype = Handle

lib.deleteGPA.argtypes = [Handle]
lib.deleteGPA.restype = None

lib.load_img.argtypes = [Handle, c_float_array, c_int_array, ctypes.c_float]
lib.load_img.restypes = None

lib.mark_spot1.argtypes = [Handle,c_int_array, ctypes.c_int, c_float_array, c_float_array]
lib.mark_spot1.restypes = None

lib.mark_spot2.argtypes = [Handle,c_int_array, ctypes.c_int ,c_float_array, c_float_array]
lib.mark_spot2.restypes = None

lib.select_ref_area.argtypes = [Handle,c_int_array, c_int_array]
lib.select_ref_area.restypes = None

lib.calc_GPA.argtypes = [Handle]
lib.calc_GPA.restypes = None

lib.apply_rotation.argtypes = [Handle, ctypes.c_float] 
lib.apply_rotation.restypes = None

lib.get.argtypes = [Handle,c_float_array,c_float_array,c_float_array,c_float_array,c_float_array,c_float_array,c_float_array,c_float_array]
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
        self.refined_spot1 = np.empty(2, dtype=np.float32)
        lib.mark_spot1(self.instance,coo,win_size,np.asarray(amp, dtype=np.float32),np.asarray(self.refined_spot1, dtype=np.float32))
        self.Sp1_marked = True
        return amp.reshape(self.dim[0],self.dim[1])
    
    def mark_spot2(self, coordinates, win_size):
        if(self.ref_load != True):
            print("Load an image first")
            return np.empty((2,2), dtype=np.float32)
        amp = np.empty(self.dim[0]*self.dim[1], dtype=np.float32)
        coo = np.asarray(coordinates,dtype = int)
        self.refined_spot2 = np.empty(2, dtype=np.float32)
        lib.mark_spot2(self.instance,coo,win_size,np.asarray(amp, dtype=np.float32),np.asarray(self.refined_spot2, dtype=np.float32))
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
        Dispx = np.empty(self.dim[0]*self.dim[1], dtype=np.float32)
        Dispy = np.empty(self.dim[0]*self.dim[1], dtype=np.float32)
        lib.get(self.instance, dxx, dyy, dxy, dyx, rot, shear, Dispx, Dispy)
        dxx = dxx.reshape(self.dim[0],self.dim[1])
        dyy = dyy.reshape(self.dim[0],self.dim[1])
        dxy = dxy.reshape(self.dim[0],self.dim[1])
        dyx = dyx.reshape(self.dim[0],self.dim[1])
        rot = rot.reshape(self.dim[0],self.dim[1])
        shear = shear.reshape(self.dim[0],self.dim[1])
        Dispx = Dispx.reshape(self.dim[0],self.dim[1])
        Dispy = Dispy.reshape(self.dim[0],self.dim[1])
        return dxx, dyy, dxy, dyx, rot, shear, Dispx, Dispy
    

def wiener_filt(A,power=1):
    dim = A.shape
    fftw = pyfftw.builders.fft2(A, threads = os.cpu_count())
    im = fftw().flatten()
    im1 = np.abs(im)**2
    #soglia = np.sqrt(max(im1[1:])*min(im1[1:]) )
    soglia = power*np.exp(np.log(im[1:]).mean())
    maskS = im1[1:]>soglia
    medS = np.sum(im1[1:]*(maskS))/maskS.sum()
    f = np.real(im1)/(np.real(im1)+medS)
    im*=f
    fftw = pyfftw.builders.ifft2(im.reshape(dim), threads = os.cpu_count())
    return np.real(fftw())

def interp(x,y,Field,X,Y):
    A = np.flip([np.interp(x, X, Field[i]) for i in range(Field.shape[0])])
    return(np.interp(y, Y, A))


class Atom:
    def __init__(self, Z, x, y, z):
        self.Z = Z    #atomic number      
        self.x = x    #x coordinate
        self.y = y    #y coordinate  
        self.z = z    #z coordinate
        
        self.Dx = 0   #atom displacements
        self.Dy = 0   #initialised to 0

def In_Box(x,y,BOX=None):
    #BOX = [x0,y0,xf,yf]
    if BOX == None:
        return True
    
    if x>BOX[0] and x<BOX[2]:
        if y>BOX[1] and y<BOX[3]:
            return True
    return False

def read_xyz(filename, BOX = None, extra=0):
    B = [0,0,0,0]
    if BOX is not None:
        B[0]=BOX[0]-extra
        B[1]=BOX[1]-extra
        B[2]=BOX[2]+extra
        B[3]=BOX[3]+extra
    
    file = open(filename, 'r')
    
    #first line containes the number of atoms
    line = file.readline()
    [Natom] = [int(f) for f in re.findall("[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?", line.strip())]

    line = file.readline() #second line is empty
    
    Alist = []    #initialize empty atom list
    for i in range(Natom):
        line = file.readline() #second line is empty
        Z = line.strip()[0:2]
        x,y,z = [float(f) for f in re.findall("[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?", line.strip())]
        if In_Box(x,y,B):
            Alist.append(Atom(Z,x,y,z))
    file.close()
    return Alist

def save_xyf(Alist,filename):
    file = open(filename, 'w')
    file.write('%i\n'%(len(Alist)))
    file.write(' \n')
    for atom in Alist:
        file.write('%s    %.6f    %.6f    %.6f\n'%(atom.Z,atom.x,atom.y,atom.z))
    file.close()
    
def Purge(Alist,BOX=None,purge_list=[]):
    NewList = []
    for atom in Alist:
        if In_Box(atom.x,atom.y,BOX):
            if atom not in purge_list:
                NewList.append(atom)
    return NewList

def Filter_errors(Alist, purge_list=[]):
    NewList = []
    for atom in Alist:
        if atom in purge_list:
            NewList.append(atom)
    return NewList

def Displace(Alist,Disp_x,Disp_y,BOX,Accuracy = 1e-3, Max_try=10, rate=0.6):
    Ly = [BOX[1],BOX[3]]
    Lx = [BOX[0],BOX[2]]
    X = np.linspace(Lx[0],Lx[1],Disp_x.shape[0])
    Y = np.linspace(Ly[0],Ly[1],Disp_x.shape[1])
    a = 0
    E_list = []
    T = len(Alist)
    for atom in Alist:
        for i in range(Max_try):
            dx = interp(atom.x,atom.y,Disp_x,X,Y)
            dy = interp(atom.x,atom.y,Disp_y,X,Y)
            deltax = (dx-atom.Dx)
            deltay = (dy-atom.Dy)
            if deltax**2 + deltay**2 < 2*Accuracy**2:
                break
            deltax*=rate
            deltay*=rate
            atom.x += deltax
            atom.Dx += deltax        
            atom.y += deltay
            atom.Dy += deltay
        if i+1 == Max_try:
            E_list.append(atom)
    if len(E_list) == 0: 
        print('Completed without problems!')
    else:
        print('Issues with %i atoms. Please check errors list.'%len(E_list))
    return E_list


fig_size = 10
path = r'E:\Arxius varis\PhD\4rth_year\Code\Strain_to_atomistic\Atomistic_models\InP_InSb_NW_Diag_110_full\GPA_unbinned_2\\'
im = Image.open(path+'image.tif')
image = np.array(im)
npx = image.shape[0]
npy = image.shape[1]
fig, ax = plt.subplots(1, 1, figsize=(fig_size, fig_size))
ax.imshow(image, cmap='gray')

L = 2143
pixsize = L/npx
# save_tiff32(image, calibration=pixsize, filename=path+'image_t32.tif')


image_flt = wiener_filt(image,power=1)

B1X = 1000
B1Y = 750 
width = 200 
height = 200 


fig, ax = plt.subplots(1, 1, figsize=(fig_size, fig_size))
ax.imshow(image_flt, cmap='gray')

Box1 = Rectangle((B1X,B1Y), width, height, angle=0.0, facecolor='none',edgecolor="red", linewidth=1, alpha=1)
ax.add_patch(Box1)

B2X = 220
B2Y = 850
width2 = 400
height2 = 400
Box2 = Rectangle((B2X,B2Y), width2, height2, angle=0.0, facecolor='none',edgecolor="blue", linewidth=1, alpha=1)
ax.add_patch(Box2)
#save_tiff32(image_flt, calibration=1, filename=path+'image_filtered_t32.tif')


fft_img = np.log(abs(FT(image_flt)))
fig, ax = plt.subplots(1, 1, figsize=(fig_size, fig_size))
ax.imshow(fft_img,interpolation='nearest', cmap='gray')

win_size = 330//2

Sp1X = 1550
Sp1Y = 857
ax.scatter(Sp1X, Sp1Y, c="red", marker="x")
circle = Circle((Sp1X, Sp1Y), win_size, facecolor='none',edgecolor="red", linewidth=1, alpha=1)
ax.add_patch(circle)

Sp2X = 1024
Sp2Y = 440
ax.scatter(Sp2X, Sp2Y, c="blue", marker="x")
circle = Circle((Sp2X, Sp2Y), win_size, facecolor='none',edgecolor="blue", linewidth=1, alpha=1)
ax.add_patch(circle)

Analysis = GPA()


Analysis.load_image(image_flt,pixsize)
Analysis.select_ref_area((B1X,B1Y),(B1X+width,B1Y+height))


#You need to remark the spots because of the different refining area
Analysis.mark_spot1((Sp1X,Sp1Y),win_size)
Analysis.mark_spot2((Sp2X,Sp2Y),win_size)
Analysis.calc_GPA()

c = 2
cm = 'ocean'
dxx, dyy, dxy, dyx, rot, shear, Dispx, Dispy = Analysis.get()
M = dxx[256:768,256:768].mean()
S = dxx[256:768,256:768].std()
fig, ax = plt.subplots(1, 1, figsize=(fig_size, fig_size))
ax.imshow(dxx,interpolation='nearest',cmap=cm, clim=(M-c*S,M+c*S))

print('Refined spot1 coordinates are:', Analysis.refined_spot1)
print('Refined spot2 coordinates are:', Analysis.refined_spot2)

ref_spot1 = Analysis.refined_spot1
ref_spot2 = Analysis.refined_spot2



fft_img = np.log(abs(FT(image_flt)))
fig, ax = plt.subplots(1, 1, figsize=(fig_size, fig_size))
ax.imshow(fft_img,interpolation='nearest', cmap='gray')

win_size = 330//2

ax.scatter(ref_spot1[0], ref_spot1[1], c="red", marker="x")
circle = Circle((ref_spot1[0], ref_spot1[1]), win_size, facecolor='none',edgecolor="red", linewidth=1, alpha=1)
ax.add_patch(circle)

ax.scatter(ref_spot2[0], ref_spot2[1], c="blue", marker="x")
circle = Circle((ref_spot2[0], ref_spot2[1]), win_size, facecolor='none',edgecolor="blue", linewidth=1, alpha=1)
ax.add_patch(circle)





fig, ax = plt.subplots(1, 1, figsize=(fig_size, fig_size))
ax.imshow(image_flt[B2Y:B2Y+width2,B2X:B2X+height2], cmap='gray')
Ref_point = [380,10]
ax.scatter(Ref_point[0],Ref_point[1],marker='*',s=400, c='r')

#Disp_x = (Dispx[B2Y:B2Y+width2,B2X:B2X+height2]-Dispx[B2Y+Ref_point[1],B2X+Ref_point[0]])*pixsize
Disp_x = (Dispx[B2Y:B2Y+width2,B2X:B2X+height2])*pixsize
fig, ax = plt.subplots(1, 1, figsize=(fig_size, fig_size))
ax.imshow(Disp_x, cmap='jet')

#Disp_y = (Dispy[B2Y:B2Y+width2,B2X:B2X+height2]-Dispy[B2Y+Ref_point[1],B2X+Ref_point[0]])*pixsize
Disp_y = (Dispy[B2Y:B2Y+width2,B2X:B2X+height2])*pixsize
fig, ax = plt.subplots(1, 1, figsize=(fig_size, fig_size))
ax.imshow(Disp_y, cmap='jet')




BOX = [B2X*pixsize,(npy-B2Y-height2)*pixsize,(B2X+height2)*pixsize,(npy-B2Y)*pixsize]
Alist = read_xyz(r'E:\Arxius varis\PhD\4rth_year\Code\Strain_to_atomistic\Atomistic_models\InP_InSb_NW_Diag_110_full\global_device_supercell.xyz',BOX,extra=10*pixsize)
save_xyf(Alist,r'E:\Arxius varis\PhD\4rth_year\Code\Strain_to_atomistic\Atomistic_models\InP_InSb_NW_Diag_110_full\cut.xyz')
print('There are %i atoms in the selected region'%len(Alist))


List = read_xyz(r'E:\Arxius varis\PhD\4rth_year\Code\Strain_to_atomistic\Atomistic_models\InP_InSb_NW_Diag_110_full\global_device_supercell.xyz',BOX,extra=10*pixsize)
t1 = time()
Errors = Displace(List,Disp_y,-Disp_x,BOX,rate=0.8,Max_try=50)
print(time()-t1)
List = Purge(List,BOX=BOX)
save_xyf(List,r'E:\Arxius varis\PhD\4rth_year\Code\Strain_to_atomistic\Atomistic_models\InP_InSb_NW_Diag_110_full\cut-rate_t.xyz')
List = Purge(List,purge_list=Errors)
save_xyf(List,r'E:\Arxius varis\PhD\4rth_year\Code\Strain_to_atomistic\Atomistic_models\InP_InSb_NW_Diag_110_full\cut-rate_t-purge.xyz')







