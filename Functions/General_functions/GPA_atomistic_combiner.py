# -*- coding: utf-8 -*-
"""
Created on Sun Feb 12 19:46:45 2023

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
import sympy
from pathlib import Path
import ase
import random

# !!! NEED to set the path to 
# Alg_Comb_Single_Image_Strain.py
# as the console working directory
Project_main_path = os.getcwd()
if 'EMicrographs_to_AtoModels' in Project_main_path:
    Project_main_path = Project_main_path[:Project_main_path.find('EMicrographs_to_AtoModels')-1]
# Project_main_path has the EMicrographs_to_AtoModels folder
sys.path.append(Project_main_path)


from EMicrographs_to_AtoModels.Functions.General_functions import ImageCalibTransf as ImCalTrans
from EMicrographs_to_AtoModels.Atomistic_model_builder import Atomistic_Model_Builder as AtomBuild
from EMicrographs_to_AtoModels.Functions.General_functions import GPA_specific as GPA_sp


'''
Script to hold the functions used for the GPA computation and to get all the
necessary variables for the application of the strain field into the atomistic model
'''


def FT(img):
    return np.fft.ifftshift(np.fft.fft2(np.fft.fftshift(img)))

def IFT(img):
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(img)))

GPA_atom_dll_path = Project_main_path + '\\EMicrographs_to_AtoModels\Strain_to_atomistic\GPA_to_atom_dll'

lib = ctypes.CDLL(GPA_atom_dll_path + '\\' + 'GPA_to_atom.dll')
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
    def __init__(self, Z, x, y, z, occ = 1.0, DW = 0.005):
        self.Z = Z    #atomic number      
        self.x = x    #x coordinate
        self.y = y    #y coordinate  
        self.z = z    #z coordinate

        self.occ = occ  # occuapncy, default 1
        self.DW = DW  # Debye - Waller factor, default 0.005
        
        self.Dx = 0   #atom displacements
        self.Dy = 0   #initialised to 0

def In_Box(x,y,BOX=np.array([0])):
    #BOX = [x0,y0,xf,yf]
    if len(BOX) == 1:
        return True
    
    if x>BOX[0] and x<BOX[2]:
        if y>BOX[1] and y<BOX[3]:
            return True
    return False

def read_xyz_Original(filename, BOX = None, extra=0):
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
    indexes_to_keep = []
    
    
    if BOX is None:
        # if box is None then the indices list makes no sense as it is all
        # the atoms, but needs to be returned anyways so just ignore
        for i in range(Natom):
            line = file.readline() #second line is empty
            Z = line.strip()[0:2]
            x,y,z = [float(f) for f in re.findall("[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?", line.strip())]
            Alist.append(Atom(Z,x,y,z))
            indexes_to_keep.append(i)
    else:
        for i in range(Natom):
            line = file.readline() #second line is empty
            Z = line.strip()[0:2]
            x,y,z = [float(f) for f in re.findall("[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?", line.strip())]
            if In_Box(x,y,B): 
                Alist.append(Atom(Z,x,y,z))
                indexes_to_keep.append(i)
    file.close()
    return Alist, indexes_to_keep



def save_xyf(Alist,filename, save_occsDW = False):
    file = open(filename, 'w')
    file.write('%i\n'%(len(Alist)))
    file.write(' \n')
    for atom in Alist:
        if save_occsDW == False:
            file.write('%s    %.6f    %.6f    %.6f\n'%(
                atom.Z, atom.x, atom.y, atom.z))
        else:
            # if save_occsDW == True,
            # save the occupancy and DW factor of the atom
            file.write('%s    %.6f    %.6f    %.6f    %.6f    %.6f\n'%(
                atom.Z, atom.x, atom.y, atom.z, atom.occ, atom.DW))
    file.close()
    
def Purge_Original(Alist,BOX=np.array([0]),purge_list=[]):
    NewList = []
    for atom in Alist:
        if In_Box(atom.x,atom.y,BOX):
            if atom not in purge_list:
                NewList.append(atom)
    return NewList

def Purge(Alist,BOX=np.array([0]),purge_list=[]):
    NewList = []
    indexes_to_purge = []
    for index, atom in enumerate(Alist):
        if In_Box(atom.x,atom.y,BOX):
            if atom not in purge_list:
                NewList.append(atom)
            else:
                indexes_to_purge.append(index)
        else:
            indexes_to_purge.append(index)
                
    return NewList, indexes_to_purge


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





def Cut_atom_identifier_Region(
        atom_models_filepath, indexes_to_keep):
    '''
    Cut the file containing the labels for every atom in the global model
    to have the 1 to 1 equivalent to the cut region
    Thus, to have a label identifier for the atoms within the box

    Parameters
    ----------
    atom_models_filepath : TYPE
        DESCRIPTION.
    indexes_to_keep : TYPE
        DESCRIPTION.

    Returns
    -------
    new_reg_filepath : TYPE
        DESCRIPTION.

    '''
    
    
    global_id_file = atom_models_filepath + '\\' + 'global_atom_identifier.txt'
        
    file_labs = open(global_id_file)
    file_labs_lines = file_labs.readlines()
    init_numb_atoms = int(file_labs_lines[0])
    
    actual_atom_labs = file_labs_lines[2:].copy()
    
    atom_labs_region = []
    for index in indexes_to_keep:
        atom_labs_region.append(actual_atom_labs[index])
        
    new_numb_atoms = int(len(indexes_to_keep))
    
    new_region_cut_file = [str(new_numb_atoms)+'\n', '\n', atom_labs_region]
    
    # create the .txt file
    new_reg_filepath = atom_models_filepath + 'region_atom_identifier.txt'
    
    filename = Path(new_reg_filepath)
    file_already_created = filename.is_file()
    
    if file_already_created == True:
        # overwrite the file
        with open(new_reg_filepath, "w+") as f:
            f.truncate(0)
            f.writelines(new_region_cut_file)
            f.close()
    else:
        # create a new file
        with open(new_reg_filepath, 'w+') as f:
            f.writelines(new_region_cut_file)
            f.close()
        
    
    return new_reg_filepath



def Purge_Indices_Labels_AtomList(
        atom_models_filepath, indexes_to_purge, new_filename):
    '''
    Purge the file containing the labels of the atoms for that specific
    cut region (box) and purge the ones whose indices come from
    the list indexes_to_purge

    Parameters
    ----------
    atom_models_filepath : path to atom cells made
    indexes_to_purge : list with the indices to purge
    new_filename : str with the addition to the name of the file
                    strained, strained_purged... whatever

    Returns
    -------
    new_purg_filepath : path to the labels file after purge

    '''
    
    region_id_file = atom_models_filepath + '\\' + 'region_atom_identifier.txt'
        
    file_labs = open(region_id_file)
    file_labs_lines = file_labs.readlines()
    init_numb_atoms = int(file_labs_lines[0])
    
    actual_atom_labs = file_labs_lines[2:].copy()
    
    # set the indexes in reverse order to first pop the biggest
    
    indexes_to_purge = sorted(indexes_to_purge, reverse = True)
    
    for index_to_pop in indexes_to_purge:
        actual_atom_labs.pop(index_to_pop)
        
        
    new_atom_number = int(init_numb_atoms - len(indexes_to_purge))
    
    new_purged_file = [str(new_atom_number)+'\n', '\n', actual_atom_labs]
    
    # create the .txt file
    new_purg_filepath = atom_models_filepath + 'region_atom_identifier_' + new_filename + '.txt'
    
    filename = Path(new_purg_filepath)
    file_already_created = filename.is_file()
    
    if file_already_created == True:
        # overwrite the file
        with open(new_purg_filepath, "w+") as f:
            f.truncate(0)
            f.writelines(new_purged_file)
            f.close()
    else:
        # create a new file
        with open(new_purg_filepath, 'w+') as f:
            f.writelines(new_purged_file)
            f.close()
    
    
    return new_purg_filepath


    
def GPA_full_AtomisticModel(
        image_array, calibration, spot1, spot2, mask_size, 
        reference, rotation_angle, display=True):
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
    
    Analysis.load_image(image_array, calibration)
    
    # 
    Analysis.select_ref_area((col_i,row_i),(col_f,row_f))

    #You need to remark the spots because of the different refining area
    amp1 = Analysis.mark_spot1((Sp1X,Sp1Y), mask_size)
    amp2 = Analysis.mark_spot2((Sp2X,Sp2Y), mask_size)
    Analysis.calc_GPA()

    # refine the spots based on the reference
    # output in format (x,y), flip to format (row, column)
    ref_spot1_ = Analysis.refined_spot1
    ref_spot2_ = Analysis.refined_spot2
    ref_spot1 = np.array([ref_spot1_[1], ref_spot1_[0]])
    ref_spot2 = np.array([ref_spot2_[1], ref_spot2_[0]])
        
    Analysis.calc_GPA()
    
    Analysis.apply_rotation(rotation_angle)
        
    exx, eyy, exy, eyx, rot, shear, Dispx, Dispy = Analysis.get()
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
        fig,ax=plt.subplots(4,2, constrained_layout=True)
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
        M = Dispx[row_i:row_f+1,col_i:col_f+1].mean()
        S = Dispx[row_i:row_f+1,col_i:col_f+1].std()
        clims_ranged=(M-c*S,M+c*S)
        im=ax[3][0].imshow(Dispx,interpolation='nearest', cmap=cm, clim=clims_fix)
        ax[3][0].set_title('Dispx')  
        fig.colorbar(im, ax=ax[3][0], shrink=colorbarshrink)
        M = Dispy[row_i:row_f+1,col_i:col_f+1].mean()
        S = Dispy[row_i:row_f+1,col_i:col_f+1].std()
        clims_ranged=(M-c*S,M+c*S)
        im=ax[3][1].imshow(Dispy,interpolation='nearest', cmap=cm, clim=clims_fix)
        ax[3][1].set_title('Dispy')  
        fig.colorbar(im, ax=ax[3][1], shrink=colorbarshrink)
        plt.show()
        
    return exx, eyy, exy, eyx, rot, shear, Dispx, Dispy, ref_spot1, ref_spot2




def Find_virtual_a_cell_c_cell_params(
        base_cell_params, hkl1, hkl2, 
        spot1_dist, spot2_dist):
    '''
    Find the a and c cell parameters of the virtual crystal given
    the double set of planes identified as 
    
    Parameters
    ----------
    base_cell_params : tuple, a, b, c, alfa, beta, gamma
        DESCRIPTION.
    hkl1 : array, h1, k1, l1
    hkl2 :  array, h2, k2, l2
    spot1_dist : exp distance found for spot 1 or plane 1
        DESCRIPTION.
    spot2_dist : exp distance found for spot 2 or plane 2
        DESCRIPTION.

    Returns
    -------
    a_v_cell, b_v_cell, c_v_cell , cell parameters of the virtual cell


    '''
    
    def F(ang1, ang2, ang3):
        
        F = np.cos(ang1*(np.pi/180))*np.cos(ang2*(np.pi/180)) -  np.cos(ang3*(np.pi/180))
        
        return F
    
    
    def Find_a_cell(
            h, k, l, spot_dist, n1, n2):
        '''
        This equation finds a_cell assuming b and c keep
        the same realtion with a as the original unit cell, 
        given by n1 and n2
        # n1 : equivalence between original a and b ; n1 = b_original/a_original
        # n2 : equivalence between original a and c ; n2 = c_original/a_original
        '''
        
        a_1 = (h**2)*(n1**2)*(n2**2)*((np.sin(alfa*(np.pi/180)))**2)
        a_2 = 2*h*k*n1*(n2**2)*F(alfa, beta, gamma)
        a_3 = 2*h*l*(n1**2)*n2*F(gamma, alfa, beta)
        a_4 = (k**2)*(n2**2)*((np.sin(beta*(np.pi/180)))**2)
        a_5 = 2*k*l*n1*n2*F(beta, gamma, alfa)
        a_6 = (l**2)*(n1**2)*((np.sin(gamma*(np.pi/180)))**2)
        
        # we keep the postiive solution, hoping the sqrts are not negative inside
        a_v_cell = (spot_dist/(n1*n2*np.sqrt(theta)))*np.sqrt(a_1 + a_2 + a_3 + a_4 + a_5 + a_6)
        
        return a_v_cell

    
    
    
    a_or, b_or, c_or, alfa, beta, gamma = base_cell_params
    
    # n : equivalence between original a and b ; n = b_original/a_original
    n = b_or/a_or
    
    h1, k1, l1 = hkl1
    h2, k2, l2 = hkl2
    
    theta = 1 - (np.cos(alfa*(np.pi/180)))**2 - (np.cos(beta*(np.pi/180)))**2 \
    - (np.cos(gamma*(np.pi/180)))**2  \
    + 2*(np.cos(alfa*(np.pi/180)))*(np.cos(beta*(np.pi/180)))*(np.cos(gamma*(np.pi/180)))
    
    # terms first equation
    c2_1 = (h1**2)*(n**2)*((np.sin(alfa*(np.pi/180)))**2) \
        + 2*h1*k1*n*F(alfa, beta, gamma) + (k1**2)*((np.sin(beta*(np.pi/180)))**2)
        
    a_c_1 = 2*h1*l1*(n**2)*F(gamma, alfa, beta) + 2*k1*l1*n*F(beta, gamma, alfa)
    
    a2_1 = (l1**2)*(n**2)*((np.sin(gamma*(np.pi/180)))**2)
    
    a2_c2_1 = -((n**2)*theta)/((spot1_dist)**2)
    
    
    # terms second equation
    c2_2 = (h2**2)*(n**2)*((np.sin(alfa*(np.pi/180)))**2) \
        + 2*h2*k2*n*F(alfa, beta, gamma) + (k2**2)*((np.sin(beta*(np.pi/180)))**2)
        
    a_c_2 = 2*h2*l2*(n**2)*F(gamma, alfa, beta) + 2*k2*l2*n*F(beta, gamma, alfa)
    
    a2_2 = (l2**2)*(n**2)*((np.sin(gamma*(np.pi/180)))**2)
    
    a2_c2_2 = -((n**2)*theta)/((spot2_dist)**2)
    
    c, a = sympy.var('c a')
    
    eq_1 = c2_1*(c**2) + a_c_1*a*c + a2_1*(a**2) + a2_c2_1*(a**2)*(c**2) 
    eq_2 = c2_2*(c**2) + a_c_2*a*c + a2_2*(a**2) + a2_c2_2*(a**2)*(c**2) 
    
    solutions = sympy.solve((eq_1, eq_2), (c, a))
    
    solutions = [list(solution) for solution in solutions]
    
    
    # check if there is a tuple where the two elements are 
    # postiive whichm means a real soltion 
    solution_found = 0
    
    for solution in solutions:
        # ensure both checked values are real to make the comparison
        solution[1] = 1+ solution[1]
        
        # convert sympy elements into numpy elements
        if type(solution[0]) == sympy.core.numbers.Float:
            solution[0] = float(solution[0])
        else:
        # elif type(solution[0]) == sympy.core.mul.Mul:
            solution[0] = complex(solution[0])
            
        # convert sympy elements into numpy elements
        if type(solution[1]) == sympy.core.numbers.Float:
            solution[1] = float(solution[1])
        else:
        # elif type(solution[0]) == sympy.core.mul.Mul:
            solution[1] = complex(solution[1])
        
        if np.iscomplex(solution[0]) == False and np.iscomplex(solution[1]) == False:
            # now just keep the two positive values solution 
            if solution[0]> 0 and solution[1]>0:
                final_solution = solution
                solution_found = 1
            
            
    
    if solution_found == 1:
        # solution found in the a != c regime
        a_v_cell = final_solution[1]
        b_v_cell = n*a_v_cell
        c_v_cell = final_solution[0]
        
    else:
        # solution not found in the a != c regime
        # this means both planes should give the same experimental distance
        # but as they are experimetal this is not happening, so just average
        # what would be obtained with both in a homogenous distortion of the
        # original cell that keeps the same relations with their cell params
        # else find the pattern with a  n1 = a/b n2 = a/c
        # so develop the recirpocal tnesor calcultation with just 1 variable and
        # then 1 equation and then make the average of both results with the two equations
    
        # n1 : equivalence between original a and b ; n1 = b_original/a_original
        n1 = b_or/a_or
        # n2 : equivalence between original a and c ; n2 = c_original/a_original
        n2 = c_or/a_or
        
        # compute the a with plane 1
        a_v_cell_1 = Find_a_cell(
            h1, k1, l1, spot1_dist, n1, n2)
        # compute the a with plane 2
        a_v_cell_2 = Find_a_cell(
            h2, k2, l2, spot2_dist, n1, n2)
        
        # !!! There is around 1% of experimental error in the estimation of the
        # unit cell parameters in this way by assuming just one of  
        # unknown, and then the relations between original cell params
        # Average both
        a_v_cell = np.mean(np.array([a_v_cell_1, a_v_cell_2]))
    
        b_v_cell = n1*a_v_cell
        c_v_cell = n2*a_v_cell
    
    return a_v_cell, b_v_cell, c_v_cell 





def Build_virtual_crystal_cif(
        cif_cells_folder_directory, base_phase_name, a_v_cell, b_v_cell, c_v_cell):
    
    base_cell_filepath = cif_cells_folder_directory + '\\' + base_phase_name + '.cif'
    
    raw_file_data = open(base_cell_filepath)
    
    raw_datalines = raw_file_data.readlines()
    
    list_split_lines = []
    
    for line in raw_datalines:
        list_split_lines.append(line.split())
    
    
    new_file_lines = []
    
    for line in raw_datalines:
        if line[:14] == '_cell_length_a':
            new_line = '_cell_length_a   ' + '{:.6f}'.format(a_v_cell) + '\n' 
            new_file_lines.append(new_line)
            
        elif line[:14] == '_cell_length_b':
            new_line = '_cell_length_b   ' + '{:.6f}'.format(b_v_cell) + '\n' 
            new_file_lines.append(new_line)
            
        elif line[:14] == '_cell_length_c':
            new_line = '_cell_length_c   ' + '{:.6f}'.format(c_v_cell) + '\n' 
            new_file_lines.append(new_line)
        else:
                    
            new_file_lines.append(line)
        
        
    # create the .cif file
    path_to_v_unitcell = cif_cells_folder_directory + '\\' + base_phase_name + '_v' + '.cif'
    
    path_to_v_unitcell_Path = Path(path_to_v_unitcell)
    file_already_created = path_to_v_unitcell_Path.is_file()
    
    if file_already_created == True:
        # overwrite the file
        with open(path_to_v_unitcell, "w+") as f:
            f.truncate(0)
            f.writelines(new_file_lines)
            f.close()
    else:
        # create a new file
        with open(path_to_v_unitcell, 'w+') as f:
            f.writelines(new_file_lines)
            f.close()    
                
        
    return path_to_v_unitcell


# !!! ONLY USE FOLLOWING FUNCTION IF WANT TO CREATE A SUPERCELL WITH THE PARAMETERS 
# AS FOUND EXPERIENTALLY FOR EACH REGION, BUT NOT FOR STRAIN APPLICATION !!!
def Build_All_Virtual_Crysts_Except_Ref(
        analysed_image, image_in_dataset_whole, 
        best_GPA_ref_spot_pair,
        image_segmented, label_of_GPA_ref,
        GPA_resolution, model_cells_filepath):
    '''
    !!! ONLY USE IF WANT TO CREATE A SUPERCELL WITH THE PARAMETERS AS FOUND
    EXPERIENTALLY FOR EACH REGION, BUT NOT FOR STRAIN APPLICATION !!!
    
    After building the virtual cell with the reference region
    the process is done with the reference mat 
    repeat with all the materias which have a spot within the mask drawn
    so checing the main spots of the reference in the gpa and
    then check the mask around these two spots, so for all the materials
    that are not the reference ones check if a spot is there  around JUST
    these two spots and build the crystals of these materials given by these
    spots with other indices  (not necessaarily the same indices as the base ones)
    then just for the others that do not fit inside the mask regime just
    keep the base unmodified crystal
    then keep a record on which materials have been checked and which is the final
    phase name for it whetehr the atomistic model builder has to check the
    modified virtual model or the base cif one

    Builds the virtual crystal based on the dsitances computed
    from the spots corresponding to the actual material, not
    from the distances corresponding to the material in the reference region
    
    Parameters
    ----------
    analysed_image : object containing the crystals found for that given image
    image_in_dataset_whole : info of the main image
    image_segmented : segmented image of the image anlaysed
    label_of_GPA_ref : label within image_segmented where the GPA reference 
                    lattice is taken from
    GPA_resolution : the resolution defines the mask inside which the other 
                    spots from the other materials need to be checked
    model_cells_filepath: str, directory where the original cif files and 
                        new ones regarding the virtual cell are created
    Returns
    -------
    
    paths_to_virt_ucells: list of strings, with the paths to the virtual cells
                            created 

    '''
    
    
    
    # Retrieve data extracted from the reference 
    crop_outputs_dict = analysed_image.Crop_outputs

    # Pixels within the whole image in which the crop of the reference is taken, 
    # so the box of the reference itself [first_row,last_row,first_col,last_col]
    scaled_reference_coords_GPA_ref = crop_outputs_dict[str(int(label_of_GPA_ref)) + '_pixel_ref_cords']

    # 
    image_crop_hs_signal_GPA_ref = crop_outputs_dict[str(int(label_of_GPA_ref)) + '_hs_signal']
    FFT_image_array_GPA_ref, _ =  ImCalTrans.Compute_FFT_ImageArray(np.asarray(image_crop_hs_signal_GPA_ref))

    crop_list_refined_cryst_spots_GPA_ref = crop_outputs_dict[str(int(label_of_GPA_ref)) + '_list_refined_cryst_spots']
    refined_pixels_GPA_ref = crop_outputs_dict[str(int(label_of_GPA_ref)) + '_refined_pixels']
    spots_int_reference_GPA_ref = crop_outputs_dict[str(int(label_of_GPA_ref)) + '_spots_int_reference']

    # Get the best cryst spot
    best_cryst_spot_GPA_ref = crop_list_refined_cryst_spots_GPA_ref[0]

    # Retrieve the best spot pair to be considered the GPA g vectors
    
    # Find the best spots considered in that crystal phase, which should 
    # be the best ones to constitute the GPA g vectors
    # and its information to help build the virutal crystal, although they
    #  need to be updated after the refining of the g vectors with the ref
    spot1_int_ref_GPA_ref = best_GPA_ref_spot_pair.spot1_int_ref
    spot2_int_ref_GPA_ref = best_GPA_ref_spot_pair.spot2_int_ref
    
    # Retrieve the info of the spots acting as g vectors
    hkl1_reference_GPA_ref = best_GPA_ref_spot_pair.hkl1_reference
    hkl2_reference_GPA_ref = best_GPA_ref_spot_pair.hkl2_reference
    spot1_dist_GPA_ref = best_GPA_ref_spot_pair.spot1_dist
    spot2_dist_GPA_ref = best_GPA_ref_spot_pair.spot2_dist
    angle_between_GPA_ref = best_GPA_ref_spot_pair.angle_between
    spot1_angle_x_GPA_ref = best_GPA_ref_spot_pair.spot1_angle_to_x
    spot2_angle_x_GPA_ref = best_GPA_ref_spot_pair.spot2_angle_to_x
    found_phase_name_GPA_ref = best_cryst_spot_GPA_ref.phase_name
    
    
    # coords of the best peaks to use as GPA g vectors in coordinartes of the crop
    # NOT of the whole image
    cord_spot_1_GPA_ref = refined_pixels_GPA_ref[int(spot1_int_ref_GPA_ref)]
    cord_spot_2_GPA_ref = refined_pixels_GPA_ref[int(spot2_int_ref_GPA_ref)]
    

    # image in dataset base
    image_array_whole = image_in_dataset_whole.image_arraynp_st
    total_pixels_whole = image_in_dataset_whole.total_pixels
    pixel_size_whole = image_in_dataset_whole.x_calibration
    FFT_calibration_whole = image_in_dataset_whole.FFT_calibration
    FFT_whole, _ =  ImCalTrans.Compute_FFT_ImageArray(np.asarray(image_in_dataset_whole.hyperspy_2Dsignal))
    
    
    # generate a 9x9 pixel square arround the best coordinate scaled to the whole image
    # so from all this pixels we can find the best one that represents the maximum of
    # the peak
    
    scaled_cord_spot_1_GPA_y = int(np.round((cord_spot_1_GPA_ref[0]/FFT_image_array_GPA_ref.shape[0])*total_pixels_whole))
    scaled_cord_spot_1_GPA_x = int(np.round((cord_spot_1_GPA_ref[1]/FFT_image_array_GPA_ref.shape[1])*total_pixels_whole))
    scaled_cord_spot_2_GPA_y = int(np.round((cord_spot_2_GPA_ref[0]/FFT_image_array_GPA_ref.shape[0])*total_pixels_whole))
    scaled_cord_spot_2_GPA_x = int(np.round((cord_spot_2_GPA_ref[1]/FFT_image_array_GPA_ref.shape[1])*total_pixels_whole))
    
    if scaled_cord_spot_1_GPA_y-1 >= 0 and scaled_cord_spot_1_GPA_y+2 <= total_pixels_whole and scaled_cord_spot_1_GPA_x-1 >= 0 and scaled_cord_spot_1_GPA_x+2 <= total_pixels_whole:
        
        FFT_crop_spot_1 = FFT_whole[scaled_cord_spot_1_GPA_y-1:scaled_cord_spot_1_GPA_y+2,scaled_cord_spot_1_GPA_x-1:scaled_cord_spot_1_GPA_x+2]
        cords_max1 = np.where(FFT_crop_spot_1 == np.max(FFT_crop_spot_1))
    
        scaled_cord_spot_1_GPA_y = scaled_cord_spot_1_GPA_y - 1 + cords_max1[0][0]
        scaled_cord_spot_1_GPA_x = scaled_cord_spot_1_GPA_x - 1 + cords_max1[1][0]
        
    
    if scaled_cord_spot_2_GPA_y-1 >= 0 and scaled_cord_spot_2_GPA_y+2 <= total_pixels_whole and scaled_cord_spot_2_GPA_x-1 >= 0 and scaled_cord_spot_2_GPA_x+2 <= total_pixels_whole:
        
        FFT_crop_spot_2 = FFT_whole[scaled_cord_spot_2_GPA_y-1:scaled_cord_spot_2_GPA_y+2,scaled_cord_spot_2_GPA_x-1:scaled_cord_spot_2_GPA_x+2]
        cords_max2 = np.where(FFT_crop_spot_2 == np.max(FFT_crop_spot_2))
    
        scaled_cord_spot_2_GPA_y = scaled_cord_spot_2_GPA_y - 1 + cords_max2[0][0]
        scaled_cord_spot_2_GPA_x = scaled_cord_spot_2_GPA_x - 1 + cords_max2[1][0]
    # else:
    #     the coordinates are the same as they were
    
    spot_1_coords = np.array([scaled_cord_spot_1_GPA_y, scaled_cord_spot_1_GPA_x])
    spot_2_coords = np.array([scaled_cord_spot_2_GPA_y, scaled_cord_spot_2_GPA_x])    
        
    
    # closer_spot_whole_coords = [spot_1_coords, spot_2_coords][closer_spot_ind]
    
    labels_unique = np.unique(image_segmented)
    labels_unique_no_ref = labels_unique[labels_unique != label_of_GPA_ref]
    
    
    
    # list to store info for the whole process, if needed
    paths_to_virt_ucells = []
    
    for compl_label in labels_unique_no_ref:
        
        # store all the coorindates from where crystals for 
        # just one
        all_possible_coords_to_check = []
        # Pixels within the whole image in which the crop of the reference is taken, 
        # so the box of the reference itself [first_row,last_row,first_col,last_col]
        scaled_reference_coords_compl_label = crop_outputs_dict[str(int(compl_label)) + '_pixel_ref_cords']
    
        # 
        image_crop_hs_signal_compl_label = crop_outputs_dict[str(int(compl_label)) + '_hs_signal']
        FFT_image_array_compl_label, _ =  ImCalTrans.Compute_FFT_ImageArray(np.asarray(image_crop_hs_signal_compl_label))
    
        crop_list_refined_cryst_spots_compl_label = crop_outputs_dict[str(int(compl_label)) + '_list_refined_cryst_spots']
        refined_pixels_compl_label = crop_outputs_dict[str(int(compl_label)) + '_refined_pixels']
        spots_int_reference_compl_label = crop_outputs_dict[str(int(compl_label)) + '_spots_int_reference']
    

        # if at least one crystal was found in that label
        if len(crop_list_refined_cryst_spots_compl_label) != 0:
            # store the coordinates of spots
            # convert them relative to the whole image
            scaled_refined_pixels_compl_label = (refined_pixels_compl_label/FFT_image_array_compl_label.shape[0])*total_pixels_whole
            all_possible_coords_to_check = all_possible_coords_to_check + list(scaled_refined_pixels_compl_label)
        else:
            continue
            
        
        # first check if the spots are close to the reference spots and use
        # these to make the indexing, in case they are not close to 
        # the spots, just use the two better ones
        
        # store the info to build the virtual crystal
        hkls_to_use_v = []
        distances_to_use_v = []
        

        
        for reference_spot in [spot_1_coords, spot_2_coords]:
            
            ref_found = 0
            
            
            # use the spot that has better score (smallest number, not closer to it)
            
        
            # now check que coordinate that is closer to closer_spot_whole_coords        
            all_possible_coords_to_check = np.asarray(all_possible_coords_to_check) 
           
            distances_y = np.abs(all_possible_coords_to_check[:,0] - reference_spot[0])*FFT_calibration_whole
            distances_x = np.abs(all_possible_coords_to_check[:,1] - reference_spot[1])*FFT_calibration_whole
        
            distances_in_nminv = np.sqrt(distances_y**2 + distances_x**2)
            distances_in_nm = 1/distances_in_nminv
        
            int_refs_close = list(spots_int_reference_compl_label[distances_in_nm >= GPA_resolution])
            
            
            if len(int_refs_close) !=0:
                
                # get the best spot, lowest score value number
                # this means finding the very first spot_pair in which 
                # that internal reference spot is found
                
                best_cryst = crop_list_refined_cryst_spots_compl_label[0]
                best_cryst_spot_pairs = best_cryst.spot_pairs_obj
                
                for spot_pair in best_cryst_spot_pairs:
                    
                    # int refs are
                    
                    spot1_int_ref = spot_pair.spot1_int_ref
                    spot2_int_ref = spot_pair.spot2_int_ref
                        
                    if spot1_int_ref in int_refs_close and ref_found == 0:
                        
                        hkl1_reference = spot_pair.hkl1_reference
                        distance_1 = spot_pair.spot1_dist
                        
                        hkls_to_use_v.append(hkl1_reference)
                        distances_to_use_v.append(distance_1)
                        
                        ref_found = 1
                        
                        break
                
                
                
                    if spot2_int_ref in int_refs_close and ref_found == 0:
                        
                        hkl2_reference = spot_pair.hkl2_reference
                        distance_2 = spot_pair.spot2_dist
                        
                        hkls_to_use_v.append(hkl2_reference)
                        distances_to_use_v.append(distance_2)
                        
                        ref_found = 1
                        
                        break
                
        
        
        if len(hkls_to_use_v) == 0:
            
            best_cryst = crop_list_refined_cryst_spots_compl_label[0]
            best_cryst_spot_pair = best_cryst.spot_pairs_obj[0]
                
            # int refs are
            spot1_int_ref = best_cryst_spot_pair.spot1_int_ref
            spot2_int_ref = best_cryst_spot_pair.spot2_int_ref
        
            hkl1_reference = best_cryst_spot_pair.hkl1_reference
            distance_1 = best_cryst_spot_pair.spot1_dist
            hkl2_reference = best_cryst_spot_pair.hkl2_reference
            distance_2 = best_cryst_spot_pair.spot2_dist
            
            hkls_to_use_v.append(hkl1_reference)
            distances_to_use_v.append(distance_1)
            hkls_to_use_v.append(hkl2_reference)
            distances_to_use_v.append(distance_2)
            
        if len(hkls_to_use_v) == 1:
            
            
            best_cryst = crop_list_refined_cryst_spots_compl_label[0]
            best_cryst_spot_pair = best_cryst.spot_pairs_obj[0]
                
            # int refs are
            spot1_int_ref = best_cryst_spot_pair.spot1_int_ref
            spot2_int_ref = best_cryst_spot_pair.spot2_int_ref
        
            hkl1_reference = best_cryst_spot_pair.hkl1_reference
            distance_1 = best_cryst_spot_pair.spot1_dist
            hkl2_reference = best_cryst_spot_pair.hkl2_reference
            distance_2 = best_cryst_spot_pair.spot2_dist
            
            # as experimental, the distances will never be exactly equal in two
            # spots, then, compare the distances
            
            if distances_to_use_v[0] == distance_1:
                hkls_to_use_v.append(hkl2_reference)
                distances_to_use_v.append(distance_2)
            else:
                hkls_to_use_v.append(hkl1_reference)
                distances_to_use_v.append(distance_1)
        
        
        # then at this point we have two planes and two distances 
        # build the crsytal based on this information 
        best_cryst = crop_list_refined_cryst_spots_compl_label[0]
        best_cryst_phase_name = best_cryst.phase_name
        
        cif_base_cell_filepath = model_cells_filepath + best_cryst_phase_name + '.cif'
        
        # build the ase crystal path
        ase_unit_cell_base = ase.io.read(cif_base_cell_filepath)
        
        spacegroup_cell_base = ase_unit_cell_base.info['spacegroup'].no
        
        a_base, b_base, c_base, alfa_base, beta_base, gamma_base = ase_unit_cell_base.cell.cellpar()
        
        base_cell_params = a_base, b_base, c_base, alfa_base, beta_base, gamma_base
        
        
        # compute the virtual cell parameters 
        a_v_cell, b_v_cell, c_v_cell = Find_virtual_a_cell_c_cell_params(
            base_cell_params, hkls_to_use_v[0], hkls_to_use_v[1], 
            distances_to_use_v[0], distances_to_use_v[1])
                

        # Build the cif file for the reference region
        path_to_v_unitcell = Build_virtual_crystal_cif(
            model_cells_filepath, best_cryst_phase_name, 
            a_v_cell, b_v_cell, c_v_cell)
        
        
        paths_to_virt_ucells.append(path_to_v_unitcell)
        
        
    return paths_to_virt_ucells
        



def Build_All_Virtual_Crysts_SameDistRef(
        analysed_image, image_in_dataset_whole, 
        best_GPA_ref_spot_pair,
        dist_spot_1_ref_subpix,
        dist_spot_2_ref_subpix,
        image_segmented, label_of_GPA_ref,
        GPA_resolution, model_cells_filepath):
    
    '''
    Builds virtual crystals based on these spots that are within the mask
    range drawn by the GPA, and then builds the virtual crystal by making
    these spots from other phases to be the exact same distance as the 
    reference region and its refined spots.
    So we search for the hkl1 and hkl2 of the spots that are close to 
    the g vectors 1 and 2 and make the virtual crystal whose hkl1 and hkl2
    have the distances we have subpixel refined for g1 and g2

    Parameters
    ----------
    analysed_image : analysed_image object
    image_in_dataset_whole : image_in_dataset object
    best_GPA_ref_spot_pair : scored_spot_pair respresenting the GPA g vects
    dist_spot_1_ref_subpix : distance of refined g1
    dist_spot_2_ref_subpix : distance of refined g2
    image_segmented : segm image
    label_of_GPA_ref : label corresponding to the region taken as reference
    GPA_resolution : resolution in nm of the mask found in the GPA
    model_cells_filepath : path to the cif cells

    Returns
    -------
    paths_to_virt_ucells : list of str, some with the virtual crystal if found
                            and created, or to base crystal if not found
    scored_spot_pairs_found: list of scored_spot_pairs_found
    scaled_cords_spots: list of pixel coordinates of the spots scaled 
                        to the full image

    '''
    
    # Retrieve data extracted from the reference 
    crop_outputs_dict = analysed_image.Crop_outputs

    # Pixels within the whole image in which the crop of the reference is taken, 
    # so the box of the reference itself [first_row,last_row,first_col,last_col]
    scaled_reference_coords_GPA_ref = crop_outputs_dict[str(int(label_of_GPA_ref)) + '_pixel_ref_cords']

    # 
    image_crop_hs_signal_GPA_ref = crop_outputs_dict[str(int(label_of_GPA_ref)) + '_hs_signal']
    FFT_image_array_GPA_ref, _ =  ImCalTrans.Compute_FFT_ImageArray(np.asarray(image_crop_hs_signal_GPA_ref))

    crop_list_refined_cryst_spots_GPA_ref = crop_outputs_dict[str(int(label_of_GPA_ref)) + '_list_refined_cryst_spots']
    refined_pixels_GPA_ref = crop_outputs_dict[str(int(label_of_GPA_ref)) + '_refined_pixels']
    spots_int_reference_GPA_ref = crop_outputs_dict[str(int(label_of_GPA_ref)) + '_spots_int_reference']

    # Get the best cryst spot
    best_cryst_spot_GPA_ref = crop_list_refined_cryst_spots_GPA_ref[0]

    # Retrieve the best spot pair to be considered the GPA g vectors
    
    # Find the best spots considered in that crystal phase, which should 
    # be the best ones to constitute the GPA g vectors
    # and its information to help build the virutal crystal, although they
    #  need to be updated after the refining of the g vectors with the ref
    spot1_int_ref_GPA_ref = best_GPA_ref_spot_pair.spot1_int_ref
    spot2_int_ref_GPA_ref = best_GPA_ref_spot_pair.spot2_int_ref
    
    # Retrieve the info of the spots acting as g vectors
    hkl1_reference_GPA_ref = best_GPA_ref_spot_pair.hkl1_reference
    hkl2_reference_GPA_ref = best_GPA_ref_spot_pair.hkl2_reference
    spot1_dist_GPA_ref = best_GPA_ref_spot_pair.spot1_dist
    spot2_dist_GPA_ref = best_GPA_ref_spot_pair.spot2_dist
    angle_between_GPA_ref = best_GPA_ref_spot_pair.angle_between
    spot1_angle_x_GPA_ref = best_GPA_ref_spot_pair.spot1_angle_to_x
    spot2_angle_x_GPA_ref = best_GPA_ref_spot_pair.spot2_angle_to_x
    found_phase_name_GPA_ref = best_cryst_spot_GPA_ref.phase_name
    
    
    # coords of the best peaks to use as GPA g vectors in coordinartes of the crop
    # NOT of the whole image
    cord_spot_1_GPA_ref = refined_pixels_GPA_ref[int(spot1_int_ref_GPA_ref)]
    cord_spot_2_GPA_ref = refined_pixels_GPA_ref[int(spot2_int_ref_GPA_ref)]
    

    # image in dataset base
    image_array_whole = image_in_dataset_whole.image_arraynp_st
    total_pixels_whole = image_in_dataset_whole.total_pixels
    pixel_size_whole = image_in_dataset_whole.x_calibration
    FFT_calibration_whole = image_in_dataset_whole.FFT_calibration
    FFT_whole, _ =  ImCalTrans.Compute_FFT_ImageArray(np.asarray(image_in_dataset_whole.hyperspy_2Dsignal))
    
    
    # generate a 9x9 pixel square arround the best coordinate scaled to the whole image
    # so from all this pixels we can find the best one that represents the maximum of
    # the peak
    
    scaled_cord_spot_1_GPA_y = int(np.round((cord_spot_1_GPA_ref[0]/FFT_image_array_GPA_ref.shape[0])*total_pixels_whole))
    scaled_cord_spot_1_GPA_x = int(np.round((cord_spot_1_GPA_ref[1]/FFT_image_array_GPA_ref.shape[1])*total_pixels_whole))
    scaled_cord_spot_2_GPA_y = int(np.round((cord_spot_2_GPA_ref[0]/FFT_image_array_GPA_ref.shape[0])*total_pixels_whole))
    scaled_cord_spot_2_GPA_x = int(np.round((cord_spot_2_GPA_ref[1]/FFT_image_array_GPA_ref.shape[1])*total_pixels_whole))
    
    if scaled_cord_spot_1_GPA_y-1 >= 0 and scaled_cord_spot_1_GPA_y+2 <= total_pixels_whole and scaled_cord_spot_1_GPA_x-1 >= 0 and scaled_cord_spot_1_GPA_x+2 <= total_pixels_whole:
        
        FFT_crop_spot_1 = FFT_whole[scaled_cord_spot_1_GPA_y-1:scaled_cord_spot_1_GPA_y+2,scaled_cord_spot_1_GPA_x-1:scaled_cord_spot_1_GPA_x+2]
        cords_max1 = np.where(FFT_crop_spot_1 == np.max(FFT_crop_spot_1))
    
        scaled_cord_spot_1_GPA_y = scaled_cord_spot_1_GPA_y - 1 + cords_max1[0][0]
        scaled_cord_spot_1_GPA_x = scaled_cord_spot_1_GPA_x - 1 + cords_max1[1][0]
        
    
    if scaled_cord_spot_2_GPA_y-1 >= 0 and scaled_cord_spot_2_GPA_y+2 <= total_pixels_whole and scaled_cord_spot_2_GPA_x-1 >= 0 and scaled_cord_spot_2_GPA_x+2 <= total_pixels_whole:
        
        FFT_crop_spot_2 = FFT_whole[scaled_cord_spot_2_GPA_y-1:scaled_cord_spot_2_GPA_y+2,scaled_cord_spot_2_GPA_x-1:scaled_cord_spot_2_GPA_x+2]
        cords_max2 = np.where(FFT_crop_spot_2 == np.max(FFT_crop_spot_2))
    
        scaled_cord_spot_2_GPA_y = scaled_cord_spot_2_GPA_y - 1 + cords_max2[0][0]
        scaled_cord_spot_2_GPA_x = scaled_cord_spot_2_GPA_x - 1 + cords_max2[1][0]
    # else:
    #     the coordinates are the same as they were
    
    spot_1_coords = np.array([scaled_cord_spot_1_GPA_y, scaled_cord_spot_1_GPA_x])
    spot_2_coords = np.array([scaled_cord_spot_2_GPA_y, scaled_cord_spot_2_GPA_x]) 
    
    
    labels_unique = np.unique(image_segmented)
    labels_unique_no_ref = labels_unique[labels_unique != label_of_GPA_ref]
    
    
    # list to store info for the whole process, if needed
    paths_to_virt_ucells = []
    
    for compl_label in labels_unique_no_ref:
        # store the hkl 1 and 2 found for every virutal cell created
        scored_spot_pairs_found = []
        scaled_cords_spots = []
        # store all the coorindates from where crystals for 
        # just one
        all_possible_coords_to_check = []
        # Pixels within the whole image in which the crop of the reference is taken, 
        # so the box of the reference itself [first_row,last_row,first_col,last_col]
        scaled_reference_coords_compl_label = crop_outputs_dict[str(int(compl_label)) + '_pixel_ref_cords']
    
        # 
        image_crop_hs_signal_compl_label = crop_outputs_dict[str(int(compl_label)) + '_hs_signal']
        FFT_image_array_compl_label, _ =  ImCalTrans.Compute_FFT_ImageArray(np.asarray(image_crop_hs_signal_compl_label))
    
        crop_list_refined_cryst_spots_compl_label = crop_outputs_dict[str(int(compl_label)) + '_list_refined_cryst_spots']
        refined_pixels_compl_label = crop_outputs_dict[str(int(compl_label)) + '_refined_pixels']
        spots_int_reference_compl_label = crop_outputs_dict[str(int(compl_label)) + '_spots_int_reference']
    
        
        # Get the best cryst spot to get all distances of this found phase
        # if at least one crystal was found in that label
        if len(crop_list_refined_cryst_spots_compl_label) != 0:
            # store the coordinates of spots
            # convert them relative to the whole image
            best_cryst_spot_compl_label = crop_list_refined_cryst_spots_compl_label[0]
            found_phase_name_compl_label = best_cryst_spot_compl_label.phase_name
        else:
            continue

        # loop through the spot pairs
        for spot_pair in best_cryst_spot_compl_label.spot_pairs_obj:
             
            spot1_int_ref_compl_label = spot_pair.spot1_int_ref
            spot2_int_ref_compl_label = spot_pair.spot2_int_ref
             
            # Retrieve the info of the spots acting as g vectors
            hkl1_reference_compl_label = spot_pair.hkl1_reference
            hkl2_reference_compl_label = spot_pair.hkl2_reference
            spot1_dist_compl_label = spot_pair.spot1_dist
            spot2_dist_compl_label = spot_pair.spot2_dist
            angle_between_compl_label = spot_pair.angle_between
            spot1_angle_x_compl_label = spot_pair.spot1_angle_to_x
            spot2_angle_x_compl_label = spot_pair.spot2_angle_to_x
             
             
            # coords of the best peaks to use as GPA g vectors in coordinartes of the crop
            # NOT of the whole image
            cord_spot_1_compl_label = refined_pixels_compl_label[int(spot1_int_ref_compl_label)]
            cord_spot_2_compl_label = refined_pixels_compl_label[int(spot2_int_ref_compl_label)]
             
            scaled_cord_spot_1_compl_label = (cord_spot_1_compl_label/FFT_image_array_compl_label.shape[0])*total_pixels_whole
            scaled_cord_spot_2_compl_label = (cord_spot_2_compl_label/FFT_image_array_compl_label.shape[0])*total_pixels_whole
             
            
            # whether, for a single phase in compl_label, the spot close to the 
            # reference spot 1 and 2 have been found (close meaning within the mask)
            spot_1_corresp_found = 0
            spot_2_corresp_found = 0
            
                
            # spot 1 ref to spot 1 trial
            distance_y = np.abs(scaled_cord_spot_1_compl_label[0] - spot_1_coords[0])*FFT_calibration_whole
            distance_x = np.abs(scaled_cord_spot_1_compl_label[1] - spot_1_coords[1])*FFT_calibration_whole
        
            distance_in_nminv = np.sqrt(distance_y**2 + distance_x**2)
            distance_in_nm = 1/distance_in_nminv
        
            if distance_in_nm >= GPA_resolution and spot_1_corresp_found == 0:
                
                # then this point is within the mask range and then 
                # is used as the plane to be considered
                # and extract the info from it
                # basically the hkl                 
                hkl_corresp_spot_1 = hkl1_reference_compl_label
                spot_1_corresp_scalcords = scaled_cord_spot_1_compl_label
                spot_1_int_ref = spot1_int_ref_compl_label
                spot_1_corresp_found = 1
                
                
            # spot 1 ref to spot 2 trial
            distance_y = np.abs(scaled_cord_spot_2_compl_label[0] - spot_1_coords[0])*FFT_calibration_whole
            distance_x = np.abs(scaled_cord_spot_2_compl_label[1] - spot_1_coords[1])*FFT_calibration_whole
        
            distance_in_nminv = np.sqrt(distance_y**2 + distance_x**2)
            distance_in_nm = 1/distance_in_nminv
        
            if distance_in_nm >= GPA_resolution and spot_1_corresp_found == 0:
                
                # then this point is within the mask range and then 
                # is used as the plane to be considered
                # and extract the info from it
                # basically the hkl                 
                hkl_corresp_spot_1 = hkl2_reference_compl_label
                spot_1_corresp_scalcords = scaled_cord_spot_2_compl_label
                spot_1_int_ref = spot2_int_ref_compl_label
                spot_1_corresp_found = 1
            
            
            # spot 2 ref to spot 1 trial
            distance_y = np.abs(scaled_cord_spot_1_compl_label[0] - spot_2_coords[0])*FFT_calibration_whole
            distance_x = np.abs(scaled_cord_spot_1_compl_label[1] - spot_2_coords[1])*FFT_calibration_whole
        
            distance_in_nminv = np.sqrt(distance_y**2 + distance_x**2)
            distance_in_nm = 1/distance_in_nminv
        
            if distance_in_nm >= GPA_resolution and spot_2_corresp_found == 0:
                
                # then this point is within the mask range and then 
                # is used as the plane to be considered
                # and extract the info from it
                # basically the hkl                 
                hkl_corresp_spot_2 = hkl1_reference_compl_label
                spot_2_corresp_scalcords = scaled_cord_spot_1_compl_label
                spot_2_int_ref = spot1_int_ref_compl_label
                spot_2_corresp_found = 1

            
            # spot 2 ref to spot 2 trial
            distance_y = np.abs(scaled_cord_spot_2_compl_label[0] - spot_2_coords[0])*FFT_calibration_whole
            distance_x = np.abs(scaled_cord_spot_2_compl_label[1] - spot_2_coords[1])*FFT_calibration_whole
        
            distance_in_nminv = np.sqrt(distance_y**2 + distance_x**2)
            distance_in_nm = 1/distance_in_nminv
        
            if distance_in_nm >= GPA_resolution and spot_2_corresp_found == 0:
                
                # then this point is within the mask range and then 
                # is used as the plane to be considered
                # and extract the info from it
                # basically the hkl                 
                hkl_corresp_spot_2 = hkl2_reference_compl_label
                spot_2_corresp_scalcords = scaled_cord_spot_2_compl_label
                spot_2_int_ref = spot2_int_ref_compl_label
                spot_2_corresp_found = 1


            # if both planes have been found, break the loop through the spot
            # pairs and build the virtual crystal based on this
            if spot_1_corresp_found == 1 and spot_2_corresp_found == 1:
                break
        
        # if both planes have been found close to the reference one,
        # then build the virtual crystal
        if spot_1_corresp_found == 1 and spot_2_corresp_found == 1:

            # then at this point we have two planes and two distances 
            # build the virtual crsytal based on this information         
            cif_base_cell_filepath = model_cells_filepath + found_phase_name_compl_label + '.cif'
            
            # build the ase crystal path
            ase_unit_cell_base = ase.io.read(cif_base_cell_filepath)
            
            spacegroup_cell_base = ase_unit_cell_base.info['spacegroup'].no
            
            a_base, b_base, c_base, alfa_base, beta_base, gamma_base = ase_unit_cell_base.cell.cellpar()
            
            base_cell_params = a_base, b_base, c_base, alfa_base, beta_base, gamma_base
            
            # compute the virtual cell parameters 
            a_v_cell, b_v_cell, c_v_cell = Find_virtual_a_cell_c_cell_params(
                base_cell_params, hkl_corresp_spot_1, hkl_corresp_spot_2, 
                dist_spot_1_ref_subpix, dist_spot_2_ref_subpix)
                    
            # Build the cif file for the reference region
            path_to_v_unitcell = Build_virtual_crystal_cif(
                model_cells_filepath, found_phase_name_compl_label, 
                a_v_cell, b_v_cell, c_v_cell)
            
            paths_to_virt_ucells.append(path_to_v_unitcell)
            
            # append the scored spot pair that contains the spots found
            for spot_pair in best_cryst_spot_compl_label.spot_pairs_obj:
                if (spot_pair.spot1_int_ref == spot_1_int_ref and spot_pair.spot2_int_ref == spot_2_int_ref):
                    scaled_cords_spots.append([spot_1_corresp_scalcords, spot_2_corresp_scalcords])
                    scored_spot_pairs_found.append(spot_pair)
                    break
                    
                if (spot_pair.spot1_int_ref == spot_2_int_ref and spot_pair.spot2_int_ref == spot_1_int_ref):
                    scaled_cords_spots.append([spot_2_corresp_scalcords, spot_1_corresp_scalcords])
                    scored_spot_pairs_found.append(spot_pair)
                    break
                    

            
        else:
            # Not enough spots 0 or 1 were found to build the virtual crystal
            # then we just indicate that no virtual is built but base is kept
            
            cif_base_cell_filepath = model_cells_filepath + found_phase_name_compl_label + '.cif'
        
            paths_to_virt_ucells.append(cif_base_cell_filepath)
            scaled_cords_spots.append([None, None])
            scored_spot_pairs_found.append(None)

    return paths_to_virt_ucells, scored_spot_pairs_found, scaled_cords_spots
                


def Purge_Atoms_InterDistance_Tolerance(
        list_atoms, min_distance_tol = 0.1):
    '''
    Function deleting the atoms which have neighbours at a distance smaller
    than the min_distance_tol defined.
    Not only does it remove the atom but selects from a pair or a group of
    close neighbours which is the best atom to remove based on the averaged
    distance between the closest neighbours (set as the 5 closest neighbours)
    by deleting the atom whose average distance is smaller, so closest to 
    the other atoms, then the function gives more credit and will always
    keep the atoms that have a bigger interatomi distance with its neighbours

    Parameters
    ----------
    list_atoms : List of Atom objects describing their position in the model
    min_distance_tol : minimum interatomic distance in Angstroms,
                    smaller distances will mean be more permissive while
                    larger distances would remove more atoms
        DESCRIPTION. The default is 0.05.

    Returns
    -------
    list_atoms_purged : list with only the atoms 

    '''

    list_atoms_purged = list_atoms.copy()
    
    # iterate over a list that never changes
    for atom in list_atoms:

        pos_x = atom.x
        pos_y = atom.y
        pos_z = atom.z
        
        # List of the rest of the atoms to compute the distances
        list_atoms_compare = list_atoms_purged.copy()
        if atom in list_atoms_compare:
            list_atoms_compare.remove(atom)
        else:
            continue
        
        positions_arr = np.array(list(map(lambda a: [a.x, a.y, a.z], list_atoms_compare)))
        
        
        distances = np.sqrt((positions_arr[:,0] - pos_x)**2 + (positions_arr[:,1] - pos_y)**2 + (positions_arr[:,2] - pos_z)**2)
    
        # filter the distances given the min_distance_tol
        list_atoms_within_tolerance = np.asarray(list_atoms_compare)
        list_atoms_within_tolerance = list_atoms_within_tolerance[distances < min_distance_tol]
        distances_within_tolerance = distances[distances < min_distance_tol]
        
        list_atoms_within_tolerance = list(list_atoms_within_tolerance)
        
        # Only keep the atom if no distance is smaller than the minimum
        # and check which to remove if some were found within the min dist
        if len(distances_within_tolerance) != 0:
  
            # in case atoms are found within the minimum distance
            
            # consider the initial target atom into the computations again
            # to have it as comparison
            list_atoms_within_tolerance.append(atom)
            # we want to keep the atom that maximises the distance
            # between their neighbouring atoms
            average_distances_neighbours = []
            
            for atom_close in list_atoms_within_tolerance:
                
                
                pos_x2 = atom_close.x
                pos_y2 = atom_close.y
                pos_z2 = atom_close.z
                
                # no need to do the check if in, as previously done, and no changes
                list_atoms_compare2 = list_atoms_purged.copy()
                list_atoms_compare2.remove(atom_close)

                positions_arr2 = np.array(list(map(lambda a: [a.x, a.y, a.z], list_atoms_compare2)))
        
                distances2 = np.sqrt((positions_arr2[:,0] - pos_x2)**2 + (positions_arr2[:,1] - pos_y2)**2 + (positions_arr2[:,2] - pos_z2)**2)
                
                # sort distances from smallest to largest and get the 5 smallest
                # (5 closest neighbours)
                
                distances2 = np.sort(distances2)
                
                neighbour_dists = distances2[:5]
                
                average_dist = np.mean(neighbour_dists)
                
                average_distances_neighbours.append(average_dist)
                
                
            # now average_distances_neighbours and list_atoms_within_tolerance
            # correlate 1 to 1 to get the atom with the smallest averaged distance
            
            # Atom to remove 
            atom_to_remove = list_atoms_within_tolerance[np.argmin(average_distances_neighbours)]
            
            list_atoms_purged.remove(atom_to_remove)
    
                
    return list_atoms_purged    
                



def Distort_AtoModel_Region(
        atom_models_filepath, Dispx, Dispy, Box_strain_pixels, 
        pixel_size_whole, total_pixels_whole,
        B_strain_aug_fact = 0.15, min_distance_tol = 0.1,
        purge_interatomic_distance = True, purge_wrong_displacements = False):
    '''
    The main function distorting the atomic models originally built by
    the rest of the functions after checking the GPA
    
    This function takes the input of the original atomistic models which are 
    built by considering the virtual crystals computed by taking into account
    the computed GPA maps and the crystal planes found, and checks 
    whether multiple models were generated for each label or region or just 
    a global one.
    If multiple are found, in order to provide overlapping in the interfaces
    to allow for security in the displacements and still have elements within
    the segmented region, we loop over them all for all the labels
    to displace them accordingly for the later check and mergin with the function
    Refine_StrainedRegion_MultiAtomBlock_Segmentation
    
    In case there is just one global cell, meaning that a single atomic block
    with the shared virtual crystal for many labels was created,
    then only this one is displaced and then for the chemical/segmentation
    checking we use the function 
    Refine_StrainedRegion_SingleAtomBlock_Segmentation
    
    Mutiple models are saved in order to keep the track of the process,
    first the cut of the region, then displacement of the region,
    and then displacement but filtering for atoms not correctly displaced
    or being to close to each other, as ruled by the booleans
    purge_interatomic_distance and purge_wrong_displacements 
    

    Parameters
    ----------
    atom_models_filepath : path to the atomistic models
    Dispx : array, x displacement field for the whole image computed with GPA
    Dispy : array, y displacement field for the whole image computed with GPA
    Box_strain_pixels : box where to apply the strain fields to
        in format 
        Box_strain_pixels = [B_strain_y_i, B_strain_y_f, B_strain_x_i, B_strain_x_f] 
    pixel_size_whole : pixel size of the image
    total_pixels_whole : total number pixels of the image
    B_strain_aug_fact : float, factor to amplify the box where the strain 
                fields are applied. The larger, the safer the process is
                as it would allow larger displacements, but more atoms are 
                moved so more computationally demanding
        DESCRIPTION. The default is 0.15.
        
    min_distance_tol : minimum interatomic distance in Angstroms,
                    smaller distances will mean be more permissive while
                    larger distances would remove more atoms
        DESCRIPTION. The default is 0.1., in Angstroms
    purge_interatomic_distance : purge by the interatomic distances between
                                the atoms, by a value defined by min_distance_tol
                                in Angstroms
        DESCRIPTION. The default is True.
    purge_wrong_displacements : purge these atoms which were not correctly
                                dispalced after all the allowed iterations
        DESCRIPTION. The default is False.
        
    Both purges can happen at the same time and applied to the same model

    Returns
    -------
    path_global_strained_purged. Path to the last file created, strained and
            purged, which will be the global one for the case of single atom
            block (which is the one interesting for inputting it into the function
            Refine_StrainedRegion_SingleAtomBlock_Segmentation)
            which requires the use of this file
            For the other case with many files (one per label) it will return 
            the latest label file, but here it is not important
            as we do not call this path if many files are to be analysed

    '''
    
    
    # Increase the box a certain factor to account for the displacements
    # that can happend at the edges of the box
    # At the very end this is reverted by having -B_strain_aug_fact
    # and filter the box with the atoms inside to have the final model
    Box_strain_pixels = GPA_sp.Mod_GPA_RefRectangle(
        Box_strain_pixels, B_strain_aug_fact)    
    
    #Disp_x = (Dispx[B2Y:B2Y+width2,B2X:B2X+height2]-Dispx[B2Y+Ref_point[1],B2X+Ref_point[0]])*pixsize
    Disp_x = (Dispx[Box_strain_pixels[0]:Box_strain_pixels[1],
                    Box_strain_pixels[2]:Box_strain_pixels[3]])*pixel_size_whole
    
    Disp_y = (Dispy[Box_strain_pixels[0]:Box_strain_pixels[1],
                    Box_strain_pixels[2]:Box_strain_pixels[3]])*pixel_size_whole


    # !!! UNITS CHANGE (nm --> angstroms)
    # The dispacement needs to be in the units of the atomic coordinates
    # in the model, so in angstroms
    Disp_x = Disp_x*10
    Disp_y = Disp_y*10
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(Disp_x, cmap='jet')
    plt.show()
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(Disp_y, cmap='jet')
    plt.show()
    # profile along displacement image
    
    profile_content = Disp_y[:, 75:125]
    
    profile_av = np.mean(profile_content, axis = 1)
    prof_x = np.arange(0, len(profile_av), 1)*pixel_size_whole*10
    
    plt.plot(prof_x, profile_av)
    plt.show()


    # Adjust the box drawn in image coordiantes, to the box defined in the 
    # atomic coords model global supercell
    
    # BOX = [x_i, y_i, x_f, y_f]  in cart coords, 0,0 bottom left corner of axis
    # the box must be an array
    
    region_to_strain_atomcords = np.array([Box_strain_pixels[2]*pixel_size_whole,
                                           (total_pixels_whole-Box_strain_pixels[1])*pixel_size_whole,
                                           Box_strain_pixels[3]*pixel_size_whole,
                                           (total_pixels_whole-Box_strain_pixels[0])*pixel_size_whole])
    
    # !!! UNITS CHANGE (nm --> angstroms)
    # Need to change the coordinates from nm to angstroms as the coords 
    # inside the file are in angstroms
    region_to_strain_atomcords = region_to_strain_atomcords*10
    

    # check if there exists a combined supercell in the path
    # as this would mean it is built upon some equally oriented structures
    global_merged = True
    for atomistic_model in os.listdir(atom_models_filepath):
        if '.xyz' in atomistic_model:
            if 'global_device_supercell' in atomistic_model:
                global_merged = True
    
    
    # Loop through the models for each label if spot separated heterostructure
    # or the global one if it comes from the single atom block
    for atomistic_model in os.listdir(atom_models_filepath):
        
        if global_merged == True:
            if 'global_device_supercell'  in atomistic_model:
                label_region = ''
                region_cut_base = 'region_cut_base'
                region_cut_strained = 'region_cut_strained'
                region_cut_strained_purged = 'region_cut_strained_purged'
    
            else:
                continue
            
        elif global_merged == False:
            if 'temp_xyz_rot_supercell_' in atomistic_model:
                label_region = atomistic_model[atomistic_model.find('supercell_')+10:]
                label_region = label_region[:label_region.find('.xyz')]
                region_cut_base = 'region_cut_base_'
                region_cut_strained = 'region_cut_strained_'
                region_cut_strained_purged = 'region_cut_strained_purged_'
            else:
                continue
            
    
        atomod_filepath = atom_models_filepath + atomistic_model
        
        # Identify the label of the region in the filname
        
        # list of atoms that are gonna handle the displacements for that given model
        region_to_strain_atom_list, indexes_to_keep_cut = read_xyz(
            atomod_filepath, region_to_strain_atomcords, extra=0)
        
        # Add name to the cutting of the atoms within the filepath
        path_region_to_strain_base = atom_models_filepath + region_cut_base + label_region +'.xyz'
        
        save_xyf(
            region_to_strain_atom_list, path_region_to_strain_base)
        print('There are %i atoms in the selected region'%len(region_to_strain_atom_list))
        
        # No need to save labels anymore with this approach
        # new_reg_filepath = GPA_AtoMod.Cut_atom_identifier_Region(
        #     atom_models_filepath, indexes_to_keep_cut)
        
        # Displace the atom list
        list_of_errors_displ = Displace(
            region_to_strain_atom_list, Disp_y, -Disp_x, region_to_strain_atomcords, 
            rate=0.8, Max_try=50)    
        
        # Purge the list of incorrectly displaced atoms by first getting just the 
        # atoms that are within the box (in case a displacement brought some outside)
        region_to_strain_atom_list, indexes_to_purge = Purge(
            region_to_strain_atom_list, BOX = region_to_strain_atomcords)
        
        path_region_to_strained = atom_models_filepath + region_cut_strained + label_region + '.xyz'
        
        save_xyf(
            region_to_strain_atom_list, path_region_to_strained)
        
        # No need to purge the list with the indices 
        # new_purg_filepath = GPA_AtoMod.Purge_Indices_Labels_AtomList(
        #     atom_models_filepath, indexes_to_purge, 'strained')
        
    
        if purge_wrong_displacements == True:
            # Remove the elements that have not been displaced correctly
            region_to_strain_atom_list, indexes_to_purge = Purge(
                region_to_strain_atom_list, purge_list = list_of_errors_displ)
            
        if purge_interatomic_distance == True:
            # Remove the elements that have not been displaced correctly
            region_to_strain_atom_list = Purge_Atoms_InterDistance_Tolerance(
                region_to_strain_atom_list, min_distance_tol = min_distance_tol)
    
        
        path_global_strained_purged = atom_models_filepath + region_cut_strained_purged + label_region + '.xyz'
        
        save_xyf(
            region_to_strain_atom_list, path_global_strained_purged)
        
        # Purge the list with the indices as well
        # new_purg_filepath = GPA_AtoMod.Purge_Indices_Labels_AtomList(
        #     atom_models_filepath, indexes_to_purge, 'strained_purged')

    return path_global_strained_purged


def Original_BoxSize_StrainedModel(
        Original_B_strain_aug_fact, Box_strain_pixels, 
        pixel_size_whole, total_pixels_whole,
        atomregmodel_path_final):
    '''
    Return the atomistic model that has been strained and finally merged
    and all the chemistry/segmentation rules to the original size
    which was originally input in Distort_AtoModel_Region
    but there was increased to make sure all the region would be full of atoms 
    despite any distortion moving them far away

    Parameters
    ----------
    Original_B_strain_aug_fact : original factor used in the 
                        Distort_AtoModel_Region function, which is an augmention
                        factor (positive), so in the function we make it
                        a reduction factor, making it negative
    Box_strain_pixels : same as used in Distort_AtoModel_Region
    atomregmodel_path_final : complete path to the final atomistic model after
                            all the strain and merging and segm/chem refinement

    Returns
    -------
    path_finalcut_strainedmodel.

    '''
    
    
    # Return the box of strained data to the size we originally wanted
    B_strain_aug_fact = -Original_B_strain_aug_fact
    Box_strain_pixels = GPA_sp.Mod_GPA_RefRectangle(
        Box_strain_pixels, B_strain_aug_fact)
    
    # With the new box (original piece we want)
    region_to_strain_atomcords = np.array([Box_strain_pixels[2]*pixel_size_whole,
                                           (total_pixels_whole-Box_strain_pixels[1])*pixel_size_whole,
                                           Box_strain_pixels[3]*pixel_size_whole,
                                           (total_pixels_whole-Box_strain_pixels[0])*pixel_size_whole])
    
    

    # list of atoms that are inside the box we want
    filtered_region_atom_list, indexes_to_keep_cut = read_xyz(
        atomregmodel_path_final, region_to_strain_atomcords, extra=0)
    
    # New final name for the cut version
    path_finalcut_strainedmodel = atomregmodel_path_final[:atomregmodel_path_final.find('.xyz')] + '_cut.xyz'
    
    save_xyf(
        filtered_region_atom_list, path_finalcut_strainedmodel)
    print('There are %i atoms in the selected region'%len(filtered_region_atom_list))
    
    
    return path_finalcut_strainedmodel




def Refine_StrainedRegion_SingleAtomBlock_Segmentation(
        analysed_image, model_cells_filepath, path_global_strained_purged, 
        label_of_GPA_ref, labels_equally_oriented_as_ref,
        conts_vertx_per_region, paths_to_virt_ucells, 
        collapse_occupancies = True):
    '''
    Function to, given the atomistic model of the single block constructed
    from just the reference virtual crystal and then strained
    for all the labels inside labels_equally_oriented_as_ref, apply
    the segmentation information to limit and substitute the atomic species
    of these segmented regions to apply the chemistry information 
    (extracted from segmentation in this function)
    It takes into account the relative postiions of the primitive cells from
    reference and the other regions by comparing them and checking their
    Wyckoff multiplicity

    Parameters
    ----------
    analysed_image : 
    model_cells_filepath : 
    path_global_strained_purged : path to the global file named global_...
                        being the final strained and purged model for the whole
                        system with all the phases and so on
    label_of_GPA_ref : 
    labels_equally_oriented_as_ref :
    conts_vertx_per_region : 
    paths_to_virt_ucells : 
    collapse_occupancies : bool, optional, whether to collapse or not the
                fractional occupancies into unitary ones
        DESCRIPTION. The default is True.

    Returns
    -------
    atomregmodel_path_final : path to the model

    '''
    
    # Get shared space group for all the regions considered here
    unitcell_to_sg = ase.io.read(paths_to_virt_ucells[0])
    shared_space_group = unitcell_to_sg.info['spacegroup'].no
    
    crop_outputs_dict = analysed_image.Crop_outputs

    # STEP 1: Get first the primitive unit cell positions of all the phases equally
    # oriented, to be able to later swap the atoms if the same position
    # they are linked 1 to 1 to the labels_equally_oriented_as_ref
    primit_cell_pos_label = []

    # for all the labels equally orietned as reference except reference
    # filter the segmentation information
    for label_region in labels_equally_oriented_as_ref:
        
        # information from reference region, label_of_GPA_ref

        # Loop through the regions and build the atomic model from each region

        image_crop_hs_signal = crop_outputs_dict[str(label_region) + '_hs_signal']
        crop_list_refined_cryst_spots = crop_outputs_dict[str(label_region) + '_list_refined_cryst_spots']

        # most likely crystal found
        best_cryst_spot = crop_list_refined_cryst_spots[0]
        found_phase_name = best_cryst_spot.phase_name

        # find the space group of the reference cell
        # in principle there should always be a virtual cell created here
        # just add the security check
        cif_virtual_cell_filepath = model_cells_filepath + found_phase_name + '_v.cif'
        
        primitcellposes = AtomBuild.Store_cif_Relative_PrimitiveCellParams(
            cif_virtual_cell_filepath)
        
        
        primit_cell_pos_label.append(primitcellposes)
        
    # STEP 2: Now that we have the primitive cell positions, loop through them
    # with the label_of_GPA_ref  as reference to compare with its
    # found primitive positions and know what to change
    # Also, to know where to change it, use the segmentation information here     
        
    # for all the labels equally orietned as reference except reference
    # filter the segmentation information
    
    # List containing the reference PrimCellPos of the ref region to be
    # the reference in the comparison with the rest of the regions
    primit_cell_pos_ref = primit_cell_pos_label[labels_equally_oriented_as_ref.index(label_of_GPA_ref)]
    
    # List of atoms holding the coordinates of the atom
    list_original_atoms_objects_region, _ = read_xyz(
        path_global_strained_purged)    
    
    # Lists to store all the elements found for all the labels to 
    # be changed, except the reference ones that need to be kept unchanged
    # each element in the list corresponds to a label of the region 
    # that is in labels_equally_oriented_as_ref and is not label_of_GPA_ref
    global_list_list_original_atoms_inside = []
    global_list_list_indices_coords_inside = []
    global_list_list_new_atoms_to_substitute = []
    
    
    for label_region in labels_equally_oriented_as_ref:
        
        if label_region != label_of_GPA_ref:
            
            # information from reference region, label_of_GPA_ref
    
            # Loop through the regions and build the atomic model from each region
    
            image_crop_hs_signal = crop_outputs_dict[str(label_region) + '_hs_signal']
            crop_list_refined_cryst_spots = crop_outputs_dict[str(label_region) + '_list_refined_cryst_spots']
    
            # most likely crystal found
            best_cryst_spot = crop_list_refined_cryst_spots[0]
            found_phase_name = best_cryst_spot.phase_name
    
            # find the space group of the reference cell
            # in principle there should always be a virtual cell created here
            # just add the security check
            cif_virtual_cell_filepath = model_cells_filepath + found_phase_name + '_v.cif'

            # Keep the PrimCellPos for the given label to compare with 
            # the reference ones stored in primit_cell_pos_ref
            primit_cell_pos_lab = primit_cell_pos_label[labels_equally_oriented_as_ref.index(label_region)]
        
            # List with the contours for that region (label)
            list_of_contours_label = conts_vertx_per_region[str(label_region) + '_contours'].copy()
            
            # STEP 2.1: Find the atoms inside the segmented region for that label
            # List to store the indices of the atoms within the contour domains
            list_atoms_inside = []
            indices_coords_inside = []
            
            # Get the maximum ys and xs to shift the atom coordinate checker
            all_ys_max = []
            for dict_entry in conts_vertx_per_region:
                if '_rel_vertexs' in dict_entry:
                    
                    list_rel_vertex = np.copy(conts_vertx_per_region[dict_entry])
                    # !!! list_rel_vertex is in format of [y,x], and the functions that use them require 
                    # format [x,y], so just interchange the columns
                    list_rel_vertex_cache = np.copy(list_rel_vertex)
                    list_rel_vertex[:,0] = list_rel_vertex_cache[:,1]
                    list_rel_vertex[:,1] = list_rel_vertex_cache[:,0]
                    list_rel_vertex = list(list_rel_vertex)
    
                    #Get dimensions of the rectangle, in nm
                    x_min_c, x_max_c, x_length, y_min_c, y_max_c, y_length = AtomBuild.Get_Smallest_Rectangular_Cell(
                        list_rel_vertex)
                    
                    all_ys_max.append(y_max_c)
            
            # Values is in nm, change it afterwards within atomic_pos_xy        
            global_y_max = np.max(np.asarray(all_ys_max))
              
            for index, atom in enumerate(list_original_atoms_objects_region):
                
                # evaluate if the elements coords are within the segmented region and z thickness
                # !!! UNITS: for that, change the units of the atom coords from angs to nm
                # as the contours are expressed in nm
                # comparing it with global_y_max which is already in nm
                
                atomic_pos_xy = [atom.x/10, global_y_max - atom.y/10]
                
                # The contour checker checks the flipped version of 
                # the coordinate but then the normal one is added
                is_coord_inside = AtomBuild.Is_coordinate_inside_contour_diagonal(
                    atomic_pos_xy, list_of_contours_label)
                
                # if coordiante is inside store the index to keep the coord from the array
                if is_coord_inside == True:
                    list_atoms_inside.append(atom)
                    indices_coords_inside.append(index)
                
            # Now in indices_coords_inside we have the indices of the list
            # of atoms that need to be kept and the atoms itself in list_atoms_inside
            
            
            global_list_list_original_atoms_inside.append(list_atoms_inside)
            global_list_list_indices_coords_inside.append(indices_coords_inside)
            
            
            # Check the relative positoins if the label ones are in the 
            # reference positions 
            # if they are make the position aware change
            # if not change the composition globally
            position_aware_replace = True
            for prim_lab in primit_cell_pos_lab:
                equiv_found = 0
                for prim_ref in primit_cell_pos_ref: 
                    if prim_ref.x == prim_lab.x and prim_ref.y == prim_lab.y and prim_ref.z == prim_lab.z:
                        equiv_found = 1
                        break
                if equiv_found == 0:
                    # not found
                    position_aware_replace = False
                    break
                
            # STEP 2.2: Get the stoichiometry of the reference phase with the    
            # given occupancies and Wyckoff multiplciites of every atom
            # Correlates 1 to 1 with the list primit_cell_pos_ref
            # so for primitve_cell_position = primit_cell_pos_ref[0]
            # we have the list of elements = elements_ref[0]
            # with the list of number of elements in the conventional unit cell
            # equal to element_indiv_stoichio[0]
            elements_ref = []
            element_indiv_stoichio = []
            
            for prim_ref in primit_cell_pos_ref: 
                
                element = prim_ref.element.strip()
                x_pos = prim_ref.x
                y_pos = prim_ref.y
                z_pos = prim_ref.z
                occup = prim_ref.occ
                
                # Compute the multiplicity of that Wyckoff position
                wyckoff_multiplictiy = AtomBuild.Wyckoff_pos_Checker(
                    shared_space_group , [x_pos, y_pos, z_pos])
                
                # Number of elements in the conventional unit cell
                # from the primitive cell and its positions
                # for that given element at that primitive position
                
                numb_elemts_primpos = wyckoff_multiplictiy*occup
                
                elements_ref.append(element)
                element_indiv_stoichio.append(numb_elemts_primpos)
                    
            # Get same relation of elements in conventional unit cell
            # for the phase inside label
            elements_lab = []
            element_indiv_stoichio_lab = []
            
            for prim_lab in primit_cell_pos_lab: 
                
                element = prim_lab.element.strip()
                x_pos = prim_lab.x
                y_pos = prim_lab.y
                z_pos = prim_lab.z
                occup = prim_lab.occ
                
                # Compute the multiplicity of that Wyckoff position
                wyckoff_multiplictiy = AtomBuild.Wyckoff_pos_Checker(
                    shared_space_group , [x_pos, y_pos, z_pos])
                
                # Number of elements in the conventional unit cell
                # from the primitive cell and its positions
                # for that given element at that primitive position
                
                numb_elemts_primpos = wyckoff_multiplictiy*occup
                
                elements_lab.append(element)
                element_indiv_stoichio_lab.append(numb_elemts_primpos)
            
            
            # list to have the atoms to be substituted in the position defined
            # by the indices_coords_inside, 
            # is a list of lists as if we do not collapse occupancies
            # multiple atoms with their occs will substitue that given entry
            # update ready x and save xyz with occupancies and DW factors 
            new_atoms_to_substitute = []
            
            # STEP 3: Substitute the atoms for the actual one belonging to
            # that position in space looping through all the atoms found 
            # inside that region
            for index, atom in zip(
                    indices_coords_inside, list_atoms_inside):
                
                
                list_atoms_inside = []
                indices_coords_inside = []
                
                
                # Can be the symbol or Z depending on how it was input
                element_original = atom.Z.strip()
                # element_original is an elemen type of the phase found in the 
                # reference region, which needs to be substituted by the
                # corresponding element in the region where it belongs
                # thus, to its real type
                
                # STEP 3.1: Make the difference on whether the equivalent positions
                # are found from the reference file to the other labels
                # and make it position aware
                if position_aware_replace == True:
                    # Make the complex elemental substitution by considering
                    # the Wyckoff multiplicty of the elements included in 
                    # the reference and other phases 

                    # Lists linked 1 to 1 to each other to array for smart indexing
                    primit_cell_pos_ref = np.asarray(primit_cell_pos_ref)
                    elements_ref = np.asarray(elements_ref)
                    element_indiv_stoichio = np.asarray(element_indiv_stoichio)
                    
                    # Compute the probability with only the same element species
                    primits_refs_atom = primit_cell_pos_ref[elements_ref == element_original]
                    elems_indstoi_atom = element_indiv_stoichio[elements_ref == element_original]
                    elems_atom = elements_ref[elements_ref == element_original]
                    
                    probab_prim_pos = elems_indstoi_atom/np.sum(elems_indstoi_atom)
                    # Get the elements to be substitued by and their prob of 
                    # being the substitution 
                    elements_to_substitute = []
                    prob_of_substitution = []
                    
                    for primit_refs_atom, prob_primpos in zip(
                            primits_refs_atom, probab_prim_pos):
                        
                        for primit_pos_lab in primit_cell_pos_lab:
                            
                            if primit_refs_atom.x == primit_pos_lab.x and primit_refs_atom.y == primit_pos_lab.y and primit_refs_atom.z == primit_pos_lab.z:
                                element_subst = primit_pos_lab.element.strip()
                                prob_subst = prob_primpos*primit_pos_lab.occ
                                
                                elements_to_substitute.append(element_subst)
                                prob_of_substitution.append(prob_subst)
                    
                    
                    # Get the atoms colapsing or not on bolean
                    if collapse_occupancies == True:
                        # Just get one atom collapsing the probabilites
                        # computed before
                        
                        element_chosen = random.choices(
                            elements_to_substitute, 
                            weights = prob_of_substitution, k=1)[0] 
                       
                        # we create a new atom with occupancy 1, same position,
                        # and new element species as found by 
                        new_colaps_atom = Atom(
                            element_chosen, atom.x, atom.y, atom.z, occ = 1)
                        
                        new_atoms_to_substitute.append([new_colaps_atom])
                        
                    else:
                        # if collapse_occupancies == False:
                        
                        new_noncolaps_atoms = []    
                        for element_i, occ_i in zip(
                                elements_to_substitute, prob_of_substitution):
                            
                            new_noncolaps_atom = Atom(
                                element_i, atom.x, atom.y, atom.z, occ = occ_i)
                            new_noncolaps_atoms.append(new_noncolaps_atom)
                            
                        new_atoms_to_substitute.append(new_noncolaps_atoms)
                            
                
                    
                # STEP 3.2: Or just give a global probability if positions are
                # not found between reference and label regions
                else:
                    
                    # do the simple one wiht just deifning global probablilities
                    
                    
                    # Lists linked 1 to 1 to each other to array for smart indexing
                    primit_cell_pos_lab = np.asarray(primit_cell_pos_lab)
                    elements_lab = np.asarray(elements_lab)
                    element_indiv_stoichio_lab = np.asarray(element_indiv_stoichio_lab)
                    elements_probs_labs = element_indiv_stoichio_lab/np.sum(element_indiv_stoichio_lab)
                    
                    
                    # Get the atoms colapsing or not on bolean
                    if collapse_occupancies == True:
                        # Just get one atom collapsing the probabilites
                        # computed before
                        
                        element_chosen = random.choices(
                            elements_lab, 
                            weights = elements_probs_labs, k=1)[0] 
                       
                        # we create a new atom with occupancy 1, same position,
                        # and new element species as found by 
                        new_colaps_atom = Atom(
                            element_chosen, atom.x, atom.y, atom.z, occ = 1)
                        
                        new_atoms_to_substitute.append([new_colaps_atom])
                        
                    else:
                        # if collapse_occupancies == False:
                        
                        new_noncolaps_atoms = []    
                        for element_i, occ_i in zip(
                                elements_lab, elements_probs_labs):
                            
                            new_noncolaps_atom = Atom(
                                element_i, atom.x, atom.y, atom.z, occ = occ_i)
                            new_noncolaps_atoms.append(new_noncolaps_atom)
                            
                        new_atoms_to_substitute.append(new_noncolaps_atoms)

                    
            # add the new elements to the global list 
            global_list_list_new_atoms_to_substitute.append(new_atoms_to_substitute)

            
            # firsts the highest indices need to be changed 
            # so reverse the loop
                
    # STEP 4: Build the final list of new atoms
        
    # List to hold all the atoms that are inside the region
    # after being substituted if needed, or the old ones from the other regions
    # that do not need to have a substitution
    final_list_of_all_new_atoms = []
    final_list_of_all_subst_orig_atoms = []
    
    
    for list_indices, list_new_atoms, list_orig_atoms in zip(
            global_list_list_indices_coords_inside,
            global_list_list_new_atoms_to_substitute, 
            global_list_list_original_atoms_inside):
        
        for index_i, new_atom_i, orig_atom_i in zip(
                list_indices, list_new_atoms, list_orig_atoms):
            
            final_list_of_all_subst_orig_atoms.append(orig_atom_i)
            
            for new_atom in new_atom_i:
                final_list_of_all_new_atoms.append(new_atom)
    
    
    # Add the atoms not substituted, so from other regions
    # that are not checked, from reference region
    for atom_to_check in list_original_atoms_objects_region:
        
        if atom_to_check not in final_list_of_all_subst_orig_atoms:
            final_list_of_all_new_atoms.append(atom_to_check)
            

    # Finally write the xyz with the final substituted atoms 
    # naming the file on whether the occupancies colapsed or not
    
    if collapse_occupancies == True:
        
        # no need to save the occupancies
        # col in the name stands for colapsed occupancies
        atomregmodel_path_final =  path_global_strained_purged[:path_global_strained_purged.find('.xyz')] + '_Col_FINAL.xyz'                
        
        save_xyf(
            final_list_of_all_new_atoms, atomregmodel_path_final, save_occsDW = False)
        
    else:
        # if collapse_occupancies == False:
        atomregmodel_path_final =  path_global_strained_purged[:path_global_strained_purged.find('.xyz')] + '_NoCol_FINAL.xyz'                
        
        save_xyf(
            final_list_of_all_new_atoms, atomregmodel_path_final, save_occsDW = True)
                
            
    return atomregmodel_path_final





def Refine_StrainedRegion_MultiAtomBlock_Segmentation(
        atom_models_filepath, conts_vertx_per_region):
    '''
    Function to build the final model after the strain application to 
    each of the separated model files for each of the labels, and 
    this function checks whether the atoms inside the model obtained in label,
    really are within the region segmented for label
    So draws the boundaries between the atoms after the strain field application

    Parameters
    ----------
    atom_models_filepath : path to where all the models for all the labels 
                            after the strain application
    conts_vertx_per_region : TYPE
        DESCRIPTION.

    Returns
    -------
    atomregmodel_path_final : TYPE
        DESCRIPTION.

    '''

    # Get the maximum ys and xs to shift the atom coordinate checker
    all_ys_max = []
    for dict_entry in conts_vertx_per_region:
        if '_rel_vertexs' in dict_entry:
            
            list_rel_vertex = np.copy(conts_vertx_per_region[dict_entry])
            # !!! list_rel_vertex is in format of [y,x], and the functions that use them require 
            # format [x,y], so just interchange the columns
            list_rel_vertex_cache = np.copy(list_rel_vertex)
            list_rel_vertex[:,0] = list_rel_vertex_cache[:,1]
            list_rel_vertex[:,1] = list_rel_vertex_cache[:,0]
            list_rel_vertex = list(list_rel_vertex)

            #Get dimensions of the rectangle, in nm
            x_min_c, x_max_c, x_length, y_min_c, y_max_c, y_length = AtomBuild.Get_Smallest_Rectangular_Cell(
                list_rel_vertex)
            
            all_ys_max.append(y_max_c) 
            
    # Values is in nm, change it afterwards within atomic_pos_xy        
    global_y_max = np.max(np.asarray(all_ys_max))
    
    # list atoms inside their region initially segmented 
    final_list_of_all_new_atoms = []
    
    for atomistic_model in os.listdir(atom_models_filepath):
        # Just keep the strained and purged models
        if 'region_cut_strained_purged_'  not in atomistic_model:
            continue
        
        # Extract the label of the file to be able to check if it 
        # is inside the segmented contour 
        
        # as the filename starts with 'region_cut_strained_purged_' 
        # the position is already known 
        label_string = atomistic_model[27: atomistic_model.find('.xyz')]
        label = int(label_string)
        atomod_filepath = atom_models_filepath + atomistic_model

        # List of atoms holding the coordinates of the atomic model
        list_original_atoms_objects_region, _ = read_xyz(
            atomod_filepath)    
        
        # Get the contours for that segmented region /label       
        list_of_contours_label = conts_vertx_per_region[str(int(label)) + '_contours'].copy()

        # Loop through the atoms found and check if they are inside the
        # segment defined by region number label
        for atom in list_original_atoms_objects_region:
            
            atomic_pos_xy = [atom.x/10, global_y_max - atom.y/10]
            
            # The contour checker checks the flipped version of 
            # the coordinate but then the normal one is added
            is_coord_inside = AtomBuild.Is_coordinate_inside_contour_diagonal(
                atomic_pos_xy, list_of_contours_label)
            
            # if coordiante is inside store the index to keep the coord from the array
            # Then from their original region, label, we check if they are still there
            # with their possibly new atomic coordinates after the strain field
            # encoded in atom.x, atom.y
            if is_coord_inside == True:
                final_list_of_all_new_atoms.append(atom)
         
            
    # build the list of atoms 
    atomregmodel_path_final =  atom_models_filepath + 'region_cut_strained_purged_FINAL.xyz'          
    
    save_xyf(
        final_list_of_all_new_atoms, atomregmodel_path_final, save_occsDW = False)

    return atomregmodel_path_final






'''

fft of the whle image
to locate points


select automatticlaly points 
but with possibility to change point if wanted
just for the trials


make the conversion and generation of a new atomic phase that
gets the differences in distances as pointed by the computed planes
modificatoin of the cif file that is built by changing the
cell parameters
something easy to do

more dificult is to generalise the changes experienced in the plane distances with 
any possible phase  to any change cell space group


what distances to modify, make only in plane differences in the distances
or what?


build the model based on this distances 
whatm odels to modify only the one in the region taken as reference 
only the regiontakes as reference is considered 

then apply strain

then rebuild the model based on the new distances
and positions
of the segmened region and cut based on this with special awareness on
the elements used
and what should be cut and wht not


when wanting to move the elements back to their original interface 
definition then the to distinguish which elements to move, whihc ones correspod
to one regionand to another, add a label to them, for iinstance, in the deby weller
coefficient, assign a specific one to each Material NOT element, so then when
we want to cut the segmented region as originally, and with the expansion of the whole region
that would overlapp regions then this way we can cut it cutting this expansion





1st approximatin 

selecting a small region of one of the segments, so just one material

the region must be selected manually as it is something one must choose to
select that specific region of interest for instance a stacking fault


where you build the base virutal crystal from is not important as soon
as the informaton of where is kept, it is comming from a region of the image
wheere the displacement is known
as the region taken as reference will define thevirtual crstal and then all
the displacement map is modified depending on where the reference is taken






2nd approximation 

the region selected contatnis two materials and then two virtual crystals need to
be computed at the same time
actually it is two materials forming two virtual crystals that rae differetn because of 
their difference in composition but should be with same cell parameters as
they should be derived from the same set of g1 and g2 vectors

perform and apply the cell variation on both materials





should be possible to cmpute the virtual crytal of both regions or all the regions
with the same infrmaotin from g1 and g2 from onlny 1 computation of the gpa:
    with just one given reference:
as then the displacement computed is from that same virtual crystal in both
positions with the same g vectors so same distances but differetnc compoiusitons
so as it computes the DISPLACEMENTS of the atoms and NOT THE POSITIONS
then building from same informatino should be enough
because it is the displacements what is being tracked and what is made converge
with the displacement field



3r approx

same as 2nd on full image



'''













