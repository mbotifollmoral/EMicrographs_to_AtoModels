# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 16:01:12 2021

@author: Marc
"""

#from TestDiff.ipynb import Crystal
import numpy as np
import ctypes


#Wrapper

#path to the dll
lib = ctypes.CDLL("E:/Arxius varis/PhD/2nd_year/Code/Diffraction_pattern_simulation_E_V/Library/difftool_2.2/diffTools.dll")
#lib = ctypes.cdll.LoadLibrary("E:/Arxius varis/PhD/2nd_year/Code/Diffraction_pattern_simulation_E_V/Library/difftool/diffTools.dll")


CrystalHandle = ctypes.POINTER(ctypes.c_char)
c_int_array = np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS')
c_int_array_2 = np.ctypeslib.ndpointer(dtype=np.int32, ndim=2, flags='C_CONTIGUOUS')
c_float_array=np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags='C_CONTIGUOUS')
c_double_array=np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS')



lib.createCrystal.argtypes = [ctypes.c_char_p]
lib.createCrystal.restype = CrystalHandle

lib.calc_d.argtypes = [CrystalHandle, ctypes.c_bool, ctypes.c_float]
lib.calc_d.restypes = ctypes.c_int

lib.FindZA.argtypes = [CrystalHandle, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float]
lib.FindZA.restype = ctypes.c_int

lib.GetZA.argtypes = [CrystalHandle, ctypes.c_int, c_int_array]
lib.GetZA.restype = None

#custom for getting the possible indexation of the pair of spots considered
lib.Gethkls1.argtypes = [CrystalHandle, ctypes.c_int, c_int_array]
lib.Gethkls1.restype = None

lib.Gethkls2.argtypes = [CrystalHandle, ctypes.c_int, c_int_array]
lib.Gethkls2.restype = None

lib.destroyCrystal.argtypes = [CrystalHandle]
lib.destroyCrystal.restype = None

lib.angle.argtypes = [CrystalHandle, c_int_array, c_int_array]
lib.angle.restype = ctypes.c_float

lib.getF.argtypes = [CrystalHandle, ctypes.c_int]
lib.getF.restype = ctypes.c_double

lib.getDistances.argtypes = [CrystalHandle, ctypes.c_int]
lib.getDistances.restype = ctypes.c_float

lib.getIndexes.argtypes = [CrystalHandle, ctypes.c_int, c_int_array]
lib.getIndexes.restype = None



class Crystal:
    
    
    def __init__(self,name):
        self.instance = lib.createCrystal(name)
        self.phase_name=name

    
    def __del__(self):
        lib.destroyCrystal(self.instance)
    
    def getZA(self,N):
        n = ctypes.c_int(N)
        hkl = np.empty(3, dtype=np.int32)
        lib.GetZA(self.instance,n,hkl)
        return hkl
    
    def gethkls1(self,N):
        n = ctypes.c_int(N)
        hkl = np.empty(3, dtype=np.int32)
        lib.Gethkls1(self.instance,n,hkl)
        return hkl
    
    def gethkls2(self,N):
        n = ctypes.c_int(N)
        hkl = np.empty(3, dtype=np.int32)
        lib.Gethkls2(self.instance,n,hkl)
        return hkl
    
    
    def Diff(self,flag,D):
        min_d = ctypes.c_float(D)
        flagd=ctypes.c_bool(flag)
        N = lib.calc_d(self.instance,flagd,min_d)
        return N
    
    def FindZA(self,D1,D2,ANG,TOL):
        d1=ctypes.c_float(D1)
        d2=ctypes.c_float(D2)
        ang=ctypes.c_float(ANG)
        tol=ctypes.c_float(TOL)
        self.n = lib.FindZA(self.instance,d1,d2,ang,tol)
        return self.n
    
    def angle(self, hkl, hkl2):
        
        angle = lib.angle(self.instance,hkl,hkl2)
        return angle
    
    def getIndexes(self,N):
        n = ctypes.c_int(N)
        hkl = np.empty(3, dtype=np.int32)
        lib.getIndexes(self.instance,n,hkl)
        return hkl
    
    def getF(self,N):
        n = ctypes.c_int(N)
        F = ctypes.c_double()
        F = lib.getF(self.instance,n)
        return F    
    
    def getDistances(self,N):
        n = ctypes.c_int(N)
        d = ctypes.c_float()
        d = lib.getDistances(self.instance,n)
        return d  
    


   
#!!! VERY IMPORTANT: The files fparams and templato, and even the dll although I am not sure about this,
#!!! must be in the same directory of the unit cell files to allow this to work.
#!!! If error OSError: exception: access violation reading 0x0000000000000180 ; spawns, then try reallocatin
#!!! the files and make the directories coincide (basically the directory of the dll and the one of the .uce file)


''' 

#Load the crystall cell into Crystal object by pointing to the string containing the name of the cell
 
Silicon = Crystal(b'E:/Arxius varis/PhD/2nd_year/Code/unit_cells/GaAs.uce')

Germanium = Crystal(b'E:/Arxius varis/PhD/2nd_year/Code/unit_cells/Ge.uce')



  
min_d=0.5    #minimum interplanar distance computed in the diffraction
forbidden = True  #Include (True) or not (False) the forbbiden reflections

N = Silicon.Diff(forbidden,min_d)
print("Calculated ",N ,"reflections")

indexes = []
distancess = []
print('silicon')
for index in range(N):
    hkls=Silicon.getIndexes(index)    
    distances=Silicon.getDistances(index)
    indexes.append(hkls)
    distancess.append(distances)
    
indexes = np.asarray(indexes)    
distancess = np.asarray(distancess)    

reduced_distances, uniqueindexes = np.unique(distancess , return_index = True)    
reduced_indexes = indexes[uniqueindexes]    
 
reduced_distances = reduced_distances[::-1]    
reduced_indexes = reduced_indexes[::-1]    

print(reduced_distances)    
print(reduced_indexes)    
       
N = Germanium.Diff(forbidden,min_d)
print("Calculated ",N ,"reflections")

print('germanium')
for index in range(N):
    hkls=Silicon.getIndexes(index)   
    print(hkls)

    
d1=0.729  #distance to spot 1
d2=0.585 #distance to spot 2
ang=161.373  #angle between both spots
tol=0.05   #tolerance: how different from theoretical values the previous values can be to get good output




n = Silicon.FindZA(d1,d2,ang,tol)
print("Found ",n,"possible Zone Axes")
ZAs_spot_pair=[]
for za in range(n):
    ZA_n=Silicon.getZA(za)
    print(ZA_n)
    ZAs_spot_pair.append(ZA_n)

ZA=Silicon.getZA(0)

print(ZA)

hkl1 = Silicon.gethkls1(0)
print(hkl1)

hkl2 = Silicon.gethkls2(0)
print(hkl2)
  
#Get distances from reflections
distances=Silicon.getDistances(forbidden,min_d)
print(distances)
 
#Get indexes from reflections
indexes=[]
for index in range(N):
    hkls=Silicon.getIndexes(index,forbidden,min_d)
    indexes.append(hkls)
    
print(indexes)

#Get F factors from reflections  (does not work in any case... not sure what else to try)

Fs=Silicon.getFfactors(forbidden,min_d)
print(Fs)
'''
