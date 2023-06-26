# -*- coding: utf-8 -*-
"""
Created on Thu May  4 12:38:23 2023

@author: Marc
"""

import numpy as np
import os
import pickle

class Scored_spot_pair:
    
    def __init__(self, spot1_int_ref, spot2_int_ref, phase_mat_name):
        self.spot1_int_ref=spot1_int_ref
        self.spot2_int_ref=spot2_int_ref
        # self.phase=crystal_object
        
        # phase_name_temp=crystal_object.phase_name
        # strin=str(phase_name_temp)
        # start=strin[::-1].find(os.path.sep)
        # end=strin[::-1].find('.')
        # phase_mat_name=strin[-start:-end-1]
        self.phase_name=phase_mat_name
        
        
    def Spot_distances_angles(self, spot1_angle_to_x, spot2_angle_to_x):
        # self.spot1_dist=refined_distances[self.spot1_int_ref]
        # self.spot2_dist=refined_distances[self.spot2_int_ref]
        self.spot1_angle_to_x=spot1_angle_to_x
        self.spot2_angle_to_x=spot2_angle_to_x
        angle_between=np.abs(self.spot2_angle_to_x - self.spot1_angle_to_x)
        if angle_between>180:
            angle_between=360-angle_between
        self.angle_between=angle_between
        
    def Zone_axis(self, ZA, hkl1_reference, hkl2_reference):
        self.ZA = ZA
        self.hkl1_reference = hkl1_reference
        self.hkl2_reference = hkl2_reference

        
    def Score(self, score):
        self.score=score
    
    
    
# class containing the crystals identified and its correspondent spots
class Crystal_spots:
    def __init__(self, spot_list):
        # the spots are referenced by the index internal reference, not by position or distance
        self.spots=spot_list
        
    def Spot_pairs(self, list_spot_pairs_obj):
        self.spot_pairs_obj=list_spot_pairs_obj
    
    def Phase(self, phase_string):
        self.phase_name=phase_string
    
    def ZA(self, ZA, ZA_priv_index):
        self.ZA=ZA
        self.ZA_string=str(ZA)
        # index used in case more than one crystal in the same ZA appears, 
        # given same phase, would be differentiated by this index
        self.ZA_priv_index=ZA_priv_index
        
    def PF_method(self, method_used):
        self.method_used = method_used    
        
        
        
# store the two cryst spots        
crystal_spots = []        
        
        

# Image info

# Image size
# 713
# refined_pixels
# np.array([[219 361]
#  [210 361]
#  [292 551]
#  [287 550]
#  [564 356]
#  [148 356]
#  [499 551]
#  [627 360]
#  [632 360]
#  [277 679]
#  [ 84 551]])
# spots_int_reference
# np.array([ 0  1  2  3  4  5  6  7  8  9 10])








scored_spot_pairs_cryst1 = []



#  Scored Spot Pair info
# ZA of scored spot pair: spot.ZA
# [0, 1, 1]
# spot.score
# 0.20366345779826867
# 1internalref
# 3
# 2internalref
# 4
# spot.hkl1_reference
# [ 1, -1,  1]
# spot.spot1_angle_to_x
# 19.57895212482911
# spot.hkl2_reference
# [ 1, -1,  1]
# spot.spot2_angle_to_x
# -90.0
# spot.angle_between
# 109.57895212482912


scored_spot_pair = Scored_spot_pair(3, 4, 'InSb')
scored_spot_pair.Spot_distances_angles(19.57895212482911, -90.0)
scored_spot_pair.Zone_axis([0, 1, 1], [ 1, -1,  1], [ 1, -1,  1])
scored_spot_pairs_cryst1.append(scored_spot_pair)



#  Scored Spot Pair info
# ZA of scored spot pair: spot.ZA
# [0 1 1]
# spot.score
# 0.20366345779826867
# 1internalref
# 3
# 2internalref
# 5
# spot.hkl1_reference
# [ 1 -1  1]
# spot.spot1_angle_to_x
# 19.57895212482911
# spot.hkl2_reference
# [ 1 -1  1]
# spot.spot2_angle_to_x
# 90.0
# spot.angle_between
# 70.42104787517088


scored_spot_pair = Scored_spot_pair(3, 5, 'InSb')
scored_spot_pair.Spot_distances_angles(19.57895212482911 , 90.0)
scored_spot_pair.Zone_axis([0, 1, 1], [ 1, -1,  1], [ 1, -1,  1])
scored_spot_pairs_cryst1.append(scored_spot_pair)




#  Scored Spot Pair info
# ZA of scored spot pair: spot.ZA
# [0 1 1]
# spot.score
# 0.35412549145076216
# 1internalref
# 2
# 2internalref
# 6
# spot.hkl1_reference
# [ 1 -1  1]
# spot.spot1_angle_to_x
# 18.170101189194654
# spot.hkl2_reference
# [0 0 2]
# spot.spot2_angle_to_x
# -36.25383773744479
# spot.angle_between
# 54.423938926639444


scored_spot_pair = Scored_spot_pair(2, 6, 'InSb')
scored_spot_pair.Spot_distances_angles(18.170101189194654, -36.25383773744479)
scored_spot_pair.Zone_axis([0, 1, 1], [ 1, -1,  1], [0, 0, 2])
scored_spot_pairs_cryst1.append(scored_spot_pair)



#  Scored Spot Pair info
# ZA of scored spot pair: spot.ZA
# [0 1 1]
# spot.score
# 0.4013494151835182
# 1internalref
# 5
# 2internalref
# 10
# spot.hkl1_reference
# [ 1 -1  1]
# spot.spot1_angle_to_x
# 90.0
# spot.hkl2_reference
# [0 2 2]
# spot.spot2_angle_to_x
# 54.362815917805975
# spot.angle_between
# 35.637184082194025

scored_spot_pair = Scored_spot_pair(5, 10, 'InSb')
scored_spot_pair.Spot_distances_angles(90.0, 54.362815917805975)
scored_spot_pair.Zone_axis([0, 1, 1], [ 1, -1,  1], [0, 2, 2])
scored_spot_pairs_cryst1.append(scored_spot_pair)




#  Scored Spot Pair info
# ZA of scored spot pair: spot.ZA
# [0 1 1]
# spot.score
# 0.4013494151835314
# 1internalref
# 4
# 2internalref
# 10
# spot.hkl1_reference
# [ 1 -1  1]
# spot.spot1_angle_to_x
# -90.0
# spot.hkl2_reference
# [0 2 2]
# spot.spot2_angle_to_x
# 54.362815917805975
# spot.angle_between
# 144.36281591780596

scored_spot_pair = Scored_spot_pair(4, 10, 'InSb')
scored_spot_pair.Spot_distances_angles(-90.0, 54.362815917805975)
scored_spot_pair.Zone_axis([0, 1, 1], [ 1, -1,  1], [0, 2, 2])
scored_spot_pairs_cryst1.append(scored_spot_pair)



#  Scored Spot Pair info
# ZA of scored spot pair: spot.ZA
# [0 1 1]
# spot.score
# 0.4941222596955421
# 1internalref
# 3
# 2internalref
# 10
# spot.hkl1_reference
# [ 1 -1  1]
# spot.spot1_angle_to_x
# 19.57895212482911
# spot.hkl2_reference
# [0 2 2]
# spot.spot2_angle_to_x
# 54.362815917805975
# spot.angle_between
# 34.78386379297686


scored_spot_pair = Scored_spot_pair(3, 10, 'InSb')
scored_spot_pair.Spot_distances_angles(19.57895212482911, 54.362815917805975)
scored_spot_pair.Zone_axis([0, 1, 1], [ 1, -1,  1], [0, 2, 2])
scored_spot_pairs_cryst1.append(scored_spot_pair)





#  Scored Spot Pair info
# ZA of scored spot pair: spot.ZA
# [0 1 1]
# spot.score
# 0.9342070879962474
# 1internalref
# 2
# 2internalref
# 10
# spot.hkl1_reference
# [ 1 -1  1]
# spot.spot1_angle_to_x
# 18.170101189194654
# spot.hkl2_reference
# [0 2 2]
# spot.spot2_angle_to_x
# 54.362815917805975
# spot.angle_between
# 36.19271472861132


scored_spot_pair = Scored_spot_pair(2, 10, 'InSb')
scored_spot_pair.Spot_distances_angles(18.170101189194654, 54.362815917805975)
scored_spot_pair.Zone_axis([0, 1, 1], [ 1, -1,  1], [0, 2, 2])
scored_spot_pairs_cryst1.append(scored_spot_pair)





#  Scored Spot Pair info
# ZA of scored spot pair: spot.ZA
# [0 1 1]
# spot.score
# 1.009170474880012
# 1internalref
# 4
# 2internalref
# 6
# spot.hkl1_reference
# [ 1 -1  1]
# spot.spot1_angle_to_x
# -90.0
# spot.hkl2_reference
# [0 0 2]
# spot.spot2_angle_to_x
# -36.25383773744479
# spot.angle_between
# 53.74616226255521


scored_spot_pair = Scored_spot_pair(4, 6, 'InSb')
scored_spot_pair.Spot_distances_angles(-90.0, -36.25383773744479)
scored_spot_pair.Zone_axis([0, 1, 1], [ 1, -1,  1], [0, 0, 2])
scored_spot_pairs_cryst1.append(scored_spot_pair)



#  Scored Spot Pair info
# ZA of scored spot pair: spot.ZA
# [0 1 1]
# spot.score
# 1.0091704748800192
# 1internalref
# 5
# 2internalref
# 6
# spot.hkl1_reference
# [ 1 -1  1]
# spot.spot1_angle_to_x
# 90.0
# spot.hkl2_reference
# [0 0 2]
# spot.spot2_angle_to_x
# -36.25383773744479
# spot.angle_between
# 126.2538377374448


scored_spot_pair = Scored_spot_pair(5, 6, 'InSb')
scored_spot_pair.Spot_distances_angles(90.0, -36.25383773744479)
scored_spot_pair.Zone_axis([0, 1, 1], [ 1, -1,  1], [0, 0, 2])
scored_spot_pairs_cryst1.append(scored_spot_pair)


#  Scored Spot Pair info
# ZA of scored spot pair: spot.ZA
# [0 1 1]
# spot.score
# 1.1110197776296924
# 1internalref
# 3
# 2internalref
# 6
# spot.hkl1_reference
# [ 1 -1  1]
# spot.spot1_angle_to_x
# 19.57895212482911
# spot.hkl2_reference
# [0 0 2]
# spot.spot2_angle_to_x
# -36.25383773744479
# spot.angle_between
# 55.8327898622739


scored_spot_pair = Scored_spot_pair(3, 6, 'InSb')
scored_spot_pair.Spot_distances_angles(19.57895212482911, -36.25383773744479)
scored_spot_pair.Zone_axis([0, 1, 1], [ 1, -1,  1], [0, 0, 2])
scored_spot_pairs_cryst1.append(scored_spot_pair)





#  Scored Spot Pair info
# ZA of scored spot pair: spot.ZA
# [0 1 1]
# spot.score
# 1.3116745683138356
# 1internalref
# 2
# 2internalref
# 4
# spot.hkl1_reference
# [ 1 -1  1]
# spot.spot1_angle_to_x
# 18.170101189194654
# spot.hkl2_reference
# [ 1 -1  1]
# spot.spot2_angle_to_x
# -90.0
# spot.angle_between
# 108.17010118919465


scored_spot_pair = Scored_spot_pair(2, 4, 'InSb')
scored_spot_pair.Spot_distances_angles(18.170101189194654, -90.0)
scored_spot_pair.Zone_axis([0, 1, 1], [ 1, -1,  1], [1, -1, 1])
scored_spot_pairs_cryst1.append(scored_spot_pair)




#  Scored Spot Pair info
# ZA of scored spot pair: spot.ZA
# [0 1 1]
# spot.score
# 1.3116745683138356
# 1internalref
# 2
# 2internalref
# 5
# spot.hkl1_reference
# [ 1 -1  1]
# spot.spot1_angle_to_x
# 18.170101189194654
# spot.hkl2_reference
# [ 1 -1  1]
# spot.spot2_angle_to_x
# 90.0
# spot.angle_between
# 71.82989881080535


scored_spot_pair = Scored_spot_pair(2, 5, 'InSb')
scored_spot_pair.Spot_distances_angles(18.170101189194654, 90.0)
scored_spot_pair.Zone_axis([0, 1, 1], [ 1, -1,  1], [1, -1, 1])
scored_spot_pairs_cryst1.append(scored_spot_pair)


#  Scored Spot Pair info
# ZA of scored spot pair: spot.ZA
# [0 1 1]
# spot.score
# 1.6810925880264087
# 1internalref
# 9
# 2internalref
# 10
# spot.hkl1_reference
# [ 2 -1 -2]
# spot.spot1_angle_to_x
# 13.743707383679068
# spot.hkl2_reference
# [ 2 -1 -2]
# spot.spot2_angle_to_x
# 54.362815917805975
# spot.angle_between
# 40.61910853412691



scored_spot_pair = Scored_spot_pair(9, 10, 'InSb')
scored_spot_pair.Spot_distances_angles(13.743707383679068, 54.362815917805975)
scored_spot_pair.Zone_axis([0, 1, 1], [ 2, -1,  -2], [ 2, -1,  -2])
scored_spot_pairs_cryst1.append(scored_spot_pair)



# Generate crysta with these spot pairs

#  Cryst 

# spot list [2, 3, 4, 5, 6, 10]
# phase name InSb
# ZA of cryst [0 1 1]


crystal_spot1 = Crystal_spots([2, 3, 4, 5, 6, 10])
crystal_spot1.Spot_pairs(scored_spot_pairs_cryst1)
crystal_spot1.Phase('InSb')
crystal_spot1.ZA([0,1,1],0) 
   
crystal_spots.append(crystal_spot1)






# new crystla generate other set of scored spot pairs

scored_spot_pairs_cryst2 = []





#  Scored Spot Pair info
# ZA of scored spot pair: spot.ZA
# [0 1 1]
# spot.score
# 0.6181820187678183
# 1internalref
# 6
# 2internalref
# 10
# spot.hkl1_reference
# [1 1 1]
# spot.spot1_angle_to_x
# -36.25383773744479
# spot.hkl2_reference
# [ 2 -1 -1]
# spot.spot2_angle_to_x
# 54.362815917805975
# spot.angle_between
# 90.61665365525076



scored_spot_pair = Scored_spot_pair(6, 10, 'InSb')
scored_spot_pair.Spot_distances_angles(-36.25383773744479, 54.362815917805975)
scored_spot_pair.Zone_axis([0, 1, 1], [1, 1 ,1], [ 2 -1 -1])
scored_spot_pairs_cryst2.append(scored_spot_pair)





#  Scored Spot Pair info
# ZA of scored spot pair: spot.ZA
# [0 1 1]
# spot.score
# 1.1018522477225206
# 1internalref
# 0
# 2internalref
# 6
# spot.hkl1_reference
# [0 0 1]
# spot.spot1_angle_to_x
# 87.90984084628933
# spot.hkl2_reference
# [1 1 1]
# spot.spot2_angle_to_x
# -36.25383773744479
# spot.angle_between
# 124.16367858373411



scored_spot_pair = Scored_spot_pair(0, 6, 'InSb')
scored_spot_pair.Spot_distances_angles(87.90984084628933, -36.25383773744479)
scored_spot_pair.Zone_axis([0, 1, 1], [0 ,0 ,1], [1, 1 ,1])
scored_spot_pairs_cryst2.append(scored_spot_pair)





#  Scored Spot Pair info
# ZA of scored spot pair: spot.ZA
# [0 1 1]
# spot.score
# 1.2032919158265367
# 1internalref
# 8
# 2internalref
# 10
# spot.hkl1_reference
# [0 0 2]
# spot.spot1_angle_to_x
# -89.169684513742
# spot.hkl2_reference
# [ 2 -1 -1]
# spot.spot2_angle_to_x
# 54.362815917805975
# spot.angle_between
# 143.532500431548



scored_spot_pair = Scored_spot_pair(8, 10, 'InSb')
scored_spot_pair.Spot_distances_angles(-89.169684513742, 54.362815917805975)
scored_spot_pair.Zone_axis([0, 1, 1], [0 ,0 ,2], [ 2, -1 ,-1])
scored_spot_pairs_cryst2.append(scored_spot_pair)



#  Scored Spot Pair info
# ZA of scored spot pair: spot.ZA
# [0 1 1]
# spot.score
# 1.219552531676339
# 1internalref
# 7
# 2internalref
# 10
# spot.hkl1_reference
# [0 0 2]
# spot.spot1_angle_to_x
# -89.15436724428038
# spot.hkl2_reference
# [ 2 -1 -1]
# spot.spot2_angle_to_x
# 54.362815917805975
# spot.angle_between
# 143.51718316208635


scored_spot_pair = Scored_spot_pair(7, 10, 'InSb')
scored_spot_pair.Spot_distances_angles(-89.15436724428038, 54.362815917805975)
scored_spot_pair.Zone_axis([0, 1, 1], [0 ,0 ,2], [ 2, -1 ,-1])
scored_spot_pairs_cryst2.append(scored_spot_pair)





#  Scored Spot Pair info
# ZA of scored spot pair: spot.ZA
# [0 1 1]
# spot.score
# 1.8201649492607872
# 1internalref
# 6
# 2internalref
# 8
# spot.hkl1_reference
# [1 1 1]
# spot.spot1_angle_to_x
# -36.25383773744479
# spot.hkl2_reference
# [0 0 2]
# spot.spot2_angle_to_x
# -89.169684513742
# spot.angle_between
# 52.91584677629721


scored_spot_pair = Scored_spot_pair(6, 8, 'InSb')
scored_spot_pair.Spot_distances_angles(-36.25383773744479, -89.169684513742)
scored_spot_pair.Zone_axis([0, 1, 1], [1, 1, 1], [ 0,0,2])
scored_spot_pairs_cryst2.append(scored_spot_pair)




#  Scored Spot Pair info
# ZA of scored spot pair: spot.ZA
# [0 1 1]
# spot.score
# 1.8361068285222668
# 1internalref
# 6
# 2internalref
# 7
# spot.hkl1_reference
# [1 1 1]
# spot.spot1_angle_to_x
# -36.25383773744479
# spot.hkl2_reference
# [0 0 2]
# spot.spot2_angle_to_x
# -89.15436724428038
# spot.angle_between
# 52.90052950683559


scored_spot_pair = Scored_spot_pair(6, 7, 'InSb')
scored_spot_pair.Spot_distances_angles(-36.25383773744479, -89.15436724428038)
scored_spot_pair.Zone_axis([0, 1, 1], [1, 1, 1], [ 0,0,2])
scored_spot_pairs_cryst2.append(scored_spot_pair)



#  Cryst 

# spot list [ 0  6  7  8 10]
# phase name Si
# ZA of cryst [0 1 1]

crystal_spot2 = Crystal_spots([0,  6,  7,  8, 10])
crystal_spot2.Spot_pairs(scored_spot_pairs_cryst2)
crystal_spot2.Phase('InSb')
crystal_spot2.ZA([0,1,1],1) 
   
crystal_spots.append(crystal_spot2)

