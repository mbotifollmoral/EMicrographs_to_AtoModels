# -*- coding: utf-8 -*-
"""
Created on Thu Dec 16 13:49:26 2021

@author: Marc
"""

#TO RUN IN PYTHON >=3.8 (nextnanopy support >=3.8)

'''
CELL intended for linking nextnanopy with the nextnano installation and licenses
REPEAT and EDIT in case the new installation modifies the paths
'''

import numpy as np
import nextnanopy as nn

conf=nn.config

print(conf)

nn.config.set('nextnano++','exe',r'C:\Program Files\nextnano\2021_09_03\nextnano++\bin 64bit\nextnano++_Intel_64bit.exe')
nn.config.set('nextnano++','license',r'C:\Users\Marc\Documents\nextnano\License\License_nnp.lic')
nn.config.set('nextnano++','database',r'C:\Program Files\nextnano\2021_09_03\nextnano++\Syntax\database_nnp.in')
nn.config.set('nextnano++','outputdirectory',r'C:\Users\Marc\Documents\nextnano\Output')

nn.config.set('nextnano3','exe',r'C:\Program Files\nextnano\2021_09_03\nextnano3\Intel 64bit\nextnano3_Intel_64bit.exe')
nn.config.set('nextnano3','license',r'C:\Users\Marc\Documents\nextnano\License\License_nnp.lic')
nn.config.set('nextnano3','database',r'C:\Program Files\nextnano\2021_09_03\nextnano3\Syntax\database_nn3.in')
nn.config.set('nextnano3','outputdirectory',r'C:\Users\Marc\Documents\nextnano\Output')

nn.config.set('nextnano.MSB','database',r'C:\Program Files\nextnano\2021_09_03\nextnano.MSB\Syntax\Materials.xml')
              
nn.config.save() #save permanently

conf=nn.config

print(conf)

#%%

'''
CELL intended for the basic commands of nextnanopy communicating with input files
'''


import numpy as np
import nextnanopy as nn


path=r'C:\Users\Marc\Downloads\3D_Cylinder_nnp.in'

#opens .in native files and .txt if acceptable file
myinput=nn.InputFile(path)

print(myinput)
print(type(myinput))
myinput.config
#myinput.execute()

# to change the output directory and threads
# my_input.execute(outputdirectory=r'C:\Users\jun-liang.wang\Downloads', threads=4)
myinput.preview()

variablesin=myinput.variables

print(variablesin)

#to call the variable, use the string of the name after the $ symbol
spacing=myinput.get_variable('SPACING').text
#get directly the value of the variable
spacing_value=myinput.get_variable('SPACING').value
#the text method makes it a string
print(spacing)
print(spacing_value)

#modify a variable
#myinput.set_variable('SPACING', value=30)
variablesin=myinput.variables
print(variablesin)

#chek line in which the variable is there
linenew=myinput.variables['SPACING'].metadata
print(linenew)


#to save the file, the object must be a input object from nextnanopy
#myinput.save()
#doesn't work if it not a input file object

pathtxt=r'C:\Users\Marc\Downloads\documenttxt.txt'

#only opens in files
myinput2=nn.InputFile(pathtxt)
print(myinput2)
myinput2.preview()

#%%

'''
CELL intended for describing the main process of loading and working with 
GDS files and creating valid inputs from these shapes with nextnanopy
'''
import matplotlib.pyplot as plt
import nextnanopy as nn
from nextnanopy.nnp.shapes import GdsPolygons


my_gds = GdsPolygons(r'E:\Arxius varis\PhD\3rd_year\Code\FEM_modelling_code\gds_examples\example3.gds')


print(f"Number of polygons: {my_gds.nb_polygons}")
my_gds.show()

x,y = my_gds.xy
#x and y hold the coordinates of the vertices
#x and y are arrays of dimension [number of polygons, number of vertices]
print(len(x))
print(x)
print(y)

fig, ax = plt.subplots(1)
my_gds.show(ax=ax)

for xyi in my_gds.xy:
    xi, yi = xyi
    ax.plot(xi,yi,'ro')
    
print(my_gds.labels)    
  

#!!! Generate polygonal prism from coordinates  
list_of_shapes = my_gds.get_polygonal_prisms(zi=0,zf=20) # initial and final z coordinates are needed for this method
for shape in list_of_shapes:
    print(shape.text)    
    
  
print(my_gds.polygons_xy)


print(list_of_shapes)


#%%

'''
CELL intended for generating gds files from a set of coordiantes and then
translate these gds coordiantes into the nextnano language
'''

import numpy as np
import matplotlib.pyplot as plt
import nextnanopy as nn
import gdspy
from nextnanopy.nnp.shapes import GdsPolygons


#the output of the segmentation is an ordered array of x,y coordinates for each
#polygon generated and identified with a given label

#set of easy coordinates, just a square:
square_array=np.array([[0,0],[100,0],[100,100],[0,100]])

#second polygon, triangle (overlapping sides with the square)
triangle_array=np.array([[0,100],[100,100],[50,200]])

#set these coordinates in the format gdspy demands

# The GDSII file is called a library, which contains multiple cells.
lib = gdspy.GdsLibrary()

# Geometry must be placed in cells.
cell = lib.new_cell('FIRST')

#labelling not mandatory but can be useful, datatype is the label
#the labels are specially useful for the gds structure but not that much
#for the nextnano geometry generation, as the labels are inferred automatically
ld_fulletch = {"layer": 0, "datatype": 1}
ld_partetch = {"layer": 0, "datatype": 2}
# Create a polygon from a list of vertices
poly_square = gdspy.Polygon(square_array, **ld_fulletch)
cell.add(poly_square)
poly_triangle = gdspy.Polygon(triangle_array, **ld_partetch)

cell.add(poly_triangle)
# Save the library in a file called 'first.gds'.
path_savegds=r'E:\Arxius varis\PhD\3rd_year\Code\FEM_modelling_code\gds_examples'
lib.write_gds(path_savegds+'\\first.gds')

# Optionally, save an image of the cell as SVG.
cell.write_svg(path_savegds+'\\first.svg')

# Display all cells using the internal viewer.
gdspy.LayoutViewer()


#create nextnano object from the generated gds

#No need to define labels to each shape as they are automatically assigned 

my_gds = GdsPolygons(r'E:\Arxius varis\PhD\3rd_year\Code\FEM_modelling_code\gds_examples\first.gds')


print(f"Number of polygons: {my_gds.nb_polygons}")
my_gds.show()

x,y = my_gds.xy
#x and y hold the coordinates of the vertices
#x and y are arrays of dimension [number of polygons, number of vertices]
print(len(x))
print(x)
print(y)

fig, ax = plt.subplots(1)
my_gds.show(ax=ax)

for xyi in my_gds.xy:
    xi, yi = xyi
    ax.plot(xi,yi,'ro')
    
print(my_gds.labels)    
  

#!!! Generate polygonal prism from coordinates  
list_of_shapes = my_gds.get_polygonal_prisms(zi=0,zf=20) # initial and final z coordinates are needed for this method
for shape in list_of_shapes:
    print(shape.text)    
    
  
print(my_gds.polygons_xy)


print(list_of_shapes)




