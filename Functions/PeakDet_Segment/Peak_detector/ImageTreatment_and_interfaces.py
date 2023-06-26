import numpy as np
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from tkinter import Tk, Scale, Label, IntVar
import customtkinter as ctk
from IPython.display import clear_output

'''Image and matrix treatment '''

#Normalize a image/matrix
def normalize_image(image):
    
    minim = np.amin(image)
    maxim = np.amax(image)
    
    if(minim != maxim):
       normalize_matrix = (image - minim) / (maxim - minim)
    elif(maxim != 0):
       normalize_matrix = image/maxim
    else:
       normalize_matrix = image
   
    return normalize_matrix

#Draw a square in the image 
def square(img, window_size, zone, thickness = 1):
    
    img_square = img + 0
    maxim = np.max(img)
    
    x0, y0 = zone
    
    if x0 > len(img) - window_size - thickness: x0 = len(img) - window_size - thickness - 1 
    if y0 > len(img) - window_size - thickness: y0 = len(img) - window_size - thickness - 1 
    if x0 < 0: x0 = 0
    if y0 < 0: y0 = 0

    x = len(img) - y0 - 1
    y = x0
    
    img_square[x - thickness : x , y : y + window_size] = maxim
    img_square[x - window_size - thickness : x - window_size  , y : y + window_size+ thickness] = maxim
    img_square[x - window_size : x  , y : y + thickness ] = maxim
    img_square[x - window_size : x , y + window_size : y + window_size + thickness ] = maxim
    
    if len(img) == window_size:
        img_square = img
    
    return img_square

#Create and save horizontal and vertical barplot of a image
def histograma(matrix,file_name):
    
    size = matrix[0].size
    
    intensity_x = np.sum(normalize_image(matrix), axis=1)
    intensity_y = np.sum(normalize_image(matrix), axis=0)
    indice_x  = np.arange(size,0,-1)
    indice_y = np.arange(size)
 
    plt.barh(indice_x,intensity_x, color = 'black')
    plt.savefig(file_name+'x')
    plt.close()
  
    plt.barh(indice_y,intensity_y, color = 'black')
    plt.savefig(file_name+'y')
    plt.close()
 
    imagen = Image.open(file_name+'x.png')
    imagen =  np.asarray(imagen)
    
    #histograma sin ejes
    cut = imagen[44:242,54:389]
    cut = Image.fromarray(cut)
    cut.save(file_name+'x.png')
  
    imagen = Image.open(file_name+'y.png')
    imagen = imagen.transpose(Image.ROTATE_270)
    imagen =  np.asarray(imagen)
    
    #histograma sin ejes
    cut = imagen[54:389,44:242]
    cut = Image.fromarray(cut)
    cut.save(file_name+'y.png')

    
#Save a matrix as .png (image)
def save_matrix_as_image(matrix,name):
    
    matrix = normalize_image(matrix)
    matrix = matrix*255
    matrix = np.uint8(matrix)
    image = Image.fromarray(matrix)
    image.save(name+'.png')

#Charge image as a matrix
def image_to_matrix(filepath):
    imagen = Image.open(filepath)
    imagen =  np.asarray(imagen)
    
    return imagen
    
    
# Show matrix/image in 3D
def image3D(image,limit = False):
    
        n_points = 20000
        n = len(image)
        print(n)
        mj = n-np.arange(n) 
        mj = np.asarray([mj]*n)
        mi = mj.T
        
        vector_i = normalize_image(mi.reshape(n**2))
        vector_j = normalize_image(mj.reshape(n**2))
        vector_image = image.reshape(n**2)
        
        if limit:
            print('reducing')
            i = (n**2-1)*np.random.random(n_points)
            i = np.uint32(i)
            vector_j,vector_i,vector_image =vector_j[i],vector_i[i],vector_image[i]

        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.scatter(vector_j,vector_i,vector_image ,c=vector_image, cmap='viridis', linewidth=0.5)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        plt.show()



'''Interfaces and jupyter settings'''

#Add a slider in a easy way
def add_slider(ventana,location,interval,label, variable,function):
    
    x_slider = location[0]
    y_slider = location[1]
    length = location[2]
    
    min_value = interval[0]
    max_value = interval[1]
    bins = interval[2]
    
       
    Scale ( ventana,
            label = label,
            orient = 'horizontal', borderwidth=3,
            length = length ,
            from_ = min_value, to = max_value,
            tickinterval = max_value - min_value,
            resolution = bins,
            variable = variable,
            command = function).place(x = x_slider, y = y_slider)
    
#Convert a matrix/image in a picture     
def matrix_to_picture(ventana, matrix, size):
    matrix = normalize_image(matrix)
    image = (255*matrix).astype(np.uint8)
    image = Image.fromarray(image).resize(size)
    picture = ImageTk.PhotoImage(image, master = ventana)
    
    return picture

#Convert a matrix/image in a label
def matrix_to_label(ventana,matrix, size, coord):

    picture = matrix_to_picture(ventana, matrix, size)
    label = Label(ventana, image = picture)
    label.place(x=coord[0], y=coord[1]) 
    
    return picture,label


#Charge bar to Jupyter notebooks
def barra_de_carga(i, size, text = ''):
 
  clear_output(wait=True)  
  porcentage = int((i+1)/size*1000)
  porcentage_string = str(porcentage/10) +'%'

  barra = '['

  if i+1 == size:
      barra = barra + 100*'#'+']'
      print(text+ ' ' +barra + ' ' + porcentage_string)

  else:
      barra = barra + porcentage//10*'#'+ str(porcentage%10) + (99-porcentage//10)*' '+']'
      print(text+ ' ' +barra + ' ' +  porcentage_string)


# App to visualize dynamic images
def show_app(img, function, **kwargs):
        
        ctk.set_appearance_mode("dark")  # Modes: system (default), light, dark
        ctk.set_default_color_theme("blue")  # Themes: blue (default), dark-blue, green
        
        ventana = Tk()
        ventana.title("Testing")
        
        n = len(kwargs)
        variables = []
        variables_val = [0]*n
        
        matrix = normalize_image(img)
        image = (255*matrix).astype(np.uint8)
        image = Image.fromarray(image).resize((600,600))
        picture = ImageTk.PhotoImage(image, master = ventana)
        label = Label(ventana, image = picture)
        label.pack()
        
        def refresh(x):
            global picture
            i = 0
            
            for var in variables:
                variables_val[i] = var.get()
                i += 1
                
            matrix = function(img,*variables_val)
            matrix = normalize_image(matrix)
            image = (255*matrix).astype(np.uint8)
            image = Image.fromarray(image).resize((600,600))
            picture = ImageTk.PhotoImage(image, master = ventana)
            label.configure(image = picture)
        
        for arg in kwargs:
            min_value = kwargs[arg][0] 
            max_value = kwargs[arg][1]
            variables.append(IntVar(ventana))
            
            Scale( ventana,
                    label = arg,
                    orient = 'horizontal', 
                    borderwidth=3,
                    length = 200 ,
                    from_ = min_value, 
                    to = max_value,
                    tickinterval = (max_value - min_value),
                    variable = variables[-1],
                    command = refresh).pack(side = 'left')
            
        ventana.mainloop()



