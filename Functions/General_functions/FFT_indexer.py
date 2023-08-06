import tkinter as tk
from tkinter import ttk
from tkinter import IntVar
from tkinter.scrolledtext import ScrolledText

# from hyperspy.api import load
from PIL import Image, ImageTk
import numpy as np
import os
import pickle
import sys
import matplotlib.pyplot as plt

# !!! NEED to set the path to 
# Alg_Comb_Single_Image_Strain.py
# as the console working directory
Project_main_path = os.getcwd()
if 'EMicrographs_to_AtoModels' in Project_main_path:
    Project_main_path = Project_main_path[:Project_main_path.find('EMicrographs_to_AtoModels')-1]
# Project_main_path has the EMicrographs_to_AtoModels folder
sys.path.append(Project_main_path)

from EMicrographs_to_AtoModels.Functions.General_functions import ImageCalibTransf as ImCalTrans
from EMicrographs_to_AtoModels.Functions.Peak_detector_Indep import PF_Peaks_detector as Peaks_detector



fft_info_data = dict()

# fft_list = [np.zeros((713,713)), np.zeros((713,713))]

# peaks1 = [[219, 361],
#   [210, 361],
#   [292, 551],
#   [287, 550],
#   [564, 356],
#   [148, 356],
#   [499, 551],
#   [627, 360],
#   [632, 360],
#   [277, 679],
#   [ 84, 551]]

# peaks2 = [[300,458]]

class Collect_data():
    def __init__(self,fft, spots_coordinates, spot_int_refs, info_obj):
        # info_obj = 'the object containing all the info you want to show'
        self.fft = fft
        self.coord = spots_coordinates
        self.spot_int_refs = spot_int_refs
        self.info_obj = info_obj
        self.info = self.get_spots_info(info_obj)
        
    def get_spots_info(self, info_obj):
            
            '''
            Extract all the info for each spot in the FFT. Here we create and exemple were the info is only the coordinates
            info_obj needs to be crop_list_refined_cryst_spots
            so all the crystal spots found for a given crop
            so inside there are lists of scored_spot_pairs
            '''
            
            
            info = ['' for i in range(len(self.spot_int_refs))]
            
            for spot_int_ref in self.spot_int_refs:
                
                info[int(spot_int_ref)] = info[int(spot_int_ref)] + 'SPOT NUMBER ' + str(int(spot_int_ref)) + '\n'
                info[int(spot_int_ref)] = info[int(spot_int_ref)] + 'Coordinate: ' + str(self.coord[spot_int_ref])+ '\n'
                for cryst_spot in info_obj:
                    # for every phase
                    
                    cryst_spot_spots_int_ref = cryst_spot.spots
                    cryst_spot_phase_name = cryst_spot.phase_name
                    cryst_spot_ZA = cryst_spot.ZA
                    cryst_spot_ZA_priv_index = cryst_spot.ZA_priv_index
                    
                    # meaning that spot was identified in that cryst spot object
                    if spot_int_ref in cryst_spot_spots_int_ref:
                        
                        info[int(spot_int_ref)] = info[int(spot_int_ref)] + '--Crystal identified with:\n'
                        info[int(spot_int_ref)] = info[int(spot_int_ref)] + 'Crystal phase: ' + str(cryst_spot_phase_name) + '\n'
                        info[int(spot_int_ref)] = info[int(spot_int_ref)] + 'Zone axis: ' + str(cryst_spot_ZA) + '\n'
                        info[int(spot_int_ref)] = info[int(spot_int_ref)] + 'Internal phase + ZA reference: ' + str(cryst_spot_ZA_priv_index) + '\n'
                        
                        for spot_pair in cryst_spot.spot_pairs_obj:
                            
                            # internal refereces of the spots defininng that spot pair
                            spot_pair_ints_i = [spot_pair.spot1_int_ref, spot_pair.spot2_int_ref]
        
                            # if that spot is present in the spot pair checked, append this information
                            if spot_int_ref in spot_pair_ints_i:
                                
                                index_found = spot_pair_ints_i.index(spot_int_ref)
                                
                                # if index_found = 0, the spot_int_ref is spot1
                                # if index_found = 1, the spot_int_ref is spot2
                                
                                if index_found == 0:
                                    other_index = 1
                                else:
                                    other_index = 0
                                    # and of course other_index = 1
                                
                                hkl_refs = [spot_pair.hkl1_reference, spot_pair.hkl2_reference]
                                dists_refs = [spot_pair.spot1_dist , spot_pair.spot2_dist]
                                angle_between_spots = spot_pair.angle_between
                                
                                info[int(spot_int_ref)] = info[int(spot_int_ref)] + 'Forming spot pair with spot number:' + str(spot_pair_ints_i[other_index]) +'\n'
                                info[int(spot_int_ref)] = info[int(spot_int_ref)] + 'where spot '+str(int(spot_int_ref))+ ' is identified as: ' + str(hkl_refs[index_found]) +'\n'
                                info[int(spot_int_ref)] = info[int(spot_int_ref)] + 'and spot '+ str(spot_pair_ints_i[other_index]) + ' is identified as: ' + str(hkl_refs[other_index]) +'\n'
                                
                                info[int(spot_int_ref)] = info[int(spot_int_ref)] + 'Spot '+str(int(spot_int_ref))+ ' has a distance of: ' + str(dists_refs[index_found]) +' Angs\n'
                                info[int(spot_int_ref)] = info[int(spot_int_ref)] + 'and spot '+str(spot_pair_ints_i[other_index])+ ' has a distance of: ' + str(dists_refs[other_index]) +' Angs\n'
                                info[int(spot_int_ref)] = info[int(spot_int_ref)] + 'forming an angle of :' + str(angle_between_spots) +' degrees\n'
                                # info[int(spot_int_ref)] = info[int(spot_int_ref)] + '(debug) ZA found :' + str(spot_pair.ZA) +'\n'
                                # info[int(spot_int_ref)] = info[int(spot_int_ref)] + '(debug) phase found :' + str(spot_pair.phase_name) +'\n'
                    
            
            
            return info    
        
        
    # def get_spots_info_ORIGINAL(self, info_obj):
        
    #     '''
    #     Extract all the info for each spot in the FFT. Here we create and exemple were the info is only the coordinates
    #     '''
    #     info = []
        
    #     for coord in self.coord:
    #         info.append(str(coord))
        
    #     return info
    
# fft_info_data = {'FFT 1' : Collect_data(fft_list[0], peaks1, info_obj = ''),
#         'FFT 2' : Collect_data(fft_list[1], peaks2, info_obj = '')
# }  
    



i_line = 1
fft_size = 800

def matrix_to_picture(ventana, matrix, size):
    matrix = normalize_image(matrix)
    image = (255*matrix).astype(np.uint8)
    image = Image.fromarray(image).resize(size)
    picture = ImageTk.PhotoImage(image, master = ventana)
    
    return picture

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

def get_info(st, text):
    if bool_text.get(): 
        overwrite(st, text)
    else:
        write(st, text) 

def write(st, text):
    global i_line
    st.insert(f'{i_line}.0', text)
    i_line += 1
    
def overwrite(st, text):
    global i_line
    i_line = 1
    st.delete("1.0","end")
    write(st, text)
    
def show_peaks():
    if bool_peaks.get():
        canvas.itemconfigure('peak', state='normal')
    else:
        canvas.itemconfigure('peak', state='hidden')

class ToolTip(object):

    def __init__(self, widget, coord):
        self.widget = widget
        self.tipwindow = None
        self.id = None
        self.x = coord[0]
        self.y = coord[1]

    def showtip(self, text):
        "Display text in tooltip window"
        self.text = text
        if self.tipwindow or not self.text:
            return
        x = self.widget.winfo_rootx() + self.x + 10
        y = self.y + self.widget.winfo_rooty() - 5
        self.tipwindow = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(1)
        tw.wm_geometry("+%d+%d" % (x, y))
        label = tk.Label(tw, text=self.text, justify='left',
                      background="#ffffe0", relief='solid', borderwidth=1,
                      font=("Arial", "8", "normal"))
        label.pack(ipadx=1)

    def hidetip(self):
        tw = self.tipwindow
        self.tipwindow = None
        if tw:
            tw.destroy()

def CreateToolTip(widget, text,coord, item):
    toolTip = ToolTip(widget,coord)
    def enter(event):
        toolTip.showtip(text)
    def leave(event):
        toolTip.hidetip()
    widget.tag_bind(item,'<Enter>', enter)
    widget.tag_bind(item,'<Leave>', leave)
    
class Interactive_spot():
    
    def __init__(self, coord, info = None):
        self.x  = coord[0]
        self.y  = coord[1]
        self.info = info

    def draw_peak(self, canvas, st, size = 3):
        peak = canvas.create_oval(self.x-size, self.y-size,self.x+size, self.y+size,
                                  fill='red', outline = 'red', tags = 'peak')
        canvas.tag_bind(peak,'<1>', lambda _: get_info(st,self.info + ' \n')) 
        CreateToolTip(canvas,self.info,(self.x,self.y),peak)
        
def coordinates_change(coord,old_size, new_size):
    if type(old_size) is int:
        old_size = (old_size, old_size)
    if type(new_size) is int:
        new_size = (new_size, new_size)
        
    x = coord[1] * new_size[0] / old_size[1]
    y = coord[0] * new_size[1] / old_size[0]
    return x, y
    

def draw_fft(e):
    global fft_picture
    fft_name = fft_combobox.get()
    canvas.delete('all')
    
    fft_picture = matrix_to_picture(root, fft_info_data[fft_name].fft, (fft_size, fft_size))
    fft_canvas = canvas.create_image(0,0,image=fft_picture, tags = 'fft', anchor = 'nw')
    
    for i in range(len(fft_info_data[fft_name].coord)):
        img_coord = coordinates_change(fft_info_data[fft_name].coord[i] ,len(fft_info_data[fft_name].fft) , fft_size)
        point = Interactive_spot(img_coord, fft_info_data[fft_name].info[i])
        point.draw_peak(canvas, st, size = 5)
    
    
'''    
Bragg filtering functions 
'''  
 

def Indexed_FFT_BraggFiltering(
        analysed_image, image_in_dataset_whole, bragg_mask_size):
    '''
    Function Bragg filtering, for all the crops extracted from it, all
    the phases found for each crop and appropiately presenting the iFFTs
    in the general_bragg_filterings (explained below)

    Parameters
    ----------
    analysed_image : 
    image_in_dataset_whole : 
    bragg_mask_size : pixels of the mask forming the circle around the 
                    central pixel

    Returns
    -------
    general_bragg_filterings : is a dictionary of dictionaries,
                    each entry, indicated with 
                    general_bragg_filterings[str(crop_index_i) + '_BG_Filt']
                    has either a 0 if no phases are found for that crop
                    with crop_index_i, or a dictionary inside with
                    many entries as phases are found for that
                    crop, so each entry is named after
                    dict_entry_string = cryst_spot.phase_name + '_' + str(cryst_spot.ZA_priv_index)
    general_bragg_filterings[str(crop_index_i) + '_BG_Filt'][dict_entry_string]  
    gives the bragg filtered image for the peaks attributted to phase  dict_entry_string
    found in crop crop_index_i

    '''
    
    
    
    total_pixels_whole = image_in_dataset_whole.total_pixels
    FFT_whole, FFT_whole_complex =  ImCalTrans.Compute_FFT_ImageArray(
        np.asarray(image_in_dataset_whole.hyperspy_2Dsignal))

    general_bragg_filterings = dict()
    
    crop_outputs_dict = analysed_image.Crop_outputs
    
    
    # dictionary keywords
    
    for crop_index_i in range(1, analysed_image.crop_index):    
        
        image_crop_hs_signal = crop_outputs_dict[str(crop_index_i) + '_hs_signal']
        FFT_crop_array, _ =  ImCalTrans.Compute_FFT_ImageArray(np.asarray(image_crop_hs_signal))
        crop_list_refined_cryst_spots = crop_outputs_dict[str(crop_index_i) + '_list_refined_cryst_spots']
        refined_pixels = crop_outputs_dict[str(crop_index_i) + '_refined_pixels']
        spots_int_reference = crop_outputs_dict[str(crop_index_i) + '_spots_int_reference']
        
        
        # If no phases are found for that crop just add dummy input
        # for keeping the order of the 
        if len(crop_list_refined_cryst_spots) == 0:
            general_bragg_filterings[str(crop_index_i) + '_BG_Filt'] = 0
            continue
        
        
        # Check if number pixels of FFT_image_array == analysed_image.totalpxiels
        # if yes keep on without scaling:
        # if they are different then scale them to the size of analysed_image.totalpxiels
        if total_pixels_whole != np.shape(FFT_crop_array)[0]:
            # scale the refined_pixels to the FFT of the full image
            
            # generate a 9x9 pixel square arround the best coordinate scaled to the whole image
            # so from all this pixels we can find the best one that represents the maximum of
            # the peak

            refined_pixs_whole = np.copy(refined_pixels)
            # problem here
            refined_pixs_whole[:,0] = np.round(((refined_pixels[:,0]/FFT_crop_array.shape[0])*total_pixels_whole))
            refined_pixs_whole[:,1] = np.round(((refined_pixels[:,1]/FFT_crop_array.shape[1])*total_pixels_whole))

            for indx, ref_pix_whole in enumerate(
                    refined_pixs_whole):
                
                ref_pix_y = ref_pix_whole[0]
                ref_pix_x = ref_pix_whole[1]
                
                # adjust the position of the pixel to the maximum of the spot in
                # the FFT of the whole image
                if ref_pix_y-1 >= 0 and ref_pix_y+2 <= total_pixels_whole and ref_pix_x-1 >= 0 and ref_pix_x+2 <= total_pixels_whole:
                    
                    FFT_crop_analys = FFT_whole[ref_pix_y-1:ref_pix_y+2,ref_pix_x-1:ref_pix_x+2]
                    cords_max = np.where(FFT_crop_analys == np.max(FFT_crop_analys))
                
                    ref_pix_y = ref_pix_y - 1 + cords_max[0][0]
                    ref_pix_x = ref_pix_x - 1 + cords_max[1][0]
                    
                    refined_pixs_whole[indx, 0] = ref_pix_y  
                    refined_pixs_whole[indx, 1] = ref_pix_x  
            
            # substitute the array as we work with the variable name refined_pixels
            # keep track also if no scaling was done
            refined_pixels = refined_pixs_whole
        
        
        # Create another dict inside the original dict
        # where each entry will be phase found
        bragg_filtered_phases = dict()
        general_bragg_filterings[str(crop_index_i) + '_BG_Filt'] = bragg_filtered_phases


        for cryst_spot in crop_list_refined_cryst_spots:
            
            
            dict_entry_string = cryst_spot.phase_name + '_' + str(cryst_spot.ZA_priv_index)
            
            # the int refs of the spots (cryst_spot.spots) 
            # are also the index so get the 
            # coordinates of the pixels within the whole image
            # which will be the center of the masks
            phase_ref_pix_cords = refined_pixels[cryst_spot.spots]
            
            
            # Create the mask with the centers of the spots based on phase_ref_pix_cords
            binary_image_centers_mask = np.zeros(np.shape(FFT_whole))
            binary_image_centers_mask[phase_ref_pix_cords[:,0], phase_ref_pix_cords[:,1]] = 1
            
            mask = Peaks_detector.cercles(
                bragg_mask_size, binary_image_centers_mask)
            
            # Apply the mask to the complex FFT which is the one needed for
            # the Compute_iFFT_ImageArray, which requires a complex array
            # specially tailord for the complex output from Compute_FFT_ImageArray
            masked_FFT = mask*FFT_whole_complex
            

            
            # Turn to real space by inverting the masked FFT
            bragg_filtered_image = ImCalTrans.Compute_iFFT_ImageArray(masked_FFT)
            
            bragg_filtered_phases[dict_entry_string] = bragg_filtered_image


            
        general_bragg_filterings[str(crop_index_i) + '_BG_Filt'] = bragg_filtered_phases
            
            
    return general_bragg_filterings



def Colour_Mix_BraggFilteredPhases(
        image_in_dataset_whole, general_bragg_filterings, 
        phases_per_crop = 1):
    '''
    Plots the Bragg filtered images, for each crop it plots the full image
    with at most phases_per_crop number of Bragg filtered phases for each crop
    all at the same time, or at different images with 1 image per filtered phase

    Parameters
    ----------
    image_in_dataset_whole : 
    general_bragg_filterings : 
    phases_per_crop : 

    Returns
    -------
    Just plots the maps

    '''
    
    # phases_per_crop =  indicates how many phases we want to retrieve
    list_colours = ['Reds', 'Greens', 'Blues', 'Oranges', 'Purples', 'spring',   
                     'Wistia', 'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 
                    'BuPu','GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']

    
    # Image array
    image_array_whole = image_in_dataset_whole.image_arraynp_st    
    
    for crop_str in general_bragg_filterings:
        
        bragg_filtered_phases = general_bragg_filterings[crop_str]
        
        # Detect when no phase was found, it was set to 0
        if bragg_filtered_phases == 0:
            continue
    
        
        # crop single figure
        
        
        fig = plt.figure(frameon=False)
                
        im1 = plt.imshow(
            image_array_whole, cmap=plt.cm.gray, alpha = 1) 
           
    
        for indx, crystal_ph in enumerate(
                bragg_filtered_phases):
            
            # limit the plotting to only phases_per_crop number of phases
            if indx >= phases_per_crop:
                break
            
            
            masked_image = bragg_filtered_phases[crystal_ph]
            
            im2 = plt.imshow(masked_image, cmap = list_colours[indx], alpha=0.3)

            
        plt.show()      
            
      
    # Get the multiple image plot      
    for crop_str in general_bragg_filterings:
        
        bragg_filtered_phases = general_bragg_filterings[crop_str]
        
        # Detect when no phase was found, it was set to 0
        if bragg_filtered_phases == 0:
            continue            
         
            
        # get for that crop the number of phases checked
        # which can be different from phases_per_crop depending on how many
        # phases are found (i.e. we want to plot 3 phases but only 1 is found)
        # then build the figure axes with this value max_phases_check
        for indx, crystal_ph in enumerate(
                bragg_filtered_phases):
            
            # limit the plotting to only phases_per_crop number of phases
            if indx >= phases_per_crop:
                indx = indx -1
                break
            
        max_phases_check = indx + 1  



        if max_phases_check == 1:
            
            fig3 = plt.figure(frameon=False)
            fig3, ax1 = plt.subplots(1, 1, constrained_layout=True)
            
            for indx, crystal_ph in enumerate(
                    bragg_filtered_phases):
                
                # limit the plotting to only phases_per_crop number of phases
                if indx >= phases_per_crop:
                    break
                
                masked_image = bragg_filtered_phases[crystal_ph]            
                im1 = ax1.imshow(
                    image_array_whole, cmap=plt.cm.gray, alpha = 1) 
                
                masked_image = bragg_filtered_phases[crystal_ph]
            
                im2 = ax1.imshow(masked_image, cmap = list_colours[indx], alpha=0.3)
                fig3.suptitle(
                    'Crop ' + crop_str[:crop_str.find('_')] + ' Bragg filtered phase: ' + crystal_ph + ', colour: ' + list_colours[indx], 
                    fontsize=18)
            
            
        else:

            # column of images per crop     
            fig2 = plt.figure(frameon=False)
            
            fig2, ax = plt.subplots(max_phases_check, 1, constrained_layout=True)
            fig2.set_size_inches(5, max_phases_check*5)
            fig2.suptitle(
                'Crop ' + crop_str[:crop_str.find('_')] + ' Bragg filtered phases', fontsize=18)
                    
        
            for indx, crystal_ph in enumerate(
                    bragg_filtered_phases):
                
                # limit the plotting to only phases_per_crop number of phases
                if indx >= phases_per_crop:
                    break
                
                masked_image = bragg_filtered_phases[crystal_ph]
                
                im = ax[indx].imshow(
                    image_array_whole, cmap=plt.cm.gray, alpha = 1)
                im = ax[indx].imshow(
                    masked_image, cmap = list_colours[indx], alpha=0.3)
                
                ax[indx].set_title(
                    'Phase: ' + crystal_ph + ', colour: ' + list_colours[indx])
    
            
        plt.show()      
                        
        

def Bragg_filter_mask_size(
        GPA_resolution, image_in_dataset, mask_reduction_factor = 5):
    '''
    Define the Bragg mask size depending on the GPA mask size,
    basically cut the GPA mask into mask_reduction_factor sectors, and this
    will act as the Bragg mask size

    Parameters
    ----------
    GPA_resolution : GPA resolution, in nm
    image_in_dataset : 
    mask_reduction_factor : int, times the GPA resolution is divided to form
                            the Bragg mask
        DESCRIPTION. The default is 5.

    Returns
    -------
    bragg_mask_size : mask size in number of pixels

    '''
    
    FFT_calibration_whole = image_in_dataset.FFT_calibration
    
    
    GPA_resol_nminv = 1/GPA_resolution
    GPA_resol_pixels = GPA_resol_nminv/FFT_calibration_whole
    
    bragg_mask_size = int(np.ceil(GPA_resol_pixels/mask_reduction_factor))
           
    
    return bragg_mask_size


    
    
def main(): 
    global root
    global fft_info_data
    root = tk.Tk()
    # root.geometry('1202x800+10+10')
    
    s = ttk.Style(root)
    s.configure('TFrame', background='#343638')
    s.configure('TCheckbutton', background='#343638', foreground='white',font=('American typewriter', 14))
    s.configure('TCombobox', background='#343638')
    
    #Frame fft image
    global canvas
    canvas = tk.Canvas(root, width= 800, height= 800)
    canvas.pack(side = 'left')
    
    #Frame settings
    global st
    st = ScrolledText(root, height = 30, width = 40)
    st.pack(fill = 'both', anchor = 'n', expand = True, side ='top')
    
    frame_select = ttk.Frame(root)
    frame_select.pack(fill = 'both', anchor = 'n', expand = True)
    
    global bool_peaks
    global bool_text
    bool_peaks = IntVar(root, 1)
    bool_text = IntVar(root, 0)
    
    check_peaks = ttk.Checkbutton(master = frame_select,text = 'Show peaks',command = show_peaks,variable= bool_peaks)
    check_multitext = ttk.Checkbutton(master = frame_select,text = 'Overwrite text',variable= bool_text)
    
    global fft_combobox
    fft_combobox = ttk.Combobox(frame_select)
    fft_combobox['values'] = list(fft_info_data.keys())
    fft_combobox['state'] = 'readonly'
    fft_combobox.set(list(fft_info_data.keys())[0])
    fft_combobox.bind('<<ComboboxSelected>>', draw_fft)  
    
    fft_combobox.pack(expand = True)
    check_peaks.pack(expand = True)
    check_multitext.pack(expand = True)
    
    
    #Layout
    for widget in root.children.values():
        widget.pack_configure(padx = 1, pady = 1)
        
    draw_fft('')
    root.mainloop()
    
    
    
if __name__ == '__main__':
    main()