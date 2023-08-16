
 
from EMicrographs_to_AtoModels.SAM_Segmentor.segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
# from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

import numpy as np
import hyperspy.api as hs
import matplotlib.pyplot as plt
import itertools
import os
import sys

# !!! NEED to set the path to 
# Alg_Comb_Single_Image_Strain.py
# as the console working directory
Project_main_path = os.getcwd()
if 'EMicrographs_to_AtoModels' in Project_main_path:
    Project_main_path = Project_main_path[:Project_main_path.find('EMicrographs_to_AtoModels')-1]
# Project_main_path has the EMicrographs_to_AtoModels folder
sys.path.append(Project_main_path)

path_to_SAM_AI_model = r'' + Project_main_path + '\\EMicrographs_to_AtoModels\SAM_Segmentor\sam_vit_h_4b8939.pth'


'''
Functions for depleting the SAM segmentation, mainly the automatic one
and its refinment to get a univocal and discrete functional segmentation
'''


def SAM_autosegment_base(
        image):
    '''
    Function calling the AI SAM model to segment an rgb image and get 
    a list of dictionaries containing the segmentations found 

    Parameters
    ----------
    image : rgb 3 channels, image, [height, width, 3], ints (from 0 to 255)

    Returns
    -------
    segmentation_dict : list of dictionaries where every entry is one 
                    segmented part of the whole process

    '''
    
    sam_checkpoint = path_to_SAM_AI_model
    model_type = "vit_h"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)

    try:
        device = "cuda"
        sam.to(device=device)
    except:
        device = "cpu"
        sam.to(device=device)

    mask_generator = SamAutomaticMaskGenerator(sam)
    segmentation_dict = mask_generator.generate(image)

    return segmentation_dict



def Maximise_num_SAM_segments(
        original_array, segmentation_dict):
    '''
    Continuation to the main function segmenting automatically by SAM,
    which maximises the number of presented segments in the final image by checking
    if there is ambiguity in the choice: if there is a segment whose pixels are
    coincident to the sum of other segments, then this one is discarded in favour
    of the other 2 or more segments that still cover the same pixels
    Long story short, the more segments the better

    Parameters
    ----------
    original_array : orignal array of the image, so [height, width] shape 
                    just two dimensions
    segmentation_dict : list of dictionaries where every entry is one 
                    segmented part of the whole process

    Returns
    -------
    final_segmented_image : 2 channel image with the segmentation, so ints
                    as type of image, each value is a label, from
                    0 (if unassigned pixels remain) to N labels

    '''
    
    segm_image = np.zeros(np.shape(original_array))
    
    # count the number of 1s each image has to sort the segments based
    # on how big they are, to then see how one can be reconstructed from others
    indiv_segms = []
    counts_of_1s = []
    
    label = 1
    for maski in segmentation_dict:
        for el in maski:
            if el == 'segmentation':
                segm = maski[el]
                indiv_segms.append(segm)
                
                unq_vals, unq_counts = np.unique(segm, return_counts=True)
                counts_of_1s.append(unq_counts[unq_vals == 1])
                
                # plt.imshow(segm)
                # plt.show()
                segm_image[segm == True] = label
                label = label + 1
                
        
    # segm_image we can see how the initial segmentation is with all the segments 
    # considered
    
    # sort the segmentated regions from the largest counts to the smallest
    # so more area/pixels assigned for the label to less        
    sorted_counts = [counts for _,counts in sorted(zip(indiv_segms, counts_of_1s), key = lambda pair: pair[1])][::-1]
    sorted_segms = [segm for segm,_ in sorted(zip(indiv_segms, counts_of_1s), key = lambda pair: pair[1])][::-1]

    
    # loop through the semgentaiotn and chcek from the biggest to the smaleles 
    # if one can be built upon the combination of other smaller segmented regions
    
    # append the segms to remove if needed
    segms_to_remove = []
    
    for index_check in range(len(sorted_segms)):
        
        segm_check = sorted_segms[index_check]
        counts_1s_check = sorted_counts[index_check][0]
        
        # do the checks with combinations of the rest of the list
        if index_check+1 == len(sorted_segms):
            continue
        
        segms_to_comb = sorted_segms[index_check+1:]
        segms_to_comb_indexs = np.arange(0, len(segms_to_comb))
        # check pairs of combs
        
        # from pairs to all the units in the remaining segment of the array
        for L in range(2,len(segms_to_comb)+1):
            for subset in itertools.combinations(segms_to_comb, L):
                # print(subset)
                
                # sum the arrays in the possible combination of elements
                
                summed_array = np.zeros(np.shape(sorted_segms[0]))
                for seg in subset:
                    summed_array[seg == 1] = 1
                    
                
                # compare the overlapping ones from the summed_array and
                # the checked segm segm_check
                
        
                # multiplying the arrays 0s 1s we keep 1s only on 
                # these having 1s on both
                overlapping_1s = summed_array*segm_check
                
                unq_vals_over1s, unq_counts_over1s = np.unique(
                    overlapping_1s, return_counts=True)
                
                counts_1s_over1s = unq_counts_over1s[unq_vals_over1s == 1]
                
                if len(counts_1s_over1s) == 0:
                    continue
                else:
                    counts_1s_over1s = counts_1s_over1s[0]
                
    
                # if a combination of some of the other
                # segmented regions makes it a coincident number of pixels
                # of more than 90% for instnac
    
                if counts_1s_over1s/counts_1s_check >= 0.9:
                    
                    segms_to_remove.append(index_check)
                
    # remove the elements that are selected to be rmeoved as can be obtained from 
    # other combined labels, identified as the index within sorted_segms
    segms_to_remove = np.unique(segms_to_remove)
    segms_to_remove = list(segms_to_remove)
    segms_to_remove = sorted(segms_to_remove)[::-1]
    
    
    for segm_rem in segms_to_remove:
        
        sorted_segms.pop(segm_rem)
        
    
    # build final segmented image
    final_segmented_image = np.zeros(np.shape(segm_image))
    
    label = 1
    for segm in sorted_segms:
        
        indiv_segms.append(segm)
        
        # plt.imshow(segm)
        # plt.show()
        final_segmented_image[segm == True] = label
        label = label + 1
    
        
    # filter if label 1 is the sum of other two, remove this one 
    # so if the are ais coincident
    # then remove the signle label and have preference of the other many
    # labels occupying this space
    
    plt.imshow(final_segmented_image)
    plt.show()
    
    return final_segmented_image



def Full_SAM_Autosegmentation(
        image_arraynp_st):
    '''
    Only function to call if we want to segment the image by the SAM in an 
    automated way, with its base automated algorithm, and with the collapsing
    of the ambiguities by maximising the number of segmented regions (labels)

    Parameters
    ----------
    image_arraynp_st : image_in_dataset object image_to_segment.image_arraynp_st
                so the normalised array (float vals 0 to 1) ready to be converted
                into ints from 0 to 255 but as an rgb 3 channel image 
                suitable as an input for the automated SAM

    Returns
    -------
    final_segmented_image : 2 channel image with the segmentation, so ints
                    as type of image, each value is a label, from
                    0 (if unassigned pixels remain) to N labels

    '''
            
    # Prepeare the image, array, needs to be int array, 3 channel image
    image_std_int_ch = np.uint8((255*image_arraynp_st)/3)

            
    image_std_int_ch.shape = (np.shape(image_std_int_ch)[0],np.shape(image_std_int_ch)[1], 1)

    rgb_img = np.concatenate((image_std_int_ch, image_std_int_ch) , axis = 2)
    rgb_img = np.concatenate((rgb_img, image_std_int_ch), axis = 2)

    # Perform the actual segmentation and extract all info as a dict
    segmentation_dict = SAM_autosegment_base(rgb_img)

    # Convert the dictionary into the segmented image by maximising the number
    # of segments found in case a region gives an ambiguous segmentation 
    # that overlaps the correct segmentation (more segments is always good)
    
    final_segmented_image = Maximise_num_SAM_segments(
        image_arraynp_st, segmentation_dict)
    
    
    return final_segmented_image


# path_imag = r'E:\Arxius varis\PhD\2nd_year\Code\trial_images\low_mag_images\nanowire3_dm3.dm3'


# hypersimage = hs.load(path_imag)

# original_array = np.asarray(hypersimage, dtype = int)

# image = np.asarray(hypersimage, dtype = int)
# image=(image-np.min(image))/np.max(image-np.min(image))


# final_segmented_image = Full_SAM_Autosegmentation(
#     image)




