# -*- coding: utf-8 -*-
"""
Created on Fri Jul 30 12:55:12 2021

@author: Marc
"""
#Manual multiscale template matching based on manual reescaling of the images and application
#of classsical tempate matching (implementation by openCV)

'''
Sample code from pyimagesearch for manual template matching

# import the necessary packages
import numpy as np
import argparse
import imutils
import glob
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-t", "--template", required=True, help="Path to template image")
ap.add_argument("-i", "--images", required=True,
	help="Path to images where template will be matched")
ap.add_argument("-v", "--visualize",
	help="Flag indicating whether or not to visualize each iteration")
args = vars(ap.parse_args())
# load the image image, convert it to grayscale, and detect edges
template = cv2.imread(args["template"])
template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
template = cv2.Canny(template, 50, 200)
(tH, tW) = template.shape[:2]
cv2.imshow("Template", template)

# loop over the images to find the template in
for imagePath in glob.glob(args["images"] + "/*.jpg"):
	# load the image, convert it to grayscale, and initialize the
	# bookkeeping variable to keep track of the matched region
	image = cv2.imread(imagePath)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	found = None
	# loop over the scales of the image
	for scale in np.linspace(0.2, 1.0, 20)[::-1]:
		# resize the image according to the scale, and keep track
		# of the ratio of the resizing
		resized = imutils.resize(gray, width = int(gray.shape[1] * scale))
		r = gray.shape[1] / float(resized.shape[1])
		# if the resized image is smaller than the template, then break
		# from the loop
		if resized.shape[0] < tH or resized.shape[1] < tW:
			break

# detect edges in the resized, grayscale image and apply template
		# matching to find the template in the image
		edged = cv2.Canny(resized, 50, 200)
		result = cv2.matchTemplate(edged, template, cv2.TM_CCOEFF)
		(_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)
		# check to see if the iteration should be visualized
		if args.get("visualize", False):
			# draw a bounding box around the detected region
			clone = np.dstack([edged, edged, edged])
			cv2.rectangle(clone, (maxLoc[0], maxLoc[1]),
				(maxLoc[0] + tW, maxLoc[1] + tH), (0, 0, 255), 2)
			cv2.imshow("Visualize", clone)
			cv2.waitKey(0)
		# if we have found a new maximum correlation value, then update
		# the bookkeeping variable
		if found is None or maxVal > found[0]:
			found = (maxVal, maxLoc, r)
	# unpack the bookkeeping variable and compute the (x, y) coordinates
	# of the bounding box based on the resized ratio
	(_, maxLoc, r) = found
	(startX, startY) = (int(maxLoc[0] * r), int(maxLoc[1] * r))
	(endX, endY) = (int((maxLoc[0] + tW) * r), int((maxLoc[1] + tH) * r))
	# draw a bounding box around the detected result and display the image
	cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)
	cv2.imshow("Image", image)
	cv2.waitKey(0)
    
    
    '''
    
    
import numpy as np
import argparse
import imutils
import glob
import cv2
import hyperspy.api as hs
import matplotlib.pyplot as plt
from PIL import Image

#hyperparams
SobelFilterSize=7

#use two images from the one dataset and work with them as trials
#Load both images

image_query_filename=r'E:\Arxius varis\PhD\2nd_year\Code\trial_images\full_device_STEM_datasets\SQ20-250-2\16.16.30 Scanning Acquire_0.dm3'
image_temhs_filename=r'E:\Arxius varis\PhD\2nd_year\Code\trial_images\full_device_STEM_datasets\SQ20-250-2\16.15.33 Scanning Acquire_0.dm3'   

image_queryhs=hs.load(image_query_filename)
image_temhs=hs.load(image_temhs_filename)

#QUERY IMAGE (image at low mag, image in which the other image (template) has to be fitted)
x_calibration_query=image_queryhs.axes_manager['x'].scale
y_calibration_query=image_queryhs.axes_manager['y'].scale
x_pixels_query=image_queryhs.axes_manager['x'].size
y_pixels_query=image_queryhs.axes_manager['y'].size
x_units_query=image_queryhs.axes_manager['x'].units
y_units_query=image_queryhs.axes_manager['y'].units
#convert to numpy array
image_querynp=np.asarray(image_queryhs)
plt.imshow(image_querynp, cmap=plt.cm.gray, vmin=image_querynp.min(), vmax=image_querynp.max())
plt.show()
#normalise
image_querynp_st=(image_querynp-np.min(image_querynp))/np.max(image_querynp-np.min(image_querynp))
#from numpy to compatible cv2
print(np.min(image_querynp_st),np.max(image_querynp_st))
#normalise after gaussian blurring

image_querynp_st = np.uint8(255*image_querynp_st)
#image_querynp_st=cv2.GaussianBlur(image_querynp_st, (5, 5), 1)


#TEMPLATE IMAGE (image at low mag, image in which the other image (template) has to be fitted)

x_calibration_tem=image_temhs.axes_manager['x'].scale
y_calibration_tem=image_temhs.axes_manager['y'].scale
x_pixels_tem=image_temhs.axes_manager['x'].size
y_pixels_tem=image_temhs.axes_manager['y'].size
x_units_tem=image_temhs.axes_manager['x'].units
y_units_tem=image_temhs.axes_manager['y'].units
#convert to numpy array
image_temnp=np.asarray(image_temhs)
plt.imshow(image_temnp, cmap=plt.cm.gray, vmin=image_temnp.min(), vmax=image_temnp.max())
plt.show()
#normalise
image_temnp_st=(image_temnp-np.min(image_temnp))/np.max(image_temnp-np.min(image_temnp))
#from numpy to compatible cv2
#normalise after gaussian blurring
image_temnp_st_int_main = np.uint8(255*image_temnp_st)
#image_temnp_st_int_main=cv2.GaussianBlur(image_temnp_st_int_main, (5, 5), 1)

#shifting contrast
alphas=[0.05,0.1,0.25,0.50,0.75,1,1.25,1.5,1.75,2,2.25,2.5,2.75,3,3.5,4]
alphas=[1]
for alpha in alphas:
    image_temnp_st_int=image_temnp_st_int_main
    beta = 0 # Brightness control (0-100)
    adjusted = cv2.convertScaleAbs(image_temnp_st_int, alpha=alpha, beta=beta)
      
    print('alpha',alpha)
    plt.imshow(adjusted, cmap=plt.cm.gray, vmin=0, vmax=255)
    plt.show()

    plt.hist(adjusted.ravel(),256,[np.min(np.array([adjusted])),np.max(np.array([adjusted]))])
    plt.show()
    image_temnp_st_int=adjusted

    #image_temnp_st_cv = Image.fromarray(image_temnp_st.astype(np.uint8))

    #template edges
    templateedge = cv2.Canny(image_temnp_st_int, 255/3, 255, apertureSize=SobelFilterSize)
    (tHbase, tWbase) = templateedge.shape[:2]
    
    
    '''
    
    # loop over the images to find the template in
    # bookkeeping variable to keep track of the matched region
    # queryedge is already in gray scale
    found = None
    # loop over the scales of the image
    
    #here the correct downscaling can be computed by the pixel sizes and fields of view of each image
    for scale in np.linspace(0.1, 1.0, 20)[::-1]:
        
      	# resize the image according to the scale, and keep track
      	# of the ratio of the resizing
        #resized is the template not the query as in the example
        resized = imutils.resize(image_temnp_st_int, width = int(image_temnp_st_int.shape[1] * scale))
        r = image_temnp_st_int.shape[1] / float(resized.shape[1])
    
      	# if the resized image is smaller than the template, then break
      	# from the loop
        # update the height and width of template image after resizing 
        (tH, tW) = resized.shape[:2]
    
        # detect edges in the resized template
        # matching to find the template in the query image which is being progressively downscaled
        templateedge = cv2.Canny(resized, 255/3, 255, apertureSize=SobelFilterSize)
        #work with the image and not edges
        templateedge =resized
        result = cv2.matchTemplate(queryedge, templateedge, cv2.TM_CCOEFF)
        (_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)
       	# check to see if the iteration should be visualized
        visualize=False
        if visualize==True:
       		# draw a bounding box around the detected region
            clone = np.dstack([queryedge, queryedge, queryedge])
            rect=cv2.rectangle(clone, (maxLoc[0], maxLoc[1]),(maxLoc[0] + tW, maxLoc[1] + tH), (0, 0, 255), 2)
            plt.imshow(clone)
            plt.imshow(rect)
            plt.show()
            #if we have found a new maximum correlation value, then update
         	# the bookkeeping variable
        if found is None or maxVal > found[0]:
       		found = (maxVal, maxLoc, r)
    # unpack the bookkeeping variable and compute the (x, y) coordinates
    # of the bounding box based on the resized ratio
    
    (_, maxLoc, r) = found
    (startX, startY) = (int(maxLoc[0]), int(maxLoc[1]))
    (endX, endY) = (int((maxLoc[0] + tWbase)/r), int((maxLoc[1] + tHbase)/r))
    # draw a bounding box around the detected result and display the image
    queryedge_copy=np.copy(queryedge)
    rect=cv2.rectangle(queryedge_copy, (startX, startY), (endX, endY), (0, 0, 255), 2)
    plt.imshow(queryedge_copy)
    plt.imshow(rect)
    plt.show()   
    
    '''
    
    
    #Try directly with the exact scale that should be applied to the template given the
    #calibration of the images (make sure they are all in nm, or same unit)
    ideal_scale=(x_calibration_tem)/(x_calibration_query)
    print(ideal_scale)
    
    
    
    #Non edged example:
    print('Not edged process')
    
    
    # resize the image according to the scale, and keep track
      	# of the ratio of the resizing
     #resized is the template not the query as in the example
    resized = imutils.resize(image_temnp_st_int, width = int(image_temnp_st_int.shape[1] * ideal_scale))
    r = image_temnp_st_int.shape[1] / float(resized.shape[1])
    
      	# if the resized image is smaller than the template, then break
      	# from the loop
     # update the height and width of template image after resizing 
    (tH, tW) = resized.shape[:2]
    
     # detect edges in the resized template
     # matching to find the template in the query image which is being progressively downscaled
    result = cv2.matchTemplate(image_querynp_st, resized, cv2.TM_CCORR_NORMED)
    print('Correlation map non edged')
    plt.imshow(result)
    plt.show()
    (_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)
    print('Coeff non edged: ', maxVal)
    coeffnon=maxVal
    	# check to see if the iteration should be visualized
    visualize=False  
    if visualize==True:
    		# draw a bounding box around the detected region
         clone = np.dstack([image_querynp_st, image_querynp_st, image_querynp_st])
         rect=cv2.rectangle(clone, (maxLoc[0], maxLoc[1]),(maxLoc[0] + tW, maxLoc[1] + tH), (0, 0, 255), 2)
         plt.imshow(clone)
         plt.imshow(rect)
         plt.show()
         #if we have found a new maximum correlation value, then update
      	# the bookkeeping variable
    
    # unpack the bookkeeping variable and compute the (x, y) coordinates
    # of the bounding box based on the resized ratio
    (startX, startY) = (int(maxLoc[0]), int(maxLoc[1]))
    (endX, endY) = (int((maxLoc[0] + tW)), int((maxLoc[1] + tH)))
    # draw a bounding box around the detected result and display the image
    image_querynp_st_cop=np.copy(image_querynp_st)
    
    rectnonedg=cv2.rectangle(image_querynp_st_cop, (startX, startY), (endX, endY), (0, 0, 255), 2)
    plt.imshow(image_querynp_st_cop)
    plt.imshow(rectnonedg)
    plt.show()   
    
    #How the edges look like in the query image in the found area
    print('crop')
    cropimagequerynonedge=image_querynp_st_cop[maxLoc[1]:maxLoc[1]+tH,maxLoc[0]:maxLoc[0]+tW]
    plt.imshow(cropimagequerynonedge)
    plt.show()  
    
    plt.hist(cropimagequerynonedge.ravel(),256,[np.min(np.array([cropimagequerynonedge])),np.max(np.array([cropimagequerynonedge]))])
    plt.show()
    
    
    #Edged example:
    print('Edged process: 7x7')
    #query edges
    queryedge = cv2.Canny(image_querynp_st, 255/3, 255, apertureSize=SobelFilterSize)
    (qH, qW) = queryedge.shape[:2]
    plt.imshow(queryedge)
    plt.show()
    
    # resize the image according to the scale, and keep track
      	# of the ratio of the resizing
     #resized is the template not the query as in the example
    resized = imutils.resize(image_temnp_st_int, width = int(image_temnp_st_int.shape[1] * ideal_scale))
    r = image_temnp_st_int.shape[1] / float(resized.shape[1])
    
      	# if the resized image is smaller than the template, then break
      	# from the loop
     # update the height and width of template image after resizing 
    (tH, tW) = resized.shape[:2]
    
     # detect edges in the resized template
     # matching to find the template in the query image which is being progressively downscaled
    templateedge = cv2.Canny(resized, 255/3, 255, apertureSize=SobelFilterSize)
    plt.imshow(templateedge)
    plt.show()
    result = cv2.matchTemplate(queryedge, templateedge, cv2.TM_CCORR_NORMED)
    print('Correlation map edged')
    plt.imshow(result)
    plt.show()
    
    (_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)
    print('Coeff edged: ', maxVal)
    coffedg=maxVal
    	# check to see if the iteration should be visualized
    visualize=False  
    if visualize==True:
    		# draw a bounding box around the detected region
         clone = np.dstack([image_querynp_st, image_querynp_st, image_querynp_st])
         rect=cv2.rectangle(clone, (maxLoc[0], maxLoc[1]),(maxLoc[0] + tW, maxLoc[1] + tH), (0, 0, 255), 2)
         plt.imshow(clone)
         plt.imshow(rect)
         plt.show()
         #if we have found a new maximum correlation value, then update
      	# the bookkeeping variable
    
    # unpack the bookkeeping variable and compute the (x, y) coordinates
    # of the bounding box based on the resized ratio
    (startX, startY) = (int(maxLoc[0]), int(maxLoc[1]))
    (endX, endY) = (int((maxLoc[0] + tW)), int((maxLoc[1] + tH)))
    # draw a bounding box around the detected result and display the image
    image_querynp_st_copy2=np.copy(image_querynp_st)
    rect=cv2.rectangle(image_querynp_st_copy2, (startX, startY), (endX, endY), (0, 0, 255), 2)
    plt.imshow(image_querynp_st_copy2)
    plt.imshow(rect)
    plt.show()   
    
    #How the edges look like in the query image in the found area
    print('crop')
    cropimagequery=queryedge[maxLoc[1]:maxLoc[1]+tH,maxLoc[0]:maxLoc[0]+tW]
    plt.imshow(cropimagequery)
    plt.show() 
    
    
    #image names
    query_name=image_query_filename[::-1][23:31][::-1]
    tem_name=image_temhs_filename[::-1][23:31][::-1]
    
    
    fig = plt.figure(figsize=(20,8))
    
    ax = fig.add_subplot(2, 4, 1)
    imgplot = plt.imshow(image_querynp_st)
    ax.set_title('Raw query '+query_name)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax = fig.add_subplot(2, 4, 2)
    imgplot = plt.imshow(image_temnp_st_int)
    ax.set_title('Raw template '+tem_name)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax = fig.add_subplot(2, 4, 3)
    imgplot = plt.imshow(image_querynp_st)
    imgplot= plt.imshow(rectnonedg)
    ax.set_title('NonEM:{:.3e}'.format(coeffnon))
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax = fig.add_subplot(2, 4, 4)
    imgplot = plt.imshow(cropimagequerynonedge)
    ax.set_title('NonE crop')
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax = fig.add_subplot(2, 4, 5)
    imgplot = plt.imshow(queryedge)
    ax.set_title('Edged query ')
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax = fig.add_subplot(2, 4, 6)
    imgplot = plt.imshow(templateedge)
    ax.set_title('Edged template')
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax = fig.add_subplot(2, 4, 7)
    imgplot = plt.imshow(image_querynp_st)
    imgplot= plt.imshow(rect)
    ax.set_title('EM:{:.3e}'.format(coffedg))
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax = fig.add_subplot(2, 4, 8)
    imgplot = plt.imshow(cropimagequery)
    ax.set_title('Edged crop')
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    
