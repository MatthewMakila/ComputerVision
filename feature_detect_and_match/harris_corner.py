# -*- coding: utf-8 -*-
"""
Created on Thu Oct 21 13:20:09 2023
Title:   491, HW2
@author: matth
"""

#%% Imports

import cv2
import numpy as np
import matplotlib.pyplot as plt


#%% HW 1. Harris Corner Detector

def harris_corner_detection(image):
    # Task 1: Compute the Harris matrix H for each 2x2 window
    im = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    dx = cv2.Sobel(im, cv2.CV_64F, 1, 0, ksize=3)
    dy = cv2.Sobel(im, cv2.CV_64F, 0, 1, ksize=3)

    kernel = cv2.getGaussianKernel(5, 0.5) # use kernel (5,5) and sigma = 0.5
    kernel = kernel.dot(kernel.T)
    
    # compute H's components Ix, Iy, Ixy using the Sobel gradients
    Ixx = cv2.filter2D(dx * dx, -1, kernel) 
    Iyy = cv2.filter2D(dy * dy, -1, kernel)
    Ixy = cv2.filter2D(dx * dy, -1, kernel)

    # Task 2: Compute the corner strength function c(H) at every pixel
    detH = Ixx * Iyy - Ixy**2
    traceH = Ixx + Iyy
    cH = detH - 0.1 * traceH**2
    
    # Task 3: Compute orientations as the angle of the gradient
    orientation_map = np.arctan2(dy, dx)  # Calculate orientation at each pixel


    # Task 4: Find local maxima in a 7x7 neighborhood
    local_maxima = []
    neighborhood_size = 7
    cH_thresh = 1000  # 1000 seemed to filter better


    for y in range(neighborhood_size, cH.shape[0] - neighborhood_size):
        for x in range(neighborhood_size, cH.shape[1] - neighborhood_size):
            center_value = cH[y, x]
            neighborhood = cH[y - neighborhood_size:y + neighborhood_size + 1, x - neighborhood_size:x + neighborhood_size + 1]
            max_value = np.max(neighborhood)

            if center_value == max_value and center_value > cH_thresh:
                local_maxima.append((x, y, orientation_map[y, x]))  # Append orientation to the keypoints


    # Task 5: Display the keypoints
    result_image = image.copy()
    for x, y, orientation in local_maxima:
        cv2.circle(result_image, (x, y), 3, (255, 255, 255), -1) # white circle
        line_length = 15  # Length of the orientation line
        angle = np.degrees(orientation)  # Convert radians to degrees
        dx = int(line_length * np.cos(angle))
        dy = int(line_length * np.sin(angle)) # add the lines to the circles for orientation mark + corner 
        cv2.line(result_image, (x, y), (x + dx, y + dy), (0, 0, 0), 2) # black line

    # Disp the imgs
    cv2.imshow('Harris Corners', result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    plt.figure()
    plt.imshow(result_image)
    

# Run examples for all imgs
image_paths = ['harris_car.jpeg', 'harris_elephant.jpeg', 'harris_sunflower.jpg',
               'tesla_2.jpg', 'elephant_2.jpg', 'sunflower_2.jpg']
for image_path in image_paths:
    image = cv2.imread(image_path)
    image = cv2.resize(image, (200,200)) #resize all im to 200x200
    harris_corner_detection(image)
    
#%% HW 3. 2D Transformations

# make a black background image
width = 300
height = 300
image = np.zeros((height, width, 3), dtype=np.uint8)
vertices = np.array([[100, 100], [100, 150], [150, 175], [150, 150]], np.int32)

# fill white irregular quadrilateral on the black background
cv2.fillPoly(image, [vertices], (255, 255, 255))

# display original quadrilateral
cv2.imshow('Quadrilateral', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# translation params
t_x, t_y = 30, 100
theta = 45

# use center point for rotation
c_x, c_y = width // 2, height // 2
rotation_matrix = cv2.getRotationMatrix2D((c_x, c_y), theta, 1)

# translate & rotate image by (tx, ty) and about original center
translated_image = cv2.warpAffine(image, np.float32([[1, 0, t_x], [0, 1, t_y]]), (width, height))
rotated_image = cv2.warpAffine(translated_image, rotation_matrix, (width, height))

# display final quadrilateral
cv2.imshow('Final Quadrilateral', rotated_image)
cv2.waitKey(0)
cv2.destroyAllWindows()


