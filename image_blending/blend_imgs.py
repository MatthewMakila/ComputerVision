# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 14:09:12 2023

@author: matth
"""
#%% Import libraries

import cv2
import numpy as np
import matplotlib.pyplot as plt

#%% Part 2.2:
"""Loading and Displaying Images"""

# load and display elephant
elephant = cv2.imread("elephant.jpeg")
cv2.imshow("Elephant", elephant)

# wait until key pressed, close img
cv2.waitKey(0)
cv2.destroyAllWindows()

"""Loading with plt"""

# display image with plt.imshow
plt.imshow(elephant)
cv2.imwrite("elephant_opencv.png", elephant)

"""Color convert"""

elephant_2 = cv2.imread("elephant_opencv.png")
rgb_elephant = cv2.cvtColor(elephant_2, cv2.COLOR_BGR2RGB)
plt.imshow(rgb_elephant)
cv2.imwrite("elephant_matplotlib.png", rgb_elephant)

#%% Task 2.3
"""Convert img to grayscale"""

gray_elephant = cv2.cvtColor(rgb_elephant, cv2.COLOR_RGB2GRAY)
plt.imshow(gray_elephant, cmap='gray')
cv2.imwrite("elephant_gray.png", gray_elephant)

"""ropping little elephant"""

og_width = 400
og_height = 570
scale_factor = 10
little_elephant = rgb_elephant[350:920, 100:500]
plt.figure()
plt.imshow(little_elephant)
cv2.imwrite("elephant_baby.png", little_elephant)

"""Resizing elephant"""

# down sample (tenth of original size)
little_elephant_down = cv2.resize(little_elephant, (og_width//scale_factor, og_height//scale_factor))
plt.figure()
plt.imshow(little_elephant_down)
cv2.imwrite("elephant_10xdown.png", little_elephant_down)

# upsample
little_elephant_NN = cv2.resize(little_elephant_down, (og_width, og_height), interpolation=cv2.INTER_NEAREST)
little_elephant_BC = cv2.resize(little_elephant_down, (og_width, og_height), interpolation=cv2.INTER_CUBIC)
plt.figure()
plt.imshow(little_elephant_NN)
plt.figure()
plt.imshow(little_elephant_BC)
cv2.imwrite("elephant_10xup_NN.png", little_elephant_NN)
cv2.imwrite("elephant_10xup_BC.png", little_elephant_BC)

# take differences of upsamples vs. original
subtracted_NN = cv2.subtract(little_elephant, little_elephant_NN)
subtracted_BC = cv2.subtract(little_elephant, little_elephant_BC)
cv2.imwrite("elephant_sub_NN.png", subtracted_NN)
cv2.imwrite("elephant_sub_BC.png", subtracted_BC)

NN_sum = np.sum(subtracted_NN)
BC_sum = np.sum(subtracted_BC)
print("Nearest neighbor diff. pixel sum (err): ", NN_sum)
print("Bicubic diff. pixel sum (err): ", BC_sum)

#%% Task 2.4
"""2D Convolution"""

def _2D_convolution(img, k):
    """
    img:    image for convolution
    k: kernel for convolution
    """    
    new_img = cv2.filter2D(src=img, ddepth=-1, kernel=k)
    
    return new_img
    

# for part 1, use edge-detected kernel
k_1 = np.array([[-1, -1, -1],
             [-1, 8, -1],
             [-1, -1, -1]])

elephant_ED = _2D_convolution(gray_elephant, k_1)

# for part 2, use box blur kernel (10 x 10 of 1's * 1/100 worked well)
k_2 = np.ones((20, 20)) / 400

elephant_blur = _2D_convolution(rgb_elephant, k_2)

plt.figure()
plt.imshow(rgb_elephant)
plt.figure()
plt.imshow(elephant_ED)
plt.figure()
plt.imshow(elephant_blur)

cv2.imwrite("elephant_ED_4.png", elephant_ED)
cv2.imwrite("elephant_blur_4.png", elephant_blur)

"""--------------------------------------------------------------"""
#%% Part 3.1:
    
"""Task 1: Phase Swapping"""
# read in the images
panda = cv2.imread("panda.png", cv2.IMREAD_GRAYSCALE)
tiger = cv2.imread("tiger.png", cv2.IMREAD_GRAYSCALE)

panda = panda[200:1800, 500:2100]   # make panda img same shape

plt.figure()
plt.imshow(panda, cmap='gray')
plt.figure()
plt.imshow(tiger, cmap='gray')

# take fft
panda_fft = np.fft.fft2(panda)
tiger_fft = np.fft.fft2(tiger)

# find magnitude and phase
mag_panda = np.abs(panda_fft)
mag_tiger = np.abs(tiger_fft)
phase_panda = np.angle(panda_fft)
phase_tiger = np.angle(tiger_fft)

# swap phase images
phase_panda_1 = mag_panda * np.exp(1j * phase_tiger)
phase_tiger_1 = mag_tiger * np.exp(1j * phase_panda)

# remake images with inverse fft
new_panda = np.real(np.fft.ifft2(phase_panda_1))
new_tiger = np.real(np.fft.ifft2(phase_tiger_1))

#display
plt.figure()
plt.imshow(phase_tiger, cmap='gray')
plt.figure()
plt.imshow(mag_tiger, cmap='gray')
plt.figure()
plt.imshow(phase_panda, cmap='gray')
plt.figure()
plt.imshow(mag_panda, cmap='gray')
plt.figure()
plt.imshow(new_panda, cmap='gray')
plt.figure()
plt.imshow(new_tiger, cmap='gray')

# cv2.imwrite("swapped_panda.png", new_panda)

#%% Task 3.2:
"""Hybrid Images"""

# use the panda and tiger to do Oliva et al. hybrid image
# equation from paper: H = I1(G1) + I2(1 - G2)
puma = cv2.imread("puma.jpg")
kang = cv2.imread("kang.jpg")
puma = cv2.cvtColor(puma, cv2.COLOR_BGR2RGB)
kang = cv2.cvtColor(kang, cv2.COLOR_BGR2RGB)

# display og images
plt.figure()
plt.imshow(puma)
plt.figure()
plt.imshow(kang)

# blur images (go filter * img)
I1_G1 = cv2.GaussianBlur(puma, (7,7), 0) # blur with 3x3 kernel and no std
I2_G2 = cv2.GaussianBlur(kang, (7,7), 0)

# I2(1 - G2) = I2 - I2*G2
# I2*G2 resolved by cv2.GaussianBlur
high = kang - I2_G2
hybrid_img = I1_G1 + high

# display blurred images and hybrid
plt.figure()
plt.imshow(I1_G1)
plt.figure()
plt.imshow(I2_G2)
plt.figure()
plt.imshow(hybrid_img)

cv2.imwrite("hybrid_1.png", hybrid_img)

"""--------------------------------------------------------------"""
#%% Part 4.1: Multiresolution Image Blending
    
"""Direct & Alpha Blending"""

apple = cv2.imread("apple.jpeg")
orange = cv2.imread("orange.jpeg")
apple = cv2.cvtColor(apple, cv2.COLOR_BGR2RGB)
orange = cv2.cvtColor(orange, cv2.COLOR_BGR2RGB)

# make the mask and fill half with 1's and half with 0's
mask = np.zeros_like(apple)
mask[:, 150:].fill(1)

# choose apple as left side and display
direct_blend = mask * orange + (1 - mask) * apple
plt.figure()
plt.imshow(direct_blend)

# alpha blend along the mask's edge
alpha_blend = direct_blend.copy()
center_dev = 10
edge_midpoint = 150
mask_edge = alpha_blend[:, edge_midpoint-center_dev:edge_midpoint+center_dev]
mask_blur = cv2.GaussianBlur(mask_edge, (7,7), 0)
alpha_blend[:, edge_midpoint-center_dev:edge_midpoint+center_dev] = mask_blur

plt.figure()
plt.imshow(alpha_blend)

# save imgs
cv2.imwrite("direct_blend.png", direct_blend)
cv2.imwrite("alpha_blend.png", alpha_blend)

#%% Task 4.2:
"""Gaussian and Laplacian Pyramids"""

def img_pyramid(img):
    # assuming symmetrical img input
    # use img pyramids list to apply Gauss and Lap filters later
    depth = int(np.floor(np.log2(img.shape[1]))) # floor log2 safely says the times img can be halved
    img_list = []
    
    # halve img size and store into list
    new_img = img
    for i in range(depth):
        img_list.append(new_img)
        #plt.figure()
        #plt.imshow(new_img)
        new_img = cv2.resize(new_img, (new_img.shape[0] // 2, new_img.shape[1] // 2))
        # print(new_img.shape)
    
    return img_list

    
def gauss_pyramid(img_list):
    # apply Gauss filters to orange, apple, and mask downsamples
    g_list = [img_list[0]]
    
    for i in range(len(img_list) - 1):
        g_img = cv2.GaussianBlur(img_list[i + 1], (5,5), 0) # don't blur og img
        g_list.append(g_img)
    return g_list

def lap_pyramid(img_list, gauss_list):
    # apply lap filters to all of the gaussian images
    l_list = [img_list[0] - gauss_list[0]]
    
    for i in range(len(gauss_list) - 1):
        l_img = img_list[i + 1] - gauss_list[i + 1]
        l_list.append(l_img)
    return l_list

# get pyramid list of downsampled images for orange, apple, & mask
o_samples = img_pyramid(orange)
a_samples = img_pyramid(apple)
m_samples = img_pyramid(mask)

# get gauss of all img set
o_gauss = gauss_pyramid(o_samples)
a_gauss = gauss_pyramid(a_samples)
m_gauss = gauss_pyramid(m_samples)

# get lap of all img set
o_lap = lap_pyramid(o_samples, o_gauss)
a_lap = lap_pyramid(a_samples, a_gauss)

#%% Task 4.3:
    
"""Multiresolution blending"""
def multiblend(im1, im2, m):
    # loop to add all gauss, mask, and lap together (but start from back of 
    # list to go from bottom up!)
    laps = []
    for i in range(len(im1) - 1, 0, -1):
        lap = im1[i] * m[i] + im2[i] * (1 - m[i])
        laps.append(lap)
        
        
    # for i in range(len(laps)):
    #     plt.figure()
    #     plt.imshow(laps[i])
        
        
    start = laps[0]
    for i in range(1, len(laps)):
        print(np.ceil(start.shape[0] * 2), np.ceil(start.shape[1] * 2))
        new_h = laps[i].shape[0]
        new_w = new_h
        start = cv2.resize(start, (new_h, new_w), interpolation=cv2.INTER_CUBIC)
        start = cv2.GaussianBlur(start, (5,5), 0)
        start = start + laps[i]
        print(i)
    
    return start

lappy = multiblend(o_lap, a_lap, m_gauss)

plt.figure()
plt.imshow(lappy)

cv2.imwrite("blend1.png", lappy)

#%% Task 4.4:
    
"""True Grit"""
grit = cv2.imread("true_grit.jpg")
dog = cv2.imread("real_dog.jpg")
grit = cv2.cvtColor(grit, cv2.COLOR_BGR2RGB)
dog = cv2.cvtColor(dog, cv2.COLOR_BGR2RGB)

dog = cv2.resize(dog, (300, 300))
grit = cv2.resize(grit, (300, 300))

plt.figure()
plt.imshow(grit)
plt.figure()
plt.imshow(dog)

# # get pyramid list of downsampled images for orange, apple, & mask
o_samples = img_pyramid(grit)
a_samples = img_pyramid(dog)
m_samples = img_pyramid(mask)

# get gauss of all img set
o_gauss = gauss_pyramid(o_samples)
a_gauss = gauss_pyramid(a_samples)
m_gauss = gauss_pyramid(m_samples)

# get lap of all img set
o_lap = lap_pyramid(o_samples, o_gauss)
a_lap = lap_pyramid(a_samples, a_gauss)

lappy_1 = multiblend(o_lap, a_lap, m_gauss)
plt.figure()
plt.imshow(lappy_1)

cv2.imwrite("blend2.png", lappy_1)
