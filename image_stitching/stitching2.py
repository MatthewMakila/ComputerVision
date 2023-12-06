# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 17:51:01 2023

@author: matth
"""

"""
USE LIKE THIS:
    python stitching2.py keble_a.jpg keble_b.jpg "C:/Users/matth/.spyder-py3/491 HW3/H_12.npy"
    
"""

import cv2
import numpy as np
import sys

def stitch_imgs(img, ref_img, h_path):
    
    H = np.load(h_path)

    
    # Get dim of images
    h1, w1 = img.shape[:2]
    h2, w2 = ref_img.shape[:2]

    # Use warpPerspective to apply the H to img
    im1_warped = cv2.warpPerspective(img, H, (w2, h2))

    # Combine images with addWeighted
    result = cv2.addWeighted(im1_warped, 0.5, ref_img, 0.5, 0.0)
    cv2.imshow("Stitched image", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # stitching 2.py
    
    if len(sys.argv) != 4:
        print("[Error] Proper usage: python stitching2.py <image-1-path> <image-2-path> <H12-path>")
        sys.exit(1)

    img1_path = sys.argv[1]
    img2_path = sys.argv[2]
    h_npy_path = sys.argv[3]
    
    image1 = cv2.imread(img1_path)
    image2 = cv2.imread(img2_path)
    
    stitch_imgs(image1, image2, h_npy_path)
    
    
    