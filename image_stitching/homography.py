# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 13:29:33 2023

@author: matthew
"""
"""
USE:
    python homography.py keble_a.jpg keble_b.jpg "C:/Users/matth/.spyder-py3/491 HW3/H_12"

"""
import cv2
import random
import numpy as np
import sys
import matplotlib.pyplot as plt
from numpy.linalg import svd

def ransac(im1_pts, im2_pts, threshold, iterations):
    best_H = None
    max_inliers = []

    # Loop [iterations] times to see what gives best H
    for i in range(iterations):
        samp_idx = np.random.choice(im1_pts.shape[1], 4, replace=False)
        samp_im1 = im1_pts[:, samp_idx]
        samp_im2 = im2_pts[:, samp_idx]

        # Estimate a H w/ the sampled matches
        H = calc_homography(samp_im1, samp_im2)

        # Calculate the re-proj error for all points
        h_im1_pts = np.vstack((im1_pts, np.ones((1, im1_pts.shape[1]))))
        proj_pts = np.dot(H, h_im1_pts)

        # Normalize coordinates
        proj_pts /= proj_pts[2, :]

        # Calc dist bet. img1 and img2
        error = np.linalg.norm(proj_pts[:2, :] - im2_pts, axis=0)

        # Count inliers within thresh + get idx of inliers!
        inliers = np.where(error < threshold)[0]

        # Update best H if the curr has more inliers (use ALL inliers)
        if len(inliers) > len(max_inliers):
            im1_inlier_points = im1_pts[:, inliers]
            im2_inlier_points = im2_pts[:, inliers]
            best_H = calc_homography(im1_inlier_points, im2_inlier_points)
            max_inliers = inliers
    
    return best_H, max_inliers


def calc_homography(im1_pts, im2_pts):
    """
    save_path:  where we save output
    output:     homography matrix of two imgs saved as .npy
    """
    # Construct A for: Ah = 0 (A = 3x3 mat)
    A = np.zeros((im1_pts.shape[1] * 2, 9)) 
    for i in range(im1_pts.shape[1]): # populate like Kriegman, odd then even between im1-2 matches
        A[2 * i, :] = [-im1_pts[0, i], -im1_pts[1, i], -1, 0, 0, 0, im1_pts[0, i] * im2_pts[0, i], im1_pts[1, i] * im2_pts[0, i], im2_pts[0, i]]
        A[2 * i + 1, :] = [0, 0, 0, -im1_pts[0, i], -im1_pts[1, i], -1, im1_pts[0, i] * im2_pts[1, i], im1_pts[1, i] * im2_pts[1, i], im2_pts[1, i]]

    # Solve Ah=0 with SVD (take V since it has right shape)
    U, S, V = svd(A)
    H = V[-1, :].reshape((3, 3))

    return H


def feat_match(img1, ref_img, save_name):
    """
    im_pair:    pair of imgs to calc features from and est. homography
    NOTE:       2ND image in the pair is REF IMAGE
    """
    # Use SIFT to detect and describe features
    sift = cv2.SIFT_create()
    kp_1, des_1 = sift.detectAndCompute(img1, None)
    kp_2, des_2 = sift.detectAndCompute(ref_img, None)
        
    """show feat in img1"""
    # cv2.imshow("img1", img1)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # feat_img1 = []
    # cv2.drawKeypoints(img1,kp_1,img1)
    # cv2.imshow("img1 feat.", img1)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
    # Use matcher to find matches
    matcher = cv2.BFMatcher()
    matches = matcher.knnMatch(des_1, des_2, k=2)
    
    # Use ratio to threshold
    keep_matches = []
    for m_1, m_2 in matches:
        # Check ratio distance is <20% bet. 1st & 2nd match, if so "good" match
        # 0.2 keeps good amount of matches; (Brown, et al) chooses ratio as 0.1-0.2 for best results
        if (m_1.distance/m_2.distance) < (0.5): 
            keep_matches.append(m_1)

    # Show tentative correspondences (FLAGS option shows inlier-matches only)
    img_matches = cv2.drawMatches(img1, kp_1, ref_img, kp_2, keep_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    # cv2.imshow("[Tentative] Correspondences", img_matches)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
    # Get points (2xN) from matches (queryIdx & trainIdx = matched kps idx in each img)
    im1_pts = np.array([kp_1[m.queryIdx].pt for m in keep_matches]).T
    im2_pts = np.array([kp_2[m.trainIdx].pt for m in keep_matches]).T
    
    # Get (4) points from these matches (2x4)
    four_im1_pts = np.array([item for row in im1_pts for item in row[:4]]).reshape((2,4))
    four_im2_pts = np.array([item for row in im2_pts for item in row[:4]]).reshape((2,4))
        
    # Calc initial homography from n=4 matches
    H = calc_homography(four_im1_pts, four_im2_pts)
    
    # RANSAC and pick BEST homography 
    final_H, max_inliers = ransac(im1_pts, im2_pts, 1, 1000)
        
    if (final_H is None):   # No better H computed
        print("[Same H] Corner case, likely not trigger-able")
        return H
    
    """Display new inliers img using final_H"""
    # final_matches = [keep_matches[idx] for idx in max_inliers]
    # img_matches = cv2.drawMatches(img1, kp_1, ref_img, kp_2, final_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    # cv2.imshow("[Final] Correspondences", img_matches)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
    # Save homography matrix as .npy file (load with np.load)
    np.save(save_name, final_H)
        
    return final_H
    

if __name__ == "__main__":
    # Official Submission script ...
    
    if len(sys.argv) != 4:
        print("[Error] Proper usage: python homography.py <image-1-path> <image-2-path> <H-path>")
        sys.exit(1)

    img1_path = sys.argv[1]
    img2_path = sys.argv[2]
    save_path = sys.argv[3]
    
    image1 = cv2.imread(img1_path)
    image2 = cv2.imread(img2_path)
    feat_match(image1, image2, save_path)

    