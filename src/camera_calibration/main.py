# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 15:18:03 2017

@author: tvieira
"""

#%% Import libraries
import numpy as np
import cv2
import glob

#%% termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

#%% prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
nwidth = 9
nheight = 6
objp = np.zeros((nheight*nwidth,3), np.float32)
objp[:,:2] = np.mgrid[0:nwidth,0:nheight].T.reshape(-1,2)

#%% Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

#%% Get the list of images
images = glob.glob('img/*.png')

#%% Find and display the corners
for fname0 in images:
    fname = fname0.replace('img/', '')
    img = cv2.imread('img/' + fname)
    print('Processing image ' + fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (nwidth,nheight),None)

    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)

        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        imgpoints.append(corners2)

        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, (nwidth, nheight), corners2,ret)
        cv2.imshow('img', img)
        cv2.imwrite('results/' + fname.replace('.png', '_points.png'), img)
        cv2.waitKey(500)

cv2.destroyAllWindows()

#%% Calibrate
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)

#%% Load some image to undistort
test_img = 'frame_12.png'
img = cv2.imread('img/' + test_img)
h, w = img.shape[:2]
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w, h)) 
dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
# crop the image
x,y,w,h = roi
dst = dst[y:y+h, x:x+w]
cv2.imwrite('results/' + test_img.replace('.png', '_corrected.png'),dst)

#%% Using remapping
## undistort
#mapx,mapy = cv2.initUndistortRectifyMap(mtx,dist,None,newcameramtx,(w,h),5)
#dst = cv2.remap(img,mapx,mapy,cv2.INTER_LINEAR)
## crop the image
#x,y,w,h = roi
#dst = dst[y:y+h, x:x+w]
#cv2.imwrite('calibresult.png',dst)

#%% Re-projection Error
mean_error = 0
tot_error = 0
for i in xrange(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv2.norm(imgpoints[i],imgpoints2, cv2.NORM_L2)/len(imgpoints2)
    tot_error += error

print "total error: ", mean_error/len(objpoints)
