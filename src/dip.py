#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 29 10:48:19 2018

@author: tvieira
"""
#%% Import modules
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

folder = '../db/'
plt.rcParams['figure.figsize'] = [20, 15]

#%% Define functions
def doNothing(x):
    pass

def plotContour(cont):
    b = np.array(cont)
    x = b[:, 0, 0]
    y = b[:, 0, 1]
    return x, y

def compute_piecewise_linear_val(val, r1, s1, r2, s2):
    output = 0
    if (0 <= val) and (val <= r1):
        output = (s1 / r1) * val
    if (r1 <= val) and (val <= r2):
        output = ((s2 - s1) / (r2 - r1)) * (val - r1) + s1
    if (r2 <= val) and (val <= 1):
        output = ((1 - s2) / (1 - r2)) * (val - r2) + s2
    return output

def bgr2rgb(bgr):
    if len(bgr.shape) < 3:
        return bgr
    rgb = np.zeros(bgr.shape, dtype=bgr.dtype)
    cv2.mixChannels([bgr], [rgb], [0, 2, 1, 1, 2, 0])
    return rgb

def createSaltAndPepperNoise(height = 100, width = 100, 
                             salt_prob = 0.05, pepper_prob = 0.05):
    """ Returns an image \in [-1, 1] containing salt (I = 1.0) and 
    pepper (I = -1.0) noise with respective probability distributions
    equal salt_prob and pepper_prob. Pixels without noise have values of 0.5.
    """
    img = np.zeros((height, width), np.float64)
    noise = np.random.rand(img.shape[0], img.shape[1])
    img[noise > 1 - salt_prob] = 1
    img[noise < pepper_prob] = -1
    return img

def addSaltAndPepperNoise2Img(img, salt_prob = 0.05, pepper_prob = 0.05):
    noise = createSaltAndPepperNoise(img.shape[0], img.shape[1],
                                     salt_prob, pepper_prob)
    img[noise == 1] = 255
    img[noise == -1] = 0    
    return img

def scaleImage2_uchar(src):
    tmp = np.copy(src)
    if src.dtype != np.float32:
        tmp = np.float32(tmp)
    cv2.normalize(tmp, tmp, 1, 0, cv2.NORM_MINMAX)
    tmp = np.uint8(255 * tmp)
    return tmp

def createWhiteDisk(height = 100, width = 100, xc = 50, yc = 50, rc = 20):
    disk = np.zeros((height, width), np.float64)
    for x in range(disk.shape[0]):
        for y in range(disk.shape[1]):
            if (x - xc) * (x - xc) + (y - yc) * (y - yc) <= rc * rc:
                disk[x][y] = 1.0
    return disk

def createWhiteDisk2(height = 100, width = 100, xc = 50, yc = 50, rc = 20):
    xx, yy = np.meshgrid(range(height), range(width))
    img = np.array(
            ( (xx - xc)**2 + (yy - yc)**2 - rc**2  ) < 0).astype('float64')
    return img

def createCosineImage(height, width, freq, theta):
    img = np.zeros((height, width), dtype=np.float64)
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            img[x][y] = np.cos(
                    2 * np.pi * freq * (x * np.cos(theta) - y * np.sin(theta)))
    return img

def createCosineImage2(height, width, freq, theta):
    img = np.zeros((height, width), dtype=np.float64)
    xx, yy = np.meshgrid(range(height), range(width))
    theta = np.deg2rad(theta)
    rho = (xx * np.cos(theta) - yy * np.sin(theta))
    img[:] = np.cos(2 * np.pi * freq * rho)
    return img

def createSineImage2(height, width, freq, theta):
    img = np.zeros((height, width), dtype=np.float64)
    xx, yy = np.meshgrid(range(height), range(width))
    theta = np.deg2rad(theta)
    rho = (xx * np.cos(theta) - yy * np.sin(theta))
    img[:] = np.sin(2 * np.pi * freq * rho)
    return img

def applyLogTransform(img):
    img2 = np.copy(img)
    img2 += 1
    img2 = np.log(img2)
    return img2

def create2DGaussian(rows = 100, 
                     cols = 100, 
                     mx = 50, 
                     my = 50, 
                     sx = 10, 
                     sy = 100,
                     theta = 0):
    """
    Create an image (rows x cols) with a 2D Gaussian with
    mx, my means in the x and y directions and standard deviations
    sx, sy respectively. The Gaussian can also be rotate of theta
    radians in clockwise direction.
    """
    
    xx0, yy0 = np.meshgrid(range(cols), range(rows))
    xx0 -= mx
    yy0 -= my
    theta = np.deg2rad(theta)
    xx = xx0 * np.cos(theta) - yy0 * np.sin(theta)
    yy = xx0 * np.sin(theta) + yy0 * np.cos(theta)
    try:
        img = np.exp( - ((xx**2)/(2*sx**2) + 
                         (yy**2)/(2*sy**2)) )
    except ZeroDivisionError:
        img = np.zeros((rows, cols), dtype='float64')
            
    cv2.normalize(img, img, 1, 0, cv2.NORM_MINMAX)
    return img

def compute_histogram_1C(src):
    # Compute the histograms:
    b_hist = cv2.calcHist([src], [0], None, [256], [0, 256], True, False)

    # Draw the histograms for B, G and R
    hist_w = 512
    hist_h = 400
    bin_w = np.round(hist_w / 256)

    histImage = np.ones((hist_h, hist_w), np.uint8)

    # Normalize the result to [ 0, histImage.rows ]
    cv2.normalize(b_hist, b_hist, 0, histImage.shape[0], cv2.NORM_MINMAX)

    # Draw for each channel
    for i in range(1, 256):
        cv2.line(histImage, (int(bin_w * (i - 1)), int(hist_h - np.round(b_hist[i - 1]))),
                 (int(bin_w * i), int(hist_h - np.round(b_hist[i]))), 255, 2, cv2.LINE_8, 0)
    return histImage, b_hist
