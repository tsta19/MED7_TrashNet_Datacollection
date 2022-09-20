import cv2
import numpy as np
import matplotlib as plt
import pandas as pd
import skimage
from skimage.io import imread, imshow
from skimage.color import rgb2gray, rgb2hsv
from skimage.measure import label, regionprops, regionprops_table
from skimage.filters import threshold_otsu
from scipy.ndimage import median_filter
from matplotlib.patches import Rectangle
from tqdm import tqdm

def maskOff(inputImg,mask):
    newImg1 = np.zeros((inputImg.shape[0], inputImg.shape[1], 3), np.uint8)
    for y in range(inputImg.shape[0]):
        for x in range(inputImg.shape[1]):
            if mask[y][x][0] == 0:
                newImg1[y][x] = inputImg[y][x]
            else:
                newImg1[y][x] = 0
    return newImg1

soda = cv2.imread('soda.jpg')
sodaT = cv2.imread('sodaT.jpg')
blueshit = cv2.imread('blueshit.jpg')
blueshitT = cv2.imread('blueshitT.jpg')
brownshit = cv2.imread('brownshit.jpg')
brownshitT = cv2.imread('brownshitT.jpg')

sodavand = maskOff(soda,sodaT)
blo = maskOff(blueshit,blueshitT)
brun = maskOff(brownshit,brownshitT)

tree_blobs = label(rgb2gray(sodavand) > 0)
imshow(tree_blobs, cmap = 'tab10')


cv2.imshow('soda', sodavand)
cv2.imshow('blo', blo)
cv2.imshow('brun', brun)

cv2.waitKey(0)