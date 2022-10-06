import cv2
import numpy as np
from skimage.io import imread, imshow
from skimage.color import rgb2gray, rgb2hsv
from skimage.measure import label, regionprops, regionprops_table
from skimage.filters import threshold_otsu
from scipy.ndimage import median_filter
from matplotlib.patches import Rectangle
from skimage.morphology import area_closing
from skimage.morphology import area_opening
import pandas as pd
import skimage
import matplotlib.pyplot as plt

image = cv2.imread('data/kop.png')
hsv = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
roi = hsv[100:600,0:1440]
threshold = 15
blur = cv2.medianBlur(roi,3)

avgHue = np.average(blur[:,:,0])
avgSat = np.average(blur[:, :, 1])
avgVal = np.average(blur[:, :, 2])

avgHueHigh = avgHue + threshold
avgSatHigh = avgSat + threshold
avgValHigh = avgVal + threshold
avgHueLow = avgHue - threshold
avgSatLow = avgSat - threshold
avgValLow = avgVal - threshold

range = cv2.inRange(blur,(avgHueLow,0,0),(avgHueHigh,255,255))
threshImg = cv2.bitwise_not(range)
f = area_opening(threshImg, 300, 1)

cv2.imshow('rekd',f)

cv2.waitKey(0)

