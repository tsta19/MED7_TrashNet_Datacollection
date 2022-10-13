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

cap = cv2.VideoCapture('data/GL010014.MP4')
imgArray = []
treshold = 40
counter = 0
ret, imgFrame = cap.read()
template_left = cv2.imread('data/left_claw.png')
template_right = cv2.imread('data/right_claw.png')

properties =['area','bbox','bbox_area']




while cap.isOpened():
    ret, frame = cap.read()
    print(frame.shape[0])
    roix = frame.shape[1]
    roiy = frame.shape[0]
    roix2 = int(frame.shape[1] / 2)


    roi = frame[0:roiy, roix2:roix]


    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    grayRoi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    grayTempLeft = cv2.cvtColor(template_left,cv2.COLOR_BGR2GRAY)
    grayTempRight = cv2.cvtColor(template_right, cv2.COLOR_BGR2GRAY)
    result_left = cv2.matchTemplate(gray, grayTempLeft, cv2.TM_CCOEFF_NORMED)
    result_right = cv2.matchTemplate(grayRoi, grayTempRight, cv2.TM_CCOEFF_NORMED)
    (minValLeft, maxValLeft, minLocLeft, maxLocLeft) = cv2.minMaxLoc(result_left)
    (minValRight, maxValRight, minLocRight, maxLocRight) = cv2.minMaxLoc(result_right)
    (startXLeft, startYLeft) = maxLocLeft
    (startXRight, startYRight) = maxLocRight
    endXLeft = startXLeft + template_left.shape[1]
    endYLeft = startYLeft + template_left.shape[0]
    endXRight = startXRight + template_right.shape[1]
    endYRight = startYRight + template_right.shape[0]

    cv2.rectangle(frame, (startXLeft, startYLeft), (endXLeft, endYLeft), (255, 0, 0), 1)
    cv2.rectangle(frame, (startXRight + roix2, startYRight), (endXRight + roix2, endYRight), (0, 255, 0), 1)

    if not ret:
        print("Cant read video")
        break




    if cv2.waitKey(1) == ('q'):
        break
    cv2.imshow('output', frame)
    cv2.imshow('result',result_left)


cap.release()
cv2.destroyAllWindows()