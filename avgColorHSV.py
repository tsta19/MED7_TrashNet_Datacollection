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


def hsvTresh(frame, treshold=20):
    imageArray = []
    if len(imageArray) < 5:
        imageArray.append(frame)
        avg_image = imageArray[0]
        avg_image[:, :, 1] = 180
        avg_image[:, :, 2] = 180
        for i in range(len(imageArray)):
            alpha = 1.0 / (i + 1)
            beta = 1.0 - alpha
            avg_image = cv2.addWeighted(imageArray[i], alpha, avg_image, beta, 0.0)

    else:
        imageArray.pop(0)



    blur = cv2.medianBlur(frame, 3)
    blur = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

    blur[:, :, 1] = 180
    blur[:, :, 2] = 180

    avgHue = np.average(avg_image[:, :, 0])

    avgHueHigh = avgHue + treshold
    avgHueLow = avgHue - treshold

    frame_treshold = cv2.inRange(blur, (avgHueLow, 0, 0), (avgHueHigh, 180, 180))
    threshInv = cv2.bitwise_not(frame_treshold)
    f = area_opening(threshInv, 256, 1)

    return f


if __name__ == '__main__':
    cap = cv2.VideoCapture('data/GL010020.mp4')
    imgArray = []
    treshold = 15
    counter = 0
    ret, imgFrame = cap.read()

    properties = ['area', 'bbox', 'bbox_area']

    # newImg = np.zeros((imgFrame.shape[0],imgFrame.shape[1],3),np.uint8)

    while cap.isOpened():
        ret, frame = cap.read()
        ret2, frame2 = cap.read()
        if not ret:
            print("Cant read video")
            break
        # cv2.imshow('frame',frame)
        hsv = cv2.cvtColor(frame2, cv2.COLOR_BGR2HSV)
        roi = frame[0:400, 300:560]
        roi2 = hsv[0:400, 300:560]

        if cv2.waitKey(1) == ('q'):
            break
        if len(imgArray) < 5:
            imgArray.append(roi2)
            avg_image = imgArray[0]
            for i in range(len(imgArray)):
                alpha = 1.0 / (i + 1)
                beta = 1.0 - alpha
                avg_image = cv2.addWeighted(imgArray[i], alpha, avg_image, beta, 0.0)
        else:
            imgArray.pop(0)
        blur = cv2.medianBlur(roi2, 3)

        avgHue = np.average(avg_image[:, :, 0])
        avgSat = np.average(avg_image[:, :, 1])
        avgVal = np.average(avg_image[:, :, 2])

        avgHueHigh = avgHue + treshold
        avgSatHigh = avgSat + treshold
        avgValHigh = avgVal + treshold
        avgHueLow = avgHue - treshold
        avgSatLow = avgSat - treshold
        avgValLow = avgVal - treshold

        counter += 1
        print(counter)
        print("Hue: " + str(avgHue))
        print("sat: " + str(avgSat))
        print("val: " + str(avgVal))

        frame_treshold = cv2.inRange(roi2, (avgHueLow, 0, 0), (avgHueHigh, 255, 255))
        threshInv = cv2.bitwise_not(frame_treshold)
        tree_blobs = label(threshInv > 0)
        df = pd.DataFrame(regionprops_table(tree_blobs, properties=properties))
        for i in range(len(df['bbox_area'])):
            if df['bbox_area'][i] == max(df['bbox_area']):
                sodaXY1 = (df['bbox-1'][i], df['bbox-0'][i])
                sodaXY2 = (df['bbox-3'][i], df['bbox-2'][i])
                cv2.rectangle(roi, sodaXY1, sodaXY2, (255, 0, 0))

        f = area_opening(threshInv, 256, 1)
        cv2.imshow('avg', avg_image)
        cv2.imshow('tresh', f)
        cv2.imshow('roi', roi)
        # cv2.imshow('roi', roi)
