import cv2
import numpy as np
import pandas as pd
import skimage
from skimage.io import imread, imshow
from skimage.color import rgb2gray, rgb2hsv
from skimage.measure import label, regionprops, regionprops_table
from skimage.filters import threshold_otsu
from scipy.ndimage import median_filter
from matplotlib.patches import Rectangle
from tqdm import tqdm

cap = cv2.VideoCapture('trimVideo.mp4')
imgArray = []
treshold = 20
counter = 0
ret, imgFrame = cap.read()

newImg = np.zeros((imgFrame.shape[0],imgFrame.shape[1],3),np.uint8)


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Cant read video")
        break
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    #cv2.imshow('frame',frame)
    roi = frame[0:850, 270:620]
    cv2.imshow('roi',roi)
    if cv2.waitKey(1) == ('q'):
        break
    if len(imgArray) < 10:
        imgArray.append(roi)
        avg_image = imgArray[0]
        for i in range(len(imgArray)):
            alpha = 1.0 / (i + 1)
            beta = 1.0 - alpha
            avg_image = cv2.addWeighted(imgArray[i], alpha, avg_image, beta, 0.0)
    else:
        imgArray.pop(0)
    blur = cv2.medianBlur(roi,7)

    avgBlue = np.average(avg_image[:,:,0])
    avgGreen = np.average(avg_image[:, :, 1])
    avgRed = np.average(avg_image[:, :, 2])

    avgBlueHigh = avgBlue + treshold
    avgGreenHigh = avgGreen + treshold
    avgRedHigh = avgRed + treshold
    avgBlueLow = avgBlue - treshold
    avgGreenLow = avgGreen - treshold
    avgRedLow = avgRed - treshold

    counter += 1
    print(counter)
    if counter == 50:
        cv2.imwrite('sodaT.jpg', frame_treshold)
        cv2.imwrite('soda.jpg', roi)
    if counter == 570:
        cv2.imwrite('clothT.jpg', frame_treshold)
        cv2.imwrite('cloth.jpg', roi)
    if counter == 850:
        cv2.imwrite('brownshitT.jpg', frame_treshold)
        cv2.imwrite('brownshit.jpg', roi)
    if counter == 1240:
        cv2.imwrite('blueshitT.jpg', frame_treshold)
        cv2.imwrite('blueshit.jpg', roi)


    frame_treshold = cv2.inRange(blur,(avgBlueLow,avgGreenLow,avgRedLow),(avgBlueHigh,avgGreenHigh,avgRedHigh))

    cv2.imshow('avg', avg_image)
    cv2.imshow('tresh', frame_treshold)



cap.release()
cv2.destroyAllWindows()