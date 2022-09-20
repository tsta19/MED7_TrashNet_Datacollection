import cv2
import numpy as np
import pandas as pd
import skimage
import matplotlib.pyplot as plt
from skimage.io import imread, imshow
from skimage.color import rgb2gray, rgb2hsv
from skimage.measure import label, regionprops, regionprops_table
from skimage.filters import threshold_otsu
from scipy.ndimage import median_filter
from matplotlib.patches import Rectangle
from skimage.morphology import area_closing
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



soda = cv2.imread('someTrashPics/soda.jpg')
sodaT = cv2.imread('someTrashPics/sodaT.jpg')

cv2.imshow('sad',sodaT)
sodavand = maskOff(soda,sodaT)


sodaNew = np.zeros((sodaT.shape[0],sodaT.shape[1],3),np.uint8)



for y in range(sodaT.shape[0]):
    for x in range(sodaT.shape[1]):
        if sodaT[y][x][0] == 0:
            sodaNew[y][x] = 255
        else:
            sodaNew[y][x] = 0
f = area_closing(sodaNew,64,1)
tree_blobs = label(rgb2gray(sodaT) > 0.5)
imshow(tree_blobs, cmap = 'tab10')
plt.show()



properties =['area','bbox','convex_area','bbox_area',
             'major_axis_length', 'minor_axis_length',
             'eccentricity']
df = pd.DataFrame(regionprops_table(tree_blobs, properties = properties))

print(df.to_string())

sodaXY1 = (df['bbox-1'][3],df['bbox-0'][3])
sodaXY2 = (df['bbox-3'][3],df['bbox-2'][3])


cv2.rectangle(soda,sodaXY1,sodaXY2,(255,0,0))

cv2.imshow('soda', sodavand)

cv2.imshow('redeufnewuifn',soda)

cv2.waitKey(0)