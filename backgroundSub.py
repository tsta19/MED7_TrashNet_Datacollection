import cv2
import numpy as np


vid = cv2.VideoCapture('samplevideo.mp4')

foreground = cv2.imread('foreground.png')
background = cv2.imread('background.png')

kernel = np.ones((5,5))

fgResize = cv2.resize(foreground,(250,250))
bgResize = cv2.resize(background,(250,250))

blurFG = cv2.blur(fgResize,(3,3))
blurBG = cv2.blur(bgResize,(3,3))


skrald = cv2.imread('Skrald.jpg')
ingenSkrald = cv2.imread('IngenSkrald.jpg')
graySkrald = cv2.cvtColor(skrald, cv2.COLOR_BGR2GRAY)
grayIngenSkrald = cv2.cvtColor(ingenSkrald, cv2.COLOR_BGR2GRAY)

xSkrald = skrald.shape[1]
ySkrald = skrald.shape[0]

subtracted2 = np.empty

subtracted = np.zeros((ySkrald, xSkrald) ,dtype=float, order = 'C')

def backgroundSubtractor(foreground_image, background_image):
    fg_image = np.copy(foreground_image)
    bg_image = np.copy(background_image)

    bgsub = cv2.createBackgroundSubtractorMOG2()

    addBG = bgsub.apply(bg_image)
    fgMask = bgsub.apply(fg_image)

    return fgMask

subtracted2 = backgroundSubtractor(blurFG,blurBG)

kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))

subtracted2 = cv2.morphologyEx(subtracted2, cv2.MORPH_OPEN, kernel2)

for y in range(subtracted2.shape[0]):
    for x in range(subtracted2.shape[1]):
        if subtracted2[y][x] == 127:
            subtracted2[y][x] = 255

cv2.imshow('yo',subtracted2)


hit = np.array([[255,255,255],
                [255,255,255],
                [255,255,255]])

print(subtracted2)

def erotion(treshimage, num):
    newImg = np.zeros((treshimage.shape[0],treshimage.shape[1],3),np.uint8)

    for i in range(num):
        for y in range(treshimage.shape[0]):
            for x in range(treshimage.shape[1]):
                crop = treshimage[y:y+3,x:x+3]
                cropArray = np.array(crop)
                if np.all(hit == cropArray):
                    newImg[y][x] = 255
    return newImg


def dialate(treshimage):
    newImg = np.zeros((treshimage.shape[0], treshimage.shape[1], 3), np.uint8)

    for y in range(treshimage.shape[0]):
        for x in range(treshimage.shape[1]):
            crop = treshimage[y:y+3,x:x+3]
            cropArray = np.array(crop)
            if np.any(hit == cropArray) and x > 1 and y > 1 and x < treshimage.shape[1]-1 and y < treshimage.shape[1]-1:
                newImg[y][x] = 255
                newImg[y][x+1] = 255
                newImg[y+1][x+1] = 255
                newImg[y+1][x] = 255
                newImg[y-1][x] = 255
                newImg[y][x-1] = 255
                newImg[y-1][x-1] = 255
    return newImg

erotion2 = erotion(subtracted2,1)
erotion3 = erotion(erotion2,1)
erotion4 = erotion(erotion3,1)
erotion5 = erotion(erotion4,1)
erotion6 = erotion(erotion5,1)
dialate2 = dialate(erotion6)
dialate3 = dialate(dialate2)
dialate4 = dialate(dialate3)

print(dialate3)

def maskOff(inputImg,mask):
    newImg = np.zeros((inputImg.shape[0], inputImg.shape[1], 3), np.uint8)
    for y in range(inputImg.shape[0]):
        for x in range(inputImg.shape[1]):
            if mask[y][x][0] == 255:
                newImg[y][x] = inputImg[y][x]
            else:
                newImg[y][x] = 0
    return newImg

getMasked = maskOff(fgResize,dialate4)

cv2.imshow('sub',erotion4)
cv2.imshow('dialate',dialate3)
cv2.imshow('mask', getMasked)


cv2.waitKey(0)