import cv2
import numpy as np
import matplotlib as plt

path = 'foreground.png'


def kMeansSegmentation(frame):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    twoDImage = img.reshape((-1, 3))
    twoDImage = np.float32(twoDImage)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 3
    attempts = 10

    ret, label, center = cv2.kmeans(twoDImage, K, None, criteria, attempts, cv2.KMEANS_PP_CENTERS)
    center = np.uint8(center)
    res = center[label.flatten()]
    result_image = res.reshape((img.shape))


    return result_image

#
# for y in range(result_image.shape[0]):
#     for x in range(result_image.shape[1]):
#         if result_image[y][x][0] == 52:
#             newImg[y][x] = 255
#
# hit = np.array([[255,255,255],
#                 [255,255,255],
#                 [255,255,255]])


def erotion(treshimage):
    newImg1 = np.zeros((treshimage.shape[0],treshimage.shape[1],3),np.uint8)

    for y in range(treshimage.shape[0]):
        for x in range(treshimage.shape[1]):
            crop = treshimage[y:y+3,x:x+3]
            cropArray = np.array(crop)
            if np.all(hit == cropArray):
                newImg1[y][x] = 255
    return newImg1


def dialate(treshimage):
    newImg = np.zeros((treshimage.shape[0], treshimage.shape[1], 3), np.uint8)

    for y in range(treshimage.shape[0]):
        for x in range(treshimage.shape[1]):
            crop = treshimage[y:y+3,x:x+3]
            cropArray = np.array(crop)
            if np.any(hit == cropArray) and x > 1 and y > 1 and x < treshimage.shape[1]-1 and y < treshimage.shape[0]-1:
                newImg[y][x] = 255
                newImg[y][x+1] = 255
                newImg[y+1][x+1] = 255
                newImg[y+1][x] = 255
                newImg[y-1][x] = 255
                newImg[y][x-1] = 255
                newImg[y-1][x-1] = 255
    return newImg

# eroted = erotion(newImg)
#
# print(newImg.shape[0])
# print(newImg.shape[1])
#
# print(eroted.shape[0])
# print(eroted.shape[1])
#
#
# dialated = dialate(eroted)
# cv2.imshow('eroted', dialated)


def maskOff(inputImg,mask):
    newImg1 = np.zeros((inputImg.shape[0], inputImg.shape[1], 3), np.uint8)
    for y in range(inputImg.shape[0]):
        for x in range(inputImg.shape[1]):
            if mask[y][x][0] == 255:
                newImg1[y][x] = inputImg[y][x]
            else:
                newImg1[y][x] = 0
    return newImg1

# maskOff1 = maskOff(img2,eroted)
#
# cv2.imshow('maskoff',maskOff1)
# cv2.imshow('mask', newImg)
#
# cv2.imshow('k-means', result_image)
#
# cv2.waitKey(0)