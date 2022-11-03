import cv2
import numpy as np
import matplotlib as plt

frame = cv2.imread('data/savedimages/segmented2.png')

img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
twoDImage = img.reshape((-1, 3))
twoDImage = np.float32(twoDImage)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
K = 5
attempts = 10
treshold = 70

ret, label, center = cv2.kmeans(twoDImage, K, None, criteria, attempts, cv2.KMEANS_PP_CENTERS)
center = np.uint8(center)
res = center[label.flatten()]
result_image = res.reshape((img.shape))

avgBlue = np.average(result_image[:,:,0])
avgGreen = np.average(result_image[:, :, 1])
avgRed = np.average(result_image[:, :, 2])

avgBlueHigh = avgBlue + treshold
avgGreenHigh = avgGreen + treshold
avgRedHigh = avgRed + treshold
avgBlueLow = avgBlue - treshold
avgGreenLow = avgGreen - treshold
avgRedLow = avgRed - treshold

thresh = cv2.inRange(result_image,(avgBlueLow,avgGreenLow,avgRedLow),(avgBlueHigh,avgGreenHigh,avgRedHigh))
threshInv = cv2.bitwise_not(thresh)


print(ret)
cv2.imshow('kmeans', result_image)
cv2.imshow('thresh',threshInv)
cv2.waitKey(0)