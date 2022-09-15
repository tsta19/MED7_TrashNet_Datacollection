# print(f"Renamed Image: {image} / {len(self.input_directory)}", end="\r")

from sys import builtin_module_names
from urllib.request import HTTPDigestAuthHandler
import cv2
import numpy as np

# Script Imports
from ImageOperator import *

imageOperator = ImageOperator()

# Kmeans color segmentation
def kmeans_color_quantization(image, clusters=8, rounds=1):
    h, w = image.shape[:2]
    samples = np.zeros([h*w,3], dtype=np.float32)
    count = 0

    for x in range(h):
        for y in range(w):
            samples[count] = image[x][y]
            count += 1

    compactness, labels, centers = cv2.kmeans(samples,
            clusters, 
            None,
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10000, 0.0001), 
            rounds, 
            cv2.KMEANS_RANDOM_CENTERS)

    centers = np.uint8(centers)
    res = centers[labels.flatten()]
    return res.reshape((image.shape))

low_green = np.array([25, 115, 12])
high_green = np.array([102, 255, 255])

# Load image and perform kmeans
image = cv2.imread('images/water_bottle.png')
rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
resized_image = imageOperator.resize_image(rgb_image, 400, 500)
original = resized_image.copy()
imgHSV = cv2.cvtColor(original, cv2.COLOR_BGR2HSV)
kmeans = kmeans_color_quantization(imgHSV, clusters=4)
outmasked_image = imageOperator.remove_mask_from_image(resized_image, low_green, high_green)

# Convert to grayscale, Gaussian blur, adaptive threshold
gray = cv2.cvtColor(kmeans, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (3,3), 0)
thresh = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,21,2)

nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(thresh, connectivity=4)
sizes = stats[:, -1]

max_label = 1
max_size = sizes[1]
for i in range(2, nb_components):
    if sizes[i] > max_size:
        max_label = i
        max_size = sizes[i]

img2 = np.zeros(output.shape)
img2[output == max_label] = 255

#cv2.imshow("Object Isolation", img2)

subtraction = thresh - img2
cv2.imshow('image', resized_image)
cv2.imshow('thresh', thresh)
#cv2.imshow('image_org', image)
#cv2.imshow('subtraction', subtraction)
cv2.imshow('Masked Image', outmasked_image)
cv2.waitKey()