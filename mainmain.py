import cv2 as cv
import numpy as np
import images.kmeans
from skimage.feature import hog

def compareMovement(frame, frames):
    frames.append(frame)
    if len(frames) > 1:
        frameDiff = cv.subtract(frames[len(frames) - 2], frames[len(frames) - 1])
        return frameDiff

    elif len(frames) > 10:
        frameDiff = cv.subtract(frames[len(frames) - 9], frames[len(frames) - 1])

    else:
        return


def regionCannyEdge(frame):
    canny = cv.Canny(frame, threshold1=100, threshold2= 200)

    return canny

def hoggers(frameGray):
    #frameGray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    hogGrad, hogIMG = hog(frameGray, orientations=8, pixels_per_cell=(16, 16),
                    cells_per_block=(1, 1), visualize=True, channel_axis=-1)



    #print(hogGrad)

    return hogGrad, hogIMG



if __name__ == '__main__':
    cap = cv.VideoCapture('data/GL010014.MP4')
    print("open  = ", cap.isOpened())
    frames = []
    while True:

        ret, frame = cap.read()

        diff = compareMovement(frame, frames)

        cv.imshow('video', frame)
        if diff is not None:
            cv.imshow('diff', diff)
            print('shape:', diff.shape)
            hogGradient, hogIMG = hoggers(diff)
            cv.imshow('hogIMG', hogIMG)
        canny = regionCannyEdge(frame)
        cv.imshow('canny', canny)



        # kmeans = images.kmeans.kMeansSegmentation(frame)
        # cv.imshow('kmeans')

        keyboard = cv.waitKey(30)
        if keyboard == 'q' or keyboard == 27:
            break
