import cv2 as cv
import numpy as np
import images.kmeans


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





if __name__ == '__main__':
    cap = cv.VideoCapture('data/GL010012.MP4')
    print("open  = ", cap.isOpened())
    frames = []
    while True:

        ret, frame = cap.read()

        diff = compareMovement(frame, frames)

        cv.imshow('video', frame)
        if diff is not None:
            cv.imshow('diff', diff)
        canny = regionCannyEdge(frame)
        cv.imshow('canny', canny)

        kmeans = images.kmeans.kMeansSegmentation(frame)
        cv.imshow('kmeans', kmeans)

        keyboard = cv.waitKey(30)
        if keyboard == 'q' or keyboard == 27:
            break
