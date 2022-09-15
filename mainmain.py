import cv2 as cv
import numpy as np


def compareMovement(frame, frames):
    frames.append(frame)
    if len(frames) > 1:
        frameDiff = cv.subtract(frames[len(frames) - 2], frames[len(frames) - 1])
        return frameDiff

    elif len(frames) > 10:
        frameDiff = cv.subtract(frames[len(frames) - 9], frames[len(frames) - 1])

    else:
        return






if __name__ == '__main__':
    cap = cv.VideoCapture('data/samplevideo.mp4')
    print("open  = ", cap.isOpened())
    frames = []
    while True:

        ret, frame = cap.read()

        diff = compareMovement(frame, frames)

        cv.imshow('video', frame)
        if diff is not None:
            cv.imshow('diff', diff)
        keyboard = cv.waitKey(30)
        if keyboard == 'q' or keyboard == 27:
            break
