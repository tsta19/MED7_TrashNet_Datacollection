from re import A
from telnetlib import AYT
from turtle import left
import cv2 as cv
import numpy as np
import images.kmeans
#from skimage.feature import hog
#from skimage.feature import match_template
import sys
import math




np.set_printoptions(threshold=sys.maxsize)


def compareMovement(frame, frames):
    frames.append(frame)
    if len(frames) > 1:
        frameDiff = cv.subtract(frames[len(frames) - 2], frames[len(frames) - 1])
        return frameDiff

    else:
        return


def regionCannyEdge(frame):
    canny = cv.Canny(frame, threshold1=50, threshold2=200, apertureSize = 3, L2gradient = True)

    return canny


# def hoggers(frameGray):
#     # frameGray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
#     hogGrad, hogIMG = hog(frameGray, orientations=8, pixels_per_cell=(16, 16),
#                           cells_per_block=(1, 1), visualize=True, channel_axis=-1)

#     # print(hogGrad)

#     return hogGrad, hogIMG


# def templateTracking(template, frame):
#     templateimage = match_template(frame, template, pad_input=False)

#     return templateimage


def tomasiTacking(frame):
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    corners = cv.goodFeaturesToTrack(gray, 30, 0.05, 15)
    corners = np.int0(corners)
    x_list = []
    y_list = []

    for i in corners:
        x, y = i.ravel()
        cv.circle(frame, (x, y), 4, 255, 4)
        x_list.append(x)
        y_list.append(y)


    return x_list, y_list


def edgeDiffTracking(frame):

    cannyImg = regionCannyEdge(frame)
    # print(canny.shape)
    x, y = tomasiTacking(frame)
    y1, x1 = np.where(cannyImg > 0)

    x, y, x1, y1 = np.asarray(x), np.asarray(y), np.asarray(x1), np.asarray(y1)

    #print(f'x, y, x1, y1: {len(x)}, {len(y)}, {len(x1)}, {len(y1)}')

    for i in range(len(x1)):
        for w in range(len(x)):
            if x[w] == x1[i] and y[w] == y1[i]:
                cv.circle(frame, (x[w], x[w]), 10, (255, 255, 255), 10)
                #cv.imwrite(f'markedImg/cirlcedImage{i}.png', frame)
                #print('Img taken')




    # for i in range(0, len(x1)):
    #     if y1[i] == y[i] and x1[i] == x[i]:
    #         cv.circle(frame, (y1[i], x1[i]), 10, (255, 255, 255), 10)
    #         cv.imwrite(f'markedImg/cirlcedImage{i}.png', frame)
    #         print('Img taken')

    # if diff[y, x] != 0 and canny[y, x] != 0:
    #     cv.circle(diff, (y, x), 5)

    cv.imshow('diffCircles', frame)
    return cannyImg

def regionBasedDetection(frame, leftArrX, leftArrY):
    leftRegion = int(frame.shape[1]*0.50)
    rightRegion = int(frame.shape[1])
    
    
    canny = regionCannyEdge(frame)
    frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    frame = cv.GaussianBlur(frame, (5,5), 0)
    
    cv.imshow('framegray', frame)
    SobelX = cv.Sobel(frame, ddepth=cv.CV_8U, dx=1, dy=0, ksize=5)
    SobelY = cv.Sobel(frame, ddepth=cv.CV_8U, dx=0, dy=1, ksize=5)


    # cv.rectangle(canny, (0, 0), (leftRegion, frame.shape[0]), (255, 0, 0), 3)
    # cv.rectangle(canny, (leftRegion, 0), (rightRegion, frame.shape[0]), (255), 3)

    leftRegionCannyIMG = canny[0:canny.shape[0], 0:int(canny.shape[1]/2)]
    rightRegionCannyIMG = canny[0:canny.shape[0], int(canny.shape[1]/2):canny.shape[1]]
    cv.imshow('leftcanny', leftRegionCannyIMG)
    cv.imshow('rightcanny', rightRegionCannyIMG)

    leftRegionIMG = frame[0:canny.shape[0], 0:int(canny.shape[1]/2)]
    rightRegionIMG = frame[0:canny.shape[0], int(canny.shape[1]/2):canny.shape[1]]
    cv.imshow('left', leftRegionIMG)
    cv.imshow('right', rightRegionIMG)

    rightRegSobelX = SobelY[0:canny.shape[0], int(canny.shape[1]/2):canny.shape[1]]
    rightRegSobelY = SobelX[0:canny.shape[0], int(canny.shape[1]/2):canny.shape[1]]

    leftRegSobelX = SobelX[0:canny.shape[0], 0:int(canny.shape[1]/2)]
    leftRegSobelY = SobelY[0:canny.shape[0], 0:int(canny.shape[1]/2)]

    leftArrX.append(leftRegSobelX)
    leftArrY.append(leftRegSobelY)

    OFimgarray = np.zeros_like(leftRegSobelX)
    if len(leftArrX) > 1:
        Ax = leftArrX[0]
        Bx = leftArrX[1]
        Cx = cv.subtract(Bx, Ax)

        Ay = leftArrY[0]
        By = leftArrY[1]
        Cy = cv.subtract(By, Ay)

        cv.imshow('cx', Cx)
        cv.imshow('cy', Cy)

        for i in range(0, Ax.shape[0]):
            for j in range(0, Ax.shape[1]):
                gradientMatrix = [[(Ax[i][j])**2, (Ax[i][j])*(Ay[i][j])], [(Ax[i][j])*(Ay[i][j]), (Ay[i][j])**2]]

                #invGradMatrix = np.linalg.inv(gradientMatrix)
                changeMatrix = [-((Ax[i][j])*(Cx[i][j])),-((Ay[i][j])*(Cy[i][j]))]
                
                OFvector = np.matmul(gradientMatrix,changeMatrix)
                #print('of', OFvector)
                ##OFimgarray[i][j] = OFvector
        
        leftArrX.clear()
        leftArrY.clear()
        return canny

    cv.waitKey(0)
    
    return canny


if __name__ == '__main__':
    #cap = cv.VideoCapture('data/trimVideo.mp4')
    cap = cv.VideoCapture('E:/MED-local/MED7/MED7_TrashNet_Datacollection/data/trimVideo.mp4')
    templateIMG = cv.imread('data/template.png')
    print("open  = ", cap.isOpened())
    frames = []
    rightRegionFrames = []
    leftRegionFrames = []
    while True:

        ret, frame = cap.read()
        croppedIMG = frame[int(frame.shape[0] * 0.40):frame.shape[0], 0:frame.shape[1]]

        diff = compareMovement(croppedIMG, frames)
        regioIMG = regionBasedDetection(croppedIMG, leftRegionFrames, rightRegionFrames)
        
        #v.imshow('region', regioIMG)


        

        #imggg = edgeDiffTracking(croppedIMG)

        if diff is not None:
            # print('shape:', diff.shape)
            # hogGradient, hogIMG = hoggers(diff)
            # cv.imshow('hogIMG', hogIMG)
            # print(hogGradient.shape)
            # print(len(hogGradient))
            # np.savetxt('test.out', hogGradient, delimiter=',')
            # np.savetxt('testimg.out', hogIMG, delimiter=',')
            pass

        canny = regionCannyEdge(croppedIMG)
        # cv.imshow('canny', canny)
        # cv.imshow('video', croppedIMG)
        # kmeans = images.kmeans.kMeansSegmentation(frame)
        # cv.imshow('kmeans')

        

        keyboard = cv.waitKey(30)
        if keyboard == 'q' or keyboard == 27:
            break

       
