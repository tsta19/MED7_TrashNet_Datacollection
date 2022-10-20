import imp
import cv2
import numpy as np
# from dense_optical_flow import *
from sparse_of import SparseOF
from scipy import ndimage
from avgColor import avgColTresh
from avgColorHSV import hsvTresh
from collections import Counter
from skimage.morphology import area_opening
from skimage.morphology import area_closing
from skimage.measure import label, regionprops, regionprops_table
import pandas as pd


def tomasiTacking(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners = cv2.goodFeaturesToTrack(gray, 30, 0.05, 15)
    corners = np.int0(corners)
    x_list = []
    y_list = []

    for i in corners:
        x, y = i.ravel()
        cv2.circle(frame, (x, y), 4, 255, 4)
        x_list.append(x)
        y_list.append(y)

    return x_list, y_list


def getROI(currentFrame):
    leftRegionIMG = currentFrame[int(currentFrame.shape[0] / 2):currentFrame.shape[0], 0:int(currentFrame.shape[1] / 2)]
    rightRegionIMG = currentFrame[int(currentFrame.shape[0] / 2): currentFrame.shape[0],
                     int(currentFrame.shape[1] / 2):currentFrame.shape[1]]

    return leftRegionIMG, rightRegionIMG

def getContourCoordinates(leftRegionIMG, rightRegionIMG):
    kernelClose = np.ones((5, 5), np.uint8)
    kernelErode = np.ones((1, 1), np.uint8)

    leftCanny = cv2.Canny(leftRegionIMG, threshold1=50, threshold2=150, apertureSize=3, L2gradient=True)
    leftCanny = cv2.morphologyEx(leftCanny, cv2.MORPH_CLOSE, kernelClose, iterations=2)

    rightCanny = cv2.Canny(rightRegionIMG, threshold1=50, threshold2=150, apertureSize=3, L2gradient=True)
    rightCanny = cv2.morphologyEx(rightCanny, cv2.MORPH_CLOSE, kernelClose, iterations=2)

    xR, yR = np.where(rightCanny < 1)
    blobsR = np.zeros_like(rightCanny)
    blobsR[xR, yR] = 255
    blobsR = cv2.erode(blobsR, kernelErode)


    xL, yL = np.where(leftCanny < 1)
    blobsL = np.zeros_like(leftCanny)
    blobsL[xL, yL] = 255
    blobsL1 = cv2.erode(blobsL, kernelErode)


    contoursR, hierarchy = cv2.findContours(blobsR, cv2.RETR_LIST, cv2.CHAIN_APPROX_TC89_KCOS)
    contoursL, hierarchy = cv2.findContours(blobsL, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    largestCntR = contoursR[0]
    largestCntL = contoursL[0]

    if len(contoursR) > 1:
        for i in range(0, len(contoursR)):
            if len(largestCntR) < len(contoursR[i]):
                largestCntR = contoursR[i]

    if len(contoursL) > 1:
        for i in range(0, len(contoursL)):
            if len(largestCntL) < len(contoursL[i]):
                largestCntL = contoursL[i]
    else:
        largestCntL = contoursL[0]

    # print(f'Contour: {largestCntL}')
    # print(f'Contour[0]: {largestCntL[0]}')
    # print(f'Contour[0][0]: {largestCntL[0][0]}')
    # test = largestCntL[0][0]
    # print(f'test[0]: {test[0]}')
    # print(f'test[1]: {test[1]}')
    # print(f'len: {len(largestCntR)}')
    # cv2.drawContours(rightRegionIMG, contours, -1, (0, 255, 0), 3)
    #cv2.drawContours(rightRegionIMG, largestCntR, -1, (255, 0, 0), 3)
    #cv2.drawContours(leftRegionIMG, largestCntL, -1, (255, 0, 0), 3)

    # cv2.imshow('blobL', blobsL)
    # cv2.imshow('blobR', blobsR)
    #
    # cv2.imshow('left', leftRegionIMG)
    # cv2.imshow('right', rightRegionIMG)
    # cv2.waitKey(0)

    # cv2.imshow('Leftcanny', leftCanny)
    # cv2.imshow('Rightcanny', rightCanny)
    return largestCntL, largestCntR, blobsL

def findMostCommonContour(contourCoordinates, image, val):
    kernelClose = np.ones((5, 5), np.uint8)
    coordsArray = []
    mostCommonCnt = []

    for i in range(0, len(contourCoordinates)):
        tempCnt = contourCoordinates[i]
        for w in range(0, len(contourCoordinates[i])):
            temp = tempCnt[w][0]
            # print(f'temp: {temp}')
            coordsArray.append(temp)

    repeatedCoords = tuple(map(tuple, coordsArray))
    counter = Counter(repeatedCoords)

    for coord in counter:
        if counter[coord] > val:
            #print(f'Coordinate: {coord}, Occurences: {counter[coord]}')
            mostCommonCnt.append(coord)
    
    mostCommonCnt = [np.asarray(mostCommonCnt)]
    #print(f'Most common coordinates and their occurences: \n {mstCommonCoords} \n and length of list: {len(mstCommonCoords)}')
    mask = np.zeros_like(image)

    for i in range(0, len(mostCommonCnt)):
        hull = cv2.convexHull(mostCommonCnt[i])
        hullList.append(hull)
        #hull = cv2.convexHull(rightcontour[i])
        
        
    for i in range(0, len(mostCommonCnt)):
        #cv2.drawContours(image, hullList, i, (255, 0, 0))
        cv2.drawContours(mask, mostCommonCnt, i, (255, 255, 255), thickness=cv2.FILLED)
    
    mask = mask[:, :, 0]

    out = np.zeros_like(image)
    out[mask == 255] = image[mask == 255]

    yI, xI = np.where(mask == 255)
    

    (topy, topx) = (np.min(yI), np.min(xI))
    (bottomy, bottomx) = (np.max(yI), np.max(xI))
    out = out[topy:bottomy+1, topx:bottomx+1]
    out = out[:, :, 0]
    yII, xII = np.where(out > 0)
    out[yII, xII] = 255
    
    
    
    print('shape:', out.shape)
    #out = cv2.morphologyEx(out, cv2.MORPH_OPEN, kernelClose, iterations=1)
    cv2.imshow('out', out) 
    #cv2.waitKey(0)
    
    #cv2.drawContours(left, leftcontour, -1, (255, 0, 0), 3)

    return mostCommonCnt, out, topy, topx, bottomy, bottomx


def colorSeg(motionVar, prevFrame, currentFrame):
    # Convert BGR to HSV
    motion = motionVar

    frame_diff = cv2.absdiff(currentFrame, prevFrame)
    cv2.imshow('framediff', frame_diff)

    hsv = cv2.cvtColor(currentFrame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (36, 0, 0), (70, 255, 255))

    kernel = np.ones((5, 5), np.uint8)

    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    segmentedFrame = cv2.bitwise_not(currentFrame, prevFrame)
    cv2.imshow('segf', segmentedFrame)
    contours, hierarchy = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    output = cv2.drawContours(segmentedFrame, contours, -1, (0, 200, 0), thickness=cv2.FILLED)

    # ROI visualization
    # start_point1 = (210, 210)
    # start_point2 = (510, 210)
    # end_point1 = (330, 335)
    # end_point2 = (630, 335)
    # r_color = (255, 255, 255)
    # cv2.rectangle(segmentedFrame, start_point1, end_point1, r_color, 6)
    # cv2.rectangle(segmentedFrame, start_point2, end_point2, r_color, 6)

    for cnt in contours:
        (x, y, w, h) = cv2.boundingRect(cnt)
        if (cv2.contourArea(cnt) <= 1000 and (x < 220 or x > 320)) and (
                (x < 520 or x > 620) or (y < 210 or y > 335)):
            cv2.rectangle(output, (x, y), (x + w, y + h), (165, 100, 0), 3)
            motion += 1
            print(motion)

    # Display the frame, saved in the file
    cv2.imshow('Garbage Picker Motion Detector', output)

    output = currentFrame

def fillColor(image):
    cntImg = np.zeros_like(image)
    close = area_closing(image,100,1)
    contours, hierachy = cv2.findContours(close, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for i in range(0,len(contours)):
        cv2.drawContours(cntImg, contours, i, (255,255,255),thickness=cv2.FILLED)
    return cntImg

def maskOff(inputImg,mask):
    newImg1 = np.zeros((inputImg.shape[0], inputImg.shape[1], 3), np.uint8)
    for y in range(inputImg.shape[0]):
        for x in range(inputImg.shape[1]):
            if mask[y][x] == 255:
                newImg1[y][x] = inputImg[y][x]
            else:
                newImg1[y][x] = 0
    return newImg1

def templateMatch(Img, Template):
    h = Template.shape[0]
    w = Template.shape[1]
    grayTemplate = cv2.cvtColor(Template, cv2.COLOR_BGR2GRAY)
    templateResult = cv2.matchTemplate(Img, grayTemplate, cv2.TM_CCOEFF_NORMED)
    (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(templateResult)
    topLeft = maxLoc
    bottomRight = (topLeft[0] + w, topLeft[1] + h)
    return topLeft, bottomRight, templateResult

def makeTransparent(image):
    tmp = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, alpha = cv2.threshold(tmp, 0, 255, cv2.THRESH_BINARY)
    b, g, r = cv2.split(image)
    rgba = [b, g, r, alpha]
    dst = cv2.merge(rgba, 4)
    cv2.imwrite('test.png',dst)
    return dst

def getHighestXY(counterArray):
    maxValX = 0
    maxValY = 0
    npp = np.asarray(counterArray)
    for i in range(npp.shape[1]):
        if maxValX < counterArray[0][i][1]:
            maxValX = counterArray[0][i][1]
        if maxValY < counterArray[0][i][0]:
            maxValY = counterArray[0][i][0]
    return maxValY, maxValX

def draw_flow(img, flow, step=16):

    h, w = img.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1).astype(int)
    fx, fy = flow[y,x].T

    lines = np.vstack([x, y, x-fx, y-fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)

    img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.polylines(img_bgr, lines, 0, (0, 255, 0))

    for (x1, y1), (_x2, _y2) in lines:
        cv2.circle(img_bgr, (x1, y1), 1, (0, 255, 0), -1)

    return img_bgr


def draw_hsv(flow):

    h, w = flow.shape[:2]
    fx, fy = flow[:,:,0], flow[:,:,1]

    ang = np.arctan2(fy, fx) + np.pi
    v = np.sqrt(fx*fx+fy*fy)

    hsv = np.zeros((h, w, 3), np.uint8)
    hsv[...,0] = ang*(180/np.pi/2)
    hsv[...,1] = 255
    hsv[...,2] = np.minimum(v*4, 255)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return bgr


if __name__ == '__main__':
    frameCount = 0
    calibrating = True
    check = True

    cap = cv2.VideoCapture('data/outside_videos/outsidecalibratoin.mp4')

    ret, frame = cap.read()
    motion = 0
    sparseOF = SparseOF()
    leftCnts = []
    rightCnts = []
    hullList = []
    contourVal = 50
    properties = ['area', 'bbox', 'bbox_area']
    kernel1 = np.ones((5,5), np.uint8)

    while cap.isOpened() and calibrating:
        previousFrame = frame[:]
        # cv2.imshow('prev', previousFrame)
        frameCount += 1
        ret, frame = cap.read()
        cv2.imshow('current', frame)
        left, right = getROI(frame)
        prevLeft, prevRight = getROI(previousFrame)
        grayLeft = cv2.cvtColor(left,cv2.COLOR_BGR2GRAY)
        grayRight = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)



        if frameCount > 150:
            # colorSeg(motion, previousFrame, frame)
            leftcnt, rightcnt, blobsL = getContourCoordinates(left, right)
            print(frameCount)
            #print(leftcnt)
            #cv2.waitKey(0)
            leftCnts.append(leftcnt)
            rightCnts.append(rightcnt)

            if frameCount > 500:
                # print(f'Len leftcnts: {len(leftCnts)}')
                # print(f'Len rightcnts: {len(rightCnts)}')
                # print(f'leftcnts[0][0]: {leftCnts[0][0]}')
                # print(f'leftcnts[1][0]: {leftCnts[1][0]}')
                # print(f'leftcnts[[0][0]]: {leftCnts[0][0]}')
                # print('------------------------')
                # temp = leftCnts[0][0]
                # temp1 = leftCnts[1][0]
                # print(f'temp[0]: {temp[0]}')
                # print(f'temp1[0]: {temp1[0]}')
                # print(f'len leftCnts[0]: {len(leftCnts[0])}')
                # temp3 = leftCnts[0]
                # print(f'temp3[0][0]: {temp3[0][0]}')
                # print(f'temp3[1][0]: {temp3[1][0]}')
                # cv2.waitKey(0)

                #leftcontour, leftout, leftYTop, leftXTop, leftYBottom, leftXBottom = findMostCommonContour(leftCnts,
                                                                                                           #left,
                                                                                                           #contourVal)
                #rightcontour, rightout, rightYTop, rightXTop, rightYBottom, rightXBottom = findMostCommonContour(
                    #rightCnts, right, contourVal)
                #roiLeft = left[leftYTop:leftYBottom + 1, leftXTop:leftXBottom + 1]
                #roiRight = right[rightYTop:rightYBottom + 1, rightXTop:rightXBottom + 1]

                if check:
                    leftcontour1, leftout1, leftYTop1, leftXTop1, leftYBottom1, leftXBottom1 = findMostCommonContour(leftCnts,
                                                                                                               left,
                                                                                                               contourVal)
                    rightcontour1, rightout1, rightYTop1, rightXTop1, rightYBottom1, rightXBottom1 = findMostCommonContour(
                        rightCnts, right, contourVal)
                    prevRoiLeft = left[leftYTop1:leftYBottom1 + 1, leftXTop1:leftXBottom1 + 1]
                    prevRoiLeftGray = cv2.cvtColor(prevRoiLeft,cv2.COLOR_BGR2GRAY)
                    blobs = label(blobsL > 0)
                    df = pd.DataFrame(regionprops_table(blobs, properties=properties))

                    closed = fillColor(leftout1)
                    closed2 = fillColor(rightout1)
                    blobsLReal = blobsL[leftYTop1:leftYBottom1 + 1, leftXTop1:leftXBottom1 + 1]
                    f = area_opening(blobsLReal, max(df['area'] - 200), 1)
                    erodeF = cv2.erode(f, kernel1, iterations=3)
                    maskedLeftPrev = maskOff(prevRoiLeft, erodeF)
                    maskedLeftPrevGray = cv2.cvtColor(maskedLeftPrev, cv2.COLOR_BGR2GRAY)

                    check = False




                roiLeft = left[leftYTop1:leftYBottom1 + 1, leftXTop1:leftXBottom1 + 1]
                roiRight = right[rightYTop1:rightYBottom1 + 1, rightXTop1:rightXBottom1 + 1]


                maskedLeft = maskOff(roiLeft, erodeF)
                maskedRight = maskOff(roiRight, closed2)
                maskedLeftGray = cv2.cvtColor(maskedLeft, cv2.COLOR_BGR2GRAY)


                grayRoiLeft = cv2.cvtColor(roiLeft,cv2.COLOR_BGR2GRAY)
                flow = cv2.calcOpticalFlowFarneback(maskedLeftPrevGray, maskedLeftGray, None, 0.5, 3, 25, 3, 5, 1.2, 0)
                maskedLeftPrevGray = maskedLeftGray



                flowHSV = draw_hsv(flow)

                flowThresh = cv2.inRange(flowHSV, (0, 0, 15), (180, 255, 255))
                blobsT = label(flowThresh > 0)
                df2 = pd.DataFrame(regionprops_table(blobsT, properties=properties))
                if len(df2) != 0 and max(df2['area']) > 300:
                    print("closing!")


                transparentLeft = makeTransparent(maskedLeft)
                transparentRight = makeTransparent(maskedRight)
                topLeftL, bottomRightL, resultL = templateMatch(grayLeft,maskedLeft)
                topLeftR, bottomRightR, resultR = templateMatch(grayRight,maskedRight)
                cv2.rectangle(left,topLeftL,bottomRightL,(0,255,0),2)
                cv2.rectangle(right, topLeftR, bottomRightR, (0, 255, 0), 2)
                #cv2.imshow('tresh',closed)



                cv2.imshow('hsvTresh', flowThresh)
                #cv2.imshow('tresh2', closed2)
                cv2.imshow('blobsL',f)
                cv2.imshow('flow', draw_flow(maskedLeftGray, flow))
                cv2.imshow('flow HSV', flowHSV)
                #cv2.imshow('getMaskedLeft', maskedLeft)
                #cv2.imshow('getMaskedRight', maskedRight)
                #cv2.imshow('left', left)
                #cv2.imshow('right', right)
                #cv2.imshow('leftout', leftout)
                #cv2.imshow('rightout', rightout)
                #cv2.imshow('templatematch', resultL)

                #cv2.circle(left, leftcontour, radius=0, color=(0, 0, 255), thickness=-1)
                
                #cv2.waitKey(0)

            # leftHSVthresh = hsvTresh(left)
            # rightHSVthresh = hsvTresh(right)
            # cv2.imshow('leftHSV', leftHSVthresh)
            # cv2.imshow('rightHSV', rightHSVthresh)
            # leftRGBthresh = avgColTresh(left)
            # rightRGBthresh = avgColTresh(right)
            # cv2.imshow('left', leftRGBthresh)
            # cv2.imshow('right', rightRGBthresh)

            # sparseOF.sparseOF(left, prevLeft)
            # sparseOF.sparseOF(right, prevRight)

        if cv2.waitKey(1) == ord('s'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # import cv2 as cv
    #
    # frameCount = 0
    #
    #
    # video = cv.VideoCapture('/Users/antonio/Downloads/test.mp4')
    #
    # # read the first frame  of the video as the initial background image
    # ret, prevFrame = video.read()
    #
    # while (video.isOpened()):
    #
    #     frameCount += 1
    #
    #     ##capture frame by frame
    #     ret, currentFrame = video.read()
    #
    #     # Find the absolute difference between the pixels of the prev_frame and current_frame
    #     # absdiff() will extract just the pixels of the objects that are moving between the two frames
    #     frame_diff = cv.absdiff(currentFrame, prevFrame)
    #     motion = 0
    #
    #     # applying Gray scale by converting the images from color to grayscale,
    #     # This will reduce noise
    #     gray = cv.cvtColor(frame_diff, cv.COLOR_BGR2GRAY)
    #
    #     # image smoothing also called blurring is applied using gauisian Blur
    #     # convert the gray to Gausioan blur to detect motion
    #     blur = cv.GaussianBlur(gray, (25, 25), 0)  # (gray, (5, 5), 0)
    #     thresh = cv.threshold(blur, 35, 255, cv.THRESH_BINARY)[1]  # (blur, 20, 255, cv2.THRESH_BINARY)[1]
    #
    #
    #     # fill the gaps by dialiting the image
    #     # Dilation is applied to binary images.
    #     dilate = cv.dilate(thresh, None, iterations=40)
    #
    #     # Contour detection
    #     (contours, _) = cv.findContours(dilate.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    #
    #     start_point1 = (210, 210)
    #     start_point2 = (510, 210)
    #
    #     end_point1 = (330, 335)
    #     end_point2 = (630, 335)
    #
    #     r_color = (255, 255, 255)
    #
    #     cv.rectangle(prevFrame, start_point1, end_point1, r_color, 6 )
    #     cv.rectangle(prevFrame, start_point2, end_point2, r_color, 6)
    #
    #
    #     #(cnt) > 700 and (x >= 250 and x <= 450) and (y >= 150 and y <= 400)
    #
    #     # Looping over the contours
    #     for cnt in contours:
    #         (x, y, w, h) = cv.boundingRect(cnt)
    #         if cv.contourArea(cnt) < 700  and (x >= 220 and x <= 320)  or (x >= 520 and x <= 620) and (y >= 210 and y <= 335):
    #             cv.rectangle(prevFrame, (x, y), (x + w, y + h), (0, 255, 0), 3)
    #             motion += 1
    #
    #     cv.imshow("motion_detector", prevFrame)
    #
    #     prevFrame = currentFrame
    #
    #     if ret == False:
    #         break
    #     if cv.waitKey(1) == ord('s'):
    #         break
    #
    # video.release()
    # cv.destroyAllWindows()
