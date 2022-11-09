import cv2
import numpy as np
from sparse_of import SparseOF
from collections import Counter
from skimage.morphology import area_opening
from skimage.morphology import area_closing
from skimage.measure import label, regionprops_table
import pandas as pd
import pixellib
from pixellib.torchbackend.instance import instanceSegmentation


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

    return largestCntL, largestCntR, blobsL


def findMostCommonContour(contourCoordinates, image, val):
    coordsArray = []
    mostCommonCnt = []

    for i in range(0, len(contourCoordinates)):
        tempCnt = contourCoordinates[i]
        for w in range(0, len(contourCoordinates[i])):
            temp = tempCnt[w][0]
            coordsArray.append(temp)

    repeatedCoords = tuple(map(tuple, coordsArray))
    counter = Counter(repeatedCoords)

    for coord in counter:
        if counter[coord] > val:
            mostCommonCnt.append(coord)

    mostCommonCnt = [np.asarray(mostCommonCnt)]
    mask = np.zeros_like(image)

    for i in range(0, len(mostCommonCnt)):
        cv2.drawContours(mask, mostCommonCnt, i, (255, 255, 255), thickness=cv2.FILLED)

    mask = mask[:, :, 0]

    out = np.zeros_like(image)
    out[mask == 255] = image[mask == 255]

    yI, xI = np.where(mask == 255)

    (topy, topx) = (np.min(yI), np.min(xI))
    (bottomy, bottomx) = (np.max(yI), np.max(xI))
    out = out[topy:bottomy + 1, topx:bottomx + 1]
    out = out[:, :, 0]
    yII, xII = np.where(out > 0)
    out[yII, xII] = 255

    cv2.imshow('out', out)

    return mostCommonCnt, out, topy, topx, bottomy, bottomx


# Draws contours around an binary image, and fills it with white.
def fillColor(image):
    cntImg = np.zeros_like(image)
    close = area_closing(image, 100, 1)
    contours, hierachy = cv2.findContours(close, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for i in range(0, len(contours)):
        cv2.drawContours(cntImg, contours, i, (255, 255, 255), thickness=cv2.FILLED)
    return cntImg


# Takes a mask and an image to make a new image, that is just the mask of the original image.
def maskOff(inputImg, mask):
    newImg1 = np.zeros((inputImg.shape[0], inputImg.shape[1], 3), np.uint8)
    for y in range(inputImg.shape[0]):
        for x in range(inputImg.shape[1]):
            if mask[y][x] == 255:
                newImg1[y][x] = inputImg[y][x]
            else:
                newImg1[y][x] = 0
    return newImg1


# draws the flow of the image, dont really understand it.
def draw_flow(img, flow, step=16):
    h, w = img.shape[:2]
    y, x = np.mgrid[step / 2:h:step, step / 2:w:step].reshape(2, -1).astype(int)
    fx, fy = flow[y, x].T

    lines = np.vstack([x, y, x - fx, y - fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)

    img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.polylines(img_bgr, lines, 0, (0, 255, 0))

    for (x1, y1), (_x2, _y2) in lines:
        cv2.circle(img_bgr, (x1, y1), 1, (0, 255, 0), -1)

    return img_bgr


# Optical flow function that maps opencv optical flow function outputs to hsv values for a visual image.
def draw_hsv(flow):
    h, w = flow.shape[:2]
    fx, fy = flow[:, :, 0], flow[:, :, 1]

    ang = np.arctan2(fy, fx) + np.pi
    v = np.sqrt(fx * fx + fy * fy)

    hsv = np.zeros((h, w, 3), np.uint8)
    hsv[..., 0] = ang * (180 / np.pi / 2)
    hsv[..., 1] = 255
    hsv[..., 2] = np.minimum(v * 4, 255)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return bgr


if __name__ == '__main__':
    # ---VARIABLES---#
    frameCount = 0
    calibrating = True
    check = True
    cap = cv2.VideoCapture('data/outside_videos/GL010030.MP4')
    ret, frame = cap.read()
    motion = 0
    sparseOF = SparseOF()
    leftCnts = []
    rightCnts = []
    hullList = []
    contourVal = 50
    properties = ['area', 'bbox', 'bbox_area']
    kernel1 = np.ones((5, 5), np.uint8)
    closing = False
    closeCounter = 0
    closeTimer = 0
    movement = False
    movementTimer = 0
    stillClosedBool = False
    yo = False
    imNum = 0
    fiftyFrame = []
    minBoundingVal = 1000
    featureImgs = []
    goodMatchesCounter = 0
    grayFrameArray = []
    normalPics = []
    normalnormalPics = []
    anotherCounter = 0

    ins = instanceSegmentation()
    ins.load_model("pointrend_resnet50.pkl", confidence=0.2)
    # ---VARIABLES---#
    while cap.isOpened() and calibrating:
        previousFrame = frame[:]
        # cv2.imshow('prev', previousFrame)
        frameCount += 1
        ret, frame = cap.read()
        cv2.imshow('current', frame)
        left, right = getROI(frame)
        prevLeft, prevRight = getROI(previousFrame)
        grayLeft = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
        grayRight = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
        fiftyFrame.append(frame)
        grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if frameCount > 150:
            # colorSeg(motion, previousFrame, frame)
            leftcnt, rightcnt, blobsL = getContourCoordinates(left, right)
            print(frameCount)
            # print(leftcnt)
            # cv2.waitKey(0)
            leftCnts.append(leftcnt)
            rightCnts.append(rightcnt)

            if frameCount > 500:
                # if statement used to make sure the code in the if statement only gets run once.
                if check:
                    # Code to find the most common contour, as well as finding the coordinates to make a bounding box around the grabber.
                    leftcontour1, leftout1, leftYTop1, leftXTop1, leftYBottom1, leftXBottom1 = findMostCommonContour(
                        leftCnts,
                        left,
                        contourVal)
                    # Same as above but for the right grabber.
                    rightcontour1, rightout1, rightYTop1, rightXTop1, rightYBottom1, rightXBottom1 = findMostCommonContour(
                        rightCnts, right, contourVal)
                    # Making ROI's for the left grabber.
                    prevRoiLeft = left[leftYTop1:leftYBottom1 + 1, leftXTop1:leftXBottom1 + 1]

                    print("top coordinates: " + str(leftYTop1) + " " + str(leftXBottom1))
                    print("bottom coordinates: " + str(rightYBottom1) + " " + str(rightXTop1 + (frame.shape[1] / 2)))
                    # label is sklearns function for doing blob analysis of a binary image
                    blobs = label(blobsL > 0)
                    # Using pandas to make a dataframe of the data to make it more easily accessable
                    df = pd.DataFrame(regionprops_table(blobs, properties=properties))
                    # Fills out both the grabber images as before this, there is a lot of holes in the image. Do it for both grabbers
                    closed = fillColor(leftout1)
                    closed2 = fillColor(rightout1)
                    # Fixes the ROI for the blobsL image, so it is the same size as the ones before, where we also made ROI's
                    blobsLReal = blobsL[leftYTop1:leftYBottom1 + 1, leftXTop1:leftXBottom1 + 1]
                    # Area_opening is sklearns function for removing any unwanted blobs. Here we say all blobs under the threshold value of 200 less than the
                    # biggest blob in the image should be removed, so we are only left with the biggest blob which should be the grabber.
                    f = area_opening(blobsLReal, max(df['area'] - 200), 1)
                    # Erode makes the threshold image smaller
                    erodeF = cv2.erode(f, kernel1, iterations=3)
                    # maskOff function, uses the binary image of the grabber to make a new image, where only the white parts of the binary image
                    # is saved from the prevRoiLeft image.
                    maskedLeftPrev = maskOff(prevRoiLeft, erodeF)
                    # make a gray version, as this is needed for the optical flow function.
                    maskedLeftPrevGray = cv2.cvtColor(maskedLeftPrev, cv2.COLOR_BGR2GRAY)

                    check = False

                # Making ROI's again, this time in the while loop so it keeps updating, and not just doing it onee as once again, we need it for the
                # optical flow function
                roiLeft = left[leftYTop1:leftYBottom1 + 1, leftXTop1:leftXBottom1 + 1]
                roiRight = right[rightYTop1:rightYBottom1 + 1, rightXTop1:rightXBottom1 + 1]
                grayFrameRoi = grayFrame[0:grayFrame.shape[0]-180, leftXBottom1:int(rightXTop1 + (grayFrame.shape[1] / 2))]
                normalFrameRoi = frame[0:grayFrame.shape[0]-180, leftXBottom1:int(rightXTop1 + (grayFrame.shape[1] / 2))]
                normalnormalPicss = frame[0:grayFrame.shape[0], leftXBottom1:int(rightXTop1 + (grayFrame.shape[1] / 2))]
                grayFrameArray.append(grayFrameRoi)
                normalPics.append(normalFrameRoi)
                normalnormalPics.append(normalnormalPicss)
                # New maskoff, only difference is this is agai in the while loop, so it keeps updating.
                maskedLeft = maskOff(roiLeft, erodeF)
                # Gray version again, needed for optical flow
                maskedLeftGray = cv2.cvtColor(maskedLeft, cv2.COLOR_BGR2GRAY)

                # The optical flow function! Uses gray images of the current frame and the frame from just before
                # Idk how the f it works. You can watch this video and still not understand it: https://www.youtube.com/watch?v=WrlH5hHv0gE&ab_channel=NicolaiNielsen-ComputerVision%26AI
                flow = cv2.calcOpticalFlowFarneback(maskedLeftPrevGray, maskedLeftGray, None, 0.5, 3, 25, 3, 5, 1.2, 0)
                # Sets the previous frame to the current so it is ready for the next frame.
                maskedLeftPrevGray = maskedLeftGray

                # Makes the hsv representation of the optical flow.
                flowHSV = draw_hsv(flow)

                # Makes binary image of the hsv image of the optical flow
                flowThresh = cv2.inRange(flowHSV, (0, 0, 5), (180, 255, 255))
                # Blob analysis again to find the biggest blob
                blobsT = label(flowThresh > 0)
                df2 = pd.DataFrame(regionprops_table(blobsT, properties=properties))
                # If the biggest blob is over 400, the grabbers are moving!!
                if len(df2) != 0 and max(df2['area']) > 400:
                    closing = True
                    closeCounter += 1
                    print(closeCounter)

                # A bunch of if statements that checks whether we got a fake closing detection or not
                if closing:
                    closeTimer += 1
                    movementTimer = 0
                    # if it only makes 2 or less detection from the first detection after 10 frames, save as false detection. Also resets detection variables
                    if closeTimer > 10 and closeCounter <= 3 and stillClosedBool == False:
                        closing = False
                        closeCounter = 0
                        closeTimer = 0
                        print("fake close")
                    # If it makes 3 or more detection in the last 10 frames from the first detection save as correct detection. Also resets detection variables
                    if closeTimer > 10 and closeCounter > 3 and stillClosedBool == False:
                        closing = False
                        closeCounter = 0
                        closeTimer = 0
                        screenVal = frame.shape[0] / 2
                        minHessian = 400
                        print("We closing!")

                        # Save image from 150 frames ago as picture of garbage. Makes ROI of the image, to filter out unnecessary noise.
                        if len(fiftyFrame) > 70:

                            saveImg = fiftyFrame[frameCount-30]

                            saveImgRoi = saveImg[0:saveImg.shape[0],leftXBottom1:int(rightXTop1 + (saveImg.shape[1] / 2))]
                            saveImgRoiGray= cv2.cvtColor(saveImgRoi,cv2.COLOR_BGR2GRAY)
                            for i in range(frameCount-700,len(grayFrameArray)-50):
                                featureImgs.append(grayFrameArray[i])
                                normalPics.append(normalPics[i])

                            for i in featureImgs:

                                detector = cv2.SIFT.create(0)
                                keypoints1, descriptors1 = detector.detectAndCompute(saveImgRoiGray, None)
                                keypoints2, descriptors2 = detector.detectAndCompute(i, None)

                                matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_FLANNBASED)
                                knn_matches = matcher.knnMatch(descriptors1, descriptors2, 2)
                                ratio_thresh = 0.9
                                good_matches = []
                                for m, n in knn_matches:
                                    if m.distance < ratio_thresh * n.distance:
                                        good_matches.append(m)
                                if len(good_matches) > goodMatchesCounter:
                                    goodMatchesCounter = len(good_matches)
                                    #print(len(good_matches))
                                    realImg = normalnormalPics[(frameCount-701) + anotherCounter]
                                    print(anotherCounter)
                                    print(len(good_matches))
                                    good_matches.clear()
                                knn_matches = 0
                                #print(len(good_matches))
                                anotherCounter += 1

                            goodMatchesCounter = 0
                            anotherCounter = 0
                            featureImgs.clear()

                            # Save the chosen image on the pc.
                        cv2.imwrite("data/savedimages/garbage" + str(imNum) + ".png", realImg)
                        # The code line for the neural network segmentation of the image.
                        results, output = ins.segmentImage("data/savedimages/garbage" + str(imNum) + ".png",
                                                           show_bboxes=True,
                                                           output_image_name="data/savedimages/segmented" + str(
                                                               imNum) + ".png")
                        # Goes through all the bounding boxes that is found by the NN on the image and chooses the one closest to the grabbers position.
                        for i in range(0, len(results['boxes'])):
                            middleObject = results['boxes'][i][3] - (
                                        results['boxes'][i][3] - results['boxes'][i][1]) / 2
                            if abs((screenVal - 80) - (results['boxes'][i][3] - middleObject)) < minBoundingVal:
                                minBoundingVal = abs((screenVal - 80) - (results['boxes'][i][3] - middleObject))
                                print("val: " + str(minBoundingVal))
                                print(imNum)
                                bbx1 = results['boxes'][i][0]
                                bby1 = results['boxes'][i][1]
                                bbx2 = results['boxes'][i][2]
                                bby2 = results['boxes'][i][3]
                        # Draws the chosen bounding box on a new image and saves it on the pc.
                        cv2.rectangle(realImg, (bbx1, bby1), (bbx2, bby2), (0, 255, 0), thickness=2)
                        cv2.imwrite("data/savedimages/bbox" + str(imNum) + ".png", realImg)
                        # Resets the minBoundingVal variable so it is ready for a new image segmentation. Also sets movement to tru to start the timer.
                        minBoundingVal = 1000
                        imNum += 1
                        movement = True
                # If correct detection happens, start a timer that resets everytime a detection happens after the first one,
                # until 100 frames passes without a new detection, so we know that the grabbers have closed
                if movement:
                    closing = False
                    stillClosedBool = True
                    movementTimer += 1
                # if the timer goes over 100 the grabbers have closed and we can start detecting for movement again.
                if movementTimer > 80:
                    closeCounter = 0
                    movement = False
                    stillClosedBool = False

                # shows the images, maybe delete some of these
                cv2.imshow('hsvTresh', flowThresh)
                # cv2.imshow('tresh2', closed2)
                cv2.imshow('blobsL', f)
                cv2.imshow('flow', draw_flow(maskedLeftGray, flow))
                cv2.imshow('flow HSV', flowHSV)
                # cv2.imshow('getMaskedLeft', maskedLeft)
                # cv2.imshow('getMaskedRight', maskedRight)
                # cv2.imshow('left', left)
                # cv2.imshow('right', right)
                # cv2.imshow('leftout', leftout)
                # cv2.imshow('rightout', rightout)
                # cv2.imshow('templatematch', resultL)

        if cv2.waitKey(1) == ord('s'):
            break

    cap.release()
    cv2.destroyAllWindows()

