import cv2
import numpy as np


def getContourCoordinates(currentFrame):
    leftRegionIMG = currentFrame[int(currentFrame.shape[0] / 2):currentFrame.shape[0], 0:int(currentFrame.shape[1] / 2)]
    rightRegionIMG = currentFrame[int(currentFrame.shape[0] / 2): currentFrame.shape[0],
                     int(currentFrame.shape[1] / 2):currentFrame.shape[1]]
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
    blobsL = cv2.erode(blobsL, kernelErode)

    contoursR, hierarchy = cv2.findContours(blobsR, cv2.RETR_LIST, cv2.CHAIN_APPROX_TC89_L1)
    contoursL, hierarchy = cv2.findContours(blobsL, cv2.RETR_LIST, cv2.CHAIN_APPROX_TC89_L1)

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

    #cv2.drawContours(rightRegionIMG, contours, -1, (0, 255, 0), 3)
    cv2.drawContours(rightRegionIMG, largestCntR, -1, (255, 0, 0), 3)
    cv2.drawContours(leftRegionIMG, largestCntL, -1, (255, 0, 0), 3)

    cv2.imshow('blobL', blobsL)
    cv2.imshow('blobR', blobsR)

    cv2.imshow('left', leftRegionIMG)
    cv2.imshow('right', rightRegionIMG)

    # cv2.imshow('Leftcanny', leftCanny)
    # cv2.imshow('Rightcanny', rightCanny)
    #return largestCntL, largestCntR


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

    output = cv2.drawContours(segmentedFrame, contours, -1, (0, 200, 0), 1)

    # ROI visualization
    # start_point1 = (210, 210)
    # start_point2 = (510, 210)
    # end_point1 = (330, 335)
    # end_point2 = (630, 335)
    # r_color = (255, 255, 255)
    # cv2.rectangle(segmentedFrame, start_point1, end_point1, r_color, 6 )
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


if __name__ == '__main__':
    frameCount = 0

    cap = cv2.VideoCapture('data/outside_videos/outsidecalibratoin.mp4')

    ret, frame = cap.read()
    motion = 0
    while cap.isOpened():
        previousFrame = frame[:]
        # cv2.imshow('prev', previousFrame)
        frameCount += 1
        ret, frame = cap.read()
        cv2.imshow('current', frame)

        if frameCount > 150:
            # colorSeg(motion, previousFrame, frame)
            getContourCoordinates(frame)

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
