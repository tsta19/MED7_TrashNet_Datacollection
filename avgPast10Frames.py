import cv2
import numpy as np

cap = cv2.VideoCapture('trimVideo.mp4')
imgArray = []
treshold = 30

ret, imgFrame = cap.read()

newImg = np.zeros((imgFrame.shape[0],imgFrame.shape[1],3),np.uint8)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Cant read video")
        break
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    cv2.imshow('frame',frame)

    if cv2.waitKey(1) == ('q'):
        break
    if len(imgArray) < 5:
        imgArray.append(frame)
        avg_image = imgArray[0]
        for i in range(len(imgArray)):
            alpha = 1.0 / (i + 1)
            beta = 1.0 - alpha
            avg_image = cv2.addWeighted(imgArray[i], alpha, avg_image, beta, 0.0)
    else:
        imgArray.pop(0)
    #blur = cv2.GaussianBlur(avg_image,(3,3),0)

    avgBlue = np.average(avg_image[:,:,0])
    avgGreen = np.average(avg_image[:, :, 1])
    avgRed = np.average(avg_image[:, :, 2])

    avgBlueHigh = avgBlue + treshold
    avgGreenHigh = avgGreen + treshold
    avgRedHigh = avgRed + treshold
    avgBlueLow = avgBlue - treshold
    avgGreenLow = avgGreen - treshold
    avgRedLow = avgRed - treshold
    print('blue' + str(avgBlue))
    print('red' + str(avgRed))
    print('green' + str(avgGreen))

    frame_treshold = cv2.inRange(frame,(avgBlueLow,avgGreenLow,avgRedLow),(avgBlueHigh,avgGreenHigh,avgRedHigh))

    #frame_treshold = cv2.inRange(gray, avgLow, avgHigh)
    cv2.imshow('avg', avg_image)
    cv2.imshow('tresh', frame_treshold)


cap.release()
cv2.destroyAllWindows()