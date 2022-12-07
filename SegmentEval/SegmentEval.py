import pandas as pd
import os
import numpy as np
from Evaluate import *

if __name__ == "__main__":

    dirCity = os.listdir("SegmentEval//CityPred//labels")
    pathCity = "SegmentEval//CityPred//labels//"

    pathValCity = "SegmentEval//CityGT//train//labels//"
    dirValCity = os.listdir("SegmentEval//CityGT//train//labels")

    dirHighway = os.listdir("SegmentEval//HighwayPred//labels")
    pathHighway = "SegmentEval//HighwayPred//labels//"

    pathValHighway= "SegmentEval//HighwayGT//train//labels//"
    dirValHighway = os.listdir("SegmentEval//HighwayGT//train//labels")

    dirPark = os.listdir("SegmentEval//ParkPred//labels")
    pathPark = "SegmentEval//ParkPred//labels//"

    pathValPark = "SegmentEval//ParkGT//train//labels//"
    dirValPark = os.listdir("SegmentEval//ParkGT//train//labels")
    
    cityPred = []
    cityGT = []

    for file in range(0, len(dirCity)):
        #Each row is: class x_center y_center width height
        with open (pathCity + dirCity[file], "r") as myfile:
            data = myfile.read().split(" ")
            #print(data)
            #print(data[2])
            x_center = float(data[2])
            y_center = float(data[3])
            width = float(data[4])
            height = float(data[5])

            xmin = x_center - width/2
            ymin = y_center - height/2
            xmax = x_center + width/2
            ymax = y_center + height/2

            bboxPred = [xmin, ymin, xmax, ymax]
            cityPred.append(bboxPred)


        with open (pathValCity + dirValCity[file], "r") as valfile:
            dataVal = valfile.read().split(" ")
            x_centerVal = float(dataVal[1])
            y_centerVal = float(dataVal[2])
            widthVal = float(dataVal[3])
            heightVal = float(dataVal[4])

            xminVal = x_centerVal - widthVal/2
            yminVal = y_centerVal - heightVal/2
            xmaxVal = x_centerVal + widthVal/2
            ymaxVal = y_centerVal + heightVal/2

            bboxGT = [xminVal, yminVal, xmaxVal, ymaxVal]
            cityGT.append(bboxGT)

    IOUs = []
    for i in range(0, len(cityGT)):
        iou = calc_iou_individual(cityPred[i], cityGT[i])
        IOUs.append(iou)
        print(f'IOU: {iou}')

    
    avgIOU = np.average(IOUs)
    print(f'average IOU of City bboxes: {avgIOU}')

    

    




    
