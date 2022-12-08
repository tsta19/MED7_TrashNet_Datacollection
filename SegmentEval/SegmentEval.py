import pandas as pd
import os
import numpy as np
from Evaluate import *
import matplotlib.pyplot as plt


def get_precision(tp, fp):
    try:
        precision = tp/(tp + fp)
    except ZeroDivisionError:
        precision = 0.0

    return precision

def get_recall(tp, fn):
    try:
        recall = tp/(tp + fn)
    except ZeroDivisionError:
        recall = 0.0

    return recall

def EvaluateResults(dirPred, pathPred, dirGT, pathGT, env):
    Pred = []
    GT = []

    for file in range(0, len(dirPred)):
        #Each row is: class x_center y_center width height
        with open (pathPred + dirPred[file], "r") as myfile:
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
            Pred.append(bboxPred)


        with open (pathGT + dirGT[file], "r") as valfile:
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
            GT.append(bboxGT)

    IOUs = []
    for i in range(0, len(GT)):
        iou = calc_iou_individual(Pred[i], GT[i])
        IOUs.append(iou)
        print(f'IOU: {iou}')

    sortedIOU = sorted(IOUs)
    
    avgIOU = np.average(IOUs)
    print(f'Average IOU of {env} bboxes: {avgIOU}')
    print(f'Lowest IOU of {env} bboxes: {np.min(IOUs)}')
    print(f'Highest IOU of {env} bboxes: {np.max(IOUs)}')
    print(f'Median IOU of {env} bboxes: {np.median(sortedIOU)}')

    dicts = {}
    resAtIIOU = []
    yaxis = []
    precisionAtIOU = []
    recallAtIOU = []
    for i in range(1, 96):
        res = get_single_image_results(gt_boxes=GT, pred_boxes=Pred, iou_thr=i/100)
        
        resAtIIOU.append(res)
        
        precision = get_precision(tp=res.get('true_pos'), fp=res.get('false_pos'))
        recall = get_recall(tp=res.get('true_pos'), fn=res.get('false_neg'))
        precisionAtIOU.append(precision)
        recallAtIOU.append(recall)
        
        yaxis.append(i/100)
        
    
    
    plt.scatter(np.arange(0, 0.95, 0.01), precisionAtIOU, s=8)
    plt.title(f"Precision at different IOUs: {env}")
    plt.xticks(np.arange(0, 1, 0.1))
    plt.xlabel("IOUs")
    plt.ylabel('Precision')
    plt.show()

    plt.scatter(np.arange(0, 0.95, 0.01), recallAtIOU, s=8)
    plt.title(f"Recall at different IOUs: {env}")
    plt.xticks(np.arange(0, 1, 0.1))
    plt.xlabel("IOUs")
    plt.ylabel('Recall')
    plt.show()
    

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

    EvaluateResults(dirCity, pathCity, dirValCity, pathValCity, "City")
    EvaluateResults(dirHighway, pathHighway, dirValHighway, pathValHighway, "Highway")
    EvaluateResults(dirPark, pathPark, dirValPark, pathValPark, "Park")



    
    
    
    
        







    

    




    
