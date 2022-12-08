import cv2
import numpy as np
import os


if __name__ == "__main__":


    dirCity = os.listdir("SegmentEval//CityPred//images")
    pathCity = "SegmentEval//CityPred//images//"



    for i in range(0, len(dirCity)):
        print(dirCity[i])
