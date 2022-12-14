import json
import cv2
import os
import numpy as np
import shutil


def filterUselessClass(dirLabels, pathLabels, newPath):
    print(len(dirLabels))
    for i in range(0, len(dirLabels)):

        with open(f"{pathLabels}/{dirLabels[i]}", "r") as input:
            with open(f"{newPath}/{dirLabels[i]}", "w") as output:
                # iterate all lines from file
                for line in input:
                    stripped = line.strip("\n")
                    templine = stripped.split(" ")
                    print(templine[0])
                    print(templine)
                    firstIndx = int(templine[0])
                    print(f"firstline: {firstIndx}")
                    print(f"firstline type: {type(firstIndx)}")
                    if firstIndx == 0:
                        output.write(line)
                    elif firstIndx == 1:
                        output.write(line)
                    elif firstIndx == 2:
                        output.write(line)

def verifyDirectory(dirLabels, pathLabels, dirImages, imgpath, oldimgpath):

    cntrLimitMetal = 0
    cntrLimitPlastic = 0
    cntrLimitRestaffald = 0

    metal = 48
    plastic = 200
    restaffald = 496
    l = 0


    print("Checking for empty txt files...")
    print(len(dirLabels))
    for i in range(0, len(dirLabels)):

        if metal <= cntrLimitMetal and plastic <= cntrLimitPlastic and restaffald <= cntrLimitRestaffald:
            print(f"All limits met - Metal: {cntrLimitMetal}, Plastic: {cntrLimitPlastic}, Restaffald: {cntrLimitRestaffald}")
            break
        if i == len(dirLabels) - 1:
            print(f"Metal: {cntrLimitMetal}, Plastic: {cntrLimitPlastic}, Restaffald: {cntrLimitRestaffald}")

        print(f"{pathLabels}/{dirLabels[i]}")
        with open(f"{pathLabels}/{dirLabels[i]}", "r") as input:
            print("Checking dataset class sizes...")
            # I already have file open at this point.. now what?
            input.seek(0) # Ensure you're at the start of the file..
            first_char = input.read(1) # Get the first character
            if not first_char:
                print ("file is empty") # The first character is the empty string..
                shutil.move(f"{imgpath}/{dirImages[i]}", oldimgpath)
                os.remove(f"{pathLabels}/{dirLabels[i]}")
                
                print(f"Deleted file empty file: {pathLabels}/{dirLabels[i]} and moved image file: {imgpath}/{dirImages[i]}")
            else:
                input.seek(0) # The first character wasn't empty. Return to the start of the file.
                # Use file now
                for line in input:
                    stripped = line.strip("\n")
                    templine = stripped.split(" ")
                    print(templine[0])
                    print(templine)
                    firstIndx = int(templine[0])
                    print(f"firstline: {firstIndx}")
                    print(f"firstline type: {type(firstIndx)}")
                    if firstIndx == 0:
                        if metal >= cntrLimitMetal:
                            cntrLimitMetal += 1
                        
                    elif firstIndx == 1:
                        if plastic >= cntrLimitPlastic:
                            cntrLimitPlastic += 1
                        
                    elif firstIndx == 2:
                        if restaffald >= cntrLimitRestaffald:
                            cntrLimitRestaffald += 1

            
                    

    
   

if __name__ == "__main__":

    # 0 = metal, 1 = plastic, 2 = restaffald
    # prep class counters: 
        # 48 Metal
        # 200 Plastic
        # 496 Restaffald
    # Loop through directory
        # Readlines
            # if class numbers is not 0, 1 or 2, delete line
    # Loop through directory
        # Readlines
            # for file in directory:
                # count classes, add to counter
                    # if counter >= limit, delete line from file
                # move file to folder
                # move corresponding images to other folder

    dirImages = os.listdir("EvaluationPrep/TrAAUsh-main/images")
    pathImages = "EvaluationPrep/TrAAUsh-main/images/"

    dirLabels = os.listdir("EvaluationPrep/txt_yolo_files2")
    pathLabels = "EvaluationPrep/txt_yolo_files2/"

    newPath = "EvaluationPrep/newTxts/"
    newDir = os.listdir("EvaluationPrep/newTxts")

    movetoImgPath = "EvaluationPrep/TrAAUsh4Yolo/images/"
    movetoImgDir = os.listdir("EvaluationPrep/TrAAUsh4Yolo/images")

    movetoLabelPath = "EvaluationPrep/TrAAUsh4Yolo/labels/"
    movetoLabelDir = os.listdir("EvaluationPrep/TrAAUsh4Yolo/labels")

    cntrLimitMetal = 0
    cntrLimitPlastic = 0
    cntrLimitRestaffald = 0

    metal = 11
    plastic = 50
    restaffald = 123
    l = 0

    #filterUselessClass(dirLabels=dirLabels, pathLabels=pathLabels, newpath=newPath)

    newLabelDir = os.listdir("EvaluationPrep/newTxts")

    

    for i in range(0, int(len(newLabelDir)*0.75)):
        if metal <= cntrLimitMetal and plastic <= cntrLimitPlastic and restaffald <= cntrLimitRestaffald:
            print(f"All limits met - Metal: {cntrLimitMetal}, Plastic: {cntrLimitPlastic}, Restaffald: {cntrLimitRestaffald}")
            break

        with open(f"{newPath}/{newDir[i]}", "r") as input:
            with open(f"{movetoLabelPath}/{newDir[i]}", "w") as output:
                # iterate all lines from file
                for line in input:
                    stripped = line.strip("\n")
                    templine = stripped.split(" ")
                    print(templine[0])
                    print(templine)
                    firstIndx = int(templine[0])
                    print(f"firstline: {firstIndx}")
                    print(f"firstline type: {type(firstIndx)}")
                    if firstIndx == 0:
                        if metal >= cntrLimitMetal:
                            output.write(line)
                            cntrLimitMetal += 1
                        
                    elif firstIndx == 1:
                        if plastic >= cntrLimitPlastic:
                            output.write(line)
                            cntrLimitPlastic += 1
                        
                    elif firstIndx == 2:
                        if restaffald >= cntrLimitRestaffald:
                            output.write(line)
                            cntrLimitRestaffald += 1
                    
                    
                        

        shutil.move(f"{pathImages}/{dirImages[i]}", movetoImgPath)
        print(f"Iteration number: {i}")
        l += 1


    print(f"Picture index used : 0-{l}")
    #verifyDirectory(dirLabels=movetoLabelDir, pathLabels=movetoLabelPath, dirImages=movetoImgDir, imgpath=movetoImgPath, oldimgpath=pathImages)

    """
    for i in range(0, len(movetoLabelDir)):
         with open(f"{movetoLabelPath}/{movetoLabelDir[i]}", "r") as input:
            print("Checking dataset class sizes...")

            for line in input:
                stripped = line.strip("\n")
                templine = stripped.split(" ")
                print(templine[0])
                print(templine)
                firstIndx = int(templine[0])
                print(f"firstline: {firstIndx}")
                print(f"firstline type: {type(firstIndx)}")
                if firstIndx == 0:
                    if metal >= cntrLimitMetal:
                        cntrLimitMetal += 1
                    
                elif firstIndx == 1:
                    if plastic >= cntrLimitPlastic:
                        cntrLimitPlastic += 1
                    
                elif firstIndx == 2:
                    if restaffald >= cntrLimitRestaffald:
                        cntrLimitRestaffald += 1

    print(f"Metal: {cntrLimitMetal}, Plastic: {cntrLimitPlastic}, Restaffald: {cntrLimitRestaffald}")

"""
    








            
