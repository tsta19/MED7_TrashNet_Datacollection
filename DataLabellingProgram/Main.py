# Directory
# Every image in directory
# 1-x classes keyboard input
# Sort image into corresponding directory

import cv2
import os
import numpy as np
import shutil
import time

class Main:
    

    def GetMainDirectory(self):
        """ Path for the images """
        main_directory = "images"
        return main_directory
    
    def GetTXTDirectory(self):
        """ Path for the txt's """
        txt_directory = "txt_files"
        return txt_directory
    
    def GetUPTXTDirectory(self):
        """ Path for the updated txt's """
        uptxt_directory = "updated_txt_files"
        return uptxt_directory
    
    def ThePathFinder(self, main_directory, folder_destination: str) -> str:
        """ Find a category from the folder_destination string """
        return f"{main_directory}/{folder_destination}/"

    def DirectoryIntegrityChecker(self):
        """ Integrity checker for directory to ensure the images are not corrupted and have the correct file extension"""
        failedChecks = []
        successfulChecks = []
        i = 0
        for image in os.listdir(self.GetMainDirectory()):
            if len(os.listdir(self.GetMainDirectory())) != 0:
                print(f"Checking: {i} / {len(os.listdir(self.GetMainDirectory()))-1}", end="\r")
                if image != None and str(image).endswith(".jpg") or str(image).endswith(".png"):
                    successfulChecks.append(image)
                else:
                    print(f"{image} Is either empty or has wrong file extension")
                    failedChecks.append(image)
                i += 1
            else:
                print("Main Directory is Empty")

        print("******** Integrity Check Finished ********")
        if len(failedChecks) > 0:
            print(f"Image(s) failed the check {failedChecks}")
            print("Fix this by ensuring the file type is jpg or png")
        else:
            print("All images satisfied the checker")
            

    def IterateOverImages(self):
        """ Iterates over images in the image directory where you can input a class from 1-6 and update a text file accordingly"""
        for image in os.listdir(self.GetMainDirectory()):
            imageString = image.replace(".jpg", ".txt")
            txtString = image.replace('bbox', "")
            txtStringEdited = txtString.replace('.png','.txt')
            replaceText = "Class here"
            print("image name: " + image)
            currImage = np.asarray(cv2.imread(os.path.join(self.GetMainDirectory(), image)))
            print(currImage.dtype)
            resizedImage = cv2.resize(currImage, (500, 500))
            with open(f"{self.GetTXTDirectory()}/{txtStringEdited}", "r") as file:
                filedata = file.read()
                cv2.imshow(f"{image} / {len(os.listdir(self.GetMainDirectory()))}", resizedImage)
                cv2.waitKey(100)

                availableClasses = ["1", "2", "3", "4", "5", "6"]
                userInput = self.check_user_input()
            
                if userInput not in availableClasses:
                    print("You fucked up, try again!")
                    self.check_user_input()
                elif userInput == "1":
                    with open(f"{self.GetUPTXTDirectory()}/{txtStringEdited}", "w") as file:
                        filedata = filedata.replace(replaceText, str(0))
                        file.write(filedata)
                        shutil.move(self.GetMainDirectory() + "/" + image, self.ThePathFinder("categories", "metal"))
                        print(f"Moved {image} to class 1")
                elif userInput == "2":
                    with open(f"{self.GetUPTXTDirectory()}/{txtStringEdited}", "w") as file:
                        filedata = filedata.replace(replaceText, str(1))
                        file.write(filedata)
                        shutil.move(self.GetMainDirectory() + "/" + image, self.ThePathFinder("categories", "plastic"))
                        print(f"Moved {image} to class 2")
                elif userInput == "3":
                    with open(f"{self.GetUPTXTDirectory()}/{txtStringEdited}", "w") as file:
                        filedata = filedata.replace(replaceText, str(2))
                        file.write(filedata)
                        shutil.move(self.GetMainDirectory() + "/" + image, self.ThePathFinder("categories", "restaffald"))
                        print(f"Moved {image} to class 3")
                elif userInput == "4":
                    shutil.move(self.GetMainDirectory() + "/" + image, self.ThePathFinder("categories", "discarded"))
                    print(f"Discarded {image}")

            cv2.destroyAllWindows()

    def check_user_input(self):
        userInput = input(
            "Keypress Options: | 1: Metal | 2: Plastic | 3: Restaffald | 4: Discard |  Input Here:  ")
        return userInput

main = Main()
main.DirectoryIntegrityChecker()
main.IterateOverImages()