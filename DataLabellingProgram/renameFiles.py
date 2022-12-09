import os

if __name__ == '__main__':

    dirIMG = os.listdir("DataLabellingProgram/categories/restaffald")
    pathIMG = "DataLabellingProgram/categories/restaffald/"

    dirTXT = os.listdir("DataLabellingProgram/txt_categories/restaffald")
    pathTXT = "DataLabellingProgram/txt_categories/restaffald/"
    #os.rename(pathIMG+dirIMG[0], pathIMG + f"metal{1}.jpg")
    #os.rename(pathTXT+dirTXT[0], pathTXT + f"metal{1}.txt")
    t = 0
    for i in range(0, len(dirIMG)):
        os.rename(pathIMG+dirIMG[t], pathIMG + f"restaffald{i}.jpg")
        os.rename(pathTXT+dirTXT[t], pathTXT + f"restaffald{i}.txt")
        print('done')
        t += 1
        


    