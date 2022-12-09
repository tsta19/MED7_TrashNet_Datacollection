import os
from natsort import natsorted

folder = r'rename\\'
num = 93
txt = ".txt"
png = ".png"

for file_name in natsorted(os.listdir(folder)):
    print(file_name)
    source = folder + file_name

    destination = folder + "im" + str(num) + png

    os.rename(source,destination)
    num += 1

print("done")