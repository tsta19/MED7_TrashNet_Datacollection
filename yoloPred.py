import torch
import cv2
import pandas as df

model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt')
model.conf = 0.1

img = cv2.imread('bottle.jpg')[..., ::-1]

results = model(img)


results.print()
results.show()