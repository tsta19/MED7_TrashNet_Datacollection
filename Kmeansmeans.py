import pixellib
from pixellib.torchbackend.instance import instanceSegmentation
import cv2
import numpy as np

img = cv2.imread("data/savedimages/garbage1.png")
roi = img[0:img.shape[0],323:545]

#cv2.imwrite("roipic.png",roi)

ins = instanceSegmentation()
ins.load_model("pointrend_resnet50.pkl", confidence=0.2)
results, output = ins.segmentImage("data/savedimages/garbage0.png", show_bboxes=True, output_image_name="data/savedimages/output_image.png")
print(results['boxes'])