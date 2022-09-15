import cv2
import numpy as np

class ImageOperator:

    def resize_image(self, image_to_resize, res_x, res_y):
        rescale_dimensions = (res_y, res_x)
        resize_image = cv2.resize(image_to_resize, rescale_dimensions, interpolation=cv2.INTER_AREA)
        resize_image = cv2.cvtColor(resize_image, cv2.COLOR_BGR2RGB)
        return resize_image

    def remove_mask_from_image(self, image, low_color_values, high_color_values):
        imgHSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(imgHSV, low_color_values, high_color_values)
        mask = 255-mask # Inverse the mask to remove the colors from the low-high range values
        outmasked_image = cv2.bitwise_and(image, image, mask=mask)
        return outmasked_image
