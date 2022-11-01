import pixellib
from pixellib.semantic import semantic_segmentation
import requests


segment_image = semantic_segmentation()
segment_image.load_pascalvoc_model("deeplabv3_xception_tf_dim_ordering_tf_kernels.h5")
segment_image.segmentAsPascalvoc("roipic.png", output_image_name = "output_image.png")