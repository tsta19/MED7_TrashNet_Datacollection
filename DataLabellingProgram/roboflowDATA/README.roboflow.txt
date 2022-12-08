
LitterDataFinal - v1 2022-12-08 2:24pm
==============================

This dataset was exported via roboflow.com on December 8, 2022 at 1:24 PM GMT

Roboflow is an end-to-end computer vision platform that helps you
* collaborate with your team on computer vision projects
* collect & organize images
* understand unstructured image data
* annotate, and create datasets
* export, train, and deploy computer vision models
* use active learning to improve your dataset over time

It includes 558 images.
Litter are annotated in YOLO v5 PyTorch format.

The following pre-processing was applied to each image:
* Auto-orientation of pixel data (with EXIF-orientation stripping)
* Resize to 250x250 (Stretch)

The following augmentation was applied to create 3 versions of each source image:

The following transformations were applied to the bounding boxes of each image:
* Equal probability of one of the following 90-degree rotations: none, clockwise, counter-clockwise
* Random shear of between -15° to +15° horizontally and -15° to +15° vertically
* Random exposure adjustment of between -20 and +20 percent
* Salt and pepper noise was applied to 10 percent of pixels


