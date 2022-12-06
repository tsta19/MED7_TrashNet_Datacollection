# --- Imports ---
import json
import cv2
import os
import numpy as np


class ImageSaver:

	mainDirectory = "EvaluationPrep\TrAAUsh-main\images"
	txtOutputDirectory = "EvaluationPrep//txt_yolo_files"

	# Category Arrays
	batteries = []
	glass = []
	hardPlastic = []
	foodDrinksContainers = []
	metal = []
	paper = []
	plastic = []
	restGarbage = []
	softPlastic = []
	textiles = []
	totalAnnotations = []

	def load_images(self):
		""" Load and read images from a directory """
		for filename in os.listdir(self.mainDirectory):
			img = cv2.imread(os.path.join(self.mainDirectory,filename))
			#print("Filename:", filename)
			return img
	
	def load_json(self):
		""" Load and opens a JSON file for editing while returning the data and path """
		jsonFilePath = "EvaluationPrep//TrAAUsh-main//annotations.json"
		f = open(jsonFilePath)
		data = json.load(f)
		return data, jsonFilePath

	def sort_labels_to_categories(self):
		""" Sorts all the trash category labels into arrays and prints the statistics of each category """
		data = self.load_json()

		for value in data["images"]:
			if value["annotations"][0]["label"] == "Batterier":
				self.batteries.append(value["annotations"][0]["label"])

			if value["annotations"][0]["label"] == "Glas":
				self.glass.append(value["annotations"][0]["label"])

			if value["annotations"][0]["label"] == "Hard Plast":
				self.hardPlastic.append(value["annotations"][0]["label"])

			if value["annotations"][0]["label"] == "MadDrikkeKartoner":
				self.foodDrinksContainers.append(value["annotations"][0]["label"])

			if value["annotations"][0]["label"] == "Metal":
				self.metal.append(value["annotations"][0]["label"])

			if value["annotations"][0]["label"] == "Papir":
				self.paper.append(value["annotations"][0]["label"])

			if value["annotations"][0]["label"] == "Plast":
				self.plastic.append(value["annotations"][0]["label"])

			if value["annotations"][0]["label"] == "Restaffald":
				self.restGarbage.append(value["annotations"][0]["label"])

			if value["annotations"][0]["label"] == "Soft Plast":
				self.softPlastic.append(value["annotations"][0]["label"])
			
			if value["annotations"][0]["label"] == "Tekstiler":
				self.textiles.append(value["annotations"][0]["label"])

		print("*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*")
		print(f"Number of images in each category:")
		print(f"Batteries: {len(self.batteries)}")
		print(f"Glass: {len(self.glass)}")
		print(f"Hard Plastic: {len(self.hardPlastic)}")
		print(f"Food & Drink Containers: {len(self.foodDrinksContainers)}")
		print(f"Metal: {len(self.metal)}")
		print(f"Paper: {len(self.paper)}")
		print(f"Plastic: {len(self.plastic)}")
		print(f"RestGarbage: {len(self.restGarbage)}")
		print(f"Soft Plastic: {len(self.softPlastic)}")
		print(f"Textiles: {len(self.textiles)}")
		print(f"")
		print(f"Results of Category Analysis:")
		print(f"Total amount of labels (unique): {len(self.batteries) + len(self.glass) + len(self.hardPlastic) + len(self.foodDrinksContainers) + len(self.metal) + len(self.paper) + len(self.plastic) + len(self.restGarbage) + len(self.softPlastic) + len(self.textiles)}")
		print(f"Total amount of images: {len(data['images'])}")
		print(f"Total annotations: {len(self.totalAnnotations)}")
		print("*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*")

	def draw_image_bounding_box(self, print_values=False, save_img=True):
		""" For every image in a directory it matches the image with the entry in the JSON and every bounding box
		is applied to the image, while also saving it """
		imagePath = "saved_images"
		data, jsonFilePath = self.load_json()
		index = 0
		print(f"Opening JSON File from Path: {jsonFilePath} for editing...")
		# For every image in directory
		for i in range(0, len(os.listdir(self.mainDirectory))):
			print(f"Saved Image {index} / {len(os.listdir(self.mainDirectory))}", end="\r")

			# For every entry in the JSON annotations 
			for entry in data["images"][index]["annotations"]:
				currentEntry = data['images'][index]["image"]
				imageArray = cv2.imread(os.path.join(self.mainDirectory, currentEntry))
				# Image rotated to account for the incorrect coordinates in the JSON
				imageArray = cv2.rotate(imageArray, cv2.ROTATE_90_COUNTERCLOCKWISE)

				# For every bounding box coordinate listed in annotations
				for i in range(0, len(data["images"][index]["annotations"])):
					x, y, w, h = data["images"][index]["annotations"][i]["coordinates"]["x"], data["images"][index]["annotations"][i]["coordinates"]["y"], data["images"][index]["annotations"][i]["coordinates"]["width"], data["images"][index]["annotations"][i]["coordinates"]["height"]
					cv2.rectangle(imageArray,(x, y), (x + w, y + h),(0,0,255),5)

				if save_img:
					imageArray = cv2.rotate(imageArray, cv2.ROTATE_90_CLOCKWISE)
					cv2.imwrite(os.path.join(imagePath , f"{currentEntry}"), imageArray)

				if print_values:
					print(f"IMAGE: ", data['images'][index]["image"], "LABEL:", entry["label"], "BB-COORIDNATES:", entry["coordinates"]["x"], entry["coordinates"]["y"], entry["coordinates"]["width"], entry["coordinates"]["height"])

			index += 1
		
	def print_directory_statistics(self):
		""" Prints statistics of the coordinates to a txt file """
		data = self.load_json()
		index = 0
		with open(f'EvaluationPrep//AnnotaitonStatistics.txt', 'w') as f:
			for entry in data["images"][index]["annotations"]:
				currentEntry = data['images'][index]["image"]
				imageArray = cv2.imread(os.path.join(self.mainDirectory,currentEntry))
				cv2.imshow(f"{currentEntry}", imageArray)
				cv2.waitKey(0)
			index += 1
		data = self.load_json()
		index = 0
		print(data["images"][index]["annotations"][index]["coordinates"]["x"])

	def rescale_images_bb(self, print_values=False, save_img=True, image_size=256):
		""" Rescales image to be neural network friendly while also saving a txt file with normalized values for the YOLO Styled network """
		imagePath = "saved_images"
		data, jsonFilePath = self.load_json()
		index = 0
		print(f"Opening JSON File from Path: {jsonFilePath} for editing...")

		# For every image in mainDirectory
		for i in range(0, len(os.listdir(self.mainDirectory))):
			print(f"Saved Image {index} / {len(os.listdir(self.mainDirectory))}", end="\r")
			# For every entry in the JSON annotations
			for entry in data["images"][index]["annotations"]:
				currentEntry = data['images'][index]["image"]
				imageArray = cv2.imread(os.path.join(self.mainDirectory, currentEntry))
				resizedImageArray = cv2.resize(imageArray, (image_size, image_size))
				resizedImageArray = cv2.rotate(resizedImageArray, cv2.ROTATE_90_COUNTERCLOCKWISE)

				imageName = currentEntry.replace(".jpg", ".txt")
				with open(f"{self.txtOutputDirectory}/{imageName}", "w") as file:
					# For every bounding box coordinate in annotations
					for i in range(0, len(data["images"][index]["annotations"])):
						scaleW = (image_size / imageArray.shape[0])
						scaleH = (image_size / imageArray.shape[1])
						####	Classes: Restaffald = 0, Plast/Hård plast/Blød plast = 1, Papir/pap = 2, Metal = 3
						if (data["images"][index]["annotations"][i]["label"]) == 'Restaffald':
							label = 2
						elif (data["images"][index]["annotations"][i]["label"]) == 'Plast' or 'Hard Plast' or 'Blød Plast':
							label = 1
						elif (data["images"][index]["annotations"][i]["label"]) == 'Papir' or 'Pap':
							label = 3
						elif (data["images"][index]["annotations"][i]["label"]) == 'Metal':
							label = 0
						else:
							label = 9
						newX = int((data["images"][index]["annotations"][i]["coordinates"]["x"] * scaleW))
						newY = int((data["images"][index]["annotations"][i]["coordinates"]["y"] * scaleH))
						newW = int((data["images"][index]["annotations"][i]["coordinates"]["width"] * scaleW))
						newH = int((data["images"][index]["annotations"][i]["coordinates"]["height"] * scaleH))
						cv2.rectangle(resizedImageArray,(newX, newY), (newX + newW, newY + newH),(0,0,255),2)
						centerX = int(newX + newW / 2)
						centerY = int(newY + newH / 2)
						# -------------- Normalized values For Yolo Network --------------
						normalizedCenterX = centerX / image_size
						normalizedCenterY = centerY / image_size
						normalizedW = newW / image_size
						normalizedH = newH / image_size
						file.write(f"{label} {normalizedCenterX} {normalizedCenterY} {normalizedW} {normalizedH}" '\n')

				if save_img:
					resizedImageArray = cv2.rotate(resizedImageArray, cv2.ROTATE_90_CLOCKWISE)
					cv2.imwrite(os.path.join(imagePath , f"{currentEntry}"), resizedImageArray)

				if print_values:
					print(f"IMAGE: ", data['images'][index]["image"], "LABEL:", entry["label"], "BB-COORIDNATES:", entry["coordinates"]["x"], entry["coordinates"]["y"], entry["coordinates"]["width"], entry["coordinates"]["height"])

			index += 1

iSave = ImageSaver()
iSave.rescale_images_bb()
