from keras.preprocessing import image
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
import numpy as np
from sklearn.cluster import KMeans
import cv2



def VGGKmeans():
    image.LOAD_TRUNCATED_IMAGES = False
    model = VGG16(weights='imagenet', include_top=False)

    # Variables
    number_clusters = 2

    img = cv2.imread('images/water_bottle.jpg')

    twoDImage = img.reshape((-1,3))
    twoDImage = np.float32(twoDImage)


    img_data = image.img_to_array(img)
    img_data = np.expand_dims(img_data, axis=0)
    img_data = preprocess_input(img_data)
    features = np.array(model.predict(img_data, use_multiprocessing=True))
    print(features)
    print(features.shape)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 2
    attempts=10

    kmeans = KMeans(n_clusters=number_clusters, random_state=0, algorithm="elkan", init="k-means++").fit(np.array(features))

    print(kmeans.labels_)

    #
    # ret,label,center=cv2.kmeans(features,K,None,criteria,attempts,cv2.KMEANS_PP_CENTERS)
    # center = np.uint8(center)
    # res = center[label.flatten()]
    # result_image = res.reshape((img.shape))


    #
    # # Loop over files and get features
    # for i, imagepath in enumerate(filelist):
    #     print("    Status: %s / %s" %(i, len(filelist)), end="\r")
    #     img = image.load_img(imagepath, target_size=(224, 224))
    #     img_data = image.img_to_array(img)
    #     img_data = np.expand_dims(img_data, axis=0)
    #     img_data = preprocess_input(img_data)
    #     features = np.array(model.predict(img_data, use_multiprocessing=True))
    #     featurelist.append(features.flatten())

    # Clustering
    kmeans = KMeans(n_clusters=number_clusters, random_state=0, algorithm="elkan", init="k-means++").fit(np.array(featurelist))

    print(kmeans.labels_)