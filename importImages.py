from imutils import paths
import random
import cv2
from keras.preprocessing.image import img_to_array
import os
from sklearn.model_selection import train_test_split
import numpy as np
from PIL import Image
import time
"""
Inspiration: https://www.pyimagesearch.com/2017/12/11/image-classification-with-keras-and-deep-learning/

"""
def new_import_images(path_to_images, split, image_size):
    seed = 42
    data = []
  #  data = np.array(dtype="float")
    labels = []
    imagePaths = list(paths.list_images(path_to_images))
  #  print(imagePaths)
    random.seed(seed)
    random.shuffle(imagePaths)
    for imagePath in imagePaths:

        image = cv2.imread(imagePath)
        image = cv2.resize(image, (image_size, image_size))
        image = img_to_array(image)
        data.append(image)

        label = imagePath.split(os.path.sep)[-2]
        label = 1 if label == "farrow_full" else 0
        labels.append(label)

    data = np.array(data, dtype="float32") / 255.0
    labels = np.array(labels)

    # partition the data into training and testing splits using 75% of
    # the data for training and the remaining 25% for testing
    testX = 0
    testY = 0
    if split:
        (trainX, testX, trainY, testY) = train_test_split(data,
                                                      labels, test_size=0.2, random_state=seed)
    else:
        #trainX = data, trainY = labels
        (trainX, testX, trainY, testY) = train_test_split(data,
                                                          labels, test_size=0.0, random_state=seed)
    return (trainX, trainY, testX, testY)
#new_get_data("D:\\porcocropped\\augmented")