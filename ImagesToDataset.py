import time
import random
####just copy and paste the below given code to your shell

# KERAS

import numpy as np
import os
import cv2
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from Definitions import *
import gc
from imutils import paths
from keras.utils import to_categorical



def get_data(path_to_images, remove_ratio):
    listing = os.listdir(path_to_images)


    imlist = os.listdir(path_to_images)
    farrowinglist = []
    nofarrowinglist = []
    for f in imlist:
        if not f.endswith(".jpg") and not f.endswith(".txt") and not f.endswith(".dat") and not f.endswith(".py"):
            f2 = os.listdir(path_to_images+f)
            for f3 in f2:
                if not f3.endswith(".jpg") and not f3.endswith(".txt") and not f3.endswith(".dat") and not f3.endswith(".py"):
               #     print(str(os.listdir(path_to_images+f+"\\"+f3)))
                   # print(path_to_images+f+"\\"+f3)
                    if f3 == "farrow_full":
                        farrowings = os.listdir(path_to_images+f+"\\"+f3)
                        tofarrowinglist = [path_to_images+f+"\\"+f3+"\\"+farrow for farrow in farrowings]
                        farrowinglist.extend(tofarrowinglist)
                    elif f3 == "nofarrow_full":
                        nofarrowings = os.listdir(path_to_images+f+"\\"+f3)
                        tonofarrowinglist = [path_to_images+f+"\\"+f3+"\\"+nofarrow for nofarrow in nofarrowings]
                        nofarrowinglist.extend(tonofarrowinglist)


    imlist = []
    #The line of code below reduces the number of no farrowings to match the number of farrowings
    nofarrowinglist = random.choices(nofarrowinglist, k=int(len(nofarrowinglist) / remove_ratio))

    imlist.extend(nofarrowinglist)
    imlist.extend(farrowinglist)
    num_samples = len(imlist)
    nb_class0 = len(nofarrowinglist)
    nb_class1 = len(farrowinglist)
    del farrowinglist, nofarrowinglist
    print("Number of farrowings: " + str(nb_class1) + "\n Number of no farrowings: " + str(nb_class0))
  #  print(imlist)
    im1 = cv2.imread(imlist[0], cv2.IMREAD_COLOR)
    img_rows = np.size(im1, 0)
    img_cols = np.size(im1, 1)  #  get the size of the images

    # create matrix to store all flattened images

    immatrix = np.array([np.array(cv2.imread(im2, cv2.IMREAD_COLOR)).flatten()
                      for im2 in imlist], 'f')

    del imlist, farrowings, nofarrowings, im1
    # immatrix = []
   # arr = []
   # for im2 in imlist:
   #     arr.append(np.array(cv2.imread(im2, cv2.IMREAD_COLOR)).flatten())
   # import sys
   # print(sys.getsizeof(arr))
   # immatrixtemp = np.array(arr[:int(len(arr)/2)], "f")
   # immatrixtemp2 = np.array(arr[int(len(arr)/2):], "f")
   # immatrix.extend(immatrixtemp)
   # immatrix.extend(immatrixtemp2)
    label = np.ones((num_samples,), dtype=int)
    label[0:(nb_class0-1)] = 0
    label[nb_class0:(nb_class0+nb_class1-1)] = 1
    data, Label = shuffle(immatrix, label, random_state=2)
    del immatrix
    train_data = [data, Label]
    del data, Label
    # STEP 1: split X and y into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(train_data[0], train_data[1], test_size=0.25, random_state=4)
    #x_train, x_val, y_train, y_val = train_test_split(train_data[0], train_data[1], test_size=0.25, random_state=4)
    del train_data
    # Return the sets to their original shapes
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, num_channels)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, num_channels)

    # Scale the values from 0 to 1 instead of 0 to 255
    gc.collect()
    x_train2 = x_train.astype('float32')
    x_test2 = x_test.astype('float32')
    x_train2 /= 255
    x_test2 /= 255

    nofarrowingclass = []
    farrowingclass = []


    return (x_train2, y_train), (x_test2, y_test)
