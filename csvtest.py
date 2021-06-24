import csv
import time
from keras.layers import Reshape, MaxPooling2D
from keras.layers import Conv2D, Dense, Flatten, Dropout
from keras.optimizers import Adam
import tensorflow as tf
from keras import backend as K
from keras.models import Sequential

def construct(classString, parameters, x_train=None):
    constructor = globals()[classString]
    instance = 0
    if classString == "Conv2D":
        if x_train is not None:
            instance = constructor(kernel_size = int(parameters[0]), strides = int(parameters[1]),
                                   filters = int(parameters[2]), padding = str(parameters[3]),
                                   activation = str(parameters[4]), name = str(parameters[5]),
                                   input_shape=x_train.shape[1:])
        else:
            instance = constructor(kernel_size=int(parameters[0]), strides=int(parameters[1]),
                                   filters=int(parameters[2]), padding=str(parameters[3]),
                                   activation=str(parameters[4]), name=str(parameters[5]))

    if classString == "Dropout":
        instance = constructor(rate = float(parameters[0]))

    if classString == "MaxPooling2D":
        instance = constructor(pool_size = int(parameters[0]), strides = int(parameters[1]))
    if classString == "Dense":
        instance = constructor(units = int(parameters[0]), activation = str(parameters[1]))
    if classString == "Adam":
        instance = constructor(lr = float(parameters[0]))
    if classString == "Flatten":
        instance = constructor()

    return instance

def read_networks_from_csv(x_train):
    networks = []
    parameters_per_layer = {"Conv2D":6, "Dropout":1, "MaxPooling2D":2, "Dense":2, "Flatten":0}
    parameters_per_optimizer = {"Adam":1}
    t1 = time.time()

    with open("models.csv", newline= '', encoding="utf-8-sig") as csvfile:
        modelno = 0
        reader = csv.reader(csvfile, delimiter = ";", dialect = "excel")
        for row in reader:
            modelno= modelno + 1
            first_conv_flag = 1
            model = Sequential(name=str(modelno))
            for i in range(len(row)):
                if row[i] in parameters_per_layer.keys():
                    parameters = row[i+1:i+parameters_per_layer[row[i]]+1]
 #                   print("Class: " + row[i] + "Parameters: " + str(parameters))
                    if first_conv_flag:
                        model.add(construct(row[i], parameters, x_train))
                        first_conv_flag = 0
                    else:
                        model.add(construct(row[i], parameters))
                    i = i + parameters_per_layer[row[i]]
                elif row[i] in parameters_per_optimizer.keys():
                    parameters = row[i + 1:i + parameters_per_optimizer[row[i]] + 1]
  #                  print("Optimizer: " + row[i] + "Parameters: " + str(parameters))
                    model.compile(optimizer = construct(row[i], parameters), loss = str(row[-2]), metrics = [str(row[-1])])
            network = ( (int(row[0]), int(row[1])), model)
            networks.append(network)
    return networks





