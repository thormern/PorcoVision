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

def construct_network_from_string(networkstring, name, x_train):
    parameters_per_layer = {"Conv2D":6, "Dropout":1, "MaxPooling2D":2, "Dense":2, "Flatten":0}
    parameters_per_optimizer = {"Adam":1}
    t1 = time.time()


    layers = networkstring.split(";")
    print(layers)

    first_conv_flag = 1
    model = Sequential(name=str(name))
    for i in range(len(layers)):
        if layers[i] in parameters_per_layer.keys():
            parameters = layers[i+1:i+parameters_per_layer[layers[i]]+1]
#                   print("Class: " + row[i] + "Parameters: " + str(parameters))
            if first_conv_flag:
                model.add(construct(layers[i], parameters, x_train))
                first_conv_flag = 0
            else:
                model.add(construct(layers[i], parameters))
            i = i + parameters_per_layer[layers[i]]
        elif layers[i] in parameters_per_optimizer.keys():
            parameters = layers[i + 1:i + parameters_per_optimizer[layers[i]] + 1]
#                  print("Optimizer: " + row[i] + "Parameters: " + str(parameters))
            model.compile(optimizer = construct(layers[i], parameters), loss = str(layers[-2]), metrics = [str(layers[-1])])
    network = ( (int(layers[0]), int(layers[1])), model)
    return network







