import matplotlib.pyplot as plt
import tensorflow as tf
#sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
import time
from keras.models import Sequential
from keras.layers import Reshape, MaxPooling2D
from keras.layers import Conv2D, Dense, Flatten, Dropout
from    keras.optimizers import Adam
from Definitions import *
from ImagesToDataset import *
import math



def pre_process_image(image, training):
    # This function takes a single image as input,
    # and a boolean whether to build the training or testing graph.

    if training:
        # For training, add the following to the TensorFlow graph.

        #  Randomly crop the input image.
        image = tf.random_crop(image, size=[img_size_cropped, img_size_cropped, num_channels])

        # Randomly flip the image horizontally.
        image = tf.image.random_flip_left_right(image)

        # Randomly adjust hue, contrast and saturation.
        image = tf.image.random_hue(image, max_delta=0.05)
        image = tf.image.random_contrast(image, lower=0.3, upper=1.0)
        image = tf.image.random_brightness(image, max_delta=0.2)
        image = tf.image.random_saturation(image, lower=0.0, upper=2.0)

        # Some of these functions may overflow and result in pixel
        # values beyond the [0, 1] range. It is unclear from the
        # documentation of TensorFlow 0.10.0rc0 whether this is
        # intended. A simple solution is to limit the range.

        # Limit the image pixels between [0, 1] in case of overflow.
        image = tf.minimum(image, 1.0)
        image = tf.maximum(image, 0.0)
    else:
        # For training, add the following to the TensorFlow graph.

        # Crop the input image around the centre so it is the same
        # size as images that are randomly cropped during training.
        image = tf.image.resize_image_with_crop_or_pad(image,
                                                       target_height=img_size_cropped,
                                                       target_width=img_size_cropped)

    return image


def pre_process(images, training):
    # Use TensorFlow to loop over all the input images and call
    # the function above which takes a single image as input.
    images = tf.map_fn(lambda image: pre_process_image(image, training), images)

    return images

save_dir = os.path.join(os.getcwd(), 'saved_models')

def pre_training(remove_farrowings):
    # get data from ImagesToDataset
    (x_train, y_train), (x_test, y_test) = get_data(path1, remove_farrowings)

    # convert class vectors to binary class matrices
    y_train_one_hot = to_categorical(y_train)
    y_test_one_hot = to_categorical(y_test)

    print("Size of:")
    print("- Training-set:\t\t{}".format(len(x_train)))
    print("- Test-set:\t\t{}".format(len(x_test)))


    # Images are stored in one-dimensional arrays of this length.
    img_size_flat = img_size * img_size * num_channels

    # Tuple with height and width of images used to reshape arrays.
    # This is used for plotting the images.
    img_shape = (img_size, img_size)

    # Tuple with height, width and depth used to reshape arrays.
    # This is used for reshaping in Keras.
    img_shape_full = (img_size, img_size, num_channels)
    print("reached")
    return (x_train, y_train), (x_test, y_test), \
           y_train_one_hot, y_test_one_hot, \
           img_size_flat, img_shape, img_shape_full

def create_model_with_params(x_train, layers, compileparams):
    print("Number of layers: " + str(len(layers)))
    model = Sequential()
    for layer in layers:
        model.add(layer)

    model.compile(optimizer=compileparams[0], loss=compileparams[1], metrics=compileparams[2])
    return model

def create_model(x_train):

    # Start construction of the Keras Sequential model.
    model = Sequential()

    # First convolutional layer with ReLU-activation and max-pooling.
    #model.add(Conv2D(kernel_size=5, strides=1, filters=16, padding='same',
    #                 activation='relu', name='layer_conv1'))
    model.add(Conv2D(kernel_size=5, strides=(1,1), filters=32, padding='same',
                      activation='relu', input_shape=x_train.shape[1:]))
    model.add(Dropout(0.5))
    model.add(MaxPooling2D(pool_size=2, strides=2))

    # Second convolutional layer with ReLU-activation and max-pooling.
    model.add(Conv2D(kernel_size=5, strides=1, filters=64, padding='same',
                     activation='relu', name='layer_conv2'))
    model.add(Dropout(0.5))
    model.add(MaxPooling2D(pool_size=2, strides=2))


    # Flatten the 4-rank output of the convolutional layers
    # to 2-rank that can be input to a fully-connected / dense layer.
    model.add(Flatten())

    # First fully-connected / dense layer with ReLU-activation.
    model.add(Dense(256, activation='relu'))

    # Last fully-connected / dense layer with softmax-activation
    # for use in classification.
    model.add(Dropout(0.7))
    model.add(Dense(num_classes, activation='sigmoid'))
    optimizer = Adam(lr=1e-3)

    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    return model

def save_model(model, Ntrainingsamples, filenamesuffix):
    save_dir = os.path.join(os.getcwd(), 'saved_models')

    model_name = "b" + str(batch_size) + "e" + str(epochs) + "n" + str(Ntrainingsamples) + filenamesuffix

    # model_name = input('Enter filename to save: ')
    if (model_name != ''):
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        model_path = os.path.join(save_dir, model_name)
        while os.path.isfile(model_path):
            print("Appending \"new\" to file name")
            model_path = model_path + "new"
        model.save(model_path)
        print('Saved trained model at %s ' % model_path)

def start_training(x_train, x_test, y_train_one_hot, y_test_one_hot, layers, compileparams, filenamesuffix):
    #model = create_model(x_train)
    model = create_model_with_params(x_train, layers, compileparams)
    t1 = time.time()
    print("Starting training...")
    model.fit(x_train, y_train_one_hot,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_test, y_test_one_hot),
              shuffle=True)
    t2 = time.time()
    result = model.evaluate(x=x_test,
                            y=y_test_one_hot)

    for name, value in zip(model.metrics_names, result):
        print(name, value)

    #print("{0}: {1:.2%}".format(model.metrics_names[1], result[1]))
    print("Model took " + str(t2 - t1) + "to train")
    save_model(model, len(x_train), filenamesuffix)
    return model


#model = create_model()
#t1 = time.time()
#print("Starting training...")
#model.fit(x_train, y_train_one_hot,
#              batch_size=batch_size,
#              epochs=epochs,
#              validation_data=(x_test, y_test_one_hot),
#              shuffle=True)
#t2 = time.time()
#result = model.evaluate(x=x_test,
#                        y=y_test_one_hot)
#t3 = time.time()
#for name, value in zip(model.metrics_names, result):
#    print(name, value)

#print("{0}: {1:.2%}".format(model.metrics_names[1], result[1]))
#print("Model took " + str(t2-t1) + "to train")
#model_name = "b" + str(batch_size) + "e" + str(epochs) + "n" + str(len(x_train)) + "3drop+sigmoid+2048dense"

#model_name = input('Enter filename to save: ')
#if (model_name != ''):
#    if not os.path.isdir(save_dir):
#        os.makedirs(save_dir)
#    model_path = os.path.join(save_dir, model_name)
#    if os.path.isfile(model_path):
#        model_path = model_path+"new"
#    model.save(model_path)
#    print('Saved trained model at %s ' % model_path)



