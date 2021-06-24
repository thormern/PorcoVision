import matplotlib.pyplot as plt

from ImagesToDataset import *
import math
from keras.models import load_model
import cv2
###########################################################
# Assumptions: It is assumed that the data used to train the model is the data returned by
#
#
###########################################################

print("Type the path of the folder that contains the saved models:")
savedModelsDir = input()
if not savedModelsDir.endswith("\\"):
    savedModelsDir+= "\\"
#savedModelsDir = 'C:\\Users\\hansk\\PycharmProjects\\PorcoVision\\saved_models\\'
############################################################


def plot_images(images, cls_true, cls_pred=None):
    assert len(images) == len(cls_true)


    # Create figure with 3x3 sub-plots.
    fig, axes = plt.subplots(3, 3)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    for i, ax in enumerate(axes.flat):
        # Plot image.
        if (i >= len(images)):
            temp = images[0].reshape(img_shape_full)
        else:
            temp = images[i].reshape(img_shape_full)
        temp = cv2.cvtColor(temp, cv2.COLOR_BGR2RGB)
        ax.imshow(temp, cmap='binary')
        # Show true and predicted classes.
        if cls_pred is None:
            if (i >= len(images)):
                xlabel = "True: {0}".format(cls_true[0])
            else:
                xlabel = "True: {0}".format(cls_true[i])
        else:
            if (i >= len(images)):
                xlabel = "True: {0}, Pred: {1}".format(cls_true[0], cls_pred[0])
            else:
                xlabel = "True: {0}, Pred: {1}".format(cls_true[i], cls_pred[i])

        # Show the classes as the label on the x-axis.
        ax.set_xlabel(xlabel)

        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])

    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()


def plot_example_errors(cls_pred):
    # cls_pred is an array of the predicted class-number for
    # all images in the test-set.

    # Boolean array whether the predicted class is incorrect.
    incorrect = (cls_pred != y_test)

    # Get the images from the test-set that have been
    # incorrectly classified.
    images = x_test[incorrect]

    # Get the predicted classes for those images.
    cls_pred = cls_pred[incorrect]

    # Get the true classes for those images.
    cls_true = y_test[incorrect]

    # Plot the first 9 images.
    plot_images(images=images[0:9],
                cls_true=cls_true[0:9],
                cls_pred=cls_pred[0:9])


def plot_conv_weights(weights, input_channel=0):
    # Get the lowest and highest values for the weights.
    # This is used to correct the colour intensity across
    # the images so they can be compared with each other.
    w_min = np.min(weights)
    w_max = np.max(weights)

    # Number of filters used in the conv. layer.
    num_filters = weights.shape[3]

    # Number of grids to plot.
    # Rounded-up, square-root of the number of filters.
    num_grids = math.ceil(math.sqrt(num_filters))

    # Create figure with a grid of sub-plots.
    fig, axes = plt.subplots(num_grids, num_grids)

    # Plot all the filter-weights.
    for i, ax in enumerate(axes.flat):
        # Only plot the valid filter-weights.
        if i < num_filters:
            # Get the weights for the i'th filter of the input channel.
            # See new_conv_layer() for details on the format
            # of this 4-dim tensor.
            img = weights[:, :, input_channel, i]

            # Plot image.
            ax.imshow(img, vmin=w_min, vmax=w_max,
                      interpolation='nearest', cmap='seismic')

        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])

    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()


def plot_conv_output(values):
    # Number of filters used in the conv. layer.
    num_filters = values.shape[3]

    # Number of grids to plot.
    # Rounded-up, square-root of the number of filters.
    num_grids = math.ceil(math.sqrt(num_filters))

    # Create figure with a grid of sub-plots.
    fig, axes = plt.subplots(num_grids, num_grids)

    # Plot the output images of all the filters.
    for i, ax in enumerate(axes.flat):
        # Only plot the images for valid filters.
        if i < num_filters:
            # Get the output image of using the i'th filter.
            img = values[0, :, :, i]

            # Plot image.
            ax.imshow(img, interpolation='nearest', cmap='binary')

        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])

    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()


def plot_image(image):
    plt.imshow(image.reshape(img_shape),
               interpolation='nearest',
               cmap='binary')

    plt.show()

# Images are stored in one-dimensional arrays of this length.
img_size_flat = img_size * img_size * num_channels

# Tuple with height and width of images used to reshape arrays.
# This is used for plotting the images.
img_shape = (img_size, img_size)

# Tuple with height, width and depth used to reshape arrays.
# This is used for reshaping in Keras.
img_shape_full = (img_size, img_size, num_channels)

model_name = ''
while (model_name == ''):
    model_name = input('Enter name of file you wish to process: ')

model = load_model(savedModelsDir + model_name)
print('reached')
print(model.summary())

choice = ''
while (choice == ''):
    choice = input('Create everything possible? (y/n)')
if(choice.lower() == 'y'):
    print("TODO: implement everything feature")

# get data from ImagesToDataset
(x_train, y_train), (x_val, y_val), (x_test, y_test) = get_data()
# Get the first images from the test-set.
images = x_test[0:9]
# Get the true classes for those images.
cls_true = y_test[0:9]
# Get the model's predictions for those images
y_pred = model.predict(x=images)
cls_pred = np.argmax(y_pred,axis=1)

plot_images(images=images,
            cls_true=cls_true,
            cls_pred=cls_pred)

y_pred = model.predict(x=x_test)
cls_pred = np.argmax(y_pred,axis=1)

plot_example_errors(cls_pred)
weights = model.layers[2].get_weights()[0]
plot_conv_weights(weights=weights, input_channel=0)