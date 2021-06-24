from PIL import Image
import os
import math
import numpy as np
from keras.preprocessing import image
from keras import backend as K
from importImages import *
import shutil
DOWN_SAMPLING_FACTOR = 8
SIZE = 200  # pixel width and height

allimages = "D:\\porcovision\\Data_fullsize\\"
outputdir = "D:\\porcocropped\\"
seed = 42
def pick_validation_data(ratio):
    imagePaths = list(paths.list_images("D:\\porcocropped\\Test\\x200"))
    random.seed(seed)
    random.shuffle(imagePaths)
    samplesize = len(imagePaths)
    validationsize = int(samplesize*ratio)
    for i in range(validationsize):
      #  print(imagePaths[i])
        outputpath = imagePaths[i].split(os.path.sep)
        outputpath[2] = "Validation"
        outputfolder = os.path.sep.join(outputpath[:-1])
        outputpath = os.path.sep.join(outputpath)
        if not os.path.exists(outputfolder):
            os.makedirs(outputfolder)
        shutil.move(imagePaths[i], outputpath)



#TODO: Downsamplingfactor (1 pixel corresponds to 8x8 area, fx. )
#TODO: (not in this file) Change final Dense layer to 1 node, and dont use one hot encoding for binary categorization.
def crop(SIZE, allimages, outputdir):
    imageno = 0
    for path in os.listdir(allimages):

        print(path)
        subpath = allimages + path
        croppings = subpath + "\croppings.txt"

        farrowfolder = subpath + "\\farrow_full"
        nofarrowfolder = subpath + "\\nofarrow_full"
        farrowimages = os.listdir(farrowfolder)
        nofarrowimages = os.listdir(nofarrowfolder)

        cropfile = open(croppings, "r")
        lines = cropfile.readlines()
        lines2 = [l.split(" ") for l in lines]
        lines = lines2

        cropfile.close()

        newpath = outputdir  + "Cropped\\" + "x" + str(SIZE) + "\\"
        print(newpath)
        if not os.path.exists(newpath + "nofarrow_full"):
            os.makedirs(newpath + "nofarrow_full")
        if not os.path.exists(newpath + "farrow_full"):
            os.makedirs(newpath + "farrow_full")

        maxX = 0
        maxY = 0
        minX = 1280
        minY = 720

        for line in lines:
            imagename = str(line[0]).zfill(8) + ".jpg"
            if imagename in farrowimages:
                imagepath = farrowfolder + "\\" + imagename
                if os.path.isfile(imagepath):
                    point = (int(line[1]), int(line[2]))
                    img = Image.open(imagepath)

                    img2 = img.crop((point[0] - (SIZE / 2),
                                     point[1] - (SIZE / 2),
                                     point[0] + (SIZE / 2),
                                     point[1] + (SIZE / 2)))
                    img2.save(newpath + "farrow_full\\" + str(imageno).zfill(8) + ".jpg")
            elif imagename in nofarrowimages:
                imagepath = nofarrowfolder + "\\" + imagename
                if os.path.isfile(imagepath):
                    point = (int(line[1]), int(line[2]))
                    img = Image.open(imagepath)
                    img2 = img.crop((point[0] - (SIZE / 2),
                                     point[1] - (SIZE / 2),
                                     point[0] + (SIZE / 2),
                                     point[1] + (SIZE / 2)))

                    img2.save(newpath + "nofarrow_full\\" + str(imageno).zfill(8) + ".jpg")
            imageno = imageno + 1

pick_validation_data(0.33)
#crop(SIZE, allimages, outputdir)