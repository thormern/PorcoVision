import os
img_size = 200
num_channels = 3
num_classes = 2
class_names = ["Farrowing", "No farrowing"]
nb_class0 = 304 #Number of pictures in class 0
nb_class1 = 333 #Number of pictures in class 1
batch_size = 32
epochs = 3
img_size_cropped = 200

def get_path_to_trainingimages():

    imagepathexists = 0
    imagespathfile = os.path.join(os.getcwd(), "path_to_images")
    imagespath = ""
    if not os.path.isfile(imagespathfile):
        file = open(imagespathfile, "w+")
    else:
        imagepathexists = 1
        file = open(imagespathfile, "r+")
        imagespath = file.readlines()[0]
        if len(imagespath) == 0:
            imagepathexists = 0
            imagespath = ""


    print("Type the full path of the croppedimages/combined folder. Leave empty to try and use that last known path")
    path1 = input()
    if path1 == "" and imagepathexists:
        path1 = imagespath
    elif path1 == "" and not imagepathexists:
        while len(path1) == 0:
            path1 = input("No previous path detected - try again")

    if not path1.endswith("\\"):
        path1 = path1 + "\\"
    imagespath = path1
    file.seek(0)
    file.write(imagespath)
    file.truncate()
    file.close()
    return path1

def get_path_to_validationimages():

    imagepathexists = 0
    imagespathfile = os.path.join(os.getcwd(), "path_to_validation")
    imagespath = ""
    if not os.path.isfile(imagespathfile):
        file = open(imagespathfile, "w+")
    else:
        imagepathexists = 1
        file = open(imagespathfile, "r+")
        imagespath = file.readlines()[0]
        if len(imagespath) == 0:
            imagepathexists = 0
            imagespath = ""


    print("Type the full path of the validation folder. Leave empty to try and use that last known path")
    path1 = input()
    if path1 == "" and imagepathexists:
        path1 = imagespath
    elif path1 == "" and not imagepathexists:
        while len(path1) == 0:
            path1 = input("No previous path detected - try again")

    if not path1.endswith("\\"):
        path1 = path1 + "\\"
    imagespath = path1
    file.seek(0)
    file.write(imagespath)
    file.truncate()
    file.close()
    return path1

path1 = get_path_to_trainingimages()
path2 = get_path_to_validationimages()
#path1 = 'C:/Users/hansk/CroppedImages/Combined'   # path to folder of images
