import numpy as np
import cv2
import math
#Left click for no farrowing, right click for farrowing, DEBUG=True to see the rectangle ( will not do anything but create rectangles.

#The path to the first image to be scaled
imagePath = 'RawImages/00000001.jpg'
#The dimensions of the new picture. Example Size = 50 will give a 50 by 50 pixels picture.
SIZE = 50
#The size of the area each pixel of the new picture covers. Example DownSamplingFactor = 20, each pixel in new picture corresponds to a 20 by 20 area of original.
DOWN_SAMPLING_FACTOR = 8
#Draw a rectangle around the area that will be made to new picture.
DEBUG = False

dst = np.zeros((SIZE,SIZE,3), np.uint8) #Initialize as black image
cv2.namedWindow('destination', cv2.WINDOW_NORMAL)

# changes imagePath to the next image. Must be changed if the naming scheme changes
def next_image_path():
    global imagePath
    pictureNumber =  int(imagePath[10:18])
    pictureNumber = pictureNumber + 1
    pictureNumber = str(pictureNumber)
    pictureNumber = pictureNumber.zfill(8)
    imagePath = 'RawImages/' + pictureNumber + '.jpg'

def saveNewPicture(flag):
    global imagePath, dst
    pictureNumber = imagePath[10:18]
    if (flag):
        cv2.imwrite('CroppedImages/Farrowing/' + pictureNumber + '.jpg', dst)
    else:
        cv2.imwrite('CroppedImages/NoFarrowing/' + pictureNumber + '.jpg', dst)

# mouse callback function
def create_new_picture(event,x,y,flags,param):
    global dst, img
    if event == cv2.EVENT_LBUTTONDOWN:
        #TODO: No farrowing
        height = np.size(img, 0)
        width = np.size(img, 1)
        topLeftX = min(max(math.floor(x-(SIZE*DOWN_SAMPLING_FACTOR/2)),0),width-SIZE*DOWN_SAMPLING_FACTOR)
        topLeftY = min(max(math.floor(y-(SIZE*DOWN_SAMPLING_FACTOR/2)),0),height-SIZE*DOWN_SAMPLING_FACTOR)
        botRightX = min(max(math.floor(x+(SIZE*DOWN_SAMPLING_FACTOR/2)),SIZE*DOWN_SAMPLING_FACTOR),width)
        botRightY = min(max(math.floor(y+(SIZE*DOWN_SAMPLING_FACTOR/2)),SIZE*DOWN_SAMPLING_FACTOR),height)
        if DEBUG:
            cv2.rectangle(img, (topLeftX, topLeftY), (botRightX, botRightY), (0, 255, 0), 3)
        else:
            dst = img[topLeftY:(topLeftY+SIZE*DOWN_SAMPLING_FACTOR), topLeftX:(topLeftX+SIZE*DOWN_SAMPLING_FACTOR)]
            dst = cv2.resize(dst, (SIZE, SIZE))
            saveNewPicture(False)
            next_image_path()
            img = cv2.imread(imagePath, cv2.IMREAD_COLOR)
    if event == cv2.EVENT_RBUTTONDOWN:
        #TODO: Farrowing
        height = np.size(img, 0)
        width = np.size(img, 1)
        topLeftX = min(max(math.floor(x - (SIZE * DOWN_SAMPLING_FACTOR / 2)), 0), width - SIZE * DOWN_SAMPLING_FACTOR)
        topLeftY = min(max(math.floor(y - (SIZE * DOWN_SAMPLING_FACTOR / 2)), 0), height - SIZE * DOWN_SAMPLING_FACTOR)
        botRightX = min(max(math.floor(x + (SIZE * DOWN_SAMPLING_FACTOR / 2)), SIZE * DOWN_SAMPLING_FACTOR), width)
        botRightY = min(max(math.floor(y + (SIZE * DOWN_SAMPLING_FACTOR / 2)), SIZE * DOWN_SAMPLING_FACTOR), height)
        if DEBUG:
            cv2.rectangle(img, (topLeftX, topLeftY), (botRightX, botRightY), (0, 255, 0), 3)
        dst = img[topLeftY:(topLeftY + SIZE * DOWN_SAMPLING_FACTOR), topLeftX:(topLeftX + SIZE * DOWN_SAMPLING_FACTOR)]
        dst = cv2.resize(dst, (SIZE, SIZE))
        saveNewPicture(True)
        next_image_path()
        img = cv2.imread(imagePath, cv2.IMREAD_COLOR)

img = cv2.imread(imagePath, cv2.IMREAD_COLOR)
cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.setMouseCallback('image',create_new_picture)
while(True):
    #TODO: Show next image
    cv2.imshow('image',img)
    cv2.imshow('destination', dst)
    if cv2.waitKey(20) & 0xFF == 27:
        #TODO: Get
        break
cv2.destroyAllWindows()