from keras.preprocessing.image import ImageDataGenerator
import SimpleModel as SM
import cv2
from PIL import Image
from keras.preprocessing import image
from keras import backend as K
from importImages import *
def augment_and_save():
    UPSAMPLING_FACTOR = 20

    (x_train, y_train, x_test, y_test) = new_import_images("D:\\porcocropped\\Cropped\\x200", False)

    train_datagen = ImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization=True,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        fill_mode="nearest",
        zoom_range=0.1
    )
    train_datagen.fit(x_train)
    pathout = "D:\\porcocropped\\augmented\\"
    iteration = 0
    batchsize = 1
    nbatches = (len(x_train)*UPSAMPLING_FACTOR)/batchsize
    for (images, classes) in train_datagen.flow(x = x_train, y = y_train, batch_size=1):
        iteration+=1
        for i in range(batchsize):
            cls = "farrow_full\\"
            if classes[i] == 0:
                cls = "nofarrow_full\\"
            img = image.array_to_img(images[i], data_format=K.image_data_format())
            img.save(pathout + cls + str(iteration+i).zfill(8) + '.jpg')
          #  cv2.imwrite(pathout + cls + str(i).zfill(8) + '.jpg', )

        if iteration > nbatches:
            break

#  def flow(self, x, y=None, batch_size=32, shuffle=True, seed=None, save_to_dir=None, save_prefix='',
#           save_format='png', subset=None) Inferred

#  type: (self
#         : ImageDataGenerator, x: {__len__, shape}, y: {__len__}, batch_size: int, shuffle: bool, seed: Any, save_to_dir: Any, save_prefix: str, save_format: str, subset: Any) -> NumpyArrayIterator

augment_and_save()

#img = Image.fromarray(x, 'RGB')