from keras.callbacks import Callback
from keras.preprocessing.image import ImageDataGenerator
import sys

#############
from ModelFromString import *
from csvtest import *
from importImages import *
config = tf.ConfigProto()
import matplotlib.pyplot as plt
import gc
# Don't pre-allocate memory; allocate as-needed
K.clear_session()
config.gpu_options.allow_growth = True

# Only allow some fraction of the GPU memory to be allocated
#config.gpu_options.per_process_gpu_memory_fraction = 0.8
#config.gpu_options.allow_growth = True

# Create a session with the above options specified.
K.tensorflow_backend.set_session(tf.Session(config=config))
K.clear_session()

###################################
class ValidationHistoryandEarlyStopping(Callback):
    def __init__(self, test_data, logger, min_delta=0, patience=1):
        self.min_delta=min_delta
        self.patience=patience
        self.test_data = test_data
        self.value = 0
        self.count = 0
        self.lastfilename = "0"
        self.logger = logger



    def on_epoch_end(self, epoch, logs={}):
        x, y = self.test_data

        loss, acc = self.model.evaluate(x, y, verbose=0)
        self.logger.vallosses.append(loss)
        self.logger.valaccuracies.append(acc)
        self.logger.testlosses.append(logs["loss"])
        self.logger.testaccuracies.append(logs["acc"])
        self.logger.trainaccuracies.append(logs["val_acc"])
        self.logger.trainlosses.append(logs["val_loss"])

        print('\nTesting loss: {}, acc: {}, epoch: {}\n'.format(loss, acc, epoch+1))
        ##Early stopping procedures:
        if acc > self.value+self.min_delta:
            self.value = acc
            if self.count > 0:
                self.count = 0
            delete_file_from_disk(self.lastfilename)
            save_model(self.model, "_acc" + str(round(acc, 4)) + "_epoch" + str(epoch+1))
            self.lastfilename = self.model.name + "_acc" + str(round(acc, 4)) + "_epoch" + str(epoch+1)
        else:
            self.count = self.count + 1
            print("Early stopping count increasing, old value: " + str(self.value) + " New value: " + str(acc) + " Count: " + str(self.count))
            if self.count > self.patience:
                self.model.stop_training = True



def delete_file_from_disk(path):
    save_dir = os.path.join(os.getcwd(), 'saved_models')
    path = os.path.join(save_dir, path)
    if os.path.isfile(path):
        os.remove(path)
    else:
        print(str(path) + " was not found")

def save_model(model, suffix):
    save_dir = os.path.join(os.getcwd(), 'saved_models')

    model_name = model.name + str(suffix)

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



#def train_existing_model(model, x_train, x_test, y_train, y_test, x_val, y_val, bsize, eps):
def train_existing_model(model, x_val, y_val, bsize, eps, patience, logger, image_size, path_to_train_data, path_to_test_data, path_to_val_data):
    t1 = time.time()


    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        featurewise_center=True,
        featurewise_std_normalization=True,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        fill_mode="nearest",
        zoom_range=0.1
    )

    test_datagen = ImageDataGenerator(
        rescale=1. / 255
    )

    train_generator = train_datagen.flow_from_directory(
        path_to_train_data,  # this is the target directory
        target_size=(image_size, image_size),  # all images will be resized to 150x150
        batch_size=bsize,
        class_mode='binary',
        classes=["nofarrow_full", "farrow_full"]
    )

    validation_generator = test_datagen.flow_from_directory(
        path_to_test_data,
        target_size=(image_size, image_size),
        batch_size=bsize,
        class_mode='binary',
        classes = ["nofarrow_full", "farrow_full"])



    model.fit_generator(
        train_generator,
        workers=8,
        max_queue_size=2,
        steps_per_epoch= (8055//bsize) + 1 if 8055%bsize is not 0 else 8055//bsize,
        #steps_per_epoch=8055//bsize,
        epochs=eps,
        validation_data=validation_generator,
        validation_steps=(895//bsize) + 1 if 895%bsize is not 0 else 895//bsize,
        callbacks=[ValidationHistoryandEarlyStopping((x_val, y_val), logger, 0, patience)],
        verbose=0



    )


    t2 = time.time()
    print("Model took " + str(t2 - t1) + "to train")

class ModelData(object):
    def __init__(self):
        self.valaccuracies = []
        self.vallosses = []
        self.trainlosses = []
        self.trainaccuracies = []
        self.testaccuracies =  []
        self.testlosses = []

    def save(self, model):
        filename = str(model.name)
        while os.path.isfile(filename + ".csv"):
            filename = filename + "_1"
        filename = filename + ".csv"
        with open(filename, "w+", newline='', encoding="utf-8-sig") as csvfile:
            writer = csv.writer(csvfile, delimiter=';', dialect="excel")
            writer.writerow(self.vallosses)
            writer.writerow(self.valaccuracies)
            writer.writerow(self.testlosses)
            writer.writerow(self.testaccuracies)
            writer.writerow(self.trainlosses)
            writer.writerow(self.trainaccuracies)

            plt.plot(self.valaccuracies)
            plt.plot(self.testaccuracies)
            plt.plot(self.trainaccuracies)

            no_points = len(self.valaccuracies)

            stepsize = 5 if no_points < 50 else 10
            minorstepsize = stepsize//5
            plt.xticks(
                np.arange(0, len(self.valaccuracies), stepsize)
            )


            plt.title('model accuracy')
            plt.ylabel('accuracy')
            plt.xlabel('epoch')
            plt.legend(["validation", "test", "training"], loc="upper left")
            fig = plt.gcf()
            fig.set_size_inches(10, 7)
            plt.savefig("./saved_models/{}_acc.png".format(model.name))
            print("Accuracies saved.")
     #       plt.show()
            plt.clf()

            plt.plot(self.vallosses)
            plt.plot(self.testlosses)
            plt.plot(self.trainlosses)
            plt.xticks(
                np.arange(0, len(self.valaccuracies), stepsize)
            )


            plt.title('model loss')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.legend(["validation", "test", "training"], loc="upper left")
            fig = plt.gcf()
            fig.set_size_inches(10, 7)
            plt.savefig("./saved_models/{}_loss.png".format(model.name))
            print("Losses saved.")
 #           plt.show()
            plt.clf()





args = sys.argv[1:]
#Args: path_to_train_data, path_to_test_data, path_to_val_data, modelname, patience, network (string)

##TODO: change path2 to parameter
image_size = int(args[6])
networkstring = args[5]
patience = int(args[4])
modelname = args[3]
path_to_val_data = args[2]
path_to_test_data = args[1]
path_to_train_data = args[0]
(x_val, y_val, val2, val2_y) = new_import_images(args[2], False, image_size)

((bsize, eps), model) = construct_network_from_string(networkstring, modelname, x_val)

#for ((bsize, eps), model) in models:
logger = ModelData()

trainable_count = int(
    np.sum([K.count_params(p) for p in set(model.trainable_weights)]))
non_trainable_count = int(
    np.sum([K.count_params(p) for p in set(model.non_trainable_weights)]))

print('Total params: {:,}'.format(trainable_count + non_trainable_count))
print('Trainable params: {:,}'.format(trainable_count))
print('Non-trainable params: {:,}'.format(non_trainable_count))

print(model.summary())
print("Starting training...")

train_existing_model(model, x_val, y_val, bsize, eps, patience, logger, image_size, path_to_train_data, path_to_test_data, path_to_val_data)

gc.collect()
gc.collect()
gc.collect()
gc.collect()
gc.collect()

logger.save(model)

#sys.stdout.close()
#sys.stderr.close()


