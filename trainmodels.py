import time
import csv
import sys
from ModelFromString import *
from importImages import *
import subprocess
def read_networks_from_csv():
    networks = []

    with open("models.csv", newline= '', encoding="utf-8-sig") as csvfile:
        reader = csv.reader(csvfile, delimiter = ";", dialect = "excel")
        for row in reader:
            string = ""
            for i in range(len(row)):
                if row[i] != "":
                    string = string + row[i]
                    if i < len(row)-1:
                        string = string + ";"

            networks.append(string)
    return networks

path_to_train_data =  "D:\\porcocropped\\Cropped\\x200"
path_to_test_data =   "D:\\porcocropped\\Test\\x200"
path_to_val_data = "D:\\porcocropped\\Validation\\"
patience = "50"
image_size = "200"
print("Using interpreter from {}".format(sys.executable))

networks = read_networks_from_csv()
print(networks)
#(x_val, y_val, val2, val2_y) = new_import_images(path_to_val_data, False)
models_with_errors = []
for i in range(len(networks)):
    modelname = str(i+1)
    network = networks[i]
 #   print(construct_network_from_string(network, "hej", x_val))
    #TODO: call subprocess.Popen() and train the specific model
# Args: path_to_train_data, path_to_test_data, path_to_val_data, modelname, patience, network (string)
    with open("{}stdout.txt".format(modelname), "wb") as out, open("{}stderr.txt".format(modelname), "wb") as err:
        process = subprocess.Popen([sys.executable, "BatchModelAndTest.py", path_to_train_data, path_to_test_data, path_to_val_data, modelname, patience, network, image_size], stderr=err, stdout=out)
        if process.wait() != 0:
            models_with_errors.append(int(modelname))
            print("Model {} failed to train.".format(int(modelname)))
        else:
            print("Model {} succeeded in training.".format(int(modelname)))



print(models_with_errors)


