from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten, Activation
from keras.layers.convolutional import Convolution2D, ZeroPadding2D, MaxPooling2D
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from keras.optimizers import SGD
from keras import backend as K
K.set_image_dim_ordering('th')
K.set_floatx('float32')
import time
from keras.callbacks import ModelCheckpoint
import pandas as pd
import numpy as np
import os
from keras.models import model_from_json

test_data = np.load('Datos/test244.npy')
test_id = np.load('Datos/test_id244.npy')

experiment_name = "Experimentos/OVO/fine_tunning_ResNet50_all244fineTuningx20OVO2-3"
submission_name = "fine_tunning_ResNet50_all244fineTuningx20OVO2-3" + ".csv"

json_file = open(experiment_name + '/modelFineTunning.json', 'r')
model_json = json_file.read()
json_file.close()

model = model_from_json(model_json)

model.load_weights(experiment_name + "/pesos-fineTunning-epoch48-val_acc0.67885.hdf5")

pred = model.predict(test_data)
df = pd.DataFrame(pred, columns=['Type_2_23','Type_3_23'])
df['image_name'] = test_id
df.to_csv('Resultados/' + submission_name, index=False)