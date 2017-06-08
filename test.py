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

test_data = np.load('test32.npy')
test_id = np.load('test_id32.npy')

experiment_name = "Experimentos/scrath_train_all32v2"
submission_name = "scrath_train_all32v2" + ".csv"

json_file = open(experiment_name + '/model.json', 'r')
model_json = json_file.read()
json_file.close()

model = model_from_json(model_json)

model.load_weights(experiment_name + "/pesos-epoch99-val_acc0.51296.hdf5")

pred = model.predict_proba(test_data)
df = pd.DataFrame(pred, columns=['Type_1','Type_2','Type_3'])
df['image_name'] = test_id
df.to_csv('Resultados/' + submission_name, index=False)