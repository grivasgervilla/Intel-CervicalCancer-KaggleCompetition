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
import pickle

train_data = np.load('Datos/train32allv2.npy')
train_target = np.load('Datos/train_target32allv2.npy')


x_train,x_val_train,y_train,y_val_train = train_test_split(train_data,train_target,test_size=0.4, random_state=17)

np.random.seed(17)

def create_model(opt_='adamax'):
	model = Sequential()
	model.add(Convolution2D(1, 3, 3, activation='relu', dim_ordering='th', input_shape=(3, 32, 32))) #use input_shape=(3, 64, 64)
	model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), dim_ordering='th'))
	model.add(Convolution2D(2, 3, 3, activation='relu', dim_ordering='th'))
	model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), dim_ordering='th'))
	model.add(Convolution2D(4, 3, 3, activation='relu', dim_ordering='th'))
	model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), dim_ordering='th'))
	#model.add(Convolution2D(8, 3, 3, activation='relu', dim_ordering='th'))
	#model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), dim_ordering='th'))
	model.add(Dropout(0.2))

	model.add(Flatten())
	model.add(Dense(24, activation='tanh'))
	model.add(Dropout(0.1))
	model.add(Dense(12, activation='tanh'))
	model.add(Dropout(0.1))
	model.add(Dense(3, activation='softmax'))

	model.compile(optimizer=opt_, loss='sparse_categorical_crossentropy', metrics=['accuracy']) 
	return model
	
if __name__ == '__main__': 
	start_time = time.time()
	#datagen = ImageDataGenerator(rotation_range=0.3, zoom_range=0.3)
	#datagen.fit(train_data)
	
	model = create_model()
	model_json = model.to_json()
	
	experiment_name = "Experimentos/scrath_train_all32v2"
	
	if not os.path.exists(experiment_name):
		os.makedirs(experiment_name)
		
	with open(experiment_name + "/" + "model.json","w") as json_file:
		json_file.write(model_json)
	
	print(x_train.shape)
	print(y_train.shape)
	
	filepath=experiment_name + "/pesos-epoch{epoch:02d}-val_acc{val_acc:.5f}.hdf5"
	checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
	callbacks_list = [checkpoint]
	#nnmodel = model.fit_generator(datagen.flow(x_train,y_train, batch_size=15, shuffle=True), nb_epoch=2, samples_per_epoch=len(x_train), callbacks=callbacks_list, verbose=2, validation_data=(x_val_train, y_val_train))
	nnmodel = model.fit(x_train,y_train, epochs=100, batch_size=15, callbacks=callbacks_list, verbose=2, validation_data=(x_val_train, y_val_train))

	pickle.dump(nnmodel.history, open((experiment_name + "/" + "history.p"), "wb"))

