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

import pandas as pd
import numpy as np

train_data = np.load('train.npy')
train_target = np.load('train_target.npy')

x_train,x_val_train,y_train,y_val_train = train_test_split(train_data,train_target,test_size=0.4, random_state=17)

np.random.seed(17)

def create_model(opt_='adamax'):
    model = Sequential()
    model.add(Convolution2D(4, 3, 3, activation='relu', dim_ordering='th', input_shape=(3, 32, 32))) #use input_shape=(3, 64, 64)
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), dim_ordering='th'))
    model.add(Convolution2D(8, 3, 3, activation='relu', dim_ordering='th'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), dim_ordering='th'))
    model.add(Dropout(0.2))
    
    model.add(Flatten())
    model.add(Dense(12, activation='tanh'))
    model.add(Dropout(0.1))
    model.add(Dense(3, activation='softmax'))

    model.compile(optimizer=opt_, loss='sparse_categorical_crossentropy', metrics=['accuracy']) 
    return model
	
if __name__ == '__main__': 

	datagen = ImageDataGenerator(rotation_range=0.3, zoom_range=0.3)
	datagen.fit(train_data)
	
	model = create_model()
	print(x_train.shape)
	print(y_train.shape)
	model.fit_generator(datagen.flow(x_train,y_train, batch_size=15, shuffle=True), nb_epoch=35, samples_per_epoch=len(x_train), verbose=20, validation_data=(x_val_train, y_val_train))
	
	test_data = np.load('test.npy')
	test_id = np.load('test_id.npy')

	pred = model.predict_proba(test_data)
	df = pd.DataFrame(pred, columns=['Type_1','Type_2','Type_3'])
	df['image_name'] = test_id
	df.to_csv('submission 0001.csv', index=False)
	print(3)
