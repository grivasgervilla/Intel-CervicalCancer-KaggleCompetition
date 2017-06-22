from keras.applications.resnet50 import ResNet50
from keras.applications.inception_v3 import InceptionV3
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.core import Dense, Dropout, Flatten, Activation
from keras import backend as K
from keras.callbacks import ModelCheckpoint

from sklearn.model_selection import train_test_split
import pickle

import numpy as np
import os

#Cargamos los datos de train
train_data = np.load('Datos/train244allOVO2-3.npy')
train_target = np.load('Datos/train_target244allOVO2-3.npy')

x_train,x_val_train,y_train,y_val_train = train_test_split(train_data,train_target,test_size=0.4, random_state=17)

np.random.seed(17)

#Cargamos la red sobre la que vamos a hacer el fine tunning
base_model = ResNet50(weights = 'imagenet', include_top = False)

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(2, activation='softmax')(x)
#K.set_image_data_format('channels_first')

model = Model(inputs=base_model.input, outputs=predictions)

for layer in base_model.layers:
    layer.trainable = False
    

model.compile(optimizer = 'adamax', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

datagen = ImageDataGenerator(
    rotation_range=180,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest')

datagen.fit(train_data)
model_json = model.to_json()

experiment_name = "Experimentos/OVO/fine_tunning_ResNet50_all244fineTuningx20OVO2-3"

if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)
	    
with open(experiment_name + "/" + "modelTransferLearning.json","w") as json_file:
    json_file.write(model_json)

if __name__ == '__main__':
    
    filepath=experiment_name + "/pesos-transferLearning-epoch{epoch:02d}-val_acc{val_acc:.5f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]

    nnmodel = model.fit_generator(datagen.flow(x_train,y_train, batch_size=15, shuffle=True),steps_per_epoch=(len(x_train)/15)*20, nb_epoch=10, samples_per_epoch=len(x_train), callbacks=callbacks_list, verbose=2, validation_data=(x_val_train, y_val_train))

    #Descongelamos algunas
    for layer in model.layers[:162]:
        layer.trainable = False
    for layer in model.layers[162:]:
        layer.trainable = True
   	 
    model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
	
    model_json = model.to_json()
	
    with open(experiment_name + "/" + "modelFineTunning.json","w") as json_file:
        json_file.write(model_json)
	
	
    filepath=experiment_name + "/pesos-fineTunning-epoch{epoch:02d}-val_acc{val_acc:.5f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]
    
    nnmodel = model.fit_generator(datagen.flow(x_train,y_train, batch_size=15, shuffle=True),steps_per_epoch=(len(x_train)/15)*20, nb_epoch=50, samples_per_epoch=len(x_train), callbacks=callbacks_list, verbose=2, validation_data=(x_val_train, y_val_train))
    
    pickle.dump(nnmodel.history, open((experiment_name + "/" + "history.p"), "wb"))


#Enlaces consultados:
# https://keras.io/applications/
# https://deeplearningsandbox.com/how-to-use-transfer-learning-and-fine-tuning-in-keras-and-tensorflow-to-build-an-image-recognition-94b0b02444f2
# https://gist.github.com/fchollet/7eb39b44eb9e16e59632d25fb3119975

