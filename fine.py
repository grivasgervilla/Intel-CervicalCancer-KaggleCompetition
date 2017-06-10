from keras.applications.resnet50 import ResNet50
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.optimizers import SGD

#Cargamos los datos de train
train_data = np.load('Datos/train244all.npy')
train_target = np.load('Datos/train_target244all.npy')

#Cargamos la red sobre la que vamos a hacer el fine tunning
base_model = ResNet50(weights = 'imagenet', include_top = False)

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(3, activation='softmax')(x)

for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer = opt_, loss='sparset_categorical_crossentropy', metrics=['accuracy'])

datagen = ImageDataGenerator(
    rotation_range=180,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest')

datagen.fin(train_data)
model_json = model.to_json()

experiment_name = "Experimentos/fine_tunning_ResNet50_all256dataAugmentationx20"

if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)
	    
with open(experiment_name + "/" + "model.json","w") as json_file:
    json_file.write(model_json)

if __name__ == '__main__':
    filepath=experiment_name + "/pesos-epoch{epoch:02d}-val_acc{val_acc:.5f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]

    nnmodel = model.fit_generator(datagen.flow(x_train,y_train, batch_size=15, shuffle=True),steps_per_epoch=(len(x_train)/15)*20, nb_epoch=10, samples_per_epoch=len(x_train), callbacks=callbacks_list, verbose=2, validation_data=(x_val_train, y_val_train))

    #Descongelamos algunas

    model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='sparset_categorical_crossentropy')

    nnmodel = model.fit_generator(datagen.flow(x_train,y_train, batch_size=15, shuffle=True),steps_per_epoch=(len(x_train)/15)*20, nb_epoch=50, samples_per_epoch=len(x_train), callbacks=callbacks_list, verbose=2, validation_data=(x_val_train, y_val_train))


#Enlaces consultados:
# https://keras.io/applications/
# https://deeplearningsandbox.com/how-to-use-transfer-learning-and-fine-tuning-in-keras-and-tensorflow-to-build-an-image-recognition-94b0b02444f2
# https://gist.github.com/fchollet/7eb39b44eb9e16e59632d25fb3119975


    
