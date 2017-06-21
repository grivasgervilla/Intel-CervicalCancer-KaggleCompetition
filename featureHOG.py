from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import numpy as np
from math import pi
from keras.preprocessing.image import ImageDataGenerator
import cv2

from sklearn.cluster import KMeans
import sklearn.preprocessing as prepro

# Generamos nuevos ejemplos
'''
datagen = ImageDataGenerator(
        rotation_range=180,
        shear_range=pi,
        fill_mode='nearest')
         
train_data = np.load('Datos/train244all.npy')
train_labels = np.load('Datos/train_target244all.npy')

datagen.fit(train_data,rounds=2)

i = 0

nuevas_imagenes = []

tam = 1

for batch in datagen.flow(train_data,train_labels,batch_size = (len(train_data))):
    i += 1
    if i > tam:
        break
        
    nuevas_imagenes.append(batch[0])

nuevas_imagenes = np.array(nuevas_imagenes)

nuevas_imagenes = np.reshape(nuevas_imagenes, (len(train_data)*tam,244,244,3))

np.save('Datos/extraRotations.npy', nuevas_imagenes, allow_pickle=True, fix_imports=True)
'''

train_data = np.load('Datos/train244all.npy')
test_data = np.load('Datos/test244.npy')

hog = cv2.HOGDescriptor()

def getHist(image):

    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    image = image * 255
    image = image.astype('uint8')    
    return hog.compute(image)
    
histograms = [getHist(img) for img in train_data]



if __name__ == '__main__':
    # Guardar los histogramas



