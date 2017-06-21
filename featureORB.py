from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import numpy as np
from math import pi
from keras.preprocessing.image import ImageDataGenerator
import cv2

from sklearn.cluster import KMeans
import sklearn.preprocessing as prepro

train_data = np.load('Datos/train244all.npy')
test_data = np.load('Datos/test244.npy')
orb = cv2.ORB_create()

def getORB(image):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    image = image * 255
    image = image.astype('uint8')
    
    kp = orb.detect(image,None)
    kp, des = orb.compute(image, kp)
    
    return des

# Obteniendo los descriptores

'''
descriptors = getORB(test_data[0])
invalid_index = []
n_descriptors = []

i = 0
for idx, image in enumerate(test_data[1:]):
    d = getORB(image)
    if d != None:
        i += 1
        descriptors = np.concatenate((descriptors, d))
        n_descriptors.append(len(descriptors))
    else:
        invalid_index.append(idx)

descriptors = np.array(descriptors)

np.save('Datos/ORBdescriptoresTest.npy', descriptors)
np.save('Datos/invalid_index_test.npy', np.array(invalid_index))
np.save('Datos/n_descriptors_test.npy', np.array(n_descriptors))
'''

if __name__ == '__main__':

    descriptoresTrain = np.load('Datos/ORBdescriptores.npy')
    descriptoresTest = np.load('Datos/ORBdescriptoresTest.npy')

    min_max_scaler = prepro.MinMaxScaler()
    descriptoresTrain = min_max_scaler.fit_transform(descriptoresTrain)
    descriptoresTest = min_max_scaler.fit_transform(descriptoresTest)

    kmeans = KMeans(n_clusters=500, random_state=0, n_init=1, verbose=1).fit(descriptoresTrain)
    
    np.save('Datos/labelsTrain.npy', kmeans.labels_)
    
    np.save('Datos/labelsTest.npy', kmeans.predict(descriptoresTest))
 



