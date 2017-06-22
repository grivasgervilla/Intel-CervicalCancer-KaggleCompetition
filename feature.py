from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input
from keras.models import Model
from keras.models import model_from_json
from sklearn.model_selection import GridSearchCV

import sklearn.preprocessing as prepro
from sklearn import svm
from sklearn.ensemble import GradientBoostingClassifier

import numpy as np

'''
train_data = np.load('Datos/train244all.npy')

experiment_name = "Experimentos/fine_tunning_ResNet50_all244fineTuningx20"

json_file = open(experiment_name + '/modelFineTunning.json', 'r')
model_json = json_file.read()
json_file.close()

base_model = model_from_json(model_json)

base_model.load_weights(experiment_name + "/pesos-fineTunning-epoch49-val_acc0.59959.hdf5")

model = Model(inputs=base_model.input, outputs=base_model.get_layer('avg_pool').output)


features = model.predict(train_data)
print(features.shape)
features = np.reshape(features, (8485, 2048))

np.save('Datos/featuresTrain.npy', features, allow_pickle=True, fix_imports=True)

#for i, layer in enumerate(base_model.layers):
#   print(i, layer.name)
'''

if __name__ == '__main__': 
    train_data = np.load('Datos/featuresTrain.npy')
    test_data = np.load('Datos/featuresTest.npy')
    train_labels = np.load('Datos/train_target244all.npy')

    f = open("Resultados/params.txt", 'w')

    print("Pasamos a normalizar")
    min_max_scaler = prepro.MinMaxScaler()
    train_data = min_max_scaler.fit_transform(train_data)
    test_data = min_max_scaler.fit_transform(test_data)
    
    '''
    print("Pasamos a entrenar la SVM")
    clf = svm.SVC(decision_function_shape='ovo', verbose=True, probability=True)
    parameters = {'kernel':['poly', 'rbf'], 'C':[1, 10], 'tol':[1e-3, 0.1]}
    clf = GridSearchCV(clf, parameters, verbose=1, n_jobs=2)
    clf.fit(train_data, train_labels)

    probs = clf.predict_proba(test_data)

    #print(probs)

    np.save('Resultados/SVMprobsTune.npy', probs, allow_pickle=True, fix_imports=True)
    print("Ha acabado SVM")
    print(clf.best_params_)
   
    f.write("Parametros SVM\n")
    f.write(clf.best_params_)
    '''
    clf = GradientBoostingClassifier(verbose=1)
    parameters = {'n_estimators':[50, 100, 200], 'max_depth':[3, 5], 'learning_rate':[1e-3, 0.1, 1]}
    clf = GridSearchCV(clf, parameters, verbose=1, n_jobs=4)
    clf.fit(train_data, train_labels)

    probs = clf.predict_proba(test_data)

    #print(probs)
    print("Ha acabado GBoost")
    print(clf.best_params_)

    np.save('Resultados/GBoostprobsTune.npy', probs, allow_pickle=True, fix_imports=True)
    
    f.write("Parametros GBoost\n")
    f.write(str(clf.best_params_))

    #http://neerajkumar.org/writings/svm/