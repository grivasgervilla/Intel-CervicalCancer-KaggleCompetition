import glob
import pandas as pd
from PIL import ImageFilter, ImageStat, Image, ImageDraw
from multiprocessing import Pool, cpu_count
import cv2
import numpy as np
from sklearn.preprocessing import LabelEncoder

def im_multi(path):
    try:
        im_stats_im_ = Image.open(path)
        return [path, {'size': im_stats_im_.size}]
    except:
        print(path)
        return [path, {'size': [0,0]}]


def im_stats(im_stats_df):
    im_stats_d = {}
    p = Pool(cpu_count())
    ret = p.map(im_multi, im_stats_df['path'])
    for i in range(len(ret)):
        im_stats_d[ret[i][0]] = ret[i][1]
    im_stats_df['size'] = im_stats_df['path'].map(lambda x: ' '.join(str(s) for s in im_stats_d[x]['size']))
    return im_stats_df
	
def get_im_cv2(path):
    img = cv2.imread(path)
    resized = cv2.resize(img, (256, 256), cv2.INTER_LINEAR) #use cv2.resize(img, (64, 64), cv2.INTER_LINEAR)
    return [path, resized]

def normalize_image_features(paths):
    imf_d = {}
    p = Pool(cpu_count())
    ret = p.map(get_im_cv2, paths)
    for i in range(len(ret)):
        imf_d[ret[i][0]] = ret[i][1]
    ret = []
    fdata = [imf_d[f] for f in paths]
    fdata = np.array(fdata, dtype=np.uint8)
    fdata = fdata.transpose((0, 3, 1, 2))
    fdata = fdata.astype('float32')
    fdata = fdata / 255
    return fdata
	
#Cargamos las imagenes de train 
#	print(train['path'][0])

if __name__ == '__main__': 
	
	#train = glob.glob('train_extra_resized/train_extra_resized/Type_1/*.jpg') + glob.glob('train_extra_resized/train_extra_resized/Type_2/*.jpg') \
	#+ glob.glob('train_extra_resized/train_extra_resized/Type_3/*.jpg')

	
	train = glob.glob('all_data_resized/all_data_resized/Type_1/*.png') + glob.glob('all_data_resized/all_data_resized/Type_2/*.png') \
	+ glob.glob('all_data_resized/all_data_resized/Type_3/*.png') + glob.glob('train_extra_resized/train_extra_resized/Type_1/*.jpg') \
	+ glob.glob('train_extra_resized/train_extra_resized/Type_2/*.jpg') + glob.glob('train_extra_resized/train_extra_resized/Type_3/*.jpg')
	
	print("Imagenes de train cargadas")
	
	
	#Creamos un dataset
	train = pd.DataFrame([[p.split('/')[2].split('\\')[0],p.split('/')[2].split('\\')[1],p] for p in train], columns = ['type','image','path']) #limit for Kaggle Demo

	#Metemos el size de las imagenes
	train = im_stats(train)
	
	train = train[train['size'] != '0 0'].reset_index(drop=True) #remove bad images
	
	train_data = normalize_image_features(train['path'])
	
	np.save('Datos/train256all.npy', train_data, allow_pickle=True, fix_imports=True)
	
	le = LabelEncoder()
	
	train_target = le.fit_transform(train['type'].values)
	
	print(le.classes_)
	
	np.save('Datos/train_target256all.npy', train_target, allow_pickle=True, fix_imports=True)
	
	'''
	test = glob.glob('test/test/*.jpg')
	
	test = pd.DataFrame([[p.split('\\')[1],p] for p in test], columns = ['image','path']) #[::20] #limit for Kaggle Demo
	
	test_data = normalize_image_features(test['path'])
	
	np.save('test32.npy', test_data, allow_pickle=True, fix_imports=True)

	test_id = test.image.values
	
	np.save('test_id32.npy', test_id, allow_pickle=True, fix_imports=True)
	'''
	




