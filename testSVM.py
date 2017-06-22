import numpy as np
import pandas as pd

probs = np.load('Resultados/GBoostprobsTune.npy')
test_id = np.load('Datos/test_id244.npy')
submission_name = 'GBoostprobsTune.csv'

df = pd.DataFrame(probs, columns=['Type_1', 'Type_2', 'Type_3'])
df['image_name'] = test_id
df.to_csv('Resultados/' + submission_name, index=False)