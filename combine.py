import pandas as pd
import numpy as np

pred12 = np.array([[0.5,0.5], [1.0, 0.0]])
pred13 = np.array([[0.2,0.8], [1., 0.0]])
pred23 = np.array([[0.6,0.4], [0.5, 0.5]])

df12 = pd.DataFrame(pred12, columns = ['Type_1_12', 'Type_2_12'])
df13 = pd.DataFrame(pred13, columns = ['Type_1_13', 'Type_3_13'])
df23 = pd.DataFrame(pred23, columns = ['Type_2_23', 'Type_3_23'])

df12['id'] = ['uno','dos']
df13['id'] = ['uno','dos']
df23['id'] = ['uno','dos']

df12.to_csv('df12.cvs', index = False)
df13.to_csv('df13.cvs', index = False)
df23.to_csv('df23.cvs', index = False)

#vector con la cantidad de cada clase
N = [100,200,100]

'''Funcion que devuelve la probabilidad de pertenencia de una instancia a una clase
:param intance: la instancia a etiquetar
:param i: la clase para la que queremos obtener la probabilidad de pertenencia
'''
def getProb(instance, i):
    R = [[0,instance['Type_1_12'],instance['Type_1_13']], [instance['Type_2_12'],0,instance['Type_2_23']], [instance['Type_3_13'],instance['Type_3_23'],0]]
    summation = 0
    
    for j in range(3):
        if (j != i):
            Pij = R[i][j] - min(R[i][j], R[j][i])
            Cij = min(R[i][j], R[j][i])
            Iij = 1.0 - max(R[i][j], R[j][i])
            summation += Pij + 0.5*Cij + Iij*N[i]/(N[i] + N[j])

    return summation


df12 = pd.read_csv('df12.cvs')
df13 = pd.read_csv('df13.cvs')
df23 = pd.read_csv('df23.cvs')


df = pd.concat([df12, df13, df23], axis=1)
df = pd.merge(df12, df13, on='id')
df = pd.merge(df, df23, on='id')
print(df)

for c in ['Type_3_12', 'Type_1_23', 'Type_2_13']:
    df[c] = 0.0

print(df)

dfProbs = pd.DataFrame()
dfProbs['Type_1'] = df.apply(lambda x : np.mean([x['Type_1_12'], x['Type_1_13'], x['Type_1_23']]), axis=1)
dfProbs['Type_2'] = df.apply(lambda x : np.mean([x['Type_2_12'], x['Type_2_13'], x['Type_2_23']]), axis=1)
dfProbs['Type_3'] = df.apply(lambda x : np.mean([x['Type_3_12'], x['Type_3_13'], x['Type_3_23']]), axis=1)
dfProbs['image_id'] = df['id']

print("Resultado con la media")
print(dfProbs)

dfProbs = pd.DataFrame()
dfProbs['Type_1'] = df.apply(lambda x : getProb(x,0)/3.0, axis=1)
dfProbs['Type_2'] = df.apply(lambda x : getProb(x,1)/3.0, axis=1)
dfProbs['Type_3'] = df.apply(lambda x : getProb(x,2)/3.0, axis=1)
dfProbs['image_id'] = df['id']

print("Resultado con el otro metodo")
print(dfProbs)
print(dfProbs['Type_1'] + dfProbs['Type_2'] + dfProbs['Type_3'])
