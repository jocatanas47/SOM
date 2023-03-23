import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

data = pd.read_csv('diabetes_novo.csv')

# pearson - linearna zavisnost
# spearman - nelinearno

corr = data.corr(method='pearson')
plt.figure()
sns.heatmap(corr, annot=True)

corr = data.corr(method='spearman')
plt.figure()
sns.heatmap(corr, annot=True)

def izracunaj_r(corr):
    k = corr.shape[0] - 1
    rzi = np.mean(corr.iloc[-1, :-1])
    rii = np.mean(np.mean(corr.iloc[:-1, :-1])) - 1/k
    
    r = k * rzi / np.sqrt(k + k*(k - 1)*rii)
    
    return r

R = izracunaj_r(corr)
print(R)

for ob in range(data.shape[1] - 1):
    data_1 = data.drop(data.columns[ob], axis=1)
    corr = data_1.corr(method='pearson')
    print(izracunaj_r(corr))
    print('-----')