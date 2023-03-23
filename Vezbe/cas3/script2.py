import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('diabetes_novo.csv')

klasa = data.Outcome

plt.figure()
plt.hist(klasa)

def izracunaj_info_d(kol):
    jedinstvene_vrednosti = np.unique(kol)
    info_d = 0;
    for vr in jedinstvene_vrednosti:
        pi = np.sum(kol == vr) / len(kol)
        info_d += -pi*np.log2(pi)
    return info_d

pom = np.array([1, 0, 0, 1, 0, 0, 1, 1, 1, 1])
print(izracunaj_info_d(pom))

# diskretizacija i racunanje informativnosti
info_d = izracunaj_info_d(klasa)
N = 20
information_gain = np.zeros((data.shape[1] - 1, 0))
information_gain = []
for i in range(data.shape[1] - 1):
    kol = data.iloc[:, i]
    korak = (np.max(kol) - np.min(kol)) / N
    nova_kol = np.floor(kol / korak) * korak
    
    uv = np.unique(nova_kol)
    info_da = 0
    for u in uv:
        pom = klasa[nova_kol == u]
        info_di = izracunaj_info_d(pom)
        di = np.sum(nova_kol == u)
        d = len(nova_kol)
        
        info_da += di * info_di / d
    
    print(info_da)
    information_gain.append(info_d - info_da)
    print(information_gain)
    print('---')
    
    data.iloc[:, i] = nova_kol
    
# najinformativnije obelezje je glukoza
