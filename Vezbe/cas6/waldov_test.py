import numpy as np
import matplotlib.pyplot as plt
import numpy.random as rnd
import pandas as pd

def izracunaj_fgv(x, m, s):
    det = np.linalg.det(s)
    inv = np.linalg.inv(s)
    
    fgv_const = 1/(np.sqrt(2*np.pi*det))
    fgv_rest = np.exp(-0.5*(x-m).T @ inv @ (x-m))
    fgv = fgv_const*fgv_rest
    return fgv

data = pd.read_csv('diabetes_novo.csv')
X = data.drop(columns='Outcome').values
Y = data.Outcome

e1 = 0.01
e2 = 0.02

a = np.log(e2 / (1 - e1))
b = np.log((1 - e2) / e1)

X1 = X[Y == 0, :]
X2 = X[Y == 1, :]

M1 = np.mean(X1, axis=0)
M2 = np.mean(X2, axis=0)

S1 = np.cov(X1.T)
S2 = np.cov(X2.T)

plt.figure()
for i in range(100):
    Sm = 0
    Sm_niz = []
    while Sm > a and Sm < b:
        ind = rnd.randint(X1.shape[0])
        tren = X1[ind, :]
        f1 = izracunaj_fgv(tren, M1, S1)
        f2 = izracunaj_fgv(tren, M2, S2)
        
        h = np.log(f2) - np.log(f1)
        
        Sm += h
        Sm_niz.append(Sm)
    
    if Sm <= a:
        plt.plot(Sm_niz, 'r')
    elif Sm >= b:
        plt.plot(Sm_niz, 'b')

plt.plot([0, 15], [a, a], 'm')
plt.plot([0, 15], [b, b], 'c')

plt.figure()
for i in range(100):
    Sm = 0
    Sm_niz = []
    while Sm > a and Sm < b:
        ind = rnd.randint(X2.shape[0])
        tren = X2[ind, :]
        f1 = izracunaj_fgv(tren, M1, S1)
        f2 = izracunaj_fgv(tren, M2, S2)
        
        h = np.log(f2) - np.log(f1)
        
        Sm += h
        Sm_niz.append(Sm)
    
    if Sm <= a:
        plt.plot(Sm_niz, 'r')
    elif Sm >= b:
        plt.plot(Sm_niz, 'b')

plt.plot([0, 15], [a, a], 'm')
plt.plot([0, 15], [b, b], 'c')