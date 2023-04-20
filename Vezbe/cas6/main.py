import numpy as np
import matplotlib.pyplot as plt
import numpy.random as rnd

#%% Definisanje klasa
N = 1000

M1 = np.array([4, 4])
S1 = np.matrix([[2, -0.5], [-0.5, 2]]) # Moze i np.array
X1 = rnd.multivariate_normal(M1, S1, N)

M2 = np.array([-3, 6])
S2 = np.matrix([[1.5, 0.5], [0.5, 1.5]]) 
X2 = rnd.multivariate_normal(M2, S2, N)

plt.figure()
plt.plot(X1[:, 0], X1[:, 1], 'o')
plt.plot(X2[:, 0], X2[:, 1], '*')
# Klase su negativno korelisane

#%% f-ja za izracunavanje fgv
def izracunaj_fgv(x, m, s):
    det = np.linalg.det(s)
    inv = np.linalg.inv(s)
    
    fgv_const = 1/(np.sqrt(2*np.pi*det))
    fgv_rest = np.exp(-0.5*(x-m).T @ inv @ (x-m))
    fgv = fgv_const*fgv_rest
    return fgv

#%% test sa odbacivanjem

p1 = X1.shape[0] / (X1.shape[0] + X2.shape[0])
p2 = X2.shape[0] / (X1.shape[0] + X2.shape[0])

xx = np.linspace(-7, 9, 100)
yy = np.linspace(-1, 10, 100)

XX, YY = np.meshgrid(xx, yy)

T1 = 0.1
T2 = 0.01

X1p = np.empty((1, 2))
X2p = np.empty((1, 2))
ostalo = np.empty((1, 2))

for i in range(100):
    for j in range(100):
        tren = np.array([XX[i, j], YY[i, j]])
        f1 = izracunaj_fgv(tren, M1, S1)
        f2 = izracunaj_fgv(tren, M2, S2)
        
        h = np.log(f2) - np.log(f1)
        
        if h < np.log(p1/p2) + np.log(T1/(1 - T1)):
            X1p = np.append(X1p, np.reshape(tren, (1, 2)), axis=0)
        elif h > np.log(p1/p2) + np.log((1- T2)/T2):
            X2p = np.append(X2p, np.reshape(tren, (1, 2)), axis=0)
        else:
            ostalo = np.append(ostalo, np.reshape(tren, (1, 2)), axis=0)

X1p = X1p[1:, :]
X2p = X2p[1:, :]
ostalo = ostalo[1:, :]
    
plt.figure()
plt.plot(X1[:, 0], X1[:, 1], 'ro')
plt.plot(X2[:, 0], X2[:, 1], 'b*')
plt.plot(X1p[:, 0], X1p[:, 1], 'ro', alpha=0.05)
plt.plot(X2p[:, 0], X2p[:, 1], 'bo', alpha=0.05)
plt.plot(ostalo[:, 0], ostalo[:, 1], 'ko', alpha=0.05)
