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

M3 = np.array([4, 10])
S3 = np.matrix([[0.9, 0.7], [0.7, 0.9]])
X3 = rnd.multivariate_normal(M3, S3, N)

plt.figure()
plt.plot(X1[:, 0], X1[:, 1], 'o')
plt.plot(X2[:, 0], X2[:, 1], '*')
plt.plot(X3[:, 0], X3[:, 1], 'd')
# Klase su negativno korelisane

#%% f-ja za izracunavanje fgv
def izracunaj_fgv(x, m, s):
    det = np.linalg.det(s)
    inv = np.linalg.inv(s)
    
    fgv_const = 1/(np.sqrt(2*np.pi*det))
    fgv_rest = np.exp(-0.5*(x-m).T @ inv @ (x-m))
    fgv = fgv_const*fgv_rest
    return fgv

xx = np.linspace(-7, 9, 100)
yy = np.linspace(-1, 15, 100)

XX, YY = np.meshgrid(xx, yy)

X1p = np.empty((1, 2))
ostalo = np.empty((1, 2))

T = 9

for i in range(100):
    for j in range(100):
        tren = np.array([XX[i, j], YY[i, j]])
        
        d2 = (tren - M1.T) @ np.linalg.inv(S1) @ (tren - M1)
        
        if d2 < T:
            X1p = np.append(X1p, np.reshape(tren, (1, 2)), axis=0)
        else:
            ostalo = np.append(ostalo, np.reshape(tren, (1, 2)), axis=0)
            
plt.figure()
plt.plot(X1p[:, 0], X1p[:, 1], 'ro', alpha=0.05)
plt.plot(ostalo[:, 0], ostalo[:, 1], 'ko', alpha=0.05)
plt.plot(X1[:, 0], X1[:, 1], 'ro')
plt.plot(X2[:, 0], X2[:, 1], 'b*')
plt.plot(X3[:, 0], X3[:, 1], 'gd')