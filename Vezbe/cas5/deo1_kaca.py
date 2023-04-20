#Bajesov test minimalne greske i test minimalne cene
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
plt.plot(X1[:,0], X1[:,1], 'o')
plt.plot(X2[:,0], X2[:,1], '*')
# Klase su negativno korelisane
#%% f-ja za izracunavanje fgv
def izracunaj_fgv(x, m, s):
    det = np.linalg.det(s)
    inv = np.linalg.inv(s)
    
    fgv_const = 1/(np.sqrt(2*np.pi*det))
    fgv_rest = np.exp(-0.5*(x-m).T @ inv @ (x-m))
    fgv = fgv_const*fgv_rest
    return fgv
#%% klasifikacija
p1 = X1.shape[0] / (X1.shape[0] + X2.shape[0])
p2 = X2.shape[0] / (X1.shape[0] + X2.shape[0])

T = np.log(p1/p2) #Bajesov test min cene

c12 = 0.1
c21 = 1
T1 = np.log(p2/p1 * c12/c21)

c12 = 10
c21 = 1
T2 = np.log(p2/p1 * c12/c21)

x1 = np.linspace(-7, 9, 100)
x2 = np.linspace(-2, 10, 100)
XX, YY = np.meshgrid(x1, x2)

X1p = np.empty((1,2))
X2p = np.empty((1,2))
h = np.zeros((100, 100))

for ix in range(XX.shape[0]):
    for iy in range(YY.shape[0]):
        pom = np.array([XX[ix, iy], YY[ix, iy]])
        
        f1 = izracunaj_fgv(pom, M1, S1)
        f2 = izracunaj_fgv(pom, M2, S2)
        
        h[ix, iy] = np.log(f2) - np.log(f1)
        
        if h[ix, iy] < T:
            X1p = np.append(X1p, np.reshape(pom, (1,2)),
                            axis=0)
        else:
            X2p = np.append(X2p, np.reshape(pom, (1,2)),
                            axis=0)
#%% Graficki prikaz klasifikacije
plt.figure()
plt.plot(X1p[:, 0], X1p[:, 1], 'r.', alpha=0.1)
plt.plot(X2p[:, 0], X2p[:, 1], 'b.', alpha=0.1)
plt.plot(X1[:,0], X1[:,1], 'ro')
plt.plot(X2[:,0], X2[:,1], 'b*')
plt.contour(XX, YY, h, [T]) #[T] oznacava da prikazujemo samo T red jer contour spaja tacke sa istom vrednoscu
plt.contour(XX, YY, h, [T1], colors='cyan')
plt.contour(XX, YY, h, [T2], colors='magenta')
#%% Konfuziona matrica

conf_mat = np.zeros((2,2))

for i in range(X1.shape[0]):
    pom = X1[i, :]
    f1 = izracunaj_fgv(pom, M1, S1)
    f2 = izracunaj_fgv(pom, M2, S2)
    h1 = np.log(f2) - np.log(f1)
    if h1<T:
        conf_mat[0,0] += 1
    else:
        conf_mat[1, 0] += 1

for i in range(X2.shape[0]):
    pom = X2[i, :]
    f1 = izracunaj_fgv(pom, M1, S1)
    f2 = izracunaj_fgv(pom, M2, S2)
    h2 = np.log(f2) - np.log(f1)
    if h2<T:
        conf_mat[0,1] += 1
    else:
        conf_mat[1, 1] += 1
        
#%% Prikaz konfuzione matrice

import seaborn as sns
plt.figure()
sns.heatmap(conf_mat, annot=True, fmt='g') #annot su anotacije tj. prikaz brojeva, a fmt='g' da se ne prikazuju kao eksponenti nego obicno

#%% Procena tacnosti

A = np.trace(conf_mat)/np.sum(conf_mat)
print(A)