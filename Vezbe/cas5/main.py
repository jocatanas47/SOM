import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import numpy.random as rnd

N1 = 1000
N2 = 2000

M1 = np.array([4, 4])
S1 = np.matrix([[2, -0.5], [-0.5, 2]])
X1 = rnd.multivariate_normal(M1, S1, N1)

M2 = np.array([-3, 6])
S2 = np.matrix([[1.5, 0.5], [0.5, 1.5]])
X2 = rnd.multivariate_normal(M2, S2, N2)

plt.figure()
plt.hist(X1)

plt.figure()
plt.hist(X2)

#%% novi podaci
import pandas as pd
data = pd.read_csv('Skin_NonSkin.txt', 
                   header=None, 
                   delimiter='\t')

X = data.iloc[:, :3].values
Y = data.iloc[:, 3].values

X1 = X[Y == 1, :]
N1 = X1.shape[0]
X2 = X[Y == 2, :]
N2 = X2.shape[0]


#%% podela na trening i test skupove
N1trening = int(0.6*N1)
X1trening = X1[:N1trening, :]
X1test = X1[N1trening:, :]

N2trening = int(0.6*N2)
X2trening = X2[:N2trening, :]
X2test = X2[N2trening:, :]

#%% estimacija parametara raspodele na trening skupu
M1est = np.mean(X1trening, axis=0)
S1est = np.cov(X1trening.T)

M2est = np.mean(X2trening, axis=0)
S2est = np.cov(X2trening.T)

#%% projektovanje klasifikatora
def izracunaj_fgv(x, m, s):
    det = np.linalg.det(s)
    inv = np.linalg.inv(s)
    
    fgv_const = 1/(np.sqrt(2*np.pi*det))
    fgv_rest = np.exp(-0.5*(x-m).T @ inv @ (x-m))
    fgv = fgv_const * fgv_rest
    return fgv

#%%
p1 = N1trening / (N1trening + N2trening)
p2 = N2trening / (N1trening + N2trening)
T = np.log(p1/p2)

odluka1 = np.zeros((N1 - N1trening, 1))
for i in range(N1 - N1trening):
    x1 = X1test[i, :]
    f1 = izracunaj_fgv(x1, M1est, S1est)
    f2 = izracunaj_fgv(x1, M2est, S2est)
    h1 = np.log(f2) - np.log(f1)
    if h1 < T:
        odluka1[i] = 1
    else:
        odluka1[i] = 2
        
odluka2 = np.zeros((N2 - N2trening, 1))
for i in range(N2 - N2trening):
    x2 = X2test[i, :]
    f1 = izracunaj_fgv(x2, M1est, S1est)
    f2 = izracunaj_fgv(x2, M2est, S2est)
    h2 = np.log(f2) - np.log(f1)
    if h2 < T:
        odluka2[i] = 1
    else:
        odluka2[i] = 2
        
odluka = np.append(odluka1, odluka2, axis=0)
true_vals = np.append(np.ones((N1 - N1trening, 1)), 2*np.ones((N2 - N2trening, 1)), axis=0)

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

cm = confusion_matrix(true_vals, odluka)
plt.figure()
sns.heatmap(cm, annot=True, fmt='g')

A = accuracy_score(true_vals, odluka)
print(A)

#%% granica odlucivanja
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
        
plt.figure()
plt.plot(X1[:, 0], X1[:, 1], 'o')
plt.plot(X2[:, 0], X2[:, 1], '*')
plt.contour(XX, YY, h, [T])