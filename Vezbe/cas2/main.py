import numpy as np
import matplotlib.pyplot as plt
import numpy.random as rnd

a = 5
b = 7
N = 1000
x = rnd.uniform(a, b, N)
print(min(x))
print(max(x))

plt.figure()
plt.hist(x, bins=50, ec='black', density=True)

fgv = 1/(b - a) * np.ones((N, 1))
plt.plot(x, fgv)

#%% normalna raspodela

mx = 5
sigmax = 0.5
N = 1000

x = rnd.normal(mx, sigmax, N)
print(min(x))
print(max(x))
print(np.mean(x))

plt.figure()
plt.hist(x, bins=30, density=True, ec='black')

x.sort()
fgv_const = 1 / (np.sqrt(2 * np.pi) * sigmax)
fgv = fgv_const * np.exp(-0.5 * (x - mx)**2 / sigmax**2)

plt.plot(x, fgv)

#%%

Mx = [6, 3]
rho = 0.8
sigma1 = 1.5
sigma2 = 3
sigmax = np.matrix([[sigma1**2, rho * sigma1 * sigma2], 
                    [rho * sigma1 * sigma2, sigma2**2]])
N = 1000

x1, x2 = rnd.multivariate_normal(Mx, sigmax, N).T

print(min(x1))
print(max(x1))

print(min(x2))
print(max(x2))

plt.figure()
plt.hist2d(x1, x2, (30, 30))
plt.colorbar()

N = 100
Mx = np.matrix([[6], [3]])
x1v = np.linspace(0, 12, N)
x2v = np.linspace(-10, 13, N)

X1, X2 = np.meshgrid(x1v, x2v)

fgv_const = 1 / (np.sqrt(2 * np.pi * np.linalg.det(sigmax)))
sigmax_inv = np.linalg.inv(sigmax)

xv = [[np.matrix([[x1v[i]], [x2v[j]]]) for j in range(len(x2v))] for i in range(len(x1v))]
fgv = [[fgv_const * np.exp(-0.5 * (xv[i][j] - Mx).T * sigmax_inv * (xv[i][j] - Mx)) for j in range(len(x2v))] for i in range(len(x1v))]

fgv = np.zeros((N, N))
for i in range(N):
    for j in range(N):
        x = np.matrix([[x1v[i]], [x2v[j]]])
        fgv[i, j] = fgv_const * np.exp(-0.5 * (x - Mx).T * sigmax_inv * (x - Mx))


plt.contour(X1, X2, fgv, cmap=plt.cm.plasma)

plt.figure()
ax = plt.axes(projection='3d')
ax.plot_surface(X1, X2, fgv)

#%%

def izracunaj_fgv(x, mx, sigmax):
    fgv_const = 1 / (np.sqrt(2 * np.pi * np.linalg.det(sigmax)))
    sigmax_inv = np.linalg.inv(sigmax)
    fgv = fgv_const * np.exp(-0.5 * (x - mx).T * sigmax_inv * (x - mx))
    return fgv

M1 = np.matrix([[4], [4]])
S1 = np.matrix([[2, -0.5], [-0.5, 2]])

M2 = np.matrix([[4], [8]])
S2 = np.matrix([[0.9, 0.7], [0.7, 0.9]])

N = 50
x1v = np.linspace(0, 8, N)
x2v = np.linspace(0, 12, N)
X1, X2 = np.meshgrid(x1v, x2v)

P1 = 0.4
P2 = 1 - P1

fgv = np.zeros((N, N))
for ix1 in range(N):
    for ix2 in range(N):
        x = np.matrix([[X1[ix1, ix2]], [X2[ix1, ix2]]])
        
        fgv1 = izracunaj_fgv(x, M1, S1)
        fgv2 = izracunaj_fgv(x, M2, S2)
        
        fgv[ix1, ix2] = P1*fgv1 + P2*fgv2

plt.contour(X1, X2, fgv, 50, cmap=plt.cm.plasma)

plt.figure()
ax = plt.axes(projection='3d')
ax.plot_surface(X1, X2, fgv)