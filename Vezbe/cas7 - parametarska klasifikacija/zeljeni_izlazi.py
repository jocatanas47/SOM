import numpy as np
import matplotlib.pyplot as plt
import numpy.random as rnd

N = 1000

M1 = [2, 2]
M2 = [12, 8]

R1 = 2*rnd.rand(N, 1)
theta1 = 2*np.pi*rnd.rand(N, 1)
x11 = M1[0] + np.cos(theta1)*R1
x12 = M1[1] + np.sin(theta1)*R1

R2 = 2*rnd.rand(N, 1) + 3
theta2 = 2*np.pi*rnd.rand(N, 1)
x21 = M2[0] + np.cos(theta2)*R2
x22 = M2[1] + np.sin(theta2)*R2

plt.figure()
plt.plot(x11, x12, 'o')
plt.plot(x21, x22, '*')

#%%
K1 = np.append(x11, x12, axis=1).T
K2 = np.append(x21, x22, axis=1).T

#%%
Z1 = np.append(-K1, -np.ones((1, N)), axis=0)
Z2 = np.append(K2, np.ones((1, N)), axis=0)

U = np.append(Z1, Z2, axis=1)
G = np.append(np.ones((N, 1)), 2*np.ones((N, 1)), axis=0)

W = np.linalg.inv(U @ U.T) @ U @ G

#%%
V = W[:-1]
V0 = W[-1]

x1t = np.array([0, 17])
x2t = -V[0]/V[1]*x1t - V0/V[1]

plt.figure()
plt.plot(x11, x12, 'o')
plt.plot(x21, x22, '*')
plt.plot(x1t, x2t)