import numpy as np
import matplotlib.pyplot as plt
import numpy.random as rnd

N = 1000

M1 = [2, 2]
M2 = [2, 2]

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
# concatenate spaja vise vektora
Z1 = np.concatenate((-x11**2, 
                     -x11*x12, 
                     -x12**2, 
                     -x11, 
                     -x12, 
                     -np.ones((N, 1))), 
                    axis=1).T
Z2 = np.concatenate((x21**2, 
                     x21*x22, 
                     x22**2, 
                     x21, 
                     x22, 
                     np.ones((N, 1))), 
                    axis=1).T

#%%
U = np.append(Z1, Z2, axis=1)
G = np.append(np.ones((N, 1)), 2*np.ones((N, 1)), axis=0)

W = np.linalg.inv(U @ U.T) @ U @ G

#%%
Q = np.array([[W[0], W[1]/2], [W[1]/2, W[2]]])
V = np.array([W[3], W[4]])
V0 = W[5]

x1t = np.linspace(-3, 7, N)
x2t = np.linspace(-3, 7, N)
X1t, X2t = np.meshgrid(x1t, x2t)

eq1 = Q[0, 0]*X1t**2 + 2*Q[0, 1]*X1t*X2t + Q[1, 1]*X2t**2
eq2 = V[0]*X1t + V[1]*X2t

eq = eq1 + eq2 + V0

plt.figure()
plt.plot(x11, x12, 'o')
plt.plot(x21, x22, '*')
plt.contour(X1t, X2t, eq, [0])