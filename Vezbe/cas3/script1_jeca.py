# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 16:28:57 2023

@author: zj190217d
"""

import numpy as np
import matplotlib.pyplot as plt
import numpy.random as rnd


Mx = [0, 0]
Sx = np.matrix([[2, 1.6], [1.6, 8]])
N = 1000

x1, x2 = rnd.multivariate_normal(Mx, Sx, N).T
#transponujemo jer ova fja vraca 2X1000, a da bismo razdvojili x1 i x2, treba nam 1000x2

plt.figure()
plt.plot(x1, x2, 'o')
# imamo pravac, tj korelaciju izmedju obelezja

X = np.array([x1, x2])

#%% sopstvene stvari

eigval, eigvec = np.linalg.eig(Sx)
print(eigval)
print(eigvec)

#%% ortonormalna transformacija

At = eigvec.T
Y = At @ X # @ je matricno mnozenje

Sy = np.cov(Y)
print(Sy)
# dovoljno dobar rez

y1 = Y[0, :]
y2 = Y[1, :]

plt.figure()
plt.plot(y1, y2, 'o')
# sve je grupisano oko 0 kao i pre, ali sada su odbirci kao krug, nekorelisani

#%% transformacija beljenja

L = np.matrix([[eigval[0], 0], [0, eigval[1]]])

import scipy.linalg as lin

At = lin.fractional_matrix_power(L, -0.5) @ eigvec.T
#mora ovako jer python ne zna da stepenuje matr sa necelim br

Y = At @ X

Sy = np.cov(Y)
print(Sy)
# zadovoljavajuc rez

y1 = Y[0, :]
y2 = Y[1, :]

plt.figure()
plt.plot(y1, y2, 'o')
# opet su centr oko 0, i svi pripadaju intervalu [-3sigma, 3sigma] tj [-3, 3]

#%% transf bojenja

My = [0, 0]
Sy = np.eye(2)
N = 1000
y1, y2 = rnd.multivariate_normal(My, Sy, N).T
Y = np.array([y1, y2])

Sx = np.matrix([[0.9, 0.7], [0.7, 0.9]])
eigval, eigvec = np.linalg.eig(Sx)

L = np.matrix([[eigval[0], 0], [0, eigval[1]]])

X = eigvec @ lin.fractional_matrix_power(L, 0.5) @ Y
Mx = np.matrix([[4], [8]])
X = X + Mx

plt.figure()
plt.plot(y1, y2, 'o')
plt.plot(X[0, :].T, X[1, :].T, 'o')
# plavi odg Y podacima koji su izbeljeni, a X su korelisani nekako
# ovo radimo jer lako mozemo da generisemo beli sum i onda da ga obojimo