import matplotlib.pyplot as plt
import numpy as np
from math import *
from numpy import linalg as LA
import cmath

t = 1
r0 = 0.04

eta = 1
sigma = 1
gamma = 1
sigma_carre = sigma ** 2
beta = [eta, sigma_carre, gamma]

epsilon = 10 ** (-9)
lambda_1 = 0.01

d = [1, 1, 1]
J = np.array([np.zeros(3) for i in range(10)])

Yt_final = np.zeros(10)
Res = np.zeros(10)
T = np.linspace(3, 30, 10)

Y_market=[0.056, 0.064, 0.074, 0.081, 0.082, 0.09, 0.087, 0.092, 0.0895, 0.091]


def fonction_B(t, T, Gamma):
    return (1 - exp(-Gamma * (T - t))) / Gamma


def fonction_A(t, T, Eta, Sigma, Gamma):
    return (fonction_B(t, T, Gamma) - (T - t)) * (Eta * Gamma - 0.5 * Sigma ** 2) / (Gamma ** 2) - (Sigma ** 2) * (
        fonction_B(t, T, Gamma)) ** 2 / (4 * Gamma)


while LA.norm(d) > epsilon: #Levenberg-Marquart
    for i in range(10):
        B = fonction_B(t, T[i], beta[2])
        A = fonction_A(t, T[i], beta[0], cmath.sqrt(beta[1]), beta[2])
        DB = ((T[i] - t) * exp(-(T[i] - t) * beta[2]) - B) / beta[2]
        DA = (beta[0] * (DB * beta[2] - B) + (T[i] - t) * beta[0] - beta[1] / 2 * (DB - 2 * B / beta[2]) - (T[i] - t) *
              beta[1] / beta[2] - beta[1] * B / 4 * (2 * beta[2] * DB - B)) / beta[2] ** 2

        J[i][0] = (B - (T[i] - t)) / (T[i] - t) / beta[0]
        J[i][1] = -((B - (T[i] - t)) / (2 * beta[2]) + (B ** 2) / 4) / ((T[i] - t) * beta[2])
        J[i][2] = (DA - r0 * DB) / (T[i] - t)
        Yt = -(A - r0 * B) / (T[i] - t)
        Res[i] = Y_market[i] - Yt

    d = np.dot(-LA.inv(np.dot(J.transpose(), J) + lambda_1 * np.identity(3)), np.dot(J.transpose(), Res))
    beta = beta + d.transpose()
    eta = eta + d[0]
    sigma_carre = sigma_carre + d[1]
    gamma = gamma + d[2]

for i in range(10):
    B = fonction_B(t, T[i], beta[2])
    A = fonction_A(t, T[i], beta[0], sqrt(beta[1]), beta[2])
    Yt_final[i] = -(A - r0 * B) / (T[i] - t)

plt.plot(T, Yt_final, label='Yt_calibrated')
plt.plot(T, Y_market, 'x', label='Yt_market')
plt.legend()
plt.xlabel('Maturité (T)')
plt.ylabel('Yield (Y)')
plt.title('Calibration de Vasicek')
plt.show()

print('Eta = ', eta)
print('Sigma² = ', sigma_carre)
print('Gamma = ', gamma)
