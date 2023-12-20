import matplotlib.pyplot as plt
import numpy as np
from math import *
from numpy import linalg as LA
import random as rm

r0 = 0
N = 1000
T = 5
delta_T = T / N

Eta = 0.6
Sigma = 0.08
Gamma = 4

r = [r0]
t = [0]

beta = [1, 1]
d = [1, 1]

lambda_1 = 0.1
epsilon = 10 ** (-4)

J = np.array([np.zeros(2) for i in range(N)])
r_final = np.zeros(N)
Res = np.zeros(N)

while LA.norm(d) > epsilon:
    t = [0]
    r = [r0]
    for i in range(N - 1):
        t.append(t[-1] + delta_T)
        r.append(r[-1] * exp(-Gamma * delta_T) + Eta / Gamma * (1 - exp(-Gamma * delta_T)) + sqrt(
            (Sigma ** 2) / (2 * Gamma) * (1 - exp(-2 * Gamma * delta_T))) * rm.gauss(0, 1))
        J[i][0] = -r[i]
        J[i][1] = -1
        Res[i] = r[-1] - (beta[0] * r[-2] + beta[1])

    d = np.dot(-LA.inv(np.dot(J.transpose(), J) + lambda_1 * np.identity(2)), np.dot(J.transpose(), Res))
    beta = beta + d.transpose()

for i in range(N):
    r_final[i] = beta[0] * r[i] + beta[1]

print('Les coefficients de la regression : [a b] = ' + str(beta))

D = sqrt(np.var(Res))
print('D = ' + str(D))

Gamma_cal = -log(beta[0]) / delta_T
print('Gamma = ' + str(Gamma_cal))

Eta_cal = Gamma_cal * beta[1] / (1 - beta[0])
print('Eta = ' + str(Eta_cal))

Sigma_cal = D * sqrt(-2 * log(beta[0]) / (delta_T * (1 - beta[0] ** 2)))
print('Sigma = ' + str(Sigma_cal))

plt.plot(t, r, label='r simulé')
plt.plot(t, r_final, label='r calibré')
plt.legend()
plt.xlabel('t')
plt.ylabel('r')
plt.title('Short term rate simulation')
plt.show()

plt.plot(r[0:N - 1], r[1:N], '.', label='r simulé')
plt.plot(r, r_final, label='r calibré')
plt.legend()
plt.xlabel('r_i')
plt.ylabel('r_(i+1)')
plt.title('Short term rate simulation')
plt.show()