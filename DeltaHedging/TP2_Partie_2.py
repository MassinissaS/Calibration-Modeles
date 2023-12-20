import numpy as np
from numpy.random import randn
import matplotlib.pyplot as plt
from math import *


def repartition(x):
    f = 1 / 2 * (1 + math.erf(x / math.sqrt(2)))
    return f


def d1(t, S, K, T, r, sigma):
    d1 = (log(S / K) + (r + ((sigma ** 2) / 2)) * (T - t)) / (sigma * sqrt(T - t))
    return d1


def d2(t, S, K, T, r, sigma):
    d2 = (log(S / K) + (r - ((sigma ** 2) / 2)) * (T - t)) / (sigma * sqrt(T - t))
    return d2


def call_black_scholes(t, S, K, T, r, sigma):
    if t == T:
        return max(S - K, 0)
    else:
        return S * repartition(d1(t, S, K, T, r, sigma)) - K * exp(-r * (T - t)) * repartition(d2(t, S, K, T, r, sigma))


def delta(t, S, K, T, sigma, r):
    return repartition(d1(t, S, K, T, sigma, r))


def profit_and_loss(Nmc):
    S0, r, K, T, N, sigma = 1, 0.05, 1.5, 5, 100, 0.5
    delta_t = T / (N + 1)
    t = np.linspace(0, T, N + 1)
    A0 = delta(0, S0, K, T, r, sigma)
    V0 = call_black_scholes(0, S0, K, T, r, sigma)
    B0 = V0 - A0 * S0
    P0 = A0 * S0 + B0
    P0_actu = V0
    A = np.zeros(N + 1)
    V = np.zeros(N + 1)
    S = np.zeros(N + 1)
    B = np.zeros(N + 1)
    P = np.zeros(N + 1)
    P_actu = np.zeros(N + 1)

    PL = np.zeros(Nmc)
    A[0], B[0], S[0], V[0], P[0], P_actu[0] = A0, B0, S0, V0, P0, P0_actu
    for j in range(Nmc):
        for i in range(0, N):
            S[i + 1] = S[i] * np.exp((r - (sigma ** 2) / 2) * delta_t + sigma * np.sqrt(delta_t) * np.random.randn())
            A[i + 1] = delta(t[i + 1], S[i + 1], K, T, r, sigma)
            B[i + 1] = (A[i] - A[i + 1]) * S[i + 1] + B[i] * (1 + r * delta_t)
            P[i + 1] = A[i + 1] * S[i + 1] + B[i + 1]
            V[i + 1] = call_black_scholes(t[i + 1], S[i + 1], K, T, r, sigma)
            P_actu[i + 1] = P[i + 1] - (P0 - V0) * np.exp(r * t[i + 1])
        PL[j] = P[N] - V[N]
    plt.figure()
    plt.plot(t, V, t, P_actu)
    plt.title("Évolution de l'option et du portefeuille de couverture")


profit_and_loss(10000);