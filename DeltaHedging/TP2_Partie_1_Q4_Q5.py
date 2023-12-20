import numpy as np
from numpy.random import randn
import matplotlib.pyplot as plt
from math import *
import random


def repartition(x):
    f = (1 + erf(x / sqrt(2))) / 2
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


def vega_black_scholes(t, S, K, T, r, sigma):
    f = (S * sqrt(T - t) * exp(-((d1(t, S, K, T, r, sigma)) ** 2) / 2)) / sqrt(2 * 3.14)


def delta(t, S, K, T, sigma, r):
    if t == T:
        return 1
    else:
        return repartition(d1(t, S, K, T, sigma, r))


def profit_and_loss(Nmc):
    S0, r, K, T, N, sigma = 1, 0.05, 1.5, 5, 60, 0.5
    B0 = 1
    delta_t = T / (N + 1)
    A0 = delta(0, S0, K, T, r, sigma)
    V0 = call_black_scholes(0, S0, K, T, r, sigma)
    P0 = A0 * S0 + B0
    P0_actu = V0
    t = np.linspace(0, T, N + 1)
    A = np.zeros(N + 1)
    V = np.zeros(N + 1)
    S = np.zeros(N + 1)
    B = np.zeros(N + 1)
    P = np.zeros(N + 1)
    P_actu = np.zeros(N + 1)
    PL = np.zeros(Nmc)
    N_trading = 10 # ou 100 ou 50 ou 25 ou 20 ou 5 ou 2 ou 1
    rebalancement = 100 / N_trading
    A[0], B[0], S[0], V[0], P[0], P_actu[0] = A0, B0, S0, V0, P0, P0_actu
    for j in range(Nmc):
        for i in range(0, N):
            S[i + 1] = S[i] * exp((r - (sigma ** 2) / 2) * delta_t + sigma * sqrt(delta_t) * np.random.randn())
            if i % rebalancement == 0:
                A[i + 1] = delta(t[i + 1], S[i + 1], K, T, r, sigma)
            else:
                A[i + 1] = A[i]
            B[i + 1] = (A[i] - A[i + 1]) * S[i + 1] + B[i] * (1 + r * delta_t)
            P[i + 1] = A[i + 1] * S[i + 1] + B[i + 1]
            V[i + 1] = call_black_scholes(t[i + 1], S[i + 1], K, T, r, sigma)
            P_actu[i + 1] = P[i + 1] - (P0 - V0) * exp(r * t[i + 1])
        PL[j] = V[N] - P_actu[N]
    plt.plot(P_actu)
    plt.plot(V)
    plt.title("Couverture Delta du portefeuille")
    plt.show()
    plt.plot(A)
    plt.plot(B)
    plt.title("Affichage du ratio A et du cash B")
    plt.legend()
    plt.show()
    plt.plot(P_actu - V)
    plt.title("Erreur")
    plt.show()
    return (PL)


def repartition_densite_pnl():
    N_x = 100
    x = np.zeros(N_x)
    repartition = np.zeros(N_x)
    a = 0.3
    Nmc = 10000
    ProfitAndLoss = np.zeros(Nmc)
    ProfitAndLoss = profit_and_loss(10_000)
    densite = np.zeros(N_x)
    for i in range(N_x):
        x[i] = -a + (2 * a) / (N_x) * (i - 1)
        compteur = 0
        for n in range(Nmc):
            if ProfitAndLoss[n] <= x[i]:
                compteur = compteur + 1
        repartition[i] = compteur / Nmc
    for j in range(N_x - 1):
        densite[j] = repartition[j + 1] - repartition[j]
    plt.figure()
    plt.plot(x, repartition)
    plt.title("Fonction de répartition du P&L")
    plt.show()
    plt.figure()
    plt.plot(x, densite)
    plt.title("Fonction de densité du P&L")
    plt.show()
    esperance = sum(ProfitAndLoss) / len(ProfitAndLoss)
    somme = 0;
    for k in ProfitAndLoss:
        somme = somme + (k - esperance) ** 2
    variance = somme / len(ProfitAndLoss)
    print("Espérance =", esperance, "; Variance =", variance)
    return ProfitAndLoss


def VaR(Nmc, alpha):
    K = alpha * Nmc
    ProfitAndLoss = np.zeros(Nmc)
    ProfitAndLoss = profit_and_loss(Nmc)
    ProfitAndLoss = sorted(ProfitAndLoss)
    print("VaR =", ProfitAndLoss[int(K)])


profit_and_loss(10000);
repartition_densite_pnl();
VaR(10000, 0.1);
