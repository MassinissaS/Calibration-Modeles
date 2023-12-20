import matplotlib.pyplot as plt
import numpy as np
import math


def repartition(x):
    f = 0.5 * (1 + math.erf(x / math.sqrt(2)))
    return f


def d1(t, S, K, T, r, sigma, cost, N):
    delta_t = T / N
    sigma_modifie = sigma * math.sqrt(1 + (cost / sigma) * math.sqrt(2 / (np.pi * delta_t)))
    d1 = (math.log(S / K) + (r + ((sigma_modifie ** 2) / 2)) * (T - t)) / (sigma_modifie * math.sqrt(T - t))
    return d1


def d2(t, S, K, T, r, sigma, cost, N):
    delta_t = T / N
    sigma_modifie = sigma * math.sqrt(1 + (cost / sigma) * math.sqrt(2 / (np.pi * delta_t)))
    d2 = (math.log(S / K) + (r - ((sigma_modifie ** 2) / 2)) * (T - t)) / (sigma_modifie * math.sqrt(T - t))
    return d2


def call_black_scholes(t, S, K, T, r, sigma, cost, N):
    if t == T:
        f = max(S - K, 0)
    else:
        f = S * repartition(d1(t, S, K, T, r, sigma, cost, N)) - K * math.exp(-r * (T - t)) * repartition(
            d2(t, S, K, T, r, sigma, cost, N))
    return f


def delta(t, S, K, T, r, sigma, cost, N):
    if t == T:
        f = 1
    else:
        f = repartition(d1(t, S, K, T, r, sigma, cost, N))
    return f


def profit_and_loss(Nmc):
    S0, r, K, T, N, sigma, cost = 100, 0.05, 100, 1, 1040, 0.25, 0.01
    delta_t = T / N
    A0 = delta(0, S0, K, T, r, sigma, cost, N)
    V0 = call_black_scholes(0, S0, K, T, r, sigma, cost, N)
    P0 = V0 - cost * np.abs(A0) * S0
    P0_actu = V0
    B0 = V0 - A0 * S0 - cost * np.abs(A0) * S0
    t = np.linspace(0, T, N + 1)
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
            S[i + 1] = S[i] * math.exp(
                (r - (sigma ** 2) / 2) * delta_t + sigma * math.sqrt(delta_t) * np.random.randn())
            if i % 4 == 0:
                A[i + 1] = delta(t[i + 1], S[i + 1], K, T, r, sigma, cost, N)
            else:
                A[i + 1] = A[i]
            # A[i+1]=delta(t[i+1],S[i+1],K,T,r,sigma,cost,N)
            B[i + 1] = (A[i] - A[i + 1]) * S[i + 1] + B[i] * (1 + r * delta_t)
            P[i + 1] = A[i] * S[i + 1] + B[i] * (1 + r * delta_t) - cost * np.abs(A[i + 1] - A[i]) * S[i + 1]
            V[i + 1] = S[i + 1] * repartition(d1(t[i + 1], S[i + 1], K, T, r, sigma, cost, N)) - K * math.exp(
                -r * (T - t[i + 1])) * repartition(d2(t[i + 1], S[i + 1], K, T, r, sigma, cost, N))
            P_actu[i + 1] = P[i + 1] - (P0 - V0) * math.exp(r * t[i + 1])
        # PL[j]=V[N]-P_actu[N]
    plt.plot(t, V)
    plt.plot(t, P_actu)
    plt.title("Couverture Delta du portefeuille")
    plt.show()
    return (PL)


def repartition_densite_pnl():
    N_x = 100
    x = np.zeros(N_x)
    repartition = np.zeros(N_x)
    a = 0.3
    Nmc = 100
    ProfitAndLoss = np.zeros(Nmc)
    ProfitAndLoss = profit_and_loss(Nmc)
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


repartition_densite_pnl();

