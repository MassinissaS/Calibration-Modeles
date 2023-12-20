import matplotlib.pyplot as plt
import numpy as np
import math


def repartition(x):
    f = 1 / 2 * (1 + math.erf(x / math.sqrt(2)))
    return f


def d1(t, S, K, T, r, sigma):
    f = (math.log(S / K) + (r + sigma ** 2 / 2) * (T - t)) / (sigma * math.sqrt(T - t))
    return f


def d2(t, S, K, T, r, sigma):
    f = (math.log(S / K) + (r - sigma ** 2 / 2) * (T - t)) / (sigma * math.sqrt(T - t))
    return f


def call_black_scholes(t, S, K, T, r, sigma):
    if t == T:
        f = max(S - K, 0)
    else:
        f = S * repartition(d1(t, S, K, T, r, sigma)) - K * math.exp(-r * (T - t)) * repartition(
            d2(t, S, K, T, r, sigma))
    return f


def delta(t, S, K, T, r, sigma):
    if t == T:
        f = 1
    else:
        f = repartition(d1(t, S, K, T, r, sigma))
    return f


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
    A[0], B[0], S[0], V[0], P[0], P_actu[0] = A0, B0, S0, V0, P0, P0_actu

    for j in range(Nmc):
        for i in range(0, N):
            S[i + 1] = S[i] * math.exp((r - (sigma ** 2) / 2) * delta_t + sigma * math.sqrt(delta_t) * np.random.randn())
            if i % 2 == 0: # Nous rebalançons le portefeuille une fois sur deux
                A[i + 1] = delta(t[i + 1], S[i + 1], K, T, r, sigma)
            else:
                A[i + 1] = A[i]
            B[i + 1] = (A[i] - A[i + 1]) * S[i + 1] + B[i] * (1 + r * delta_t)
            P[i + 1] = A[i + 1] * S[i + 1] + B[i + 1]
            V[i + 1] = call_black_scholes(t[i + 1], S[i + 1], K, T, r, sigma)
            P_actu[i + 1] = P[i + 1] - (P0 - V0) * math.exp(r * t[i + 1])
        PL[j] = V[N] - P_actu[N]
        plt.plot(t, V, t, P_actu)

    return (PL)


def repartition_densite_pnl():
    N_x = 100
    x = np.zeros(N_x)
    repartition = np.zeros(N_x)
    a = 0.3
    Nmc = 10000
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
    esperance = sum(ProfitAndLoss) / len(ProfitAndLoss)
    somme = 0;
    for k in ProfitAndLoss:
        somme = somme + (k - esperance) ** 2
    variance = somme / len(ProfitAndLoss)
    print("Esperance =", esperance, "; Variance =", variance)
    return ProfitAndLoss



def VaR(Nmc, alpha):
    K = alpha * Nmc
    ProfitAndLoss = np.zeros(Nmc)
    ProfitAndLoss = profit_and_loss(Nmc)
    ProfitAndLoss = sorted(ProfitAndLoss)
    print("Nous sommes certains à ", (1-alpha)*100, "% que nous n'allons pas perdre plus de", ProfitAndLoss[int(K)], "euros")


repartition_densite_pnl()
VaR(10000, 0.05)






