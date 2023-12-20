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


def pnl_modele_1(Nmc):
    S0, r, K, T, N = 1, 0.05, 1.5, 5, 100
    B0 = 1
    t = np.linspace(0, T, N + 1)
    A = np.zeros(N + 1)
    V = np.zeros(N + 1)
    S = np.zeros(N + 1)
    B = np.zeros(N + 1)
    P = np.zeros(N + 1)
    sigma = np.zeros(N + 1)
    delta_t = T / (N + 1)
    sigma1 = 0.3;
    sigma2 = 0.5;
    if np.random.rand() < 0.8:
        sigma[0] = sigma1
    else:
        sigma[0] = sigma2
    P_actu = np.zeros(N + 1)
    PL = np.zeros(Nmc)
    P_actu = np.zeros(N + 1)
    A0 = delta(0, S0, K, T, r, sigma[0])
    V0 = call_black_scholes(0, S0, K, T, r, sigma[0])
    P0 = A0 * S0 + B0
    P0_actu = V0
    A[0], B[0], S[0], V[0], P[0], P_actu[0] = A0, B0, S0, V0, P0, P0_actu

    for j in range(Nmc):
        for i in range(0, N):
            U = np.random.rand()
            if U < 0.8:
                sigma[i + 1] = sigma1
            else:
                sigma[i + 1] = sigma2
            S[i + 1] = S[i] * math.exp(
                (r - (sigma[i + 1] ** 2) / 2) * delta_t + sigma[i + 1] * math.sqrt(delta_t) * np.random.randn())
            # si on veut rebalancer une fois sur deux
            if i % 4 == 0:
                A[i + 1] = delta(t[i + 1], S[i + 1], K, T, r, sigma[i + 1])
            else:
                A[i + 1] = A[i]
            B[i + 1] = (A[i] - A[i + 1]) * S[i + 1] + B[i] * (1 + r * delta_t)
            P[i + 1] = A[i + 1] * S[i + 1] + B[i + 1]
            V[i + 1] = call_black_scholes(t[i + 1], S[i + 1], K, T, r, sigma[i + 1])
            P_actu[i + 1] = P[i + 1] - (P0 - V0) * math.exp(r * t[i + 1])
        PL[j] = V[N] - P_actu[N]
    plt.plot(sigma)
    plt.title("Vérification du modèle 1")
    plt.show()
    plt.plot(V)
    plt.plot(P_actu)
    plt.title("Couverture Delta du portefeuille (Modèle 1)")
    plt.show()
    #plt.plot(A)
    #plt.plot(B)
    #plt.title("Affichage de A et B")
    #plt.legend()
    #plt.show()
    plt.plot(P_actu)
    plt.title("Portefeuille actualisé")
    plt.show()
    plt.plot(P_actu - V)
    plt.title("Erreur du modèle 1")
    plt.show()
    return (PL)


def pnl_modele_2(Nmc):
    S0, r, K, T, N = 1, 0.05, 1.5, 5, 100
    B0 = 1
    t = np.linspace(0, T, N + 1)
    A = np.zeros(N + 1)
    V = np.zeros(N + 1)
    S = np.zeros(N + 1)
    B = np.zeros(N + 1)
    P = np.zeros(N + 1)
    sigma = np.zeros(N + 1)
    delta_t = T / (N + 1)
    sigma1 = 0.3;
    sigma2 = 0.5;
    sigma[0] = sigma1;
    P_actu = np.zeros(N + 1)
    PL = np.zeros(Nmc)
    P_actu = np.zeros(N + 1)
    A0 = delta(0, S0, K, T, r, sigma[0])
    V0 = call_black_scholes(0, S0, K, T, r, sigma[0])
    P0 = A0 * S0 + B0
    P0_actu = V0
    A[0], B[0], S[0], V[0], P[0], P_actu[0] = A0, B0, S0, V0, P0, P0_actu

    for j in range(Nmc):
        for i in range(0, N):
            U = np.random.rand()
            if U < 0.05:
                if sigma[i] == sigma1:
                    sigma[i + 1] = sigma2
                else:
                    sigma[i + 1] = sigma1
            else:
                sigma[i + 1] = sigma[i]
            S[i + 1] = S[i] * math.exp(
                (r - (sigma[i + 1] ** 2) / 2) * delta_t + sigma[i + 1] * math.sqrt(delta_t) * np.random.randn())
            if i % 4 == 0: # si on veut rebalancer une fois sur deux
                A[i + 1] = delta(t[i + 1], S[i + 1], K, T, r, sigma[i + 1])
            else:
                A[i + 1] = A[i]
            # A[i+1]=delta(t[i+1],S[i+1],K,T,r,sigma[i+1])
            B[i + 1] = (A[i] - A[i + 1]) * S[i + 1] + B[i] * (1 + r * delta_t)
            P[i + 1] = A[i + 1] * S[i + 1] + B[i + 1]
            V[i + 1] = call_black_scholes(t[i + 1], S[i + 1], K, T, r, sigma[i + 1])
            P_actu[i + 1] = P[i + 1] - (P0 - V0) * math.exp(r * t[i + 1])
        PL[j] = V[N] - P_actu[N]
    plt.plot(sigma)
    plt.title("Vérification du modèle 2")
    plt.show()
    plt.plot(V)
    plt.plot(P_actu)
    plt.title("Couverture Delta du portefeuille (Modèle 2)")
    plt.show()
    #plt.plot(A)
    #plt.plot(B)
    #plt.title("Affichage de A et B")
    #plt.legend()
    #plt.show()
    plt.plot(P_actu)
    plt.title("Portefeuille actualisé")
    plt.show()
    plt.plot(P_actu - V)
    plt.title("Erreur du modèle 2")
    plt.show()
    return (PL)


def repartition_densite_pnl_sigma_variable_modele_1():
    N_x = 100
    x = np.zeros(N_x)
    repartition = np.zeros(N_x)
    a = 0.3
    Nmc = 1000
    ProfitAndLoss = pnl_modele_1(Nmc)
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
    plt.title("Fonction de répartition du P&L pour le modèle 1")
    plt.show()
    plt.figure()
    plt.plot(x, densite)
    plt.title("Fonction de densité du P&L pour le modèle 1")
    plt.show()


def repartition_densite_pnl_sigma_variable_modele_2():
    N_x = 100
    x = np.zeros(N_x)
    repartition = np.zeros(N_x)
    a = 0.3
    Nmc = 1000
    ProfitAndLoss = pnl_modele_2(Nmc)
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
    plt.title("Fonction de répartition du P&L pour le modèle 2")
    plt.show()
    plt.figure()
    plt.plot(x, densite)
    plt.title("Fonction de densité du P&L pour le modèle 2")
    plt.show()


pnl_modele_1(10000);
pnl_modele_2(10000);
repartition_densite_pnl_sigma_variable_modele_1();
repartition_densite_pnl_sigma_variable_modele_2();