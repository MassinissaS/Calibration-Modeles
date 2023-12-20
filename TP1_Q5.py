import numpy as np
import matplotlib.pyplot as plt
import math
import xlrd


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
        f = S * repartition(d1(t, S, K, T, r, sigma)) - K * math.exp(-r * (T - t)) * repartition(d2(t, S, K, T, r, sigma))
    return f


def vega_black_scholes(t, S, K, T, r, sigma):
    f = (S * math.sqrt(T - t) * math.exp(-(d1(t, S, K, T, r, sigma)) ** 2 / 2)) / math.sqrt(2 * np.pi)
    return f


def put_black_scholes(t, S, K, T, r, sigma):
    if t == T:
        f = max(K - S, 0)
    else:
        f = -S * repartition(-d1(t, S, K, T, r, sigma)) + K * math.exp(-r * (T - t)) * repartition(-d2(t, S, K, T, r, sigma))
    return f


def F(marche, t, S, K, T, r, sigma):
    f = call_black_scholes(t, S, K, T, r, sigma) - marche
    return f


def F_Put(marche, t, S, K, T, r, sigma):
    f = put_black_scholes(t, S, K, T, r, sigma) - marche
    return f


def tracer_volatilite_implicite_call_2D_fichier_Google():
    document = xlrd.open_workbook('GoogleOrig2.xlsx')
    feuille_1 = document.sheet_by_index(0)
    feuille_1 = document.sheet_by_name('Orig')
    cols = feuille_1.ncols
    N = feuille_1.nrows
    q = 0.0217
    eps = 0.0001
    t = 0
    K = []
    r = np.zeros(N)
    T = []
    S0 = np.zeros(N)
    M = [];
    sigma = np.zeros(N)
    volatilite_implicite = np.zeros(N)
    for i in range(0, N):
        r[i] = 0.06
        S0[i] = 591.66
        T += [feuille_1.cell_value(rowx=i, colx=0)]
        K += [feuille_1.cell_value(rowx=i, colx=1)]
        M += [feuille_1.cell_value(rowx=i, colx=2)]
        sigma[i] = (math.sqrt(2 * np.abs((math.log(S0[i] / K[0]) + r[i] * T[i]) / T[i])))
        if M[i] < S0[i] and M[i] >= max(S0[i] - K[i] * math.exp(-r[i] * T[i]), 0):
            while (np.abs(F(M[i], t, S0[i], K[i], T[i], r[i], sigma[i])) > eps):
                sigma[i] = sigma[i] - F(M[i], t, S0[i], K[i], T[i], r[i], sigma[i]) / vega_black_scholes(t, S0[i], K[i], T[i], r[i], sigma[i])
                volatilite_implicite[i] = sigma[i]
        else:
            volatilite_implicite[i] = (0)
    volat_nules = np.where(volatilite_implicite == 0)
    K = np.delete(K, volat_nules)
    T = np.delete(T, volat_nules)
    volatilite_implicite = np.delete(volatilite_implicite, volat_nules)
    plt.plot(K, volatilite_implicite, '*')
    plt.xlabel('Strike (en $)')
    plt.ylabel('Volatilité Implicite (en % par an)')
    plt.title("Smile de l'option Call en 2D (GoogleOrig.xlsx)")
    plt.show()


def tracer_volatilite_implicite_call_3D_fichier_Google():
    document = xlrd.open_workbook('GoogleOrig2.xlsx')
    feuille_1 = document.sheet_by_index(0)
    feuille_1 = document.sheet_by_name('Orig')
    cols = feuille_1.ncols
    N = feuille_1.nrows
    q = 0.0217
    eps = 0.0001
    t = 0
    K = []
    r = np.zeros(N)
    T = []
    S0 = np.zeros(N)
    M = [];
    sigma = np.zeros(N)
    volatilite_implicite = np.zeros(N)
    for i in range(0, N):
        r[i] = 0.06
        S0[i] = 591.66
        T += [feuille_1.cell_value(rowx=i, colx=0)]
        K += [feuille_1.cell_value(rowx=i, colx=1)]
        M += [feuille_1.cell_value(rowx=i, colx=2)]
        sigma[i] = (math.sqrt(2 * np.abs((math.log(S0[i] / K[0]) + r[i] * T[i]) / T[i])))
        if M[i] < S0[i] and M[i] >= max(S0[i] - K[i] * math.exp(-r[i] * T[i]), 0):
            while (np.abs(F(M[i], t, S0[i], K[i], T[i], r[i], sigma[i])) > eps):
                sigma[i] = sigma[i] - F(M[i], t, S0[i], K[i], T[i], r[i], sigma[i]) / vega_black_scholes(t, S0[i], K[i], T[i], r[i], sigma[i])
                volatilite_implicite[i] = sigma[i]
        else:
            volatilite_implicite[i] = (0)
    volat_nules = np.where(volatilite_implicite == 0)
    K = np.delete(K, volat_nules)
    T = np.delete(T, volat_nules)
    volatilite_implicite = np.delete(volatilite_implicite, volat_nules)
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')  # Affichage de points en 3d
    ax.set_xlabel('Strike (en $)')
    ax.set_ylabel('Temps')
    ax.set_zlabel('Volatilité Implicite (en % par an)')
    plt.plot(K, T, volatilite_implicite, '*')
    plt.title("Smile de l'option Call en 3D (GoogleOrig.xlsx)")
    plt.show()


tracer_volatilite_implicite_call_2D_fichier_Google();
tracer_volatilite_implicite_call_3D_fichier_Google();