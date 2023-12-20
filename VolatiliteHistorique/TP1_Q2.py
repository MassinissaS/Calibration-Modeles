import numpy as np
import matplotlib.pyplot as plt
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
        f = S * repartition(d1(t, S, K, T, r, sigma)) - K * math.exp(-r * (T - t)) * repartition(d2(t, S, K, T, r, sigma))
    return f


def vega_black_scholes(t, S, K, T, r, sigma):
    f = (S * math.sqrt(T - t) * math.exp(-(d1(t, S, K, T, r, sigma)) ** 2 / 2)) / math.sqrt(2 * np.pi)
    return f


def F_call(marche, t, S, K, T, r, sigma):
    f = call_black_scholes(t, S, K, T, r, sigma) - marche
    return f

def put_black_scholes(t, S, K, T, r, sigma):
    if t == T:
        f = max(K - S, 0)
    else:
        f = -S * repartition(-d1(t, S, K, T, r, sigma)) + K * math.exp(-r * (T - t)) * repartition(-d2(t, S, K, T, r, sigma))
    return f

def F_Put(marche, t, S, K, T, r, sigma):
    f = put_black_scholes(t, S, K, T, r, sigma) - marche
    return f


def tracer_volatilite_implicite_call_3D_fichiertxt():
    link = "sp-index.txt"
    fichier = np.loadtxt(link)
    q = 0.0217
    eps = 0.0001
    t = 0
    S_0 = 1260.36
    N = len(fichier)
    print(N)
    K = np.zeros(N)
    r = np.zeros(N)
    T = np.zeros(N)
    S0 = np.zeros(N)
    M = np.zeros(N)
    sigma = np.zeros(N)
    volatilite_implicite = np.zeros(N)

    for i in range(N):
        Ca = fichier[i][3]
        Cb = fichier[i][2]
        M[i] = (Ca + Cb) / 2
        K[i] = fichier[i][1]
        T[i] = fichier[i][0]
        r[i] = fichier[i][6] * 0.01
        S0[i] = S_0 * math.exp(-q * T[i])
        sigma[i] = (math.sqrt(2 * np.abs((math.log(S0[i] / K[0]) + r[i] * T[i]) / T[i])))

        if M[i] < S0[i] and M[i] >= max(S0[i] - K[i] * math.exp(-r[i] * T[i]), 0):
            while (np.abs(F_call(M[i], t, S0[i], K[i], T[i], r[i], sigma[i])) > eps):
                sigma[i] = sigma[i] - F_call(M[i], t, S0[i], K[i], T[i], r[i], sigma[i]) / vega_black_scholes(t, S0[i], K[i], T[i], r[i], sigma[i])
                volatilite_implicite[i] = sigma[i]
        else:
            volatilite_implicite[i] = (0)
    volat_nules = np.where(volatilite_implicite == 0)
    K = np.delete(K, volat_nules)
    T = np.delete(T, volat_nules)
    volatilite_implicite = np.delete(volatilite_implicite, volat_nules)
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')  # Affichage de points en 3d
    plt.plot(K, T, volatilite_implicite, '*')
    ax.set_xlabel('Strike (en $)')
    ax.set_ylabel('Temps')
    ax.set_zlabel('Volatilité Implicite (en % par an)')
    plt.title("Smile de l'option Call en 3D (sp-index.txt)")
    plt.show()


def tracer_volatilite_implicite_put_3D_fichiertxt():
    link = "sp-index.txt"
    fichier = np.loadtxt(link)
    q = 0.0217
    eps = 0.0001
    t = 0
    S_0 = 1260.36
    N = len(fichier)
    print(N)
    K = np.zeros(N)
    r = np.zeros(N)
    T = np.zeros(N)
    S0 = np.zeros(N)
    M = np.zeros(N)
    sigma = np.zeros(N)
    volatilite_implicite = np.zeros(N)
    for i in range(N):
        Ca = fichier[i][4]
        Cb = fichier[i][5]
        M[i] = (Ca + Cb) / 2
        K[i] = fichier[i][1]
        T[i] = fichier[i][0]
        r[i] = fichier[i][6] * 0.01
        S0[i] = S_0 * math.exp(-q * T[i])
        sigma[i] = (math.sqrt(2 * np.abs((math.log(S0[i] / K[0]) + r[i] * T[i]) / T[i])))

        if (M[i] < K[i] * math.exp(-r[i] * (T[i] - t))) and (M[i] >= max(S0[i] - K[i] * math.exp(-r[i] * T[i]), 0) - S0[i] + K[i] * math.exp(-r[i] * (T[i] - t))):
            while (np.abs(F_Put(M[i], t, S0[i], K[i], T[i], r[i], sigma[i])) > eps):
                sigma[i] = sigma[i] - F_Put(M[i], t, S0[i], K[i], T[i], r[i], sigma[i]) / vega_black_scholes(t, S0[i], K[i], T[i], r[i], sigma[i])
                volatilite_implicite[i] = sigma[i]
        else:
            volatilite_implicite[i] = (0)

    volat_nules = np.where(volatilite_implicite == 0)
    K = np.delete(K, volat_nules)
    T = np.delete(T, volat_nules)
    volatilite_implicite = np.delete(volatilite_implicite, volat_nules)
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    plt.plot(K, T, volatilite_implicite, '*')
    ax.set_xlabel('Strike (en $)')
    ax.set_ylabel('Temps')
    ax.set_zlabel('Volatilité Implicite (en % par an)')
    plt.title("Smile de l'option Put en 3D (sp-index.txt)")
    plt.show()


def tracer_call_black_scholes():
    S = np.zeros(100)
    call_test = np.zeros(100)
    for i in range(1, 100):
        S[i] = 0.2 * i;
        call_test[i] = call_black_scholes(0, S[i], 10, 1, 0.1, 0.5)
    plt.plot(S, call_test, '*')
    plt.xlabel('XXX')
    plt.ylabel('YYY')
    plt.title("Courbe Call")
    plt.show()


def tracer_put_black_scholes():
    S = np.zeros(100)
    put_test = np.zeros(100)
    for i in range(1, 100):
        S[i] = 0.2 * i;
        put_test[i] = put_black_scholes(0, S[i], 10, 1, 0.1, 0.5)
    plt.plot(S, put_test, '*')
    plt.xlabel('XXX')
    plt.ylabel('YYY')
    plt.title("Courbe Put")
    plt.show()

def tracer_vega_black_scholes():
    S = np.zeros(100)
    vega_test = np.zeros(100)
    for i in range(1, 100):
        S[i] = 0.2 * i;
        vega_test[i] = vega_black_scholes(0, S[i], 10, 1, 0.1, 0.5)
    plt.plot(S, vega_test, '*')
    plt.xlabel('XXX')
    plt.ylabel('YYY')
    plt.title("Courbe Vega")
    plt.show()

tracer_volatilite_implicite_call_3D_fichiertxt();
tracer_volatilite_implicite_put_3D_fichiertxt();

#tracer_vega_black_scholes();
#tracer_call_black_scholes();
#tracer_put_black_scholes();