import numpy as np
import pandas as pd
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
    return call_black_scholes(t, S, K, T, r, sigma) - marche

def tracer_volatilite_implicite_call_3D_fichier_CAC40():
    # Chargement et préparation des données CAC40
    document = pd.read_csv("spx_quotedata.csv", delimiter=",")
    q = 0.0217
    S_0 = 3932.69
    document["S0"] = S_0
    document["Marche"] = (document.iloc[:, 4] + document.iloc[:, 5]) / 2
    document["Time"] = 0.0

    # Configuration du temps jusqu'à l'échéance pour chaque option
    TIndex = document.drop_duplicates(subset=['Expiration Date']).index.tolist()
    TIndex.append(9677)
    for i in range(1, len(TIndex)):
        for j in range(TIndex[i-1], TIndex[i]):
            document.at[j, "Time"] = i / 365


    # Nouvelles colonnes df et algorithme de Newton
    document["r"] = 0.0255
    document["Sigma"] = 0.0
    document["Strike"] = document.iloc[:, 11]
    t = 0.0

    initial_sigma = np.sqrt(2 * np.abs(np.log(document["S0"] / document["Strike"]) + document["r"] * document["Time"]) / document["Time"])
    document["Sigma"] = initial_sigma

    for i in range(len(document)):
        if np.max(document.at[i, "S0"] - document.at[i, "Strike"] * np.exp(-document.at[i, "r"] * document.at[i, "Time"]), 0) < document.at[i, "Marche"] < document.at[i, "S0"]:
            while np.abs(F_call(document.at[i, "Marche"], t, document.at[i, "S0"], document.at[i, "Strike"],
                                document.at[i, "Time"], document.at[i, "r"], document.at[i, "Sigma"])) > 0.0001:
                document.at[i, "Sigma"] -= F_call(document.at[i, "Marche"], t, document.at[i, "S0"],
                                                  document.at[i, "Strike"], document.at[i, "Time"], document.at[i, "r"],
                                                  document.at[i, "Sigma"]) / vega_black_scholes(
                    t, document.at[i, "S0"], document.at[i, "Strike"], document.at[i, "Time"], document.at[i, "r"], document.at[i, "Sigma"])
        else:
            document.at[i, "Sigma"] = np.nan

    # Suppression des valeurs d'arbitrage et visualisation des "smiles"
    document = document.dropna(subset=["Sigma"])

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(document["Time"], document["Strike"], document["Sigma"], c='blue', marker='o')
    ax.view_init(10, 50)
    ax.set_xlabel('Temps')
    ax.set_ylabel('Strike (en $)')
    ax.set_zlabel('Volatilité Implicite (en % par an)')
    ax.set_title("Smile de l'option Call en 3D (spx_quotedata.csv)")
    plt.show()

tracer_volatilite_implicite_call_3D_fichier_CAC40();