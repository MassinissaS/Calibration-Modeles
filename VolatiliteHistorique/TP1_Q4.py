import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

def regression(T,K,C,P,r):
    c=0
    S_=[]
    T_=[]
    S=C[0]-P[0]+math.exp(-r[0]*T[0])*K[0]

    for i in range(0, len(T)-1):
        if T[i]==T[i+1]:
            c+=1
            S+=C[i+1]-P[i+1]+math.exp(-r[i+1]*T[i+1])*K[i+1]
        else :
            S_.append(S/(c+1))
            T_.append(T[i])
            c = 0
            S = C[i+1]-P[i+1]+math.exp(-r[i+1]*T[i+1])*K[i+1]
    S_.append(S/(c+1))
    T_.append(T[-1])

    for i in range(len(S_)) :
        S_[i] = math.log(S_[i])
    X = np.array(T_).reshape(-1,1)
    Y = np.array(S_)
    model= LinearRegression()
    model.fit(X, Y)
    Beta1 = model.coef_[0]
    Beta2 = model.intercept_
    plt.scatter(X, Y)
    plt.plot(X, Beta1*X+Beta2, color='red')
    plt.xlabel("Tj")
    plt.ylabel("Ln(Sj)")
    plt.title("Logarithme de Sj en fonction de Tj")
    plt.show()
    return -Beta1, math.exp(Beta2)

link = "sp-index.txt"
fichier = np.loadtxt(link)

n=len(fichier)
P=np.zeros(n)
C=np.zeros(n)
K=np.zeros(n)
T=np.zeros(n)
r=np.zeros(n)

for i in range(n):
    Pa=fichier[i][4]
    Pb=fichier[i][5]
    P[i]=(Pa+Pb)/2
    Ca = fichier[i][2]
    Cb = fichier[i][3]
    C[i] = (Ca + Cb) / 2
    K[i] = fichier[i][1]
    T[i] = fichier[i][0]
    r[i] = fichier[i][6] / 100

print(regression(T,K,C,P,r));


