import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate

fichier = open("liste_rho.txt")
lignes = fichier.readlines()
lignes.pop(0)
fichier.close()

L = []
for i in lignes:
    L.append(i.split(" "))

profondeur = []
rho = []

for i in L:
    i[1] = i[1][:-1]
    profondeur.append(float(i[0]))
    rho.append(float(i[1]))


f = interpolate.interp1d(profondeur, rho, kind = 'quadratic')
x = np.linspace(min(profondeur), max(profondeur), 50)
y = f(x)

def tracer_rho():
    plt.plot(np.array(profondeur), np.array(rho), '+', label = 'expérimentale')
    plt.plot(x, y, label = 'interpolation')
    plt.title('rho = f(z)')
    plt.xlabel("profondeur en m")
    plt.ylabel("rho en kg/m3")
    plt.grid()
    plt.legend()
    plt.show()

#tracer_rho()

mf = 20 #en kg
V = 19.8e-3 #en m3
D = 11.3e-2 #en m
S = np.pi*(D/2)**2 #en m3
C =1
N=1000
h = 0.05
t = np.linspace(0, 800, N+1)
g = 9.8 #m/s-2


def g(X):
    depth = -X[0]
    if depth.any() < min(profondeur) or depth.any() > max(profondeur):
        # Gestion des cas où la profondeur est en dehors de la plage des données
        # Vous pouvez ajuster cela selon vos besoins
        if depth.any() < min(profondeur):
            depth = min(profondeur)
        else:
            depth = max(profondeur)
        rho_interp = f(depth)
    else:
        rho_interp = f(depth)
    ma = 0.5#rho_interp*4*np.pi*(D/2)**3 /3
    m = ma+mf
    return np.array([X[1], (-rho_interp*V/m + 1)*9.8 - rho_interp*S*C*X[1]*abs(X[1])/(2*m)])


y = np.zeros(N+1)
X = np.zeros((N+1, 2))
v = np.zeros(N+1)

X[0] = [x[0]+0.5, 0]
y[0] = X[0][0]
v[0] = X[0][1]
def euler():
    for i in range(N):
        X[i+1] = X[i] + h*g(X[i])
        y[i+1] = X[i+1][0]
        v[i+1] = X[i+1][1]
    print(v[-1])
    plt.plot(t, np.array(y), '+', label='Profondeur')
    plt.plot(t, v, label='vitesse')
    plt.xlabel("temps")
    plt.title("z = f(t)")
    plt.grid()
    plt.legend()
    plt.show()

euler()

