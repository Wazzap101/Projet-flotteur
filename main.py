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


plt.plot(np.array(profondeur), np.array(rho), '+', label = 'expérimentale')
plt.plot(x, y, label = 'interpolation')
plt.title('rho = f(z)')
plt.xlabel("profondeur en m")
plt.ylabel("rho en kg/m3")
plt.grid()
plt.legend()
plt.show()

