import matplotlib.pyplot as plt
import numpy as np

fichier = open("liste_rho.txt")
lignes = fichier.readlines()
lignes.pop(0)
fichier.close()

L = []
for i in lignes:
    L.append(i.split(" "))

for i in L:
    i[1] = i[1][:-1]

profondeur = []
rho = []

for i in L:
    profondeur.append(float(i[0]))
    rho.append(float(i[1]))

plt.plot(np.array(profondeur), np.array(rho), '+')
plt.xlabel("profondeur en m")
plt.ylabel("rho en kg/m3")
plt.show()