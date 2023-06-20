import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

import pandas as pd
from tqdm import tqdm

from sklearn.decomposition import PCA

#--------------------------------
# ARCHIVOS
#----------
filename = 'circulo.csv'
#filename = 'te.csv'
#filename = 'irisbin.csv'

#--------------------------------
# PARAMETROS DEL MODELO
#-----------------------
SOM_size = [10,10]  # Filas, Columnas
mu = [0.8, 0.1, 0.01]  # tasa de aprendizaje
EPOCAS = [10, 150, 200]

MAX_SIZE = np.max((SOM_size[0],SOM_size[1]))//2


patterns = pd.read_csv(filename, header=None).to_numpy()


#===============================================
def build_SOM(n_rows, n_cols, n_dim):

    idx_grid = np.arange(n_rows*n_cols).reshape(n_rows,n_cols)

    x = np.arange(n_cols)
    y = np.arange(n_rows)
    X,Y= np.meshgrid(x,y)
    pos_grid = np.vstack((X.flatten(),Y.flatten())).T

    codebook = np.random.rand(n_rows*n_cols, n_dim)

    return (idx_grid, pos_grid, codebook)
#===============================================


#===============================================
def distancia(pattern, codebook, p=2):
    '''
    Normas: [1:cityblock | 2:euclidea | "p"]
    '''

    if p > 1:
        d = np.sum((codebook-pattern)**p, axis=1)**(1/p)
    else:
        d = np.sum(np.abs(codebook-pattern), axis=1)

    idx_winner = d.argmin()

    return idx_winner
#==============================================


#==============================================
def get_idx_winner(winner, n_rows, n_cols):
    idxr = winner // n_rows
    idxc = winner % n_cols
    return (idxr, idxc)
#==============================================



#==============================================
def get_neighbours(pos_grid, idx_winner, nb_size):

    output = np.sum(np.abs(pos_grid-pos_grid[idx_winner,:]), axis=1)

    neighborhood = output <= nb_size

    return neighborhood
#==============================================


# REDUZCO DIMENSIONES
if patterns.shape[1] > 2:
    pca = PCA(n_components=2)
    pca.fit(patterns)
    X = pca.transform(patterns)
else:
    X = patterns.copy()


# INICIALIZO SOM
idx_grid, pos_grid, codebook = build_SOM(SOM_size[0], SOM_size[1], patterns.shape[1])

plt.ion()

ENERGIA = 1E6

for epoca in tqdm(range(EPOCAS[-1])):

    #==============================================================================
    # ORDENAMIENTO TOPOLOGICO
    if (epoca <= EPOCAS[0]):
        eta = mu[0]
        nb_size = MAX_SIZE

    # TRANSICION
    elif (epoca > EPOCAS[0]) and (epoca <= EPOCAS[1]):
        eta = ( ( mu[1] - mu[0] ) / (EPOCAS[1] - EPOCAS[0]) ) * epoca + mu[0]
        nb_size = ( ( 1 - MAX_SIZE ) / (EPOCAS[1] - EPOCAS[0]) ) * epoca + MAX_SIZE

    # CONVERGENCIA
    else:
        eta = mu[2]
        nb_size = 0
    #==============================================================================

    # ALEATORIZO PATRONES
    idxs = np.arange(len(patterns))
    np.random.shuffle(idxs)
    patterns = patterns[idxs,:]

    for pattern in patterns:

        # CALCULO LAS DISTANCIAS DE CADA PATRON A CADA NEURONA Y DETERMINO GANADORA
        idx_winner = distancia(pattern, codebook)

        # OBTENGO FILA Y COLUMNA DE LA MALLA
        idxr, idxc = get_idx_winner(idx_winner, SOM_size[0], SOM_size[1])

        # IDENTIFICO GANADORA Y VECINDAD
        neighborhood = get_neighbours(pos_grid, idx_winner, nb_size)

        # ACTUALIZO GANADORA Y VECINDAD
        codebook[neighborhood,:] += (eta * (pattern - codebook[neighborhood,:]))


    # REDUCIR CODEBOOK DE DIMENSIONES (VER EN EL ESPACIO REDUCIDO DE LOS DATOS)
    if patterns.shape[1] > 2:
        Xc = pca.transform(codebook)
    else:
        Xc = codebook.copy()

    # RECONSTRUIR MALLA
    SOM_grid_x = Xc[:,0].reshape(SOM_size[0],SOM_size[1])
    SOM_grid_y = Xc[:,1].reshape(SOM_size[0],SOM_size[1])

    ############################
    # GRAFICAR DATOS Y MALLA
    ############################
    DELTA_E = np.abs(ENERGIA - (codebook**2).flatten().sum())

    plt.cla()
    plt.scatter(X[:,0], X[:,1], 30, 'gray', alpha=0.5)  # GRAFICO LOS DATOS
    segs1 = np.stack((SOM_grid_x,SOM_grid_y), axis=2)
    segs2 = segs1.transpose(1,0,2)
    plt.gca().add_collection(LineCollection(segs1))
    plt.gca().add_collection(LineCollection(segs2))
    plt.scatter(Xc[:,0], Xc[:,1], 30, 'r', marker='o', alpha=1)  # GRAFICO LOS DATOS
    plt.xticks([])
    plt.yticks([])
    plt.title(f'Epoca: {epoca}/{EPOCAS[-1]} - DELTA ENERGIA: {DELTA_E:.4}')
    plt.draw()
    plt.pause(0.05)

    ENERGIA = (codebook**2).flatten().sum()

plt.waitforbuttonpress(0)


#DISTANCES = np.zeros_like((2*SOM_size[0])-1, (2*SOM_size[1])-1)

# PARA LA POSICION DE LA NEURONA SE CALCULAN LOS PROMEDIOS DE LOS VECINOS, Y LA DISTANCIA ENTRE PARES PARA CADA POSIBLE CONEXION
# >>>>>>>>>>> https://stackoverflow.com/questions/13631673/how-do-i-make-a-u-matrix


#for idx_winner in tqdm(range(SOM_size[0]*SOM_size[1])):
    #nb = get_neighbours(pos_grid, idx_winner, 1)
    #d = distancia(codebook[idx_winner,:], codebook[nb,:], p=2)
    #r,c = get_idx_winner(idx_winner, SOM_size[0], SOM_size[1])
    #DISTANCES[r,c] = d.mean()

#plt.ioff()
#plt.figure()
#plt.imshow(DISTANCES)
#plt.colorbar()
#plt.grid(True)
#plt.show()
