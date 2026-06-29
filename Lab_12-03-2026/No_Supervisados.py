import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
from math import ceil, floor, pi
from seaborn import color_palette

def centroide(num_cluster, datos, clusters):
    ind = clusters == num_cluster
    return(pd.DataFrame(datos[ind].mean()).T)

def recodificar(col, nuevo_codigo):
    col_cod = pd.Series(col, copy=True)
    for llave, valor in nuevo_codigo.items():
        col_cod.replace(llave, valor, inplace=True)
    return col_cod

def bar_plot(centros, labels, scale = False,cluster = None, var = None):
    fig, ax = plt.subplots(1,1, figsize = (15,8), dpi = 200)
    
    centros = np.copy(centros)
    
    if scale:
        for col in range(centros.shape[1]):
            centros[:,col] = centros[:,col] / max(centros[:,col])
    
    colores = color_palette()
    minimo = floor(centros.min()) if floor(centros.min()) < 0 else 0
    def inside_plot(valores, labels, titulo):
        plt.barh(range(len(valores)), valores, 1/1.5, color = colores)
        plt.xlim(minimo, ceil(centros.max()))
        plt.title(titulo)
    if var is not None:
        centros = np.array([n[[x in var for x in labels]] for n in centros])
        colores = [colores[x % len(colores)] for x, i in enumerate(labels) if i in var]
        labels = labels[[x in var for x in labels]]
    if cluster is None:
        for i in range(centros.shape[0]):
            plt.subplot(1, centros.shape[0], i + 1)
            inside_plot(centros[i].tolist(), labels, ('Cluster ' + str(i)))
            plt.yticks(range(len(labels)), labels) if i == 0 else plt.yticks([]) 
    else:
        pos = 1
        for i in cluster:
            plt.subplot(1, len(cluster), pos)
            inside_plot(centros[i].tolist(), labels, ('Cluster ' + str(i)))
            plt.yticks(range(len(labels)), labels) if pos == 1 else plt.yticks([]) 
            pos += 1
            
def bar_plot_detail(centros,columns_names = [], columns_to_plot = [],figsize = (10,7),dpi = 150):
  fig, ax = plt.subplots(1,1, figsize = (15,8), dpi = 200)
  numClusters = centros.shape[0]
  labels = ["Cluster "+ str(i) for i in range(numClusters)]
  centros = pd.DataFrame(centros,columns=columns_names,index= labels)
  
  plots = len(columns_to_plot) if len(columns_to_plot) != 0 else len(columns_names)
  rows, cols = ceil(plots/2),2
  
  plt.figure(1, figsize = figsize,dpi = dpi)
  plt.subplots_adjust(hspace=1,wspace = 0.5)
  columns = columns_names
  if len(columns_to_plot) > 0: 
    if type(columns_to_plot[0]) is str:
      columns = columns_to_plot
    else:
      columns = [columns_names[i] for i in columns_to_plot]
  var = 0
  for numRow in range(rows):
    for numCol in range(cols):
      if var < plots:
        ax = plt.subplot2grid((rows, cols), (numRow, numCol), colspan=1, rowspan=1)
        sb.barplot(y = labels, x=columns[var] ,data=centros ,ax=ax)
        var += 1    
              
def radar_plot(centros, labels):
    
    fig, ax = plt.subplots(1,1, figsize = (15,8), dpi = 200)
    centros = np.array([((n - min(n)) / (max(n) - min(n)) * 100) if 
                        max(n) != min(n) else (n/n * 50) for n in centros.T])
    angulos = [n / float(len(labels)) * 2 * pi for n in range(len(labels))]
    angulos += angulos[:1]
    ax = plt.subplot(111, polar = True)
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)
    
    plt.xticks(angulos[:-1], labels)
    ax.set_rlabel_position(0)
    plt.yticks([10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
           ["10%", "20%", "30%", "40%", "50%", "60%", "70%", "80%", "90%", "100%"], 
           color = "grey", size = 8)
    plt.ylim(-10, 100)
    for i in range(centros.shape[1]):
        valores = centros[:, i].tolist()
        valores += valores[:1]
        ax.plot(angulos, valores, linewidth = 1, linestyle = 'solid', 
                label = 'Cluster ' + str(i))
        ax.fill(angulos, valores, alpha = 0.3)
    plt.legend(loc='upper right', bbox_to_anchor = (0.1, 0.1))
