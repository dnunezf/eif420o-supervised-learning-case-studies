#Paquete ver1.0: Analisis de Datos Exploratorios (EDA)


# Seccion de las librerias que se estaran utilizando:
import numpy as np
import pandas as pd
import seaborn as sns
import math
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from prince import PCA as PCA_Prince
import umap as um
from sklearn_extra.cluster import KMedoids
from seaborn import color_palette
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, ward, single, complete,average,linkage, fcluster
from math import ceil, floor, pi
import umap.umap_ as umap
from sklearn.manifold import TSNE
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix



# Clase EDA:
class analisisEDA():
# Aqui inicio y cargo el Dataset desde archivo CSV.
    def __init__(self,path, num):
        self.__df = self.__datosCargados(path, num)
    
    @property
    def df(self):
        return self.__df
    
    @df.setter
    def df(self, p_df):
        self.__df = p_df

# Cargar el dataset
    def __datosCargados(self, path, num):
        if num == 1:
            return pd.read_csv(path,
            sep = ",",
            decimal = ".",
            index_col = 0)
        if num == 2:
            return pd.read_csv(path,
            sep = ";",
            decimal = ".")
        
# Mostrar el tipo de datos
    def tipoDatos(self):
        print(self.__df.dtypes)

# Filtra las columnas num
    def analisisNumerico(self):
        self.__df = self.__df.select_dtypes(include=["number"])

# Cambiar variables categoricas a dummies
    def analisisCompleto(self):
        columnas_categoricas = self.__df.select_dtypes(include=["object", "category"]).columns.tolist()
        print(f"Columnas categóricas convertidas a dummies: {columnas_categoricas}")
        self.__df = pd.get_dummies(self.__df, columns=columnas_categoricas, drop_first=True).astype(int)


# Eliminar las columnas que no sirven
    def eliminarColumnas(self, columnas):
        self.__df.drop(columns=columnas, inplace=True)

# Cambiar el nombre de las columnas
    def renombrarColumnas(self, nuevos_nombres):
        self.__df.rename(columns=nuevos_nombres, inplace=True)

# Valores unicos
    def valores_unicos(self, v):
        unique_values = self.__df[v].unique()
        print("Valores unicos en", v,":")
        for value in unique_values:
            count = (self.__df[v] == value).sum()
            print(f"{value}: {count}")

# Valores Faltantes
    def valores_faltantes(self):
        missing_values = self.__df.isna().sum()
        print("Missing values by column:")
        print(missing_values)
        print('\n')

# Eliminar los datos duplicados
    def eliminarDuplicados(self):
        antes = self.__df.shape[0]
        self.__df.drop_duplicates(inplace=True)
        despues = self.__df.shape[0]
        print(f"Se eliminaron {antes - despues} filas duplicadas. Total actual: {despues} filas.")

# Eliminar los datos Nulos
    def eliminarNulos(self):
        nulos_antes = self.__df.isnull().sum().sum()
        filas_antes = self.__df.shape[0]
        print(f"Valores nulos totales antes de eliminar: {nulos_antes}")
        self.__df.dropna(inplace=True)
        nulos_despues = self.__df.isnull().sum().sum()
        filas_despues = self.__df.shape[0]
        print(f"Filas eliminadas por contener nulos: {filas_antes - filas_despues}")
        print(f"Valores nulos restantes: {nulos_despues}")

# Analisis de los datos del dataset
    def analisis(self):
        print("Dimensiones:", self.__df.shape)
        print(self.__df.head())
        print("="*40)
        print("Estadisticas Descriptivas Generales")
        print("="*40)
        print("Media:\n", self.__df.mean(numeric_only=True))
        print("="*40)
        print("Mediana:\n", self.__df.median(numeric_only=True))
        print("="*40)
        print("Desviación estándar:\n", self.__df.std(numeric_only=True))
        print("="*40)
        print("Máximos:\n", self.__df.max(numeric_only=True))
        print("="*40)
        print("Mínimos:\n", self.__df.min(numeric_only=True))
        print("="*40)
        print("Cuantiles:\n", self.__df.quantile([0, 0.25, 0.5, 0.75, 1], numeric_only=True))

# Boxplot
    def graficoBoxplot(self):
        columnas_numericas = self.__df.select_dtypes(include='number').columns
        n = len(columnas_numericas)
        columnas = 3
        filas = math.ceil(n / columnas)
        fig, axes = plt.subplots(filas, columnas, figsize=(5 * columnas, 4 * filas), dpi=150)
        axes = axes.flatten()
        colores = sns.color_palette("Set3", n)
        for i, col in enumerate(columnas_numericas):
            sns.boxplot(y=self.__df[col], ax=axes[i], color=colores[i])
            axes[i].set_title(f"Boxplot de {col}", fontsize=10)
            axes[i].set_ylabel(col)
            axes[i].grid(True, linestyle='--', alpha=0.5)
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])
        plt.tight_layout()
        plt.show()

# Histograma (sin la clase predictoria)
    def histogramas(self):
        columnas_numericas = self.__df.select_dtypes(include='number').columns
        n = len(columnas_numericas)
        columnas = 3
        filas = math.ceil(n / columnas)
        fig, axes = plt.subplots(filas, columnas, figsize=(5 * columnas, 4 * filas), dpi=150)
        axes = axes.flatten()  # Para acceder como una lista plana
        colores = sns.color_palette("Set2", n)
        for i, col in enumerate(columnas_numericas):
            ax = axes[i]
            ax.hist(self.__df[col], bins=30, color=colores[i], edgecolor='black', alpha=0.7)
            ax.set_title(f"Histograma de {col}", fontsize=10)
            ax.set_xlabel(col)
            ax.set_ylabel("Frecuencia")
            ax.grid(True, linestyle='--', alpha=0.5)
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])
        plt.tight_layout()
        plt.show()

# Gráficos de distribución (histogramas) 
    def distribucionVariables(self):
        columnas_numericas = self.__df.select_dtypes(include='number').columns
        n = len(columnas_numericas)
        columnas = 3
        filas = math.ceil(n / columnas)
        fig, axes = plt.subplots(filas, columnas, figsize=(5 * columnas, 4 * filas), dpi=150)
        axes = axes.flatten()
        colores = sns.color_palette("coolwarm", n)
        for i, col in enumerate(columnas_numericas):
            sns.histplot(self.__df[col], kde=True, ax=axes[i], color=colores[i], bins=30)
            axes[i].set_title(f"Distribución de {col}", fontsize=10)
            axes[i].set_xlabel(col)
            axes[i].set_ylabel("Frecuencia")
            axes[i].grid(True, linestyle='--', alpha=0.5)
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])
        plt.tight_layout()
        plt.show()

# Histograma de la clase predictora
    def histogramaClase(self, columna_objetivo):
        if columna_objetivo in self.__df.columns:
            plt.figure(figsize=(8, 5), dpi=150)
            colores = sns.color_palette("pastel")
            self.__df[columna_objetivo].value_counts().plot(kind='bar', color=colores)
            plt.title(f"Distribución de la Clase: {columna_objetivo}")
            plt.xlabel(columna_objetivo)
            plt.ylabel("Frecuencia")
            plt.grid(axis='y', linestyle='--', alpha=0.5)
            plt.tight_layout()
            plt.show()
        else:
            print(f"La columna '{columna_objetivo}' no existe en el DataFrame.")

# Funcion de la Densidad de los datos
    def datosDensidad(self):
        columnas_numericas = self.__df.select_dtypes(include='number').columns
        n = len(columnas_numericas)  
        columnas = 3
        filas = math.ceil(n / columnas)
        fig, axes = plt.subplots(filas, columnas, figsize=(5 * columnas, 4 * filas), dpi=150)
        axes = axes.flatten()
        colores = sns.color_palette("husl", n)
        for i, col in enumerate(columnas_numericas):
            sns.kdeplot(data=self.__df, x=col, fill=True, ax=axes[i], color=colores[i], linewidth=2)
            axes[i].set_title(f"Densidad de {col}", fontsize=10)
            axes[i].set_xlabel(col)
            axes[i].set_ylabel("Densidad")
            axes[i].grid(True, linestyle='--', alpha=0.5)
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])
        plt.tight_layout()
        plt.show()

# Correlacion
    def correlaciones(self):
        corr = self.__df.corr(numeric_only=True)
        print("Matriz de correlaciones:\n")
        print(corr)

# Grafico de correlacion
    def graficoCorrelacion(self):
        corr = self.__df.corr(numeric_only=True)
        plt.figure(figsize=(12, 8), dpi=150)
        cmap = sns.diverging_palette(240, 10, as_cmap=True).reversed() 
        sns.heatmap(corr, vmin=-1, vmax=1, cmap=cmap, annot=True,fmt=".2f",linewidths=0.5,linecolor='white', square=True, cbar_kws={"shrink": 0.8, "label": "Correlación"},annot_kws={"size": 10, "color": "black"})
        plt.title("Mapa de Calor de Correlaciones", fontsize=16)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()

# Graficos de dispersion
    def graficosDispersion(self):
        columnas_numericas = self.__df.select_dtypes(include='number').columns
        if len(columnas_numericas) >= 2:
            sns.pairplot(self.__df[columnas_numericas])
            plt.suptitle("Gráficos de Dispersión por Pares", y=1.02)
            plt.tight_layout()
            plt.show()
        else:
            print("No hay suficientes variables numéricas para graficar dispersión.")

    def centroide(num_cluster, datos, clusters):
        ind = clusters == num_cluster
        return pd.DataFrame(datos[ind].mean()).T

    def recodificar(col, nuevo_codigo):
        col_cod = pd.Series(col, copy=True)
        for llave, valor in nuevo_codigo.items():
            col_cod.replace(llave, valor, inplace=True)
        return col_cod

# EXTRA SECCION PARA LOS GRAFICOS EN HAC   
    def bar_plot(centros, labels, scale = False,cluster = None, var = None):
        fig, ax = plt.subplots(1,1, figsize = (15,8), dpi = 200)
        centros = np.copy(centros)
        if scale:
            for col in range(centros.shape[1]):
                centros[:,col] = centros[:,col] / max(centros[:,col])
        colores = color_palette("Set2")
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

    def __str__(self):
        return f'AnalisisDatosExploratorio: {self.__df} {self.__path}'


# SECCION ACP BASICA
class ACPmain:
    def __init__(self, datos, n_componentes = 2): 
        self.__datos = datos
        self.__modelo = PCA_Prince(n_components = n_componentes).fit(self.__datos)
        self.__correlacion_var = self.__modelo.column_correlations
        self.__coordenadas_ind = self.__modelo.row_coordinates(self.__datos)
        self.__contribucion_ind = self.__modelo.row_contributions_
        self.__cos2_ind = self.__modelo.row_cosine_similarities(self.__datos)
        self.__var_explicada = self.__modelo.percentage_of_variance_
    @property
    def datos(self):
        return self.__datos
    @datos.setter
    def datos(self, datos):
        self.__datos = datos
    @property
    def modelo(self):
        return self.__modelo
    @property
    def correlacion_var(self):
        return self.__correlacion_var
    @property
    def coordenadas_ind(self):
        return self.__coordenadas_ind
    @property
    def contribucion_ind(self):
        return self.__contribucion_ind
    @property
    def cos2_ind(self):
        return self.__cos2_ind
    @property
    def var_explicada(self):
        return self.__var_explicada
    @var_explicada.setter
    def var_explicada(self, var_explicada):
        self.__var_explicada = var_explicada
    @modelo.setter
    def modelo(self, modelo):
        self.__modelo = modelo
    @correlacion_var.setter
    def correlacion_var(self, correlacion_var):
        self.__correlacion_var = correlacion_var
    @coordenadas_ind.setter
    def coordenadas_ind(self, coordenadas_ind):
        self.__coordenadas_ind = coordenadas_ind
    @contribucion_ind.setter
    def contribucion_ind(self, contribucion_ind):
        self.__contribucion_ind = contribucion_ind
    @cos2_ind.setter
    def cos2_ind(self, cos2_ind):
        self.__cos2_ind = cos2_ind
    def plot_plano_principal(self, ejes = [0, 1], ind_labels = True, titulo = 'Plano Principal'):
        x = self.coordenadas_ind[ejes[0]].values
        y = self.coordenadas_ind[ejes[1]].values
        plt.style.use('ggplot')
        plt.scatter(x, y, color = 'gray')
        plt.title(titulo)
        plt.axhline(y = 0, color = 'dimgrey', linestyle = '--')
        plt.axvline(x = 0, color = 'dimgrey', linestyle = '--')
        inercia_x = round(self.var_explicada[ejes[0]], 2)
        inercia_y = round(self.var_explicada[ejes[1]], 2)
        plt.xlabel('Componente ' + str(ejes[0]) + ' (' + str(inercia_x) + '%)')
        plt.ylabel('Componente ' + str(ejes[1]) + ' (' + str(inercia_y) + '%)')
        if ind_labels:
            for i, txt in enumerate(self.coordenadas_ind.index):
                plt.annotate(txt, (x[i], y[i]))
    def plot_circulo(self, ejes = [0, 1], var_labels = True, titulo = 'Círculo de Correlación'):
        cor = self.correlacion_var.iloc[:, ejes].values
        plt.style.use('ggplot')
        c = plt.Circle((0, 0), radius = 1, color = 'steelblue', fill = False)
        plt.gca().add_patch(c)
        plt.axis('scaled')
        plt.title(titulo)
        plt.axhline(y = 0, color = 'dimgrey', linestyle = '--')
        plt.axvline(x = 0, color = 'dimgrey', linestyle = '--')
        inercia_x = round(self.var_explicada[ejes[0]], 2)
        inercia_y = round(self.var_explicada[ejes[1]], 2)
        plt.xlabel('Componente ' + str(ejes[0]) + ' (' + str(inercia_x) + '%)')
        plt.ylabel('Componente ' + str(ejes[1]) + ' (' + str(inercia_y) + '%)')
        for i in range(cor.shape[0]):
            plt.arrow(0, 0, cor[i, 0] * 0.95, cor[i, 1] * 0.95, color = 'steelblue', 
                      alpha = 0.5, head_width = 0.05, head_length = 0.05)
            if var_labels:
                plt.text(cor[i, 0] * 1.05, cor[i, 1] * 1.05, self.correlacion_var.index[i], 
                         color = 'steelblue', ha = 'center', va = 'center')
    def plot_sobreposicion(self, ejes = [0, 1], ind_labels = True, 
                      var_labels = True, titulo = 'Sobreposición Plano-Círculo'):
        x = self.coordenadas_ind[ejes[0]].values
        y = self.coordenadas_ind[ejes[1]].values
        cor = self.correlacion_var.iloc[:, ejes]
        scale = min((max(x) - min(x)/(max(cor[ejes[0]]) - min(cor[ejes[0]]))), 
                    (max(y) - min(y)/(max(cor[ejes[1]]) - min(cor[ejes[1]])))) * 0.7
        cor = self.correlacion_var.iloc[:, ejes].values
        plt.style.use('ggplot')
        plt.axhline(y = 0, color = 'dimgrey', linestyle = '--')
        plt.axvline(x = 0, color = 'dimgrey', linestyle = '--')
        inercia_x = round(self.var_explicada[ejes[0]], 2)
        inercia_y = round(self.var_explicada[ejes[1]], 2)
        plt.xlabel('Componente ' + str(ejes[0]) + ' (' + str(inercia_x) + '%)')
        plt.ylabel('Componente ' + str(ejes[1]) + ' (' + str(inercia_y) + '%)')
        plt.scatter(x, y, color = 'gray')
        if ind_labels:
            for i, txt in enumerate(self.coordenadas_ind.index):
                plt.annotate(txt, (x[i], y[i]))
        for i in range(cor.shape[0]):
            plt.arrow(0, 0, cor[i, 0] * scale, cor[i, 1] * scale, color = 'steelblue', 
                      alpha = 0.5, head_width = 0.05, head_length = 0.05)
            if var_labels:
                plt.text(cor[i, 0] * scale * 1.15, cor[i, 1] * scale * 1.15, 
                         self.correlacion_var.index[i], 
                         color = 'steelblue', ha = 'center', va = 'center')

# SECCION NoSupervisado
class NoSupervisado(analisisEDA):
    def __init__(self, df):
        self.__df = df
        self.__model = []
    @property
    def df(self):
        return self.__df 
    @df.setter
    def df(self, p_df):
        self.__df = p_df
        
    @property
    def model(self):
        return self.__model
    @model.setter
    def model(self, p_model):
        self.__model = p_model
        
    def benchmark(self):
        df = pd.DataFrame(self.__model, columns=['Algoritmo', 'Num de Clusters', 'Silhouette Score'])
        return df
    
    def __agregar_modelo(self, algoritmo, n_clusters, silhouette_score):
        self.__model.append([algoritmo, n_clusters, silhouette_score])


    def ACP(self, n_componentes): 
        p_acp = ACPmain(self.__df,n_componentes) 
        self.__ploteoGraficos(p_acp,1)
        self.__ploteoGraficos(p_acp,2)
        self.__ploteoGraficos(p_acp,3)
        
    def __ploteoGraficos(self,p_acp, tipo):
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize = (8,4), dpi = 200)
        if tipo==1:
            p_acp.plot_plano_principal()
        elif tipo==2:
            p_acp.plot_circulo()
        elif tipo==3:
            p_acp.plot_sobreposicion()     
            plt.show()
        

    def HAC(self):
        p_hac = self.__df
        ward_res = ward(self.__df)      
        average_res = average(self.__df)  
        single_res  = single(self.__df)    
        complete_res = complete(self.__df)
        self.__ploteoGraficosHAC(p_hac, ward_res, 1)
        self.__ploteoGraficosHAC(p_hac, average_res, 2)
        self.__ploteoGraficosHAC(p_hac, single_res, 3)
        self.__ploteoGraficosHAC(p_hac, complete_res, 4)
        self.__clusterHAC(1)
        self.__clusterHAC(2)

    # Silhouette Score
        metodos = ['ward', 'average', 'single', 'complete']
        for metodo in metodos:
            silhouette_scores = []
            for k in range(2, 11):
                modelo = AgglomerativeClustering(n_clusters=k, linkage=metodo, metric='euclidean')
                etiquetas = modelo.fit_predict(self.__df)
                score = silhouette_score(self.__df, etiquetas)
                silhouette_scores.append(score)
            mejor_k = np.argmax(silhouette_scores) + 2
            mejor_score = max(silhouette_scores)
            plt.figure(figsize=(10, 6))
            plt.plot(range(2, 11), silhouette_scores, marker='o')
            plt.title(f'Silhouette Score ({metodo})')
            plt.xlabel('Número de clusters')
            plt.ylabel('Silhouette Score')
            plt.grid(True)
            plt.show()
            if hasattr(self, '_NoSupervisado__agregar_modelo'):
                self.__agregar_modelo(f'HAC-{metodo}', mejor_k, mejor_score)

    def __ploteoGraficosHAC(self, p_hac, res, tipo):
        fig, ax = plt.subplots(1, 1, figsize=(12, 8), dpi=200)
        dendrogram(res, labels=self.__df.index.tolist(), ax=ax)

        mensajes = ['Metodo Ward:', 'Average:', 'Single:', 'Complete:']
        print(mensajes[tipo - 1])

        if tipo == 4:
            plt.savefig('dendrogram.png')
        ax.grid(False)
        plt.show()

    def __clusterHAC(self, tipo):
        grupos = fcluster(linkage(self.__df, method = 'ward', metric='euclidean'), 3, criterion = 'maxclust')
        grupos = grupos-1 
        centros = np.array(pd.concat([analisisEDA.centroide(0, self.__df, grupos), 
                              analisisEDA.centroide(1, self.__df, grupos),
                              analisisEDA.centroide(2, self.__df, grupos)]))
        if tipo == 1:
            analisisEDA.bar_plot(centros, self.__df.columns, scale=True)
        plt.show()
        

# SECCION K-MEDIAS
    def KMedia(self):
        self.__ploteoGraficosKMEDIAS(1)
        self.__ploteoGraficosKMEDIAS(2)

    def __ploteoGraficosKMEDIAS(self, tipo):
        if tipo == 1:
            self.__ploteoKmedias()
        elif tipo == 2:
            self.__ploteoKmedoids()
    
    def __ploteoKmedias(self):
        kmedias = KMeans(n_clusters=3, max_iter=500, n_init=150)
        kmedias.fit(self.__df)
        pca = PCA(n_components=2)
        componentes = pca.fit_transform(self.__df)
        fig, ax = plt.subplots(1,1, figsize = (15,8), dpi = 200)
        colores = ['red', 'green', 'blue']
        colores_puntos = [colores[label] for label in kmedias.predict(self.__df)]
        ax.scatter(componentes[:, 0], componentes[:, 1],c=colores_puntos)
        ax.set_xlabel('componente 1')
        ax.set_ylabel('componente 2')
        ax.set_title('3 Cluster K-Medias')
        ax.grid(False)
        plt.show()

        centros = np.array(kmedias.cluster_centers_)
        analisisEDA.radar_plot(centros, self.__df.columns)
        plt.show()

    def __ploteoKmedoids(self):
        kmedoids = KMedoids(n_clusters=3, max_iter=500, metric='cityblock')
        kmedoids.fit(self.__df)
        pca = PCA(n_components=2)
        componentes = pca.fit_transform(self.__df)
        fig, ax = plt.subplots(1,1, figsize = (15,8), dpi = 200)
        colores = ['red', 'green', 'blue']
        colores_puntos = [colores[label] for label in kmedoids.predict(self.__df)]
        ax.scatter(componentes[:, 0], componentes[:, 1],c=colores_puntos)
        ax.set_xlabel('componente 1')
        ax.set_ylabel('componente 2')
        ax.set_title('3 Cluster K-Medoids')
        ax.grid(False)
        plt.show()

        centros = np.array(kmedoids.cluster_centers_)
        analisisEDA.radar_plot(centros, self.__df.columns)
        plt.show()
        
# UMAP
    def UMAP(self, n_componentes=2, n_neighbors=15):
        if self.__df is None or self.__df.empty:
            raise ValueError("El DataFrame está vacío o no ha sido definido.")
        if self.__df.isnull().any().any():
            raise ValueError("El DataFrame contiene valores nulos.")

    # Normalización previa
        df_scaled = StandardScaler().fit_transform(self.__df)

    # UMAP
        modelo_umap = um.UMAP(n_components=n_componentes, n_neighbors=n_neighbors)
        componentes = modelo_umap.fit_transform(df_scaled)

    # Gráfico
        fig, ax = plt.subplots(1, 1, figsize=(10, 6), dpi=200)
        ax.scatter(componentes[:, 0], componentes[:, 1], alpha=0.7)
        ax.set_xlabel('Componente 1')
        ax.set_ylabel('Componente 2')
        ax.set_title('UMAP')
        ax.grid(False)
        plt.show()

        return componentes

#T-SNE
    def TSNE(self):
        tsne = TSNE(n_components=2, random_state=42)
        df_tsne = tsne.fit_transform(self.__df)
        
        silhouette_scores = []
        for n_clusters in range(2, 11):
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
            cluster_labels = kmeans.fit_predict(df_tsne)
            silhouette_avg = silhouette_score(df_tsne, cluster_labels)
            silhouette_scores.append(silhouette_avg)
    
        optimal_n_clusters = silhouette_scores.index(max(silhouette_scores)) + 2
    
        kmeans = KMeans(n_clusters=optimal_n_clusters, random_state=42, n_init='auto')
        cluster_labels = kmeans.fit_predict(df_tsne)
        
        print('Silhouette Score:', max(silhouette_scores))
        self.__plotTSNE(df_tsne, cluster_labels, silhouette_scores)
        
        self.__agregar_modelo('TSNE', optimal_n_clusters, max(silhouette_scores))


    def __plotTSNE(self, df, cluster_labels, silhouette_scores):
        plt.plot(range(2, 11), silhouette_scores, marker='o')
        plt.xlabel('Número de Clusters')
        plt.ylabel('Puntuación de la Silueta')
        plt.title('Silhouette Score')
        plt.grid(True)
        plt.show()
        
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=df[:, 0], y=df[:, 1], hue=cluster_labels, palette='tab10', legend='full')
        plt.title('TSNE')
        plt.xlabel('TSNE Dimensión 1')
        plt.ylabel('TSNE Dimensión 2')
        plt.legend(title='Cluster')
        plt.show()

# SECCION SUPERVISADOS
class Supervisado(analisisEDA):
    def __init__(self, df):
        self.__df = df
    @property
    def df(self):
        return self.__df
    @df.setter
    def df(self, p_df):
        self.__df = p_df
    
    def preparar_datos(self):
        X = self.__df.drop(columns=['target'])
        X = pd.DataFrame(StandardScaler().fit_transform(X), columns=X.columns)
        y = self.__df['target']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
        return X_train, X_test, y_train, y_test, y

# KNN  
    def modeloKNN(self, X_train, y_train, n_neighbors, algorithm):
        model = KNeighborsClassifier(n_neighbors=n_neighbors, algorithm=algorithm)
        model.fit(X_train, y_train)
        return model
    
    def KNN(self, n_neighbors):
        algorithms = ['auto', 'ball_tree', 'kd_tree', 'brute']
        for algorithm in algorithms:
            print(f"\nUsando el algoritmo: {algorithm}")
            X_train, X_test, y_train, y_test, y = self.preparar_datos()
            model = self.modeloKNN(X_train, y_train, n_neighbors, algorithm)
            y_pred = self.predecir(model, X_test)
            self.evaluar(y_test, y_pred, y)

    def prediccionesKNN(self):
        print("=== Predicciones KNN ===")
        for n in range(1, 11):
            for algorithm in ['auto', 'ball_tree', 'kd_tree', 'brute']:
                print(f"\nKNN - Vecinos: {n}, Algoritmo: {algorithm}")
                print("                                              ")
                X_train, X_test, y_train, y_test, y = self.preparar_datos()
                model = self.modeloKNN(X_train, y_train, n_neighbors=n, algorithm=algorithm)
                y_pred = self.predecir(model, X_test)
                print("Predicciones en Testing:", y_pred)
                print("---------------------------------------------------")
                print("Matriz de Confusión:\n", confusion_matrix(y_test, y_pred))
                print("______________________________________________")

# Decision Tree
    def modeloDT(self, X_train, y_train, min_samples_split, max_depth):
        model = DecisionTreeClassifier(min_samples_split=min_samples_split, max_depth=max_depth)
        model.fit(X_train, y_train)
        return model

    def DT(self, min_samples_split, max_depth):
        X_train, X_test, y_train, y_test, y = self.preparar_datos()
        model = self.modeloDT(X_train, y_train, min_samples_split, max_depth)
        y_pred = self.predecir(model, X_test)
        self.evaluar(y_test, y_pred, y)
  
    def prediccionesDT(self):
        print("=== Predicciones Árbol de Decisión ===")
        for depth in range(1, 11):
            print(f"\nDT - max_depth: {depth}")
            print("                                              ")
            X_train, X_test, y_train, y_test, y = self.preparar_datos()
            model = self.modeloDT(X_train, y_train, min_samples_split=2, max_depth=depth)
            y_pred = self.predecir(model, X_test)
            print("Predicciones en Testing:", y_pred)
            print("---------------------------------------------------")
            print("Matriz de Confusión:\n", confusion_matrix(y_test, y_pred))
            print("______________________________________________")

# Random Forest
    def modeloRF(self, X_train, y_train, n_estimators, min_samples_split, max_depth):
        model = RandomForestClassifier(n_estimators=n_estimators, min_samples_split=min_samples_split, max_depth=max_depth)
        model.fit(X_train, y_train)
        return model

    def RF(self, n_estimators, min_samples_split, max_depth):
        X_train, X_test, y_train, y_test, y = self.preparar_datos()
        model = self.modeloRF(X_train, y_train, n_estimators, min_samples_split, max_depth)
        y_pred = self.predecir(model, X_test)
        self.evaluar(y_test, y_pred, y)

    def prediccionesRF(self):
        print("=== Predicciones Random Forest ===")
        for depth in range(1, 11):
            print(f"\nRF - max_depth: {depth}")
            print("                                              ")
            X_train, X_test, y_train, y_test, y = self.preparar_datos()
            model = self.modeloRF(X_train, y_train, n_estimators=100, min_samples_split=2, max_depth=depth)
            y_pred = self.predecir(model, X_test)
            print("Predicciones en Testing:", y_pred)
            print("---------------------------------------------------")
            print("Matriz de Confusión:\n", confusion_matrix(y_test, y_pred))
            print("______________________________________________")

# XGBoost
    def modeloXG(self, X_train, y_train, n_estimators, min_samples_split, max_depth):
        model = GradientBoostingClassifier(n_estimators=n_estimators, min_samples_split=min_samples_split, max_depth=max_depth)
        model.fit(X_train, y_train)
        return model

    def XG(self, n_estimators, min_samples_split, max_depth):
        X_train, X_test, y_train, y_test, y = self.preparar_datos()
        model = self.modeloXG(X_train, y_train, n_estimators, min_samples_split, max_depth)
        y_pred = self.predecir(model, X_test)
        self.evaluar(y_test, y_pred, y)

    def prediccionesXG(self):
        print("=== Predicciones Gradient Boosting ===")
        for depth in range(1, 11):
            print(f"\nXGBoost - max_depth: {depth}")
            print("                                              ")
            X_train, X_test, y_train, y_test, y = self.preparar_datos()
            model = self.modeloXG(X_train, y_train, n_estimators=100, min_samples_split=2, max_depth=depth)
            y_pred = self.predecir(model, X_test)
            print("Predicciones en Testing:", y_pred)
            print("---------------------------------------------------")
            print("Matriz de Confusión:\n", confusion_matrix(y_test, y_pred))
            print("______________________________________________")

# ADABoost
    def modeloADA(self, X_train, y_train, estimator, n_estimators):
        model = AdaBoostClassifier(estimator=estimator, n_estimators=n_estimators)
        model.fit(X_train, y_train)
        return model

    def ADA(self, n_estimators):
        estimators = {"Decision Tree": DecisionTreeClassifier(min_samples_split=2, max_depth=4),
                    "Random Forest": RandomForestClassifier(n_estimators=100, min_samples_split=2, max_depth=4),
                    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, min_samples_split=2, max_depth=4)
                    }
        for name, estimator in estimators.items():
            print(f"\nUsando el metodo: {name}")
            X_train, X_test, y_train, y_test, y = self.preparar_datos()
            model = self.modeloADA(X_train, y_train, estimator, n_estimators)
            y_pred = self.predecir(model, X_test)
            self.evaluar(y_test, y_pred, y)

    def prediccionesADA(self):
        print("=== Predicciones AdaBoost ===")
        for n in range(10, 101, 10):
            print(f"\nAdaBoost - n_estimators: {n}")
            print("                                              ")
            X_train, X_test, y_train, y_test, y = self.preparar_datos()
            estimator = DecisionTreeClassifier(min_samples_split=2, max_depth=4)
            model = self.modeloADA(X_train, y_train, estimator=estimator, n_estimators=n)
            y_pred = self.predecir(model, X_test)
            print("Predicciones en Testing:", y_pred)
            print("---------------------------------------------------")
            print("Matriz de Confusión:\n", confusion_matrix(y_test, y_pred))
            print("______________________________________________")

    def predecir(self, model, X_test):
        return model.predict(X_test)
    
    def evaluar(self, y_test, y_pred, y):
        MC = confusion_matrix(y_test, y_pred)
        indices = self.indices_general(MC,list(np.unique(y)))
        for k in indices:
            print("\n%s:\n%s"%(k,str(indices[k])))
    
    def indices_general(self, MC, nombres = None):
        precision_global = np.sum(MC.diagonal()) / np.sum(MC)
        error_global     = 1 - precision_global
        precision_categoria  = pd.DataFrame(MC.diagonal()/np.sum(MC,axis = 1)).T
        if nombres!=None:
            precision_categoria.columns = nombres
        return {"Matriz de Confusión":MC, 
                "Precisión Global":   precision_global, 
                "Error Global":       error_global, 
                "Precisión por categoría":precision_categoria}
        

    def __KNNBM(self, n_neighbors=3, algorithm="ball_tree"):
            X_train, X_test, y_train, y_test, y = self.preparar_datos()
            model = self.modeloKNN(X_train, y_train, n_neighbors, algorithm)
            y_pred = self.predecir(model, X_test)
            MCknn = confusion_matrix(y_test, y_pred)
            indices = self.indices_general(MCknn,list(np.unique(y)))
            return indices
    
    def __DTBM(self, min_samples_split=8, max_depth=1):
            X_train, X_test, y_train, y_test, y = self.preparar_datos()
            model = self.modeloDT(X_train, y_train, min_samples_split, max_depth)
            y_pred = self.predecir(model, X_test)
            MCdt = confusion_matrix(y_test, y_pred)
            indices = self.indices_general(MCdt,list(np.unique(y)))
            return indices
    
    def __RFBM(self, n_estimators=200, min_samples_split=9, max_depth=8):
            X_train, X_test, y_train, y_test, y = self.preparar_datos()
            model = self.modeloRF(X_train, y_train, n_estimators, min_samples_split, max_depth)
            y_pred = self.predecir(model, X_test)
            MCdt = confusion_matrix(y_test, y_pred)
            indices = self.indices_general(MCdt,list(np.unique(y)))
            return indices

    def __XGBM(self, n_estimators=100, min_samples_split=5, max_depth=1):
            X_train, X_test, y_train, y_test, y = self.preparar_datos()
            model = self.modeloXG(X_train, y_train, n_estimators, min_samples_split, max_depth)
            y_pred = self.predecir(model, X_test)
            MCdt = confusion_matrix(y_test, y_pred)
            indices = self.indices_general(MCdt,list(np.unique(y)))
            return indices
    
    def __ADABM(self, estimator= GradientBoostingClassifier(n_estimators=100, min_samples_split=5, max_depth=1), n_estimators=10):
            X_train, X_test, y_train, y_test, y = self.preparar_datos()
            model = self.modeloADA(X_train, y_train, estimator, n_estimators)
            y_pred = self.predecir(model, X_test)
            MCdt = confusion_matrix(y_test, y_pred)
            indices = self.indices_general(MCdt,list(np.unique(y)))
            return indices

    def BM(self):
        datos = {"PG": [0,0,0,0,0],"EG": [0,0,0,0,0],"PP": [0,0,0,0,0],"PN": [0,0,0,0,0]}
        Tdatos = pd.DataFrame(datos,index=["AlgKnn","AlgDT","AlgRF","AlgXGBoost","AlgADABoost"],columns=["PG","EG","PP","PN"])

        indices = self.__KNNBM()
        PP=indices['Precisión por categoría']
        PN=indices['Precisión por categoría']
        Tdatos.loc["AlgKnn", "PG"] = indices['Precisión Global']
        Tdatos.loc["AlgKnn", "EG"] = indices['Error Global']
        Tdatos.loc["AlgKnn", "PP"] = PP.loc[0,'Diabetes']
        Tdatos.loc["AlgKnn", "PN"] = PN.loc[0,'No_diabetes']
        
        indices = self.__DTBM()
        PP=indices['Precisión por categoría']
        PN=indices['Precisión por categoría']
        Tdatos.loc["AlgDT", "PG"] = indices['Precisión Global']
        Tdatos.loc["AlgDT", "EG"] = indices['Error Global']
        Tdatos.loc["AlgDT", "PP"] = PP.loc[0,'Diabetes']
        Tdatos.loc["AlgDT", "PN"] = PN.loc[0,'No_diabetes']

        indices = self.__RFBM()
        PP=indices['Precisión por categoría']
        PN=indices['Precisión por categoría']
        Tdatos.loc["AlgRF", "PG"] = indices['Precisión Global']
        Tdatos.loc["AlgRF", "EG"] = indices['Error Global']
        Tdatos.loc["AlgRF", "PP"] = PP.loc[0,'Diabetes']
        Tdatos.loc["AlgRF", "PN"] = PN.loc[0,'No_diabetes']

        indices = self.__XGBM()
        PP=indices['Precisión por categoría']
        PN=indices['Precisión por categoría']
        Tdatos.loc["AlgXGBoost", "PG"] = indices['Precisión Global']
        Tdatos.loc["AlgXGBoost", "EG"] = indices['Error Global']
        Tdatos.loc["AlgXGBoost", "PP"] = PP.loc[0,'Diabetes']
        Tdatos.loc["AlgXGBoost", "PN"] = PN.loc[0,'No_diabetes']

        indices = self.__ADABM()
        PP=indices['Precisión por categoría']
        PN=indices['Precisión por categoría']
        Tdatos.loc["AlgADABoost", "PG"] = indices['Precisión Global']
        Tdatos.loc["AlgADABoost", "EG"] = indices['Error Global']
        Tdatos.loc["AlgADABoost", "PP"] = PP.loc[0,'Diabetes']
        Tdatos.loc["AlgADABoost", "PN"] = PN.loc[0,'No_diabetes']

        print(Tdatos)
