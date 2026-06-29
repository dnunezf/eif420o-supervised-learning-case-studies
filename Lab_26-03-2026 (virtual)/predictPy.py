import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import matplotlib.ticker as mticker
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sns


class Analisis_Predictivo:

    def __init__(self, datos: pd.DataFrame, predecir: str, predictoras=[],
                 modelo=None, estandarizar = True,train_size=0.75, random_state=None):
        '''
        datos: Datos completos y listos para construir un modelo   

        predecir: Nombre de la variable a predecir

        predictoras: Lista de los nombres de las variables predictoras.
        Si vacío entonces utiliza todas las variables presentes excepto la variable a predecir.

        modelo: Instancia de una Clase de un método de clasificación(KNN,Árboles,SVM,etc).
        Si no especifica un modelo no podrá utilizar el método fit

        estandarizar: Indica si se debe o no estandarizar los datos que utilizará el estimador.

        train_size: Si el valor es tipo flotante entre 0.0 y 1.0 entonces representa la proporción de la tabla de entrenamiento.
        Si el valor es un entero entonces representa el valor absoluto de la tabla de entrenamiento.

        random_state: Semilla aleatoria para la división de datos(training-testing).
        '''
        self.__datos = datos
        self.__predecir = predecir
        self.__predictoras = predictoras
        self.__nombre_clases = list(np.unique(self.__datos[predecir].values))
        self.__modelo = modelo
        self.__random_state = random_state
        if modelo != None:
            self.__train_size = train_size
            self._training_testing(estandarizar)
    
    @property
    def datos(self):
        return self.__datos
    
    @property
    def predecir(self):
        return self.__predecir
    
    @property
    def predictoras(self):
        return self.__predictoras
    
    @property
    def nombre_clases(self):
        return self.__nombre_clases
    
    @property
    def modelo(self):
        return self.__modelo
    
    @property
    def random_state(self):
        return self.__random_state
    
    @property
    def train_size(self):
        return self.__train_size
    
    @datos.setter
    def datos(self, datos):
        self.__datos = datos
    
    @predecir.setter
    def predecir(self, predecir):
        self.__predecir = predecir
        
    @predictoras.setter
    def predictoras(self, predictoras):
        self.__predictoras = predictoras
        
    @nombre_clases.setter
    def nombre_clases(self, nombre_clases):
        self.__nombre_clases = nombre_clases
        
    @modelo.setter
    def modelo(self, modelo):
        self.__modelo = modelo
        
    @random_state.setter
    def random_state(self, random_state):
        self.__random_state = random_state
        
    @train_size.setter
    def train_size(self, train_size):
        self.__train_size = train_size
        
    def _training_testing(self, estandarizar = True):
        if len(self.predictoras) == 0:
            X = self.datos.drop(columns=[self.predecir])
            self.predictoras = list(X.columns.values)
        else:
            X = self.datos[self.predictoras]

        if estandarizar:
            X = pd.DataFrame(StandardScaler().fit_transform(X), columns= X.columns)

        y = self.datos[self.predecir].values

        train_test = train_test_split(X, y, train_size=self.train_size,
                                      random_state=self.random_state)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test

    def fit_predict(self):
        if(self.modelo != None):
            self.modelo.fit(self.X_train, self.y_train)
            return self.modelo.predict(self.X_test)

    def fit_predict_resultados(self, imprimir=True):
        if(self.modelo != None):
            prediccion = self.fit_predict()
            MC = confusion_matrix(self.y_test, prediccion, labels= self.nombre_clases)
            indices = self.indices_general(MC, self.nombre_clases)
            if imprimir == True:
                for k in indices:
                    print("\n%s:\n%s" % (k, str(indices[k])))

            return indices

    def indices_general(self, MC, nombres=None):
        "Método para calcular los índices de calidad de la predicción"
        precision_global = np.sum(MC.diagonal()) / np.sum(MC)
        error_global = 1 - precision_global
        precision_categoria = pd.DataFrame(MC.diagonal()/np.sum(MC, axis=1)).T
        if nombres != None:
            precision_categoria.columns = nombres
        return {"Matriz de Confusión": MC,
                "Precisión Global": precision_global,
                "Error Global": error_global,
                "Precisión por categoría": precision_categoria}

    def distribucion_variable_predecir(self, ax=None):
        "Método para graficar la distribución de la variable a predecir"
        variable_predict = self.predecir
        data = self.datos
        if ax == None:
            fig, ax = plt.subplots(1, 1, figsize=(12, 6), dpi=200)
        colors = list(dict(**mcolors.CSS4_COLORS))
        df = pd.crosstab(index=data[variable_predict],
                         columns="valor") / data[variable_predict].count()
        countv = 0
        titulo = "Distribución de la variable %s" % variable_predict
        for i in range(df.shape[0]):
            ax.barh(1, df.iloc[i], left=countv, align='center',
                    color=colors[11+i], label=df.iloc[i].name)
            countv = countv + df.iloc[i]
        ax.set_xlim(0, 1)
        ax.set_yticklabels("")
        ax.set_ylabel(variable_predict)
        ax.set_title(titulo)
        ticks_loc = ax.get_xticks().tolist()
        ax.xaxis.set_major_locator(mticker.FixedLocator(ticks_loc))
        ax.set_xticklabels(['{:.0%}'.format(x) for x in ticks_loc])
        countv = 0
        for v in df.iloc[:, 0]:
            ax.text(np.mean([countv, countv+v]) - 0.03, 1,
                    '{:.1%}'.format(v), color='black', fontweight='bold')
            countv = countv + v
        ax.legend(loc='upper center', bbox_to_anchor=(
            1.08, 1), shadow=True, ncol=1)

    def poder_predictivo_categorica(self, var: str, ax=None):
        "Método para ver la distribución de una variable categórica respecto a la predecir"
        data = self.datos
        variable_predict = self.predecir
        if ax == None:
            fig, ax = plt.subplots(1, 1, figsize=(12, 6), dpi=200)
        df = pd.crosstab(index=data[var], columns=data[variable_predict])
        df = df.div(df.sum(axis=1), axis=0)
        titulo = "Distribución de la variable %s según la variable %s" % (
            var, variable_predict)
        df.plot(kind='barh', stacked=True, legend=True,
                ax=ax, xlim=(0, 1), title=titulo, width=0.8)
        ticks_loc = ax.get_xticks().tolist()
        ax.xaxis.set_major_locator(mticker.FixedLocator(ticks_loc))
        ax.set_xticklabels(['{:.0%}'.format(x) for x in ticks_loc])
        ax.legend(loc='upper center', bbox_to_anchor=(
            1.08, 1), shadow=True, ncol=1)
        for bars in ax.containers:
            plt.setp(bars, width=.9)
        for i in range(df.shape[0]):
            countv = 0
            for v in df.iloc[i]:
                ax.text(np.mean([countv, countv+v]) - 0.03, i, '{:.1%}'.format(v),
                        color='black', fontweight='bold')
                countv = countv + v

    def poder_predictivo_numerica(self, var: str):
        "Función para ver la distribución de una variable numérica respecto a la predecir"
        sns.FacetGrid(self.datos, hue=self.predecir, height=8, aspect=1.8).map(
            sns.kdeplot, var, shade=True).add_legend()
def indices_general(MC, nombres = None):
    precision_global = np.sum(MC.diagonal()) / np.sum(MC)
    error_global     = 1 - precision_global
    precision_categoria  = pd.DataFrame(MC.diagonal()/np.sum(MC,axis = 1)).T
    if nombres!=None:
        precision_categoria.columns = nombres
    return {"Matriz de Confusión":MC, 
            "Precisión Global":   precision_global, 
            "Error Global":       error_global, 
            "Precisión por categoría":precision_categoria}