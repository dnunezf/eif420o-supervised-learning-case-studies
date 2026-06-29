# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 18:49:47 2024

@author: JUANMURILLOMORERA
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix 
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from math import ceil, floor, pi
from IPython.display import display
from sklearn.model_selection import GridSearchCV
import warnings

warnings.filterwarnings("ignore")


class Supervisado():
    def __init__(self, df):
        self.__df = df
    @property
    def df(self):
        return self.__df
    @df.setter
    def df(self, p_df):
        self.__df = p_df
        
    def ADA_grid_search(self, param_grid):
        X_train, X_test, y_train, y_test, y = self.__preparar_datos()
        ada = AdaBoostClassifier(base_estimator=DecisionTreeClassifier())
        grid_search = GridSearchCV(ada, param_grid, cv=5, scoring='accuracy', verbose=1, n_jobs=-1)
        grid_search.fit(X_train, y_train)

        print("Mejores hiperparámetros encontrados:")
        print(grid_search.best_params_)

        print("\nMejor puntuación de validación cruzada:")
        print(grid_search.best_score_)

        print("\nRendimiento en el conjunto de prueba:")
        y_pred = grid_search.predict(X_test)
        self.__evaluar(y_test, y_pred, y)
    
    def __preparar_datos(self):
        X = self.__df.drop(columns=['target'])
        X = pd.DataFrame(StandardScaler().fit_transform(X), columns=X.columns)
        #y = self.__df['target']
        y = self.__df['target'].astype(int)  # Convertir a tipo entero  # Convertir a tipo entero
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
        return X_train, X_test, y_train, y_test, y
    
    def __modeloKNN(self, X_train, y_train, n_neighbors, algorithm):
        model = KNeighborsClassifier(n_neighbors=n_neighbors, algorithm=algorithm)
        model.fit(X_train, y_train)
        return model
    
    def __modeloDT(self, X_train, y_train, min_samples_split, max_depth):
        model = DecisionTreeClassifier(min_samples_split=min_samples_split, max_depth=max_depth)
        model.fit(X_train, y_train)
        return model
    
    def __modeloRF(self, X_train, y_train, n_estimators, min_samples_split, max_depth):
        model = RandomForestClassifier(n_estimators=n_estimators, min_samples_split=min_samples_split, max_depth=max_depth)
        model.fit(X_train, y_train)
        return model
    
    def __modeloXG(self, X_train, y_train, n_estimators, min_samples_split, max_depth):
        model = GradientBoostingClassifier(n_estimators=n_estimators, min_samples_split=min_samples_split, max_depth=max_depth)
        model.fit(X_train, y_train)
        return model
    
    def __modeloADA(self, X_train, y_train, estimator, n_estimators):
        model = AdaBoostClassifier(estimator=estimator, n_estimators=n_estimators)
        model.fit(X_train, y_train)
        return model
    
    def __predecir(self, model, X_test):
        return model.predict(X_test)
    
    def __evaluar(self, y_test, y_pred, y):
        MC = confusion_matrix(y_test, y_pred)
        indices = self.__indices_general(MC,list(np.unique(y)))
        for k in indices:
            print("\n%s:\n%s"%(k,str(indices[k])))
    
    def __indices_general(self, MC, nombres = None):
        precision_global = np.sum(MC.diagonal()) / np.sum(MC)
        error_global     = 1 - precision_global
        precision_categoria  = pd.DataFrame(MC.diagonal()/np.sum(MC,axis = 1)).T
        if nombres!=None:
            precision_categoria.columns = nombres
        return {"Matriz de Confusión":MC, 
                "Precisión Global":   precision_global, 
                "Error Global":       error_global, 
                "Precisión por categoría":precision_categoria}
        
    def KNN(self, n_neighbors):
        algorithms = ['auto', 'ball_tree', 'kd_tree', 'brute']
        for algorithm in algorithms:
            print(f"\nUsando algoritmo: {algorithm}")
            X_train, X_test, y_train, y_test, y = self.__preparar_datos()
        
            # Convertir las etiquetas de clase a tipo entero si es necesario
            y_train = y_train.astype(int)
            y_test = y_test.astype(int)
        
            model = self.__modeloKNN(X_train, y_train, n_neighbors, algorithm)
            y_pred = self.__predecir(model, X_test)
            self.__evaluar(y_test, y_pred, y)
        
        
    def DT(self, min_samples_split, max_depth):
        X_train, X_test, y_train, y_test, y = self.__preparar_datos()
        model = self.__modeloDT(X_train, y_train, min_samples_split, max_depth)
        y_pred = self.__predecir(model, X_test)
        self.__evaluar(y_test, y_pred, y)
    
    def RF(self, n_estimators, min_samples_split, max_depth):
        X_train, X_test, y_train, y_test, y = self.__preparar_datos()
        model = self.__modeloRF(X_train, y_train, n_estimators, min_samples_split, max_depth)
        y_pred = self.__predecir(model, X_test)
        self.__evaluar(y_test, y_pred, y)
    
    def XG(self, n_estimators, min_samples_split, max_depth):
        X_train, X_test, y_train, y_test, y = self.__preparar_datos()
        model = self.__modeloXG(X_train, y_train, n_estimators, min_samples_split, max_depth)
        y_pred = self.__predecir(model, X_test)
        self.__evaluar(y_test, y_pred, y)

    def ADA(self, n_estimators):
        estimators = {"Decision Tree": DecisionTreeClassifier(min_samples_split=2, max_depth=4),
                    "Random Forest": RandomForestClassifier(n_estimators=100, min_samples_split=2, max_depth=4),
                    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, min_samples_split=2, max_depth=4)
                    }
        for name, estimator in estimators.items():
            print(f"\nUsando metodo: {name}")
            X_train, X_test, y_train, y_test, y = self.__preparar_datos()
            model = self.__modeloADA(X_train, y_train, estimator, n_estimators)
            y_pred = self.__predecir(model, X_test)
            self.__evaluar(y_test, y_pred, y)

    def __KNNBM(self, n_neighbors=5, algorithm="auto"):
            X_train, X_test, y_train, y_test, y = self.__preparar_datos()
            model = self.__modeloKNN(X_train, y_train, n_neighbors, algorithm)
            y_pred = self.__predecir(model, X_test)
            MCknn = confusion_matrix(y_test, y_pred)
            indices = self.__indices_general(MCknn,list(np.unique(y)))
            return indices
    
    def __DTBM(self, min_samples_split=2, max_depth=4):
            X_train, X_test, y_train, y_test, y = self.__preparar_datos()
            model = self.__modeloDT(X_train, y_train, min_samples_split, max_depth)
            y_pred = self.__predecir(model, X_test)
            MCdt = confusion_matrix(y_test, y_pred)
            indices = self.__indices_general(MCdt,list(np.unique(y)))
            return indices
    
    def __RFBM(self, n_estimators=100, min_samples_split=2, max_depth=4):
            X_train, X_test, y_train, y_test, y = self.__preparar_datos()
            model = self.__modeloRF(X_train, y_train, n_estimators, min_samples_split, max_depth)
            y_pred = self.__predecir(model, X_test)
            MCdt = confusion_matrix(y_test, y_pred)
            indices = self.__indices_general(MCdt,list(np.unique(y)))
            return indices

    def __XGBM(self, n_estimators=100, min_samples_split=2, max_depth=4):
            X_train, X_test, y_train, y_test, y = self.__preparar_datos()
            model = self.__modeloXG(X_train, y_train, n_estimators, min_samples_split, max_depth)
            y_pred = self.__predecir(model, X_test)
            MCdt = confusion_matrix(y_test, y_pred)
            indices = self.__indices_general(MCdt,list(np.unique(y)))
            return indices
    
    def __ADABM(self, estimator= DecisionTreeClassifier(min_samples_split=2, max_depth=4), n_estimators=100):
            X_train, X_test, y_train, y_test, y = self.__preparar_datos()
            model = self.__modeloADA(X_train, y_train, estimator, n_estimators)
            y_pred = self.__predecir(model, X_test)
            MCdt = confusion_matrix(y_test, y_pred)
            indices = self.__indices_general(MCdt,list(np.unique(y)))
            return indices

    def BM(self):
        datos = {"PG": [0, 0, 0, 0, 0], "EG": [0, 0, 0, 0, 0], "PP": [0, 0, 0, 0, 0], "PN": [0, 0, 0, 0, 0]}
        Tdatos = pd.DataFrame(datos, index=["AlgKnn", "AlgDT", "AlgRF", "AlgXGBoost", "AlgADABoost"],
                          columns=["PG", "EG", "PP", "PN"])

        algoritmos = [("AlgKnn", self.__KNNBM), ("AlgDT", self.__DTBM), ("AlgRF", self.__RFBM),
                  ("AlgXGBoost", self.__XGBM), ("AlgADABoost", self.__ADABM)]

        for alg_name, alg_method in algoritmos:
            indices = alg_method()
            PP = indices['Precisión por categoría']
            PN = indices['Precisión por categoría']
            Tdatos.loc[alg_name, "PG"] = indices['Precisión Global']
            Tdatos.loc[alg_name, "EG"] = indices['Error Global']
            Tdatos.loc[alg_name, "PP"] = PP.loc[0, 1]
            Tdatos.loc[alg_name, "PN"] = PN.loc[0, 0]

        print(Tdatos)