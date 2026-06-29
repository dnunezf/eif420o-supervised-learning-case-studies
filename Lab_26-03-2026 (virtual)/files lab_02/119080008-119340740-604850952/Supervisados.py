#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LABORATORIO 02 - INTELIGENCIA ARTIFICIAL
Integración de Análisis de Clasificación (KNN, Árboles de Decisión, RandomForest y Boosting)
Dataset: BankChurners (Bank Customer Churn Prediction)
"""

# =============================================================================
# SECCIÓN 1: ModuloSupLab02.py (Módulos de Boosting)
# =============================================================================

import warnings
warnings.filterwarnings("ignore")

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
    roc_auc_score,
    roc_curve,
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier

try:
    from xgboost import XGBClassifier
    XGBOOST_DISPONIBLE = True
except Exception:
    XGBClassifier = None
    XGBOOST_DISPONIBLE = False


@dataclass
class ConfiguracionModelo:
    """Estructura simple para definir cada configuracion."""
    nombre: str
    algoritmo: str
    imputador: str = "median"
    escalador: Optional[str] = None
    k_mejores: Optional[int] = None
    parametros_modelo: Optional[Dict[str, Any]] = None


class ClasificacionBoostingAnalisis:
    """
    Ajusta y evalua una sola configuracion de XGBoost o AdaBoost.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        columna_objetivo: str,
        configuracion: ConfiguracionModelo,
        columnas_eliminar: Optional[List[str]] = None,
        test_size: float = 0.25,
        random_state: int = 42,
        cv_splits: int = 5
    ):
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df debe ser un DataFrame de pandas.")

        if columna_objetivo not in df.columns:
            raise ValueError(f"La columna objetivo '{columna_objetivo}' no existe en el DataFrame.")

        self.__df = df.copy()
        self.__objetivo = columna_objetivo
        self.__config = configuracion
        self.__columnas_eliminar = columnas_eliminar or []
        self.__test_size = test_size
        self.__random_state = random_state
        self.__cv_splits = cv_splits

        self.__pipeline = None
        self.__X_train = None
        self.__X_test = None
        self.__y_train = None
        self.__y_test = None
        self.__y_pred = None
        self.__y_prob = None
        self.__label_encoder = None
        self.__cv_metricas = None
        self.__metricas_test = None
        self.__matriz_confusion = None
        self.__feature_names = None
        self.__importancias = None
        self.__reporte_clasificacion = None

        self.__ajustar()

    @property
    def nombre(self):
        return self.__config.nombre

    @property
    def algoritmo(self):
        return self.__config.algoritmo

    @property
    def pipeline(self):
        return self.__pipeline

    @property
    def cv_metricas(self):
        return self.__cv_metricas

    @property
    def metricas_test(self):
        return self.__metricas_test

    @property
    def matriz_confusion(self):
        return self.__matriz_confusion

    @property
    def reporte_clasificacion(self):
        return self.__reporte_clasificacion

    @property
    def importancias(self):
        return self.__importancias

    @property
    def feature_names(self):
        return self.__feature_names

    def __construir_preprocesamiento(self, X: pd.DataFrame) -> ColumnTransformer:
        columnas_numericas = X.select_dtypes(include=["number"]).columns.tolist()
        columnas_categoricas = X.select_dtypes(exclude=["number"]).columns.tolist()

        pasos_num = [("imputer", SimpleImputer(strategy=self.__config.imputador))]
        if self.__config.escalador == "standard":
            pasos_num.append(("scaler", StandardScaler()))
        elif self.__config.escalador == "minmax":
            pasos_num.append(("scaler", MinMaxScaler()))
        elif self.__config.escalador == "robust":
            pasos_num.append(("scaler", RobustScaler()))

        pipeline_num = Pipeline(pasos_num)

        if len(columnas_categoricas) > 0:
            pipeline_cat = Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore"))
            ])

            preprocesador = ColumnTransformer([
                ("num", pipeline_num, columnas_numericas),
                ("cat", pipeline_cat, columnas_categoricas)
            ])
        else:
            preprocesador = ColumnTransformer([
                ("num", pipeline_num, columnas_numericas)
            ])

        return preprocesador

    def __construir_modelo(self, y_codificado: np.ndarray):
        n_clases = len(np.unique(y_codificado))
        params = self.__config.parametros_modelo.copy() if self.__config.parametros_modelo else {}

        if self.__config.algoritmo.lower() == "adaboost":
            base_estimator = params.pop(
                "estimator",
                DecisionTreeClassifier(max_depth=1, random_state=self.__random_state)
            )
            modelo = AdaBoostClassifier(
                estimator=base_estimator,
                random_state=self.__random_state,
                **params
            )
            return modelo

        if self.__config.algoritmo.lower() == "xgboost":
            if not XGBOOST_DISPONIBLE:
                raise ImportError(
                    "No se pudo importar xgboost. Instale el paquete con: pip install xgboost"
                )

            objetivo = "binary:logistic" if n_clases == 2 else "multi:softprob"
            params_limpios = {k: v for k, v in params.items() if v is not None}

            modelo = XGBClassifier(
                objective=objetivo,
                num_class=n_clases if n_clases > 2 else None,
                eval_metric="logloss",
                random_state=self.__random_state,
                n_jobs=-1,
                tree_method="hist",
                **params_limpios
            )
            return modelo

        raise ValueError("algoritmo debe ser 'AdaBoost' o 'XGBoost'.")

    def __armar_pipeline(self, X: pd.DataFrame, y_codificado: np.ndarray) -> Pipeline:
        preprocesador = self.__construir_preprocesamiento(X)
        modelo = self.__construir_modelo(y_codificado)
        pasos = [("preprocessor", preprocesador)]
        if self.__config.k_mejores is not None:
            pasos.append(("selector", SelectKBest(score_func=mutual_info_classif, k=self.__config.k_mejores)))
        pasos.append(("model", modelo))
        return Pipeline(pasos)

    def __obtener_scoring(self, y_codificado: np.ndarray):
        n_clases = len(np.unique(y_codificado))
        if n_clases == 2:
            return {
                "accuracy": "accuracy",
                "balanced_accuracy": "balanced_accuracy",
                "precision": "precision",
                "recall": "recall",
                "f1": "f1",
                "roc_auc": "roc_auc"
            }
        return {
            "accuracy": "accuracy",
            "balanced_accuracy": "balanced_accuracy",
            "precision": "precision_weighted",
            "recall": "recall_weighted",
            "f1": "f1_weighted"
        }

    def __extraer_feature_names(self):
        pre = self.__pipeline.named_steps["preprocessor"]
        nombres = pre.get_feature_names_out()
        if "selector" in self.__pipeline.named_steps:
            selector = self.__pipeline.named_steps["selector"]
            soporte = selector.get_support()
            nombres = np.array(nombres)[soporte]
        self.__feature_names = np.array(nombres)

    def __extraer_importancias(self):
        modelo = self.__pipeline.named_steps["model"]
        if hasattr(modelo, "feature_importances_"):
            importancias = modelo.feature_importances_
            self.__importancias = pd.DataFrame({
                "Variable": self.__feature_names,
                "Importancia": importancias
            }).sort_values("Importancia", ascending=False).reset_index(drop=True)
        else:
            self.__importancias = None

    def __ajustar(self):
        columnas_a_descartar = [c for c in self.__columnas_eliminar if c in self.__df.columns and c != self.__objetivo]
        df_trabajo = self.__df.drop(columns=columnas_a_descartar).copy()
        X = df_trabajo.drop(columns=[self.__objetivo]).copy()
        y = df_trabajo[self.__objetivo].copy()

        self.__label_encoder = LabelEncoder()
        y_codificado = self.__label_encoder.fit_transform(y)

        self.__X_train, self.__X_test, self.__y_train, self.__y_test = train_test_split(
            X, y_codificado,
            test_size=self.__test_size,
            random_state=self.__random_state,
            stratify=y_codificado
        )

        self.__pipeline = self.__armar_pipeline(X, y_codificado)
        cv = StratifiedKFold(n_splits=self.__cv_splits, shuffle=True, random_state=self.__random_state)
        scoring = self.__obtener_scoring(y_codificado)

        cv_resultados = cross_validate(
            clone(self.__pipeline), self.__X_train, self.__y_train,
            cv=cv, scoring=scoring, n_jobs=None, return_train_score=False
        )

        self.__cv_metricas = {
            f"cv_{k}": float(np.mean(v))
            for k, v in cv_resultados.items()
            if k.startswith("test_")
        }

        self.__pipeline.fit(self.__X_train, self.__y_train)
        self.__y_pred = self.__pipeline.predict(self.__X_test)

        if hasattr(self.__pipeline, "predict_proba"):
            self.__y_prob = self.__pipeline.predict_proba(self.__X_test)
        else:
            self.__y_prob = None

        self.__matriz_confusion = confusion_matrix(self.__y_test, self.__y_pred)
        self.__reporte_clasificacion = classification_report(
            self.__y_test, self.__y_pred,
            target_names=self.__label_encoder.classes_.astype(str),
            output_dict=True, zero_division=0
        )

        promedio = "binary" if len(np.unique(self.__y_test)) == 2 else "weighted"
        self.__metricas_test = {
            "test_accuracy": accuracy_score(self.__y_test, self.__y_pred),
            "test_balanced_accuracy": balanced_accuracy_score(self.__y_test, self.__y_pred),
            "test_precision": precision_score(self.__y_test, self.__y_pred, average=promedio, zero_division=0),
            "test_recall": recall_score(self.__y_test, self.__y_pred, average=promedio, zero_division=0),
            "test_f1": f1_score(self.__y_test, self.__y_pred, average=promedio, zero_division=0)
        }

        if self.__y_prob is not None and len(np.unique(self.__y_test)) == 2:
            self.__metricas_test["test_roc_auc"] = roc_auc_score(self.__y_test, self.__y_prob[:, 1])
        else:
            self.__metricas_test["test_roc_auc"] = np.nan

        self.__extraer_feature_names()
        self.__extraer_importancias()

    def resumen(self) -> pd.DataFrame:
        fila = {
            "Modelo": self.nombre,
            "Algoritmo": self.algoritmo,
            "Imputador": self.__config.imputador,
            "Escalador": self.__config.escalador if self.__config.escalador is not None else "None",
            "K mejores": self.__config.k_mejores if self.__config.k_mejores is not None else "Todas"
        }
        fila.update(self.__cv_metricas)
        fila.update(self.__metricas_test)
        return pd.DataFrame([fila])

    def top_importancias(self, top_n: int = 15) -> Optional[pd.DataFrame]:
        if self.__importancias is None:
            return None
        return self.__importancias.head(top_n).copy()

    def matriz_confusion_dataframe(self) -> pd.DataFrame:
        clases = self.__label_encoder.classes_.astype(str)
        return pd.DataFrame(
            self.__matriz_confusion,
            index=[f"Real_{c}" for c in clases],
            columns=[f"Pred_{c}" for c in clases]
        )

    def plot_matriz_confusion(self, guardar: Optional[str] = None):
        clases = self.__label_encoder.classes_.astype(str)
        fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
        disp = ConfusionMatrixDisplay(confusion_matrix=self.__matriz_confusion, display_labels=clases)
        disp.plot(ax=ax, cmap="Blues", colorbar=False)
        ax.set_title(f"Matriz de confusión - {self.nombre}")
        plt.tight_layout()
        if guardar:
            Path(guardar).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(guardar, bbox_inches="tight")
        plt.show()

    def plot_roc(self, guardar: Optional[str] = None):
        if self.__y_prob is None or len(np.unique(self.__y_test)) != 2:
            print("La curva ROC solo se genera para problemas binarios con predict_proba.")
            return
        fpr, tpr, _ = roc_curve(self.__y_test, self.__y_prob[:, 1])
        auc = self.__metricas_test["test_roc_auc"]
        fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
        ax.plot(fpr, tpr, linewidth=2, label=f"AUC = {auc:.4f}")
        ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
        ax.set_xlabel("Falsos positivos")
        ax.set_ylabel("Verdaderos positivos")
        ax.set_title(f"Curva ROC - {self.nombre}")
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        if guardar:
            Path(guardar).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(guardar, bbox_inches="tight")
        plt.show()

    def plot_importancias(self, top_n: int = 15, guardar: Optional[str] = None):
        if self.__importancias is None:
            print("Este modelo no expone importancias de variables.")
            return
        top = self.__importancias.head(top_n).sort_values("Importancia", ascending=True)
        fig, ax = plt.subplots(figsize=(9, 6), dpi=150)
        ax.barh(top["Variable"], top["Importancia"])
        ax.set_title(f"Importancia de variables - {self.nombre}")
        ax.set_xlabel("Importancia")
        ax.set_ylabel("Variable")
        ax.grid(axis="x", alpha=0.3)
        plt.tight_layout()
        if guardar:
            Path(guardar).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(guardar, bbox_inches="tight")
        plt.show()

    def __str__(self):
        return (
            f'{self.nombre} | {self.algoritmo} | '
            f'CV F1={self.__cv_metricas.get("cv_test_f1", np.nan):.4f} | '
            f'Test F1={self.__metricas_test.get("test_f1", np.nan):.4f}'
        )

    def __repr__(self):
        return self.__str__()


class BoostingExperimento:
    """
    Ejecuta multiples configuraciones de AdaBoost y XGBoost.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        columna_objetivo: str,
        nombre_dataset: str = "dataset",
        carpeta: Optional[str] = None,
        columnas_eliminar: Optional[List[str]] = None,
        test_size: float = 0.25,
        random_state: int = 42,
        cv_splits: int = 5
    ):
        self.__df = df.copy()
        self.__objetivo = columna_objetivo
        self.__nombre_dataset = nombre_dataset
        self.__carpeta = carpeta
        self.__columnas_eliminar = columnas_eliminar or []
        self.__test_size = test_size
        self.__random_state = random_state
        self.__cv_splits = cv_splits
        self.__modelos: Dict[str, ClasificacionBoostingAnalisis] = {}
        self.__mejor_nombre: Optional[str] = None

    @property
    def modelos(self):
        return self.__modelos

    @property
    def mejor_nombre(self):
        return self.__mejor_nombre

    @property
    def mejor_modelo(self):
        if self.__mejor_nombre is None:
            return None
        return self.__modelos[self.__mejor_nombre]

    def resumen_dataset(self) -> Dict[str, Any]:
        df = self.__df.copy()
        columnas_modelo = [c for c in df.columns if c not in self.__columnas_eliminar]
        df_modelo = df[columnas_modelo].copy()
        n = int(df_modelo.shape[0])
        p_total = int(df_modelo.shape[1])
        p_pred = int(p_total - 1)
        faltantes = int(df_modelo.isna().sum().sum())
        dist = df_modelo[self.__objetivo].value_counts(dropna=False)
        return {
            "n_observaciones": n,
            "n_variables_totales": p_total,
            "n_predictoras": p_pred,
            "faltantes_totales": faltantes,
            "variables_numericas": df_modelo.drop(columns=[self.__objetivo]).select_dtypes(include=["number"]).columns.tolist(),
            "variables_categoricas": df_modelo.drop(columns=[self.__objetivo]).select_dtypes(exclude=["number"]).columns.tolist(),
            "distribucion_objetivo": dist.to_dict()
        }

    def configuraciones_por_defecto(self) -> List[ConfiguracionModelo]:
        return [
            ConfiguracionModelo(
                nombre="AdaBoost_Standard", algoritmo="AdaBoost",
                parametros_modelo={"n_estimators": 100, "learning_rate": 1.0}
            ),
            ConfiguracionModelo(
                nombre="XGBoost_Standard", algoritmo="XGBoost",
                parametros_modelo={"n_estimators": 200, "learning_rate": 0.10, "max_depth": 4}
            )
        ]

    def ejecutar(self, configuraciones: Optional[List[ConfiguracionModelo]] = None):
        if configuraciones is None:
            configuraciones = self.configuraciones_por_defecto()
        print("=" * 70)
        print(f"EXPERIMENTO BOOSTING - {self.__nombre_dataset}")
        print("=" * 70)
        for config in configuraciones:
            print(f"\nAjustando: {config.nombre}")
            try:
                modelo = ClasificacionBoostingAnalisis(
                    df=self.__df, columna_objetivo=self.__objetivo,
                    configuracion=config, columnas_eliminar=self.__columnas_eliminar,
                    test_size=self.__test_size, random_state=self.__random_state, cv_splits=self.__cv_splits
                )
                self.__modelos[config.nombre] = modelo
                print(modelo)
            except Exception as e:
                print(f"  No se pudo ajustar {config.nombre}: {e}")
        if len(self.__modelos) == 0:
            raise RuntimeError("No se pudo ajustar ningun modelo.")
        self.__seleccionar_mejor()

    def __seleccionar_mejor(self):
        tabla = self.benchmark()
        columnas_orden = ["cv_test_f1", "cv_test_roc_auc", "test_f1"]
        tabla = tabla.sort_values(by=columnas_orden, ascending=[False, False, False]).reset_index(drop=True)
        self.__mejor_nombre = tabla.iloc[0]["Modelo"]

    def benchmark(self) -> pd.DataFrame:
        filas = [modelo.resumen().iloc[0].to_dict() for _, modelo in self.__modelos.items()]
        tabla = pd.DataFrame(filas)
        return tabla

    def analisis_mejor(self):
        modelo = self.mejor_modelo
        if modelo is None: return
        print("\n" + "=" * 70)
        print("ANALISIS DEL MEJOR MODELO")
        print("=" * 70)
        print(modelo)
        modelo.plot_matriz_confusion()

# =============================================================================
# SECCIÓN 2: RandomForest.py
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


class ClaseRandomForest():
    
    def __init__(self, dataframe, target_col='Attrition_Flag'):
        self.__df = dataframe.copy()
        self.__target = target_col
        self.__model = None
        self.__scaler = StandardScaler()
        self.__classes = None
        self.__best_model = None
        self.__best_config = None
        self.__X_train = None
        self.__X_test = None
        self.__y_test = None

    def preparar_datos(self, test_size=0.25):
        df = self.__df.copy()
        df[self.__target] = df[self.__target].map({
            "Existing Customer": 0, "Attrited Customer": 1
        })
        df = pd.get_dummies(df, drop_first=True)
        X = df.drop(columns=[self.__target])
        y = df[self.__target]
        self.__classes = list(np.unique(y))
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        X_train = pd.DataFrame(self.__scaler.fit_transform(X_train), columns=X.columns)
        X_test = pd.DataFrame(self.__scaler.transform(X_test), columns=X.columns)
        return X_train, X_test, y_train, y_test

    def entrenar(self, X_train, y_train, config):
        model = RandomForestClassifier(
            n_estimators=config["n_estimators"],
            max_depth=config["max_depth"],
            min_samples_split=config.get("min_samples_split", 2),
            random_state=42
        )
        model.fit(X_train, y_train)
        return model

    def evaluar(self, model, X_test, y_test):
        y_pred = model.predict(X_test)
        MC = confusion_matrix(y_test, y_pred)
        precision_global = np.sum(MC.diagonal()) / np.sum(MC)
        return precision_global, MC

    def probar_configuraciones(self, configuraciones):
        X_train, X_test, y_train, y_test = self.preparar_datos()
        resultados = []
        for config in configuraciones:
            model = self.entrenar(X_train, y_train, config)
            precision, MC = self.evaluar(model, X_test, y_test)
            resultados.append({"config": config, "precision": precision, "modelo": model})
            print(f"Config: {config} -> Precision: {precision:.4f}")
        mejor = max(resultados, key=lambda x: x["precision"])
        self.__best_model = mejor["modelo"]
        self.__best_config = mejor["config"]
        self.__X_train, self.__X_test, self.__y_test = X_train, X_test, y_test
        return resultados

    def importancia_variables(self):
        if self.__best_model is None: return
        importancias = self.__best_model.feature_importances_
        indices = np.argsort(importancias)[::-1]
        plt.figure(figsize=(10, 6))
        plt.title("Importancia de Variables (Mejor Modelo)")
        plt.bar(range(len(importancias)), importancias[indices])
        plt.xticks(range(len(importancias)), [self.__X_train.columns[i] for i in indices], rotation=45)
        plt.tight_layout()
        plt.show()

    def matriz_confusion(self, guardar=True, ruta="images/rf_cm_standard.png"):
        from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
        import os
        if self.__best_model is None: raise ValueError("Entrene el modelo primero.")
        y_pred = self.__best_model.predict(self.__X_test)
        cm = confusion_matrix(self.__y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        plt.title("Matriz de Confusión - Random Forest")
        if guardar:
            os.makedirs(os.path.dirname(ruta), exist_ok=True)
            plt.savefig(ruta)
        plt.show()
        return cm

# =============================================================================
# SECCIÓN 3: knn_analysis (1).py
# =============================================================================

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier

class Supervisado:
    def __init__(self, df):
        self.__df = df

    def preparar_datos(self, target_col='target', test_size=0.25, random_state=42):
        X = self.__df.drop(columns=[target_col])
        X = pd.DataFrame(StandardScaler().fit_transform(X), columns=X.columns)
        y = self.__df[target_col]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        return X_train, X_test, y_train, y_test, y

    def modeloKNN(self, X_train, y_train, n_neighbors=5, algorithm='auto', weights='uniform', metric='minkowski', p=2):
        model = KNeighborsClassifier(n_neighbors=n_neighbors, algorithm=algorithm, weights=weights, metric=metric, p=p)
        model.fit(X_train, y_train)
        return model

    def predecir(self, model, X_test):
        return model.predict(X_test)

    def evaluar(self, y_test, y_pred, y):
        MC = confusion_matrix(y_test, y_pred)
        return self.indices_general(MC, list(np.unique(y)))

    def indices_general(self, MC, nombres=None):
        precision_global = np.sum(MC.diagonal()) / np.sum(MC)
        error_global = 1 - precision_global
        precision_categoria = pd.DataFrame(MC.diagonal() / np.sum(MC, axis=1)).T
        if nombres is not None: precision_categoria.columns = nombres
        return {"Matriz de Confusión": MC, "Precisión Global": precision_global, "Error Global": error_global, "Precisión por categoría": precision_categoria}

    def KNN_standard(self, target_col='target', random_state=42):
        X_train, X_test, y_train, y_test, y = self.preparar_datos(target_col=target_col, random_state=random_state)
        model = self.modeloKNN(X_train, y_train, n_neighbors=5)
        y_pred = self.predecir(model, X_test)
        indices = self.evaluar(y_test, y_pred, y)
        return {'model': model, 'indices': indices, 'y_test': y_test, 'y_pred': y_pred}

    def KNN_variaciones(self, target_col='target', random_state=42):
        X_train, X_test, y_train, y_test, y = self.preparar_datos(target_col=target_col, random_state=random_state)
        resultados = []
        for k in range(1, 21):
            model = self.modeloKNN(X_train, y_train, n_neighbors=k)
            y_pred = self.predecir(model, X_test)
            indices = self.evaluar(y_test, y_pred, y)
            resultados.append({'k': k, 'precision_global': indices['Precisión Global'], 'y_pred': y_pred, 'model': model})
        df_results = pd.DataFrame([{'k': r['k'], 'precision_global': r['precision_global']} for r in resultados])
        return resultados, df_results, X_train, X_test, y_train, y_test, y

def preparar_dataset_knn(path):
    df = pd.read_csv(path)
    if 'ID' in df.columns: df = df.drop(columns=['ID'])
    if 'CLIENTNUM' in df.columns: df = df.drop(columns=['CLIENTNUM'])
    le_target = LabelEncoder()
    df['target'] = le_target.fit_transform(df['Attrition_Flag'])
    df = df.drop(columns=['Attrition_Flag'])
    cat_cols = df.select_dtypes(include=['object', 'string']).columns.tolist()
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
    return df, le_target

if __name__ == '__main__':
    print("\n--- INICIANDO ANÁLISIS KNN ---")
    try:
        df_knn, le_knn = preparar_dataset_knn('BankChurners.csv')
        sup_knn = Supervisado(df_knn)
        res_knn = sup_knn.KNN_standard(target_col='target')
        print(f"KNN Standard Precision: {res_knn['indices']['Precisión Global']:.4f}")
    except Exception as e:
        print(f"No se pudo completar KNN: {e}")

# =============================================================================
# SECCIÓN 4: dt_analysis (1).py
# =============================================================================

from sklearn.tree import DecisionTreeClassifier, plot_tree

# Redefinición de clase para Árboles (comportamiento específico de Árboles)
class SupervisadoDT:
    def __init__(self, df):
        self.__df = df

    def preparar_datos(self, target_col='target', test_size=0.25, random_state=42):
        X = self.__df.drop(columns=[target_col])
        y = self.__df[target_col]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
        return X_train, X_test, y_train, y_test, y

    def modeloDT(self, X_train, y_train, criterion='gini', max_depth=None):
        model = DecisionTreeClassifier(criterion=criterion, max_depth=max_depth, random_state=42)
        model.fit(X_train, y_train)
        return model

    def evaluar(self, y_test, y_pred, y):
        MC = confusion_matrix(y_test, y_pred)
        precision_global = np.sum(MC.diagonal()) / np.sum(MC)
        return {"Precisión Global": precision_global, "Matriz": MC}

if __name__ == '__main__':
    print("\n--- INICIANDO ANÁLISIS ÁRBOLES DE DECISIÓN ---")
    try:
        df_dt, le_dt = preparar_dataset_knn('BankChurners.csv')
        sup_dt = SupervisadoDT(df_dt)
        X_tr, X_te, y_tr, y_te, y_all = sup_dt.preparar_datos()
        model_dt = sup_dt.modeloDT(X_tr, y_tr, max_depth=5)
        y_p = model_dt.predict(X_te)
        eval_dt = sup_dt.evaluar(y_te, y_p, y_all)
        print(f"DT (depth=5) Precision: {eval_dt['Precisión Global']:.4f}")
    except Exception as e:
        print(f"No se pudo completar DT: {e}")

# =============================================================================
# SECCIÓN FINAL: Ejemplo de Boosting y RF (Integración)
# =============================================================================

if __name__ == '__main__':
    print("\n--- INICIANDO ANÁLISIS BOOSTING Y RANDOM FOREST ---")
    try:
        df_final = pd.read_csv('BankChurners.csv')
        # Boosting
        exp = BoostingExperimento(df_final, columna_objetivo='Attrition_Flag', columnas_eliminar=['CLIENTNUM', 'ID'])
        exp.ejecutar()
        exp.analisis_mejor()
        
        # Random Forest
        rf = ClaseRandomForest(df_final)
        rf.probar_configuraciones([{"n_estimators": 100, "max_depth": 10}])
        rf.matriz_confusion(guardar=False)
    except Exception as e:
        print(f"Error en ejecución final: {e}")