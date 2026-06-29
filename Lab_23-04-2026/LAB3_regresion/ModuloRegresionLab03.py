# -*- coding: utf-8 -*-
"""
ModuloRegresionLab03.py
=======================
Laboratorio 03 - Aprendizaje Supervisado para Regresión
Curso: EIF420O Inteligencia Artificial

Enfoque:
- Estudio comparativo de modelos de regresión supervisada.
- Configuración estándar + variaciones de preprocesamiento, selección de atributos e hiperparámetros.
- Selección automática del mejor modelo por validación cruzada.
- Evaluación final en conjunto de prueba.
- Gráficos y tablas para documentar el estudio de caso.

Modelos incluidos:
LinearRegression, Ridge, RidgeCV, Lasso, LassoCV, KNN Regressor, SVR,
DecisionTreeRegressor, RandomForestRegressor, AdaBoostRegressor,
GradientBoostingRegressor y XGBoostRegressor cuando el paquete xgboost está instalado.
"""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.model_selection import train_test_split, KFold, cross_validate
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV, Lasso, LassoCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.datasets import load_diabetes

try:
    from xgboost import XGBRegressor
    XGBOOST_DISPONIBLE = True
except Exception:
    XGBRegressor = None
    XGBOOST_DISPONIBLE = False


def cargar_diabetes_sklearn() -> pd.DataFrame:
    """Carga el dataset Diabetes de scikit-learn como DataFrame con target continuo."""
    data = load_diabetes(as_frame=True)
    df = data.frame.copy()
    df = df.rename(columns={"target": "target"})
    return df


def cargar_csv_o_diabetes(path: Optional[str] = None, columna_objetivo: str = "target") -> Tuple[pd.DataFrame, str]:
    """
    Carga un CSV si existe; si no existe, carga el dataset Diabetes de sklearn.

    Retorna:
        df, columna_objetivo
    """
    if path is not None and Path(path).exists():
        df = pd.read_csv(path)
        if columna_objetivo not in df.columns:
            candidatos = ["target", "Target", "DiseaseProgression", "y", "Y", "medv", "price"]
            for c in candidatos:
                if c in df.columns:
                    columna_objetivo = c
                    break
        if columna_objetivo not in df.columns:
            raise ValueError(f"No se encontró la columna objetivo '{columna_objetivo}' en el CSV.")
        return df, columna_objetivo
    return cargar_diabetes_sklearn(), "target"


def resumen_dataset(df: pd.DataFrame, columna_objetivo: str) -> Dict[str, Any]:
    """Devuelve una descripción compacta del dataset."""
    if columna_objetivo not in df.columns:
        raise ValueError(f"La columna objetivo '{columna_objetivo}' no existe.")
    X = df.drop(columns=[columna_objetivo])
    y = df[columna_objetivo]
    return {
        "n_observaciones": int(df.shape[0]),
        "n_variables_totales": int(df.shape[1]),
        "n_predictoras": int(X.shape[1]),
        "faltantes_totales": int(df.isna().sum().sum()),
        "variables_numericas": X.select_dtypes(include=[np.number]).columns.tolist(),
        "variables_categoricas": X.select_dtypes(exclude=[np.number]).columns.tolist(),
        "objetivo": columna_objetivo,
        "target_tipo": str(y.dtype),
        "target_min": float(pd.to_numeric(y, errors="coerce").min()),
        "target_max": float(pd.to_numeric(y, errors="coerce").max()),
        "target_media": float(pd.to_numeric(y, errors="coerce").mean()),
        "target_desv": float(pd.to_numeric(y, errors="coerce").std())
    }


def detectar_problema_regresion(df: pd.DataFrame, columna_objetivo: str) -> bool:
    """Valida de forma simple si la variable objetivo parece continua."""
    y = pd.to_numeric(df[columna_objetivo], errors="coerce")
    if y.isna().all():
        return False
    return y.nunique(dropna=True) > 10


@dataclass
class ConfiguracionRegresion:
    """Estructura para declarar una configuración experimental."""
    nombre: str
    algoritmo: str
    imputador: str = "median"
    escalador: Optional[str] = None
    k_mejores: Optional[int] = None
    parametros_modelo: Optional[Dict[str, Any]] = None


class RegresionAnalisis:
    """Ajusta, valida y evalúa una configuración de regresión."""

    def __init__(
        self,
        df: pd.DataFrame,
        columna_objetivo: str,
        configuracion: ConfiguracionRegresion,
        columnas_eliminar: Optional[List[str]] = None,
        test_size: float = 0.25,
        random_state: int = 42,
        cv_splits: int = 3,
        carpeta: Optional[str] = None
    ):
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df debe ser un DataFrame de pandas.")
        if columna_objetivo not in df.columns:
            raise ValueError(f"La columna objetivo '{columna_objetivo}' no existe.")

        self.__df = df.copy()
        self.__objetivo = columna_objetivo
        self.__config = configuracion
        self.__columnas_eliminar = columnas_eliminar or []
        self.__test_size = test_size
        self.__random_state = random_state
        self.__cv_splits = cv_splits
        self.__carpeta = carpeta

        self.__pipeline: Optional[Pipeline] = None
        self.__X_train = None
        self.__X_test = None
        self.__y_train = None
        self.__y_test = None
        self.__y_pred = None
        self.__cv_metricas: Dict[str, float] = {}
        self.__metricas_test: Dict[str, float] = {}
        self.__feature_names: Optional[np.ndarray] = None
        self.__importancias: Optional[pd.DataFrame] = None
        self.__coeficientes: Optional[pd.DataFrame] = None

        self.__ajustar()

    @property
    def nombre(self) -> str:
        return self.__config.nombre

    @property
    def algoritmo(self) -> str:
        return self.__config.algoritmo

    @property
    def pipeline(self) -> Pipeline:
        return self.__pipeline

    @property
    def metricas_test(self) -> Dict[str, float]:
        return self.__metricas_test

    @property
    def cv_metricas(self) -> Dict[str, float]:
        return self.__cv_metricas

    @property
    def y_test(self):
        return self.__y_test

    @property
    def y_pred(self):
        return self.__y_pred

    @property
    def feature_names(self):
        return self.__feature_names

    @property
    def importancias(self):
        return self.__importancias

    @property
    def coeficientes(self):
        return self.__coeficientes

    def __construir_preprocesamiento(self, X: pd.DataFrame) -> ColumnTransformer:
        columnas_numericas = X.select_dtypes(include=[np.number]).columns.tolist()
        columnas_categoricas = X.select_dtypes(exclude=[np.number]).columns.tolist()

        pasos_num = [("imputer", SimpleImputer(strategy=self.__config.imputador))]
        if self.__config.escalador == "standard":
            pasos_num.append(("scaler", StandardScaler()))
        elif self.__config.escalador == "minmax":
            pasos_num.append(("scaler", MinMaxScaler()))
        elif self.__config.escalador == "robust":
            pasos_num.append(("scaler", RobustScaler()))
        pipeline_num = Pipeline(pasos_num)

        transformadores = [("num", pipeline_num, columnas_numericas)]
        if len(columnas_categoricas) > 0:
            pipeline_cat = Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore"))
            ])
            transformadores.append(("cat", pipeline_cat, columnas_categoricas))

        return ColumnTransformer(transformadores)

    def __construir_modelo(self):
        params = self.__config.parametros_modelo.copy() if self.__config.parametros_modelo else {}
        algoritmo = self.__config.algoritmo.lower()

        if algoritmo == "linearregression":
            return LinearRegression(**params)
        if algoritmo == "ridge":
            return Ridge(random_state=self.__random_state, **params)
        if algoritmo == "ridgecv":
            params.setdefault("alphas", np.logspace(-4, 4, 30))
            return RidgeCV(**params)
        if algoritmo == "lasso":
            return Lasso(random_state=self.__random_state, max_iter=20000, **params)
        if algoritmo == "lassocv":
            return LassoCV(random_state=self.__random_state, max_iter=20000, cv=5, **params)
        if algoritmo == "knn":
            return KNeighborsRegressor(**params)
        if algoritmo == "svr":
            return SVR(**params)
        if algoritmo == "decisiontree":
            return DecisionTreeRegressor(random_state=self.__random_state, **params)
        if algoritmo == "randomforest":
            return RandomForestRegressor(random_state=self.__random_state, n_jobs=1, **params)
        if algoritmo == "adaboost":
            return AdaBoostRegressor(random_state=self.__random_state, **params)
        if algoritmo == "gradientboosting":
            return GradientBoostingRegressor(random_state=self.__random_state, **params)
        if algoritmo == "xgboost":
            if not XGBOOST_DISPONIBLE:
                raise ImportError("xgboost no está instalado. Use GradientBoostingRegressor como alternativa.")
            return XGBRegressor(random_state=self.__random_state, n_jobs=1, objective="reg:squarederror", **params)
        raise ValueError("Algoritmo no soportado.")

    def __armar_pipeline(self, X: pd.DataFrame) -> Pipeline:
        pasos = [("preprocessor", self.__construir_preprocesamiento(X))]
        if self.__config.k_mejores is not None:
            pasos.append(("selector", SelectKBest(score_func=f_regression, k=self.__config.k_mejores)))
        pasos.append(("model", self.__construir_modelo()))
        return Pipeline(pasos)

    @staticmethod
    def __rmse(y_true, y_pred) -> float:
        return float(np.sqrt(mean_squared_error(y_true, y_pred)))

    @staticmethod
    def __mape(y_true, y_pred) -> float:
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        mask = np.abs(y_true) > 1e-12
        return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)

    def __extraer_feature_names(self):
        pre = self.__pipeline.named_steps["preprocessor"]
        try:
            nombres = pre.get_feature_names_out()
        except Exception:
            nombres = np.array([f"x{i}" for i in range(self.__X_train.shape[1])])
        if "selector" in self.__pipeline.named_steps:
            selector = self.__pipeline.named_steps["selector"]
            nombres = np.array(nombres)[selector.get_support()]
        self.__feature_names = np.array([str(n).replace("num__", "").replace("cat__", "") for n in nombres])

    def __extraer_importancias_o_coeficientes(self):
        modelo = self.__pipeline.named_steps["model"]
        self.__importancias = None
        self.__coeficientes = None

        if hasattr(modelo, "feature_importances_"):
            self.__importancias = pd.DataFrame({
                "Variable": self.__feature_names,
                "Importancia": modelo.feature_importances_
            }).sort_values("Importancia", ascending=False).reset_index(drop=True)
        elif hasattr(modelo, "coef_"):
            coef = np.ravel(modelo.coef_)
            if len(coef) == len(self.__feature_names):
                self.__coeficientes = pd.DataFrame({
                    "Variable": self.__feature_names,
                    "Coeficiente": coef,
                    "AbsCoeficiente": np.abs(coef)
                }).sort_values("AbsCoeficiente", ascending=False).reset_index(drop=True)

    def __ajustar(self):
        columnas_a_descartar = [c for c in self.__columnas_eliminar if c in self.__df.columns and c != self.__objetivo]
        df_trabajo = self.__df.drop(columns=columnas_a_descartar).copy()
        X = df_trabajo.drop(columns=[self.__objetivo]).copy()
        y = pd.to_numeric(df_trabajo[self.__objetivo], errors="coerce")
        mask = y.notna()
        X = X.loc[mask].copy()
        y = y.loc[mask].copy()

        self.__X_train, self.__X_test, self.__y_train, self.__y_test = train_test_split(
            X, y, test_size=self.__test_size, random_state=self.__random_state
        )

        self.__pipeline = self.__armar_pipeline(X)
        cv = KFold(n_splits=self.__cv_splits, shuffle=True, random_state=self.__random_state)
        scoring = {
            "neg_rmse": "neg_root_mean_squared_error",
            "neg_mae": "neg_mean_absolute_error",
            "r2": "r2"
        }
        cv_resultados = cross_validate(
            clone(self.__pipeline), self.__X_train, self.__y_train,
            cv=cv, scoring=scoring, n_jobs=None, return_train_score=False
        )
        self.__cv_metricas = {
            "cv_rmse": float(-np.mean(cv_resultados["test_neg_rmse"])),
            "cv_mae": float(-np.mean(cv_resultados["test_neg_mae"])),
            "cv_r2": float(np.mean(cv_resultados["test_r2"]))
        }

        self.__pipeline.fit(self.__X_train, self.__y_train)
        self.__y_pred = self.__pipeline.predict(self.__X_test)

        self.__metricas_test = {
            "test_rmse": self.__rmse(self.__y_test, self.__y_pred),
            "test_mae": float(mean_absolute_error(self.__y_test, self.__y_pred)),
            "test_r2": float(r2_score(self.__y_test, self.__y_pred)),
            "test_mape": self.__mape(self.__y_test, self.__y_pred)
        }
        self.__extraer_feature_names()
        self.__extraer_importancias_o_coeficientes()

    def resumen(self) -> pd.DataFrame:
        fila = {
            "Modelo": self.nombre,
            "Algoritmo": self.algoritmo,
            "Imputador": self.__config.imputador,
            "Escalador": self.__config.escalador if self.__config.escalador else "None",
            "K mejores": self.__config.k_mejores if self.__config.k_mejores is not None else "Todas"
        }
        fila.update(self.__cv_metricas)
        fila.update(self.__metricas_test)
        return pd.DataFrame([fila])

    def predicciones_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame({"Real": np.asarray(self.__y_test), "Predicho": np.asarray(self.__y_pred), "Residuo": np.asarray(self.__y_test) - np.asarray(self.__y_pred)})

    def top_variables(self, top_n: int = 15) -> Optional[pd.DataFrame]:
        if self.__importancias is not None:
            return self.__importancias.head(top_n).copy()
        if self.__coeficientes is not None:
            return self.__coeficientes.head(top_n).copy()
        return None

    def plot_predicho_vs_real(self, guardar: Optional[str] = None):
        dfp = self.predicciones_dataframe()
        fig, ax = plt.subplots(figsize=(6.5, 5), dpi=150)
        ax.scatter(dfp["Real"], dfp["Predicho"], alpha=0.75)
        minimo = min(dfp["Real"].min(), dfp["Predicho"].min())
        maximo = max(dfp["Real"].max(), dfp["Predicho"].max())
        ax.plot([minimo, maximo], [minimo, maximo], linestyle="--")
        ax.set_title(f"Predicho vs. real - {self.nombre}")
        ax.set_xlabel("Valor real")
        ax.set_ylabel("Valor predicho")
        ax.grid(alpha=0.3)
        plt.tight_layout()
        if guardar:
            Path(guardar).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(guardar, bbox_inches="tight")
        plt.close(fig)

    def plot_residuos(self, guardar: Optional[str] = None):
        dfp = self.predicciones_dataframe()
        fig, ax = plt.subplots(figsize=(6.5, 5), dpi=150)
        ax.scatter(dfp["Predicho"], dfp["Residuo"], alpha=0.75)
        ax.axhline(0, linestyle="--")
        ax.set_title(f"Residuos - {self.nombre}")
        ax.set_xlabel("Valor predicho")
        ax.set_ylabel("Residuo")
        ax.grid(alpha=0.3)
        plt.tight_layout()
        if guardar:
            Path(guardar).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(guardar, bbox_inches="tight")
        plt.close(fig)

    def plot_top_variables(self, top_n: int = 15, guardar: Optional[str] = None):
        top = self.top_variables(top_n)
        if top is None or top.empty:
            return
        fig, ax = plt.subplots(figsize=(8.5, 6), dpi=150)
        if "Importancia" in top.columns:
            valores = top.sort_values("Importancia", ascending=True)
            ax.barh(valores["Variable"], valores["Importancia"])
            ax.set_xlabel("Importancia")
        else:
            valores = top.sort_values("AbsCoeficiente", ascending=True)
            ax.barh(valores["Variable"], valores["Coeficiente"])
            ax.set_xlabel("Coeficiente")
        ax.set_title(f"Variables relevantes - {self.nombre}")
        ax.grid(axis="x", alpha=0.3)
        plt.tight_layout()
        if guardar:
            Path(guardar).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(guardar, bbox_inches="tight")
        plt.close(fig)

    def __str__(self):
        return f"{self.nombre} | CV RMSE={self.__cv_metricas.get('cv_rmse', np.nan):.4f} | Test RMSE={self.__metricas_test.get('test_rmse', np.nan):.4f} | Test R2={self.__metricas_test.get('test_r2', np.nan):.4f}"

    def __repr__(self):
        return self.__str__()


class ExperimentoRegresion:
    """Ejecuta múltiples configuraciones, compara y selecciona el mejor modelo."""

    def __init__(
        self,
        df: pd.DataFrame,
        columna_objetivo: str,
        nombre_dataset: str = "Diabetes",
        carpeta: Optional[str] = None,
        columnas_eliminar: Optional[List[str]] = None,
        test_size: float = 0.25,
        random_state: int = 42,
        cv_splits: int = 3
    ):
        self.__df = df.copy()
        self.__objetivo = columna_objetivo
        self.__nombre_dataset = nombre_dataset
        self.__carpeta = carpeta
        self.__columnas_eliminar = columnas_eliminar or []
        self.__test_size = test_size
        self.__random_state = random_state
        self.__cv_splits = cv_splits
        self.__modelos: Dict[str, RegresionAnalisis] = {}
        self.__mejor_nombre: Optional[str] = None

    @property
    def modelos(self):
        return self.__modelos

    @property
    def mejor_nombre(self):
        return self.__mejor_nombre

    @property
    def mejor_modelo(self) -> Optional[RegresionAnalisis]:
        if self.__mejor_nombre is None:
            return None
        return self.__modelos[self.__mejor_nombre]

    def configuraciones_por_defecto(self) -> List[ConfiguracionRegresion]:
        configs = [
            ConfiguracionRegresion("Regresion_Lineal", "LinearRegression", escalador="standard"),
            ConfiguracionRegresion("Regresion_Lineal_SelectK", "LinearRegression", escalador="standard", k_mejores=6),
            ConfiguracionRegresion("Ridge_alpha_1", "Ridge", escalador="standard", parametros_modelo={"alpha": 1.0}),
            ConfiguracionRegresion("RidgeCV", "RidgeCV", escalador="standard"),
            ConfiguracionRegresion("Lasso_alpha_0_01", "Lasso", escalador="standard", parametros_modelo={"alpha": 0.01}),
            ConfiguracionRegresion("LassoCV", "LassoCV", escalador="standard"),
            ConfiguracionRegresion("KNN_k5", "KNN", escalador="standard", parametros_modelo={"n_neighbors": 5, "weights": "uniform", "metric": "minkowski", "p": 2}),
            ConfiguracionRegresion("KNN_k7_distance_manhattan", "KNN", escalador="standard", parametros_modelo={"n_neighbors": 7, "weights": "distance", "metric": "manhattan"}),
            ConfiguracionRegresion("SVR_RBF", "SVR", escalador="standard", parametros_modelo={"kernel": "rbf", "C": 10.0, "epsilon": 0.1, "gamma": "scale"}),
            ConfiguracionRegresion("SVR_RBF_Robust", "SVR", escalador="robust", parametros_modelo={"kernel": "rbf", "C": 100.0, "epsilon": 5.0, "gamma": "scale"}),
            ConfiguracionRegresion("Arbol_Standard", "DecisionTree", parametros_modelo={"max_depth": None, "min_samples_split": 2, "min_samples_leaf": 1}),
            ConfiguracionRegresion("Arbol_Podado", "DecisionTree", parametros_modelo={"max_depth": 4, "min_samples_split": 10, "min_samples_leaf": 5}),
            ConfiguracionRegresion("RandomForest_100", "RandomForest", parametros_modelo={"n_estimators": 100, "max_depth": None, "min_samples_leaf": 1}),
            ConfiguracionRegresion("RandomForest_Regularizado", "RandomForest", parametros_modelo={"n_estimators": 120, "max_depth": 5, "min_samples_leaf": 4, "max_features": "sqrt"}),
            ConfiguracionRegresion("AdaBoost_Standard", "AdaBoost", parametros_modelo={"n_estimators": 100, "learning_rate": 0.05, "loss": "linear"}),
            ConfiguracionRegresion("AdaBoost_Profundo", "AdaBoost", parametros_modelo={"estimator": DecisionTreeRegressor(max_depth=3, random_state=self.__random_state), "n_estimators": 80, "learning_rate": 0.05, "loss": "linear"}),
            ConfiguracionRegresion("GradientBoosting", "GradientBoosting", parametros_modelo={"n_estimators": 120, "learning_rate": 0.05, "max_depth": 2, "subsample": 0.9})
        ]
        if XGBOOST_DISPONIBLE:
            configs.extend([
                ConfiguracionRegresion("XGBoost_Standard", "XGBoost", parametros_modelo={"n_estimators": 120, "learning_rate": 0.05, "max_depth": 2, "subsample": 0.9, "colsample_bytree": 0.9, "reg_lambda": 1.0}),
                ConfiguracionRegresion("XGBoost_Regularizado", "XGBoost", parametros_modelo={"n_estimators": 150, "learning_rate": 0.03, "max_depth": 2, "min_child_weight": 3, "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 2.0})
            ])
        return configs

    def ejecutar(self, configuraciones: Optional[List[ConfiguracionRegresion]] = None, verbose: bool = True):
        if configuraciones is None:
            configuraciones = self.configuraciones_por_defecto()
        if verbose:
            print("=" * 72)
            print(f"EXPERIMENTO DE REGRESION - {self.__nombre_dataset}")
            print("=" * 72)
        for config in configuraciones:
            if verbose:
                print(f"\nAjustando: {config.nombre}")
            try:
                modelo = RegresionAnalisis(
                    df=self.__df,
                    columna_objetivo=self.__objetivo,
                    configuracion=config,
                    columnas_eliminar=self.__columnas_eliminar,
                    test_size=self.__test_size,
                    random_state=self.__random_state,
                    cv_splits=self.__cv_splits,
                    carpeta=self.__carpeta
                )
                self.__modelos[config.nombre] = modelo
                if verbose:
                    print(modelo)
            except Exception as e:
                if verbose:
                    print(f"  No se pudo ajustar {config.nombre}: {e}")
        if not self.__modelos:
            raise RuntimeError("No se pudo ajustar ningún modelo.")
        self.__seleccionar_mejor()
        if verbose:
            print("\n" + "=" * 72)
            print(f"MEJOR MODELO: {self.__mejor_nombre}")
            print(self.mejor_modelo)
            print("=" * 72)

    def __seleccionar_mejor(self):
        tabla = self.benchmark()
        tabla = tabla.sort_values(by=["cv_rmse", "test_rmse", "cv_r2"], ascending=[True, True, False]).reset_index(drop=True)
        self.__mejor_nombre = str(tabla.iloc[0]["Modelo"])

    def benchmark(self) -> pd.DataFrame:
        filas = [m.resumen().iloc[0].to_dict() for m in self.__modelos.values()]
        tabla = pd.DataFrame(filas)
        if tabla.empty:
            return tabla
        columnas = ["Modelo", "Algoritmo", "Escalador", "K mejores", "cv_rmse", "cv_mae", "cv_r2", "test_rmse", "test_mae", "test_r2", "test_mape"]
        columnas = [c for c in columnas if c in tabla.columns]
        return tabla[columnas].sort_values(by=["cv_rmse", "test_rmse"], ascending=[True, True]).reset_index(drop=True)

    def mejores_por_algoritmo(self) -> pd.DataFrame:
        tabla = self.benchmark()
        if tabla.empty:
            return tabla
        return tabla.sort_values("cv_rmse").groupby("Algoritmo", as_index=False).first().sort_values("cv_rmse")

    def guardar_resultados(self, carpeta: str):
        path = Path(carpeta)
        path.mkdir(parents=True, exist_ok=True)
        self.benchmark().to_csv(path / "benchmark_regresion.csv", index=False)
        self.mejores_por_algoritmo().to_csv(path / "mejores_por_algoritmo.csv", index=False)
        if self.mejor_modelo is not None:
            self.mejor_modelo.predicciones_dataframe().to_csv(path / "predicciones_mejor_modelo.csv", index=False)
            top = self.mejor_modelo.top_variables(20)
            if top is not None:
                top.to_csv(path / "top_variables_mejor_modelo.csv", index=False)

    def plot_benchmark_rmse(self, guardar: Optional[str] = None, top_n: int = 15):
        tabla = self.benchmark().head(top_n).sort_values("test_rmse", ascending=True)
        fig, ax = plt.subplots(figsize=(9, 6), dpi=150)
        ax.barh(tabla["Modelo"], tabla["test_rmse"])
        ax.set_title("Comparación de modelos por RMSE de prueba")
        ax.set_xlabel("RMSE de prueba")
        ax.grid(axis="x", alpha=0.3)
        plt.tight_layout()
        if guardar:
            Path(guardar).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(guardar, bbox_inches="tight")
        plt.close(fig)

    def plot_benchmark_r2(self, guardar: Optional[str] = None, top_n: int = 15):
        tabla = self.benchmark().head(top_n).sort_values("test_r2", ascending=True)
        fig, ax = plt.subplots(figsize=(9, 6), dpi=150)
        ax.barh(tabla["Modelo"], tabla["test_r2"])
        ax.set_title("Comparación de modelos por R² de prueba")
        ax.set_xlabel("R² de prueba")
        ax.grid(axis="x", alpha=0.3)
        plt.tight_layout()
        if guardar:
            Path(guardar).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(guardar, bbox_inches="tight")
        plt.close(fig)

    def plot_familias(self, guardar: Optional[str] = None):
        tabla = self.mejores_por_algoritmo().sort_values("test_rmse", ascending=True)
        fig, ax = plt.subplots(figsize=(9, 5.5), dpi=150)
        ax.barh(tabla["Algoritmo"], tabla["test_rmse"])
        ax.set_title("Mejor configuración por familia de algoritmo")
        ax.set_xlabel("RMSE de prueba")
        ax.grid(axis="x", alpha=0.3)
        plt.tight_layout()
        if guardar:
            Path(guardar).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(guardar, bbox_inches="tight")
        plt.close(fig)

    def analisis_mejor(self, carpeta_figuras: Optional[str] = None):
        if self.mejor_modelo is None:
            print("Ejecute .ejecutar() primero.")
            return
        print("\n" + "=" * 72)
        print("ANALISIS DEL MEJOR MODELO")
        print("=" * 72)
        print(self.mejor_modelo)
        print("\nMétricas de prueba:")
        print(pd.DataFrame([self.mejor_modelo.metricas_test]).round(4))
        print("\nTop variables:")
        top = self.mejor_modelo.top_variables(15)
        print(top.round(4) if top is not None else "No disponible")
        if carpeta_figuras:
            p = Path(carpeta_figuras)
            p.mkdir(parents=True, exist_ok=True)
            self.mejor_modelo.plot_predicho_vs_real(str(p / "predicho_vs_real_mejor.png"))
            self.mejor_modelo.plot_residuos(str(p / "residuos_mejor.png"))
            self.mejor_modelo.plot_top_variables(15, str(p / "top_variables_mejor.png"))


def graficar_eda_basica(df: pd.DataFrame, columna_objetivo: str, carpeta: str):
    """Genera gráficos mínimos de EDA para el reporte."""
    p = Path(carpeta)
    p.mkdir(parents=True, exist_ok=True)
    y = pd.to_numeric(df[columna_objetivo], errors="coerce")

    fig, ax = plt.subplots(figsize=(7, 5), dpi=150)
    ax.hist(y.dropna(), bins=25, edgecolor="black", alpha=0.75)
    ax.set_title(f"Distribución de la variable objetivo: {columna_objetivo}")
    ax.set_xlabel(columna_objetivo)
    ax.set_ylabel("Frecuencia")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(p / "distribucion_target.png", bbox_inches="tight")
    plt.close(fig)

    num = df.select_dtypes(include=[np.number])
    if num.shape[1] >= 2:
        corr = num.corr(numeric_only=True)
        fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
        im = ax.imshow(corr.values, aspect="auto", vmin=-1, vmax=1)
        ax.set_xticks(range(len(corr.columns)))
        ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(corr.index)))
        ax.set_yticklabels(corr.index, fontsize=8)
        ax.set_title("Matriz de correlación de variables numéricas")
        fig.colorbar(im, ax=ax, shrink=0.85)
        plt.tight_layout()
        plt.savefig(p / "correlacion_numericas.png", bbox_inches="tight")
        plt.close(fig)


def ejecutar_lab03_demo(carpeta_salida: str = "outputs_lab03") -> ExperimentoRegresion:
    """Ejecuta el laboratorio con el dataset Diabetes de sklearn y guarda resultados."""
    carpeta = Path(carpeta_salida)
    fig = carpeta / "figures"
    res = carpeta / "results"
    df = cargar_diabetes_sklearn()
    graficar_eda_basica(df, "target", str(fig))
    exp = ExperimentoRegresion(df, "target", nombre_dataset="Diabetes sklearn", carpeta=str(carpeta))
    exp.ejecutar(verbose=True)
    exp.guardar_resultados(str(res))
    exp.plot_benchmark_rmse(str(fig / "benchmark_rmse.png"))
    exp.plot_benchmark_r2(str(fig / "benchmark_r2.png"))
    exp.plot_familias(str(fig / "benchmark_familias.png"))
    exp.analisis_mejor(str(fig))
    return exp


if __name__ == "__main__":
    ejecutar_lab03_demo("outputs_lab03")
