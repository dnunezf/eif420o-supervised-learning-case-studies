from __future__ import annotations

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, LassoCV, LinearRegression, Ridge, RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

from ..supervisado_base import SupervisadoBase


class RegresionModelos(SupervisadoBase):
    def __init__(
        self,
        df,
        target: str,
        test_size: float = 0.25,
        random_state: int = 42,
        scale: bool = True,
    ) -> None:
        super().__init__(df=df, target=target, test_size=test_size, random_state=random_state, scale=scale, stratify=False)

    def lineal(self):
        model = LinearRegression()
        model, split, y_pred = self._fit_predict(model)
        return {"model": model, "metricas": self.evaluar_regresion(split.y_test, y_pred)}

    def lasso(self, alpha: float = 0.1):
        model = Lasso(alpha=alpha, random_state=self.random_state)
        model, split, y_pred = self._fit_predict(model)
        return {"model": model, "metricas": self.evaluar_regresion(split.y_test, y_pred)}

    def lasso_cv(self, alphas=None, cv: int = 10):
        if alphas is None:
            alphas = np.logspace(-6, 3, 80)
        split = self.prepare_data()
        selector = LassoCV(alphas=alphas, cv=cv)
        selector.fit(split.X_train, split.y_train)
        model = Lasso(alpha=float(selector.alpha_), random_state=self.random_state)
        model.fit(split.X_train, split.y_train)
        y_pred = model.predict(split.X_test)
        return {
            "model": model,
            "best_alpha": float(selector.alpha_),
            "metricas": self.evaluar_regresion(split.y_test, y_pred),
        }

    def ridge(self, alpha: float = 1.0):
        model = Ridge(alpha=alpha, random_state=self.random_state)
        model, split, y_pred = self._fit_predict(model)
        return {"model": model, "metricas": self.evaluar_regresion(split.y_test, y_pred)}

    def ridge_cv(self, alphas=None, cv: int = 10):
        if alphas is None:
            alphas = np.logspace(-10, 2, 80)
        split = self.prepare_data()
        selector = RidgeCV(alphas=alphas, cv=cv)
        selector.fit(split.X_train, split.y_train)
        model = Ridge(alpha=float(selector.alpha_), random_state=self.random_state)
        model.fit(split.X_train, split.y_train)
        y_pred = model.predict(split.X_test)
        return {
            "model": model,
            "best_alpha": float(selector.alpha_),
            "metricas": self.evaluar_regresion(split.y_test, y_pred),
        }

    def svr(self):
        split = self.prepare_data()
        model = make_pipeline(StandardScaler(), SVR(kernel="rbf", C=30, epsilon=0.1))
        model.fit(split.X_train, split.y_train)
        y_pred = model.predict(split.X_test)
        return {"model": model, "metricas": self.evaluar_regresion(split.y_test, y_pred)}

    def decision_tree(self, max_depth: int = 4):
        model = DecisionTreeRegressor(max_depth=max_depth, random_state=self.random_state)
        model, split, y_pred = self._fit_predict(model)
        return {"model": model, "metricas": self.evaluar_regresion(split.y_test, y_pred)}

    def random_forest(self, n_estimators: int = 250, max_depth: int | None = None):
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=self.random_state,
            n_jobs=-1,
        )
        model, split, y_pred = self._fit_predict(model)
        return {"model": model, "metricas": self.evaluar_regresion(split.y_test, y_pred)}

    def gradient_boosting(self, n_estimators: int = 400, max_depth: int = 3):
        model = GradientBoostingRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=self.random_state,
        )
        model, split, y_pred = self._fit_predict(model)
        return {"model": model, "metricas": self.evaluar_regresion(split.y_test, y_pred)}

    def xgboost(self):
        try:
            from xgboost import XGBRegressor
        except Exception as exc:
            raise ImportError("Install 'xgboost' to use XGBoostRegressor.") from exc

        model = XGBRegressor(
            n_estimators=160,
            learning_rate=0.05,
            max_depth=3,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            random_state=self.random_state,
            n_jobs=1,
        )
        model, split, y_pred = self._fit_predict(model)
        return {"model": model, "metricas": self.evaluar_regresion(split.y_test, y_pred)}

    def comparar_basico(self):
        modelos = {
            "lineal": self.lineal(),
            "lasso": self.lasso(),
            "lasso_cv": self.lasso_cv(),
            "ridge": self.ridge(),
            "ridge_cv": self.ridge_cv(),
            "svr": self.svr(),
            "arbol": self.decision_tree(),
            "bosque": self.random_forest(),
            "boosting": self.gradient_boosting(),
        }
        try:
            modelos["xgboost"] = self.xgboost()
        except Exception:
            pass
        return modelos
