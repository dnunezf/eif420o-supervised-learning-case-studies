from __future__ import annotations

from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from ..supervisado_base import SupervisadoBase


class ClasificacionModelos(SupervisadoBase):
    def knn(self, n_neighbors: int = 5, algorithm: str = "auto"):
        model = KNeighborsClassifier(n_neighbors=n_neighbors, algorithm=algorithm)
        model, split, y_pred = self._fit_predict(model)
        return {"model": model, "metricas": self.evaluar_clasificacion(split.y_test, y_pred)}

    def decision_tree(self, min_samples_split: int = 2, max_depth: int | None = None):
        model = DecisionTreeClassifier(min_samples_split=min_samples_split, max_depth=max_depth)
        model, split, y_pred = self._fit_predict(model)
        return {"model": model, "metricas": self.evaluar_clasificacion(split.y_test, y_pred)}

    def random_forest(
        self,
        n_estimators: int = 200,
        min_samples_split: int = 2,
        max_depth: int | None = None,
    ):
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            min_samples_split=min_samples_split,
            max_depth=max_depth,
            random_state=self.random_state,
        )
        model, split, y_pred = self._fit_predict(model)
        return {"model": model, "metricas": self.evaluar_clasificacion(split.y_test, y_pred)}

    def gradient_boosting(self, n_estimators: int = 120, max_depth: int = 2):
        model = GradientBoostingClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=self.random_state)
        model, split, y_pred = self._fit_predict(model)
        return {"model": model, "metricas": self.evaluar_clasificacion(split.y_test, y_pred)}

    def adaboost(self, n_estimators: int = 80):
        model = AdaBoostClassifier(n_estimators=n_estimators, random_state=self.random_state)
        model, split, y_pred = self._fit_predict(model)
        return {"model": model, "metricas": self.evaluar_clasificacion(split.y_test, y_pred)}

    def adaboost_grid(self, param_grid: dict):
        split = self.prepare_data()
        base = AdaBoostClassifier(random_state=self.random_state)
        grid = GridSearchCV(base, param_grid, cv=5, scoring="accuracy", n_jobs=-1)
        grid.fit(split.X_train, split.y_train)
        y_pred = grid.predict(split.X_test)
        return {
            "model": grid.best_estimator_,
            "best_params": grid.best_params_,
            "best_cv_score": float(grid.best_score_),
            "metricas": self.evaluar_clasificacion(split.y_test, y_pred),
        }

    def comparar_basico(self):
        return {
            "knn": self.knn(),
            "decision_tree": self.decision_tree(),
            "random_forest": self.random_forest(),
            "gradient_boosting": self.gradient_boosting(),
            "adaboost": self.adaboost(),
        }
