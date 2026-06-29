from __future__ import annotations

from typing import Any

from src.eda.eda import EDAEngine, SplitData
from src.shared.metricas.metricas_supervisado import metricas_clasificacion, metricas_regresion


class SupervisadoBase(EDAEngine):
    """Shared logic for classification and regression models."""

    def __init__(
        self,
        df,
        target: str,
        test_size: float = 0.25,
        random_state: int = 42,
        scale: bool = False,
        stratify: bool = False,
    ) -> None:
        super().__init__(df=df, target=target)
        self.test_size = test_size
        self.random_state = random_state
        self.scale = scale
        self.stratify = stratify
        self._split_key = f"ts={test_size}|rs={random_state}|scale={scale}|strat={stratify}"

    def prepare_data(self, force: bool = False) -> SplitData:
        if force:
            self._cache.pop(f"split::{self._split_key}", None)
        return self.split_supervised(
            target=self.target,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=self.stratify,
            use_scaled=self.scale,
            cache_key=self._split_key,
        )

    def _fit_predict(self, model: Any) -> tuple[Any, Any, Any]:
        split = self.prepare_data()
        model.fit(split.X_train, split.y_train)
        y_pred = model.predict(split.X_test)
        return model, split, y_pred

    def evaluar_clasificacion(self, y_true, y_pred) -> dict[str, Any]:
        return metricas_clasificacion(y_true, y_pred)

    def evaluar_regresion(self, y_true, y_pred) -> dict[str, float]:
        return metricas_regresion(y_true, y_pred)

