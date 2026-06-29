from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


@dataclass
class SplitData:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series


class EDAEngine:
    """Base layer: load, clean, encode, scale and cache reusable data objects."""

    def __init__(
        self,
        df: pd.DataFrame | None = None,
        path: str | None = None,
        sep: str = ",",
        decimal: str = ".",
        index_col: int | None = None,
        target: str | None = None,
    ) -> None:
        if df is not None:
            self._df = df.copy()
        elif path is not None:
            self._df = pd.read_csv(path, sep=sep, decimal=decimal, index_col=index_col)
        else:
            self._df = pd.DataFrame()

        self._target = target
        self._cache: dict[str, Any] = {}

    @staticmethod
    def _as_model_matrix(data: pd.DataFrame) -> pd.DataFrame:
        """Return a fully numeric feature matrix ready for sklearn models."""
        X = data.copy()
        if X.empty:
            raise ValueError("No hay columnas de entrada para construir la matriz de features.")

        # Normalize string-like values early (trim spaces from CSVs).
        string_cols = X.select_dtypes(include=["object", "string", "category"]).columns
        for col in string_cols:
            X[col] = X[col].astype(str).str.strip()

        # Try to recover numeric columns that came as text.
        for col in X.columns:
            if pd.api.types.is_object_dtype(X[col]) or pd.api.types.is_string_dtype(X[col]):
                maybe_numeric = pd.to_numeric(X[col], errors="coerce")
                if maybe_numeric.notna().sum() >= max(1, int(0.8 * X[col].notna().sum())):
                    X[col] = maybe_numeric

        numeric_cols = X.select_dtypes(include=["number", "bool"]).columns.tolist()
        cat_cols = [c for c in X.columns if c not in numeric_cols]

        for col in numeric_cols:
            if X[col].isna().any():
                X[col] = X[col].fillna(X[col].median())

        for col in cat_cols:
            if X[col].isna().any():
                mode = X[col].mode(dropna=True)
                fill_value = mode.iloc[0] if not mode.empty else "missing"
                X[col] = X[col].fillna(fill_value)

        if cat_cols:
            X = pd.get_dummies(X, columns=cat_cols, drop_first=False, dtype=float)

        bool_cols = X.select_dtypes(include=["bool"]).columns
        for col in bool_cols:
            X[col] = X[col].astype(int)

        if X.shape[1] == 0:
            raise ValueError("No se pudieron generar features numéricas con los datos actuales.")
        return X

    @property
    def df(self) -> pd.DataFrame:
        return self._df

    @property
    def target(self) -> str | None:
        return self._target

    @target.setter
    def target(self, value: str | None) -> None:
        self._target = value

    def set_dataframe(self, df: pd.DataFrame) -> None:
        self._df = df.copy()
        self._cache.clear()

    def load_csv(
        self,
        path: str,
        sep: str = ",",
        decimal: str = ".",
        index_col: int | None = None,
    ) -> pd.DataFrame:
        self._df = pd.read_csv(path, sep=sep, decimal=decimal, index_col=index_col)
        self._cache.clear()
        return self._df

    def get_df(self, copy: bool = True) -> pd.DataFrame:
        return self._df.copy() if copy else self._df

    def drop_columns(self, columns: list[str]) -> pd.DataFrame:
        self._df = self._df.drop(columns=columns, errors="ignore")
        self._cache.clear()
        return self._df

    def rename_columns(self, mapping: dict[str, str]) -> pd.DataFrame:
        self._df = self._df.rename(columns=mapping)
        self._cache.clear()
        return self._df

    def drop_duplicates(self) -> pd.DataFrame:
        self._df = self._df.drop_duplicates()
        self._cache.clear()
        return self._df

    def drop_na(self) -> pd.DataFrame:
        self._df = self._df.dropna()
        self._cache.clear()
        return self._df

    def to_numeric_only(self) -> pd.DataFrame:
        self._df = self._df.select_dtypes(include=["number"])
        self._cache.clear()
        return self._df

    def encode_categoricals(self, drop_first: bool = True) -> pd.DataFrame:
        categoricals = self._df.select_dtypes(include=["object", "category"]).columns.tolist()
        if categoricals:
            self._df = pd.get_dummies(self._df, columns=categoricals, drop_first=drop_first)
        self._cache.clear()
        return self._df

    def get_feature_matrix(
        self,
        use_scaled: bool = False,
        target: str | None = None,
        scaled_key: str = "scaled",
    ) -> pd.DataFrame:
        target_name = target or self._target
        if target_name and target_name in self._df.columns:
            X = self._df.drop(columns=[target_name])
        else:
            X = self._df

        X = self._as_model_matrix(X)
        if use_scaled:
            return self.scale_numeric(cache_key=scaled_key, source=X)
        return X

    def get_target(self, target: str | None = None) -> pd.Series:
        target_name = target or self._target
        if target_name is None:
            raise ValueError("Target column is not defined.")
        if target_name not in self._df.columns:
            raise KeyError(f"Target column '{target_name}' does not exist.")
        return self._df[target_name]

    def scale_numeric(
        self,
        cache_key: str = "scaled",
        source: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        key = f"scale::{cache_key}"
        if source is None and key in self._cache:
            return self._cache[key]

        raw = source if source is not None else self._df
        data = self._as_model_matrix(raw)
        scaler = StandardScaler()
        scaled = pd.DataFrame(scaler.fit_transform(data), columns=data.columns, index=data.index)

        if source is None:
            self._cache[key] = scaled
            self._cache[f"{key}::scaler"] = scaler
        return scaled

    def split_supervised(
        self,
        target: str | None = None,
        test_size: float = 0.25,
        random_state: int = 42,
        stratify: bool = False,
        use_scaled: bool = False,
        cache_key: str = "default",
    ) -> SplitData:
        key = f"split::{cache_key}"
        if key in self._cache:
            return self._cache[key]

        y = self.get_target(target=target)
        X = self.get_feature_matrix(use_scaled=use_scaled, target=target)

        valid_mask = ~y.isna()
        X = X.loc[valid_mask]
        y = y.loc[valid_mask]
        if X.empty or y.empty:
            raise ValueError("No hay datos válidos después de remover filas con target nulo.")

        stratify_arg = None
        if stratify:
            counts = y.value_counts(dropna=False)
            if counts.size > 1 and counts.min() >= 2:
                stratify_arg = y

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify_arg,
        )

        split = SplitData(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
        self._cache[key] = split
        return split

    def summary(self) -> dict[str, Any]:
        return {
            "shape": self._df.shape,
            "columns": self._df.columns.tolist(),
            "dtypes": self._df.dtypes.astype(str).to_dict(),
            "missing": self._df.isna().sum().to_dict(),
        }
