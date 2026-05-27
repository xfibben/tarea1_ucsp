"""Preprocesamiento del dataset Bank Marketing para un pipeline E2E de ML."""

from __future__ import annotations

from dataclasses import dataclass, field
from inspect import signature

import pandas as pd
from pandas.api.types import is_string_dtype
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def read_raw_data(csv_path: str) -> pd.DataFrame:
    """Lee el archivo CSV detectando separadores comunes como coma o punto y coma."""
    return pd.read_csv(csv_path, sep=None, engine="python")


def load_data(path: str) -> pd.DataFrame:
    """Carga el dataset desde la ruta indicada."""
    return read_raw_data(path)


@dataclass
class BankMarketingPreprocessor:
    """Construye features limpias a partir de datos crudos de marketing bancario."""

    target_column: str = "y"
    columns_to_remove: tuple[str, ...] = ()
    fitted_transformer: ColumnTransformer | None = field(default=None, init=False)

    def transform_dataset(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """Devuelve un DataFrame numerico listo para monitoreo y entrenamiento."""
        self._validate_input(raw_data)

        engineered = self._add_domain_features(raw_data)
        target = self._encode_target(engineered[self.target_column])
        features = engineered.drop(columns=[self.target_column])

        transformer = self._build_transformer(features)
        transformed_values = transformer.fit_transform(features)
        feature_names = transformer.get_feature_names_out()

        processed = pd.DataFrame(
            transformed_values,
            columns=feature_names,
            index=raw_data.index,
        )
        processed[self.target_column] = target.to_numpy()
        self.fitted_transformer = transformer
        return processed

    def _validate_input(self, raw_data: pd.DataFrame) -> None:
        if self.target_column not in raw_data.columns:
            raise ValueError(f"No existe la columna objetivo: {self.target_column}")
        if raw_data.empty:
            raise ValueError("El dataset esta vacio; no se puede preprocesar.")

    def _encode_target(self, target: pd.Series) -> pd.Series:
        """Convierte la respuesta yes/no del banco a una etiqueta binaria."""
        if target.dtype == "object" or is_string_dtype(target):
            normalized = target.str.strip().str.lower()
            return normalized.map({"no": 0, "yes": 1}).astype(int)
        return target.astype(int)

    def _add_domain_features(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        data = raw_data.copy()

        if "pdays" in data.columns:
            data["was_previously_contacted"] = (data["pdays"] != -1).astype(int)

        if {"housing", "loan"}.issubset(data.columns):
            data["has_any_loan"] = (
                (data["housing"].str.lower() == "yes")
                | (data["loan"].str.lower() == "yes")
            ).astype(int)

        if "balance" in data.columns:
            data["negative_balance"] = (data["balance"] < 0).astype(int)

        if {"campaign", "previous"}.issubset(data.columns):
            data["total_contacts"] = data["campaign"] + data["previous"]

        removable = [col for col in self.columns_to_remove if col in data.columns]
        return data.drop(columns=removable)

    def _build_transformer(self, features: pd.DataFrame) -> ColumnTransformer:
        numeric_columns = features.select_dtypes(include=["number"]).columns.tolist()
        categorical_columns = [
            col for col in features.columns if col not in set(numeric_columns)
        ]

        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", _make_one_hot_encoder()),
            ]
        )

        return ColumnTransformer(
            transformers=[
                ("num", numeric_pipeline, numeric_columns),
                ("cat", categorical_pipeline, categorical_columns),
            ],
            remainder="drop",
            verbose_feature_names_out=False,
        )


def _make_one_hot_encoder() -> OneHotEncoder:
    """Crea un encoder compatible con versiones nuevas y antiguas de sklearn."""
    encoder_params = {"handle_unknown": "ignore"}
    if "sparse_output" in signature(OneHotEncoder).parameters:
        encoder_params["sparse_output"] = False
    else:
        encoder_params["sparse"] = False
    return OneHotEncoder(**encoder_params)


def preprocess(raw_data: pd.DataFrame) -> pd.DataFrame:
    """Funcion de conveniencia para ejecutar todo el preprocesamiento."""
    return BankMarketingPreprocessor().transform_dataset(raw_data)
