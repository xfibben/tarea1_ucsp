"""Modulo de preprocesamiento de datos."""

import pandas as pd
from pandas.api.types import is_string_dtype
from sklearn.preprocessing import StandardScaler


TARGET_COL = "y"


def load_data(path: str) -> pd.DataFrame:
    """Carga el dataset desde la ruta indicada."""
    return pd.read_csv(path, sep=None, engine="python")


def add_business_features(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega variables utiles para el caso de campanas bancarias."""
    df = df.copy()

    if "pdays" in df.columns:
        df["was_previously_contacted"] = (df["pdays"] != -1).astype(int)

    if {"housing", "loan"}.issubset(df.columns):
        df["has_any_loan"] = (
            (df["housing"].str.lower() == "yes")
            | (df["loan"].str.lower() == "yes")
        ).astype(int)

    if "balance" in df.columns:
        df["negative_balance"] = (df["balance"] < 0).astype(int)

    if {"campaign", "previous"}.issubset(df.columns):
        df["total_contacts"] = df["campaign"] + df["previous"]

    return df


def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Imputa valores nulos con mediana para numericas y moda para categoricas."""
    df = df.copy()

    for column in df.select_dtypes(include=["number"]).columns:
        df[column] = df[column].fillna(df[column].median())

    categorical_columns = df.select_dtypes(include=["object", "category"]).columns
    for column in categorical_columns:
        mode = df[column].mode()
        fill_value = mode.iloc[0] if not mode.empty else "unknown"
        df[column] = df[column].fillna(fill_value)

    return df


def encode_target(df: pd.DataFrame) -> pd.DataFrame:
    """Convierte la variable objetivo yes/no a 1/0."""
    df = df.copy()

    if TARGET_COL not in df.columns:
        raise ValueError(f"No existe la columna objetivo: {TARGET_COL}")

    target = df[TARGET_COL]
    if target.dtype == "object" or is_string_dtype(target):
        df[TARGET_COL] = target.str.strip().str.lower().map({"no": 0, "yes": 1})

    df[TARGET_COL] = df[TARGET_COL].astype(int)
    return df


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """Codifica variables categoricas con one-hot encoding."""
    categorical_columns = [
        column
        for column in df.select_dtypes(include=["object", "category"]).columns
        if column != TARGET_COL
    ]

    if not categorical_columns:
        return df.copy()

    return pd.get_dummies(df, columns=categorical_columns, drop_first=True, dtype=int)


def scale_features(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza variables numericas sin modificar la columna objetivo."""
    df = df.copy()
    numeric_columns = [
        column
        for column in df.select_dtypes(include=["number"]).columns
        if column != TARGET_COL
    ]

    if numeric_columns:
        scaler = StandardScaler()
        df[numeric_columns] = scaler.fit_transform(df[numeric_columns])

    return df


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Pipeline completo de preprocesamiento."""
    df = add_business_features(df)
    df = handle_missing(df)
    df = encode_target(df)
    df = encode_features(df)
    df = scale_features(df)
    return df
