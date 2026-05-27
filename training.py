"""Modulo de entrenamiento y evaluacion del modelo."""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split


TARGET_COL = "y"


def split_data(df: pd.DataFrame):
    """Divide en features (X) y etiqueta (y), luego train/test."""
    x = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL].astype(int)
    return train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )


def build_model():
    """Instancia y retorna el modelo."""
    return RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )


def train(df: pd.DataFrame):
    """Entrena el modelo y retorna (modelo, metricas)."""
    x_train, x_test, y_train, y_test = split_data(df)
    model = build_model()
    model.fit(x_train, y_train)
    metrics = {"X_test": x_test, "y_test": y_test}
    return model, metrics


def evaluate(model, metrics: dict) -> None:
    """Imprime el reporte de clasificacion."""
    y_pred = model.predict(metrics["X_test"])

    print("\nENTRENAMIENTO Y EVALUACION")
    print("-" * 42)
    print(f"precision clase positiva: {precision_score(metrics['y_test'], y_pred):.4f}")
    print(f"recall clase positiva: {recall_score(metrics['y_test'], y_pred):.4f}")
    print(f"f1-score clase positiva: {f1_score(metrics['y_test'], y_pred):.4f}")
    print("\nREPORTE DE EVALUACION")
    print(classification_report(metrics["y_test"], y_pred))
