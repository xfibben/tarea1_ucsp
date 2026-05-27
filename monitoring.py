"""Validaciones estadisticas para datos raw y datos transformados."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from scipy.stats import kstest, ks_2samp


@dataclass(frozen=True)
class AuditResult:
    """Resultado resumido de monitoreo para una version del dataset."""

    name: str
    rows: int
    columns: int
    duplicated_rows: int
    missing_values: int
    alerts: list[str]


def audit_raw_data(raw_data: pd.DataFrame, target_column: str = "y") -> AuditResult:
    """Ejecuta controles de calidad sobre el dataset sin transformar."""
    alerts = []
    missing_by_column = raw_data.isna().sum()

    for column, missing_count in missing_by_column.items():
        if missing_count > 0:
            alerts.append(f"{column}: {missing_count} valores nulos")

    if target_column in raw_data.columns:
        target_share = raw_data[target_column].value_counts(normalize=True).to_dict()
        minority_share = min(target_share.values())
        if minority_share < 0.25:
            alerts.append(f"{target_column}: conversion baja ({minority_share:.2%})")

    duplicated_rows = int(raw_data.duplicated().sum())
    if duplicated_rows:
        alerts.append(f"filas duplicadas detectadas: {duplicated_rows}")

    return AuditResult(
        name="raw",
        rows=len(raw_data),
        columns=len(raw_data.columns),
        duplicated_rows=duplicated_rows,
        missing_values=int(missing_by_column.sum()),
        alerts=alerts,
    )


def audit_processed_data(
    processed_data: pd.DataFrame,
    target_column: str = "y",
) -> AuditResult:
    """Valida que el dataset procesado sea apto para entrenar."""
    alerts = []
    feature_data = processed_data.drop(columns=[target_column], errors="ignore")

    missing_values = int(processed_data.isna().sum().sum())
    if missing_values:
        alerts.append(f"quedaron {missing_values} nulos despues del preprocesamiento")

    non_numeric_columns = (
        feature_data.select_dtypes(exclude=["number"]).columns.tolist()
    )
    if non_numeric_columns:
        alerts.append(f"features no numericas: {', '.join(non_numeric_columns)}")

    constant_features = [
        column for column in feature_data.columns if feature_data[column].nunique() <= 1
    ]
    if constant_features:
        alerts.append(f"features constantes: {', '.join(constant_features[:8])}")

    duplicated_rows = int(processed_data.duplicated().sum())
    if duplicated_rows:
        alerts.append(f"filas duplicadas despues de transformar: {duplicated_rows}")

    return AuditResult(
        name="processed",
        rows=len(processed_data),
        columns=len(processed_data.columns),
        duplicated_rows=duplicated_rows,
        missing_values=missing_values,
        alerts=alerts,
    )


def compare_numeric_distributions(
    raw_data: pd.DataFrame,
    processed_data: pd.DataFrame,
    max_columns: int = 8,
) -> pd.DataFrame:
    """
    Compara distribuciones numericas antes y despues del escalamiento.

    El test KS no decide por si solo si hay drift real en este caso, porque se
    compara el mismo dataset antes y despues de transformar. Sirve como control
    estadistico del cambio producido por el pipeline.
    """
    raw_numeric = raw_data.select_dtypes(include=["number"])
    processed_numeric = processed_data.select_dtypes(include=["number"])
    common_columns = [
        column
        for column in raw_numeric.columns
        if column in processed_numeric.columns and column != "y"
    ][:max_columns]

    rows = []
    for column in common_columns:
        left = raw_numeric[column].dropna()
        right = processed_numeric[column].dropna()
        statistic, p_value = ks_2samp(left, right)
        rows.append(
            {
                "column": column,
                "ks_statistic": round(float(statistic), 4),
                "p_value": round(float(p_value), 4),
                "raw_mean": round(float(left.mean()), 4),
                "processed_mean": round(float(right.mean()), 4),
            }
        )

    return pd.DataFrame(rows)


def print_distribution_report(distribution_report: pd.DataFrame) -> None:
    """Muestra el reporte KS si existen columnas comparables."""
    if distribution_report.empty:
        print("\nNo hay columnas numericas comparables para el test KS.")
        return

    print("\nTEST KS RAW VS PROCESADO")
    print("-" * 42)
    print(distribution_report.to_string(index=False))


def _print_ks_by_numeric_column(df: pd.DataFrame, max_columns: int = 8) -> None:
    """Imprime KS contra una normal teorica para columnas numericas."""
    numeric_columns = df.select_dtypes(include=["number"]).columns[:max_columns]
    if len(numeric_columns) == 0:
        print("\nTest KS: no hay columnas numericas.")
        return

    print("\nTest KS por columna numerica:")
    for column in numeric_columns:
        values = df[column].dropna()
        std = values.std()
        if len(values) < 2 or std == 0:
            print(f"  {column}: no evaluable")
            continue
        statistic, p_value = kstest(values, "norm", args=(values.mean(), std))
        print(f"  {column}: KS={statistic:.4f}, p={p_value:.4f}")


def monitor_raw(df: pd.DataFrame) -> None:
    """
    Analiza la calidad de la data cruda.
    Reporta: forma, nulos, duplicados, estadisticas basicas y balance del target.
    """
    audit = audit_raw_data(df)

    print("=" * 50)
    print("MONITOREO — DATA CRUDA")
    print(f"  Filas x Columnas : {df.shape}")
    print(f"  Valores nulos    : {df.isnull().sum().sum()}")
    print(f"  Duplicados       : {df.duplicated().sum()}")
    print("  Estadisticas:")
    print(df.describe().to_string())

    if audit.alerts:
        print("  Alertas:")
        for alert in audit.alerts:
            print(f"    - {alert}")

    _print_ks_by_numeric_column(df)
    print("=" * 50)


def monitor_processed(df: pd.DataFrame) -> None:
    """
    Verifica la calidad de la data post-preprocesamiento.
    Reporta: forma, nulos restantes, tipos de datos y distribucion de columnas.
    """
    audit = audit_processed_data(df)

    print("=" * 50)
    print("MONITOREO — DATA PROCESADA")
    print(f"  Filas x Columnas : {df.shape}")
    print(f"  Nulos restantes  : {df.isnull().sum().sum()}")
    print("  Tipos de datos:")
    print(df.dtypes.value_counts().to_string())

    if audit.alerts:
        print("  Alertas:")
        for alert in audit.alerts:
            print(f"    - {alert}")

    _print_ks_by_numeric_column(df)
    print("=" * 50)
