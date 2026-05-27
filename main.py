"""
TAREA: Pipeline de Machine Learning
=====================================
Rubrica de evaluacion:
- Codigo completo (todas las etapas funcionando) : 10 puntos
- Codigo ordenado (estructura modular clara)      :  7 puntos
- Buenas practicas (funciones, docstrings, PEP8)  :  3 puntos

Dataset: bank-full.csv
Objetivo: predecir si un cliente acepta un deposito a plazo.
"""

import monitoring as mn
import preprocessing as pp
import training as tr

DATA_PATH = "bank-full.csv"


def main():
    """Ejecuta el pipeline completo en el orden indicado."""
    # 1 Cargar y preprocesar
    df_raw = pp.load_data(DATA_PATH)
    df_processed = pp.preprocess(df_raw)

    # 2 Monitoreo
    mn.monitor_raw(df_raw)
    mn.monitor_processed(df_processed)

    # 3 entrenamiento
    model, metrics = tr.train(df_processed)
    tr.evaluate(model, metrics)


if __name__ == "__main__":
    main()
