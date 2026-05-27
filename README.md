# Pipeline ML - con dataset Bank Marketing

```text
Arturo Eyck Tapia Ramos
```

```text
https://github.com/xfibben/tarea1_ucsp
```

Este proyecto implementa un pipeline simple de machine learning usando el dataset
`bank-full.csv`. El objetivo es predecir si un cliente acepta o no un deposito a
plazo ofrecido por campaña de un banco.

## Archivos

- `main.py`: ejecuta todo el flujo.
- `preprocessing.py`: carga datos, crea algunas variables, imputa, codifica y escala.
- `monitoring.py`: revisa la data cruda y la data procesada.
- `training.py`: separa train/test, entrena el modelo y muestra metricas.
- `bank-full.csv`: dataset usado.

## Como ejecutar

```bash
pip install -r requirements.txt
python main.py
```

## Resultado

- Dataset raw: 45211 filas y 17 columnas.
- Dataset procesado: 45211 filas y 47 columnas.
- Valores nulos finales: 0.
- Duplicados finales: 0.
- La clase positiva tiene baja conversion: 11.70%.

Metricas del modelo con RandomForest:

```text
accuracy: 0.91
precision clase positiva: 0.7050
recall clase positiva: 0.3365
f1-score clase positiva: 0.4555
```

El modelo tiene buena precision para la clase positiva: cuando predice que un
cliente acepta el producto, suele acertar. El recall es menor, por lo que todavia
se pierden varios clientes que si aceptarian. Para una campana bancaria puede
servir como una primera version del scoring comercial.
