# Forecast de ventas

Proyecto de machine learning para preparar datos, entrenar un modelo de prediccion y consultar el forecast desde una app de Streamlit.

## Objetivo

El proyecto procesa ventas historicas, incorpora variables temporales y de competidores, ajusta un modelo de regresion y genera predicciones de unidades vendidas para un periodo de inferencia posterior.

## Estructura

- `data/raw/`: datos originales de ventas y competencia.
- `data/processed/`: datasets transformados y listos para entrenamiento/inferencia.
- `models/`: artefactos entrenados, incluido el modelo final serializado.
- `notebooks/`: exploracion, entrenamiento y validacion del pipeline.
- `src/`: codigo reutilizable para cargas, transformacion y prediccion.
- `app/`: dashboard en Streamlit para consulta del forecast.
- `docs/`: documentacion adicional del proyecto.

## Inicio rapido

```bash
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

## Flujo principal

1. Cargar datos historicos de ventas y competencia.
2. Construir variables de calendario, estacionalidad y lags.
3. Generar features one-hot y preparar el dataset de entrenamiento.
4. Entrenar el modelo final con HistGradientBoostingRegressor.
5. Guardar el artefacto en `models/modelo_final.joblib`.
6. Ejecutar la app para consultar predicciones del periodo de inferencia.

## Archivos clave

- `data/processed/df.csv`: dataset de entrenamiento preparado.
- `data/processed/inferencia_df_transformado.csv`: datos de inferencia ya transformados.
- `models/modelo_final.joblib`: modelo final entrenado.
- `app/streamlit_app.py`: interfaz web del forecast.
- `src/forecasting.py`: funciones reutilizables para cargar datos y generar predicciones.

## Requisitos

El proyecto usa las librerias definidas en `requirements.txt` y evita dependencias adicionales.
