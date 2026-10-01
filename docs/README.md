# Documentacion

## Resumen del proyecto

Este repositorio implementa un flujo completo de forecasting de ventas con:

- preparacion y fusion de datos de ventas y competencia,
- construccion de features temporales y de estacionalidad,
- generacion de lags y medias moviles,
- entrenamiento de un modelo de regresion basado en boosting historico,
- exportacion de artefactos para uso en produccion y consulta.

## Modo de uso

1. Revisa el notebook de entrenamiento para entender el pipeline completo.
2. Verifica que los CSV procesados esten disponibles en `data/processed/`.
3. Ejecuta la app con:

```bash
streamlit run app/streamlit_app.py
```

4. El dashboard te permite elegir un producto y ver la prediccion de unidades vendidas.

## Nota tecnica importante

La app usa la lista de features que el modelo entrenado esperaba en `model.feature_names_in_` para alinear los datos de inferencia. Esto evita errores de feature mismatch y garantiza predicciones reproducibles.

