# Documentacion del ejercicio de proyeccion de ventas

## Resumen del proyecto

Este repositorio es un ejercicio demostrativo de ciencia de datos e IA aplicado a la proyeccion de ventas. No es un sistema listo para tomar decisiones comerciales ni garantiza resultados. Incluye:

- preparacion y fusion de datos de ventas y competencia,
- construccion de features temporales y de estacionalidad,
- generacion de lags y medias moviles,
- entrenamiento de un modelo de regresion basado en boosting historico,
- exportacion de artefactos para probar la inferencia y explorar escenarios en la app.

## Modo de uso

El repositorio incluye los CSV sinteticos de entrada, los datasets procesados y `models/modelo_final.joblib`, asi que la app puede ejecutarse despues de clonar sin descargar artefactos aparte.

Para regenerar los artefactos, usa las versiones de `requirements.txt`, inicia Jupyter desde `notebooks/` y ejecuta primero todas las celdas de `entrenamiento.ipynb`; despues ejecuta `forecasting.ipynb`. El primero guarda `data/processed/df.csv` y `models/modelo_final.joblib`; el segundo guarda `data/processed/inferencia_df_transformado.csv`.

1. Revisa el notebook de entrenamiento para entender el pipeline completo.
2. Verifica que los CSV procesados esten disponibles en `data/processed/`.
3. Activa el entorno Conda del proyecto e instala las dependencias si aun no lo has hecho:

```powershell
conda activate forecast
python -m pip install -r requirements.txt
```

4. Ejecuta la app con:

```powershell
python -m streamlit run app/streamlit_app.py
```

5. Abre la URL local que te muestre Streamlit, normalmente `http://localhost:8503`.

6. El dashboard te permite elegir un producto y explorar una proyeccion ilustrativa de unidades vendidas.

Si el puerto `8503` ya esta ocupado, abre la instancia existente o detenla con `Ctrl+C` en su terminal. Para usar un puerto alternativo, ejecuta `python -m streamlit run app/streamlit_app.py --server.port 8504` y abre la URL que indique la terminal.

## Nota tecnica importante

La app usa la lista de features que el modelo entrenado esperaba en `model.feature_names_in_` para alinear los datos de inferencia y evitar errores por columnas incompatibles. La reproduccion de resultados tambien depende de disponer del mismo modelo y versiones compatibles de las dependencias.

