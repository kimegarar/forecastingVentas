# Ejercicio de ciencia de datos e IA para proyeccion de ventas de negocio

Prototipo demostrativo que aplica ciencia de datos y aprendizaje automatico a la proyeccion de ventas. El dashboard permite explorar escenarios por producto para noviembre de 2025.

Este repositorio es un ejercicio educativo, no un sistema de planificacion comercial listo para produccion. Las predicciones son estimaciones experimentales y no una recomendacion ni una garantia de ventas o ingresos.

## Objetivo

El proyecto prepara datos historicos de ventas y competencia, crea variables temporales y de comportamiento, ajusta un modelo de regresion y permite explorar predicciones de unidades vendidas.

## Estructura

- `data/raw/`: datos originales de ventas y competencia.
- `data/processed/`: datasets transformados y listos para entrenamiento/inferencia.
- `models/`: artefactos entrenados, incluido el modelo final serializado.
- `notebooks/`: exploracion, entrenamiento y validacion del pipeline.
- `src/`: codigo reutilizable para cargas, transformacion y prediccion.
- `app/`: dashboard en Streamlit para consulta del forecast.
- `docs/`: documentacion adicional del proyecto.

## Inicio rapido

Antes de arrancar, comprueba que estan disponibles los dos artefactos locales indicados en la seccion "Datos y artefactos locales".

```bash
conda activate forecast
python -m pip install -r requirements.txt
python -m streamlit run app/streamlit_app.py
```

## Como iniciar la app

1. Abre PowerShell en la raiz del proyecto.
2. Activa el entorno Conda del proyecto:

```powershell
conda activate forecast
```

3. Instala las dependencias:

```powershell
python -m pip install -r requirements.txt
```

4. Arranca la aplicacion:

```powershell
python -m streamlit run app/streamlit_app.py
```

5. Abre la URL que te muestre Streamlit en el navegador, normalmente:

```text
http://localhost:8503
```

Si el puerto `8503` ya esta ocupado, abre la URL de la instancia que ya esta en ejecucion o detienela con `Ctrl+C` en su terminal. Tambien puedes iniciar otra instancia en un puerto alternativo y abrir la URL que imprima Streamlit:

```powershell
python -m streamlit run app/streamlit_app.py --server.port 8504
```

## Configuracion de Streamlit

El repositorio incluye la configuracion de la app en `.streamlit/config.toml` para:

- desactivar la recogida de estadisticas de uso,
- dejar la app en modo headless,
- fijar el puerto `8503` como URL local predeterminada.

La configuracion desactiva la recopilacion de estadisticas y ejecuta el servidor en modo headless.

## Datos y artefactos locales

Git excluye los CSV de `data/raw/`, los datasets de `data/processed/` y los modelos de `models/`. Por eso, una clonacion limpia de GitHub no contiene por si sola los archivos que necesita la app. Para iniciarla deben existir localmente:

- `data/processed/inferencia_df_transformado.csv`
- `models/modelo_final.joblib`

Para reproducir el pipeline tambien se necesitan los archivos de entrada que leen los notebooks, entre ellos `data/raw/entrenamiento/ventas.csv`, `data/raw/entrenamiento/competencia.csv` y `data/raw/inferencia/ventas_2025_inferencia.csv`. Ejecuta primero `notebooks/entrenamiento.ipynb` para preparar el esquema y guardar el modelo; luego ejecuta `notebooks/forecasting.ipynb` para transformar la inferencia de noviembre. Los datos de entrada no se publican en este repositorio.

## Flujo principal

1. Cargar datos historicos de ventas y competencia.
2. Construir variables de calendario, estacionalidad y lags.
3. Generar features one-hot y preparar el dataset de entrenamiento.
4. Entrenar el modelo de demostracion con HistGradientBoostingRegressor.
5. Guardar el artefacto en `models/modelo_final.joblib`.
6. Ejecutar la app para explorar la proyeccion ilustrativa de noviembre de 2025.

## Archivos clave

- `data/processed/df.csv`: dataset de entrenamiento preparado.
- `data/processed/inferencia_df_transformado.csv`: datos de inferencia ya transformados.
- `models/modelo_final.joblib`: modelo final entrenado.
- `app/streamlit_app.py`: interfaz web del forecast.
- `src/forecasting.py`: funciones reutilizables para cargar datos y generar predicciones.

## Requisitos

El proyecto usa las librerias definidas en `requirements.txt` y evita dependencias adicionales.
