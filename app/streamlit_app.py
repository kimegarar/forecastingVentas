# Aplicacion Streamlit para consultar la prediccion de unidades vendidas
# usando el modelo final guardado en models/modelo_final.joblib.
# La app carga el CSV de inferencia procesado, alinea las columnas del
# modelo y muestra un resumen por producto y fecha.

import os

import pandas as pd
import streamlit as st

from src.forecasting import cargar_datos_inferencia, cargar_modelo, predecir_ventas


# Ruta relativa al repositorio para cargar model y datos desde cualquier entorno.
RUTA_BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RUTA_MODELO = os.path.join(RUTA_BASE, "models", "modelo_final.joblib")
RUTA_INFERENCIA = os.path.join(
    RUTA_BASE,
    "data",
    "processed",
    "inferencia_df_transformado.csv",
)


@st.cache_data
def cargar_predicciones():
    """Carga el modelo y genera la prediccion para noviembre de 2025."""
    modelo = cargar_modelo(RUTA_MODELO)
    datos_inferencia = cargar_datos_inferencia(RUTA_INFERENCIA)
    predicciones = predecir_ventas(modelo, datos_inferencia)
    return predicciones


st.set_page_config(page_title="Forecast de ventas", page_icon="📈")
st.title("Forecast de ventas")
st.caption("Predicciones por producto para el periodo de inferencia 2025.")

predicciones = cargar_predicciones()

producto_seleccionado = st.sidebar.selectbox(
    "Selecciona un producto",
    options=sorted(predicciones["nombre"].drop_duplicates().tolist()),
)

# Filtra el resultado para el producto elegido y presenta un resumen compacto.
producto_filtrado = predicciones.loc[
    predicciones["nombre"].eq(producto_seleccionado),
    ["fecha", "producto_id", "nombre", "prediccion_unidades"],
].sort_values("fecha").reset_index(drop=True)

st.subheader(f"Producto: {producto_seleccionado}")

if producto_filtrado.empty:
    st.warning("No hay registros disponibles para este producto en la inferencia.")
else:
    st.metric(
        label="Prediccion media por dia",
        value=f"{producto_filtrado['prediccion_unidades'].mean():.2f} uds",
    )
    st.dataframe(producto_filtrado, use_container_width=True)

st.subheader("Resumen del forecast")
resumen_general = predicciones.groupby("nombre", as_index=False)[
    "prediccion_unidades"
].sum().sort_values("prediccion_unidades", ascending=False)

st.dataframe(resumen_general, use_container_width=True)

st.caption(
    "El forecast usa la misma feature engineering aplicada durante el entrenamiento "
    "y alinea automaticamente las columnas esperadas por el modelo final."
)
