# Carga y prepara el modelo final para generar predicciones de ventas.
# El punto clave es alinear exactamente las columnas de inferencia con la
# lista de features que el modelo entrenado esperó durante el fit.

import joblib
import pandas as pd


def cargar_modelo(ruta_modelo):
    """Carga el artefacto entrenado desde disco."""
    return joblib.load(ruta_modelo)


def cargar_datos_inferencia(ruta_csv):
    """Carga el CSV de inferencia con la columna fecha en formato datetime."""
    datos = pd.read_csv(ruta_csv, parse_dates=["fecha"])
    return datos.copy()


def preparar_features_para_prediccion(datos, modelo):
    """Selecciona solo las columnas esperadas por el modelo entrenado."""
    columnas_esperadas = list(getattr(modelo, "feature_names_in_", []))
    if not columnas_esperadas:
        raise ValueError(
            "El modelo cargado no expone feature_names_in_. "
            "No se puede validar la estructura de entrada."
        )

    columnas_faltantes = [
        columna for columna in columnas_esperadas if columna not in datos.columns
    ]
    if columnas_faltantes:
        raise ValueError(
            "Faltan columnas en los datos de inferencia para el modelo: "
            f"{columnas_faltantes}"
        )

    return datos[columnas_esperadas].copy()


def predecir_ventas(modelo, datos_inferencia):
    """Genera predicciones de unidades vendidas para cada fila de inferencia."""
    X_inferencia = preparar_features_para_prediccion(datos_inferencia, modelo)
    predicciones = modelo.predict(X_inferencia)

    salida = datos_inferencia[["fecha", "producto_id", "nombre", "categoria", "subcategoria"]].copy()
    salida["prediccion_unidades"] = predicciones
    salida = salida.sort_values(["producto_id", "fecha"]).reset_index(drop=True)
    return salida
