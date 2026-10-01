# Aplicacion de simulacion del forecast de ventas para noviembre 2025.
# La logica recursiva usa el dataframe ya preparado y actualiza lags y media
# movil dia a dia para emular la evolucion de las ventas dentro del mes.

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st


ROOT_DIR = Path(__file__).resolve().parent.parent if (Path(__file__).resolve().parent / "models").exists() is False else Path(__file__).resolve().parent
MODEL_PATH = ROOT_DIR / "models" / "modelo_final.joblib"
DATA_PATH = ROOT_DIR / "data" / "processed" / "inferencia_df_transformado.csv"


def cargar_modelo_y_datos():
    """Carga el modelo entrenado y el dataset de inferencia ya preparado."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"No se encuentra el modelo en: {MODEL_PATH}")
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"No se encuentra el CSV de inferencia: {DATA_PATH}")

    modelo = joblib.load(MODEL_PATH)
    datos = pd.read_csv(DATA_PATH, parse_dates=["fecha"])
    return modelo, datos


def obtener_columnas_lag():
    """Devuelve la lista exacta de columnas de lag del dataset."""
    return [f"unidades_vendidas_lag_{numero}" for numero in range(1, 8)]


def expandir_horizonte_noviembre(producto_df):
    """Expande el registro base del producto a los 30 dias de noviembre cuando el CSV trae un solo registro por producto."""
    producto_df = producto_df.sort_values("fecha").reset_index(drop=True).copy()
    if len(producto_df) >= 30:
        return producto_df

    base_fila = producto_df.iloc[0].copy()
    registros = []
    for dia in range(1, 31):
        fila = base_fila.copy()
        fecha = pd.Timestamp(f"2025-11-{dia:02d}")
        fila["fecha"] = fecha
        fila["anio"] = 2025
        fila["mes"] = 11
        fila["dia_mes"] = dia
        fila["dia_semana"] = int(fecha.weekday())
        fila["dia_del_anio"] = int(fecha.dayofyear)
        fila["semana_anio"] = int(fecha.isocalendar().week)
        fila["trimestre"] = 4
        fila["dias_del_mes"] = 30
        fila["es_fin_de_semana"] = bool(fecha.weekday() >= 5)
        fila["es_inicio_mes"] = bool(dia <= 3)
        fila["es_fin_mes"] = bool(dia >= 28)
        fila["es_black_friday"] = bool(dia == 28)
        fila["es_cyber_monday"] = bool(dia == 30)
        fila["es_festivo"] = bool(dia == 1)
        fila["nombre_festivo"] = "Todos los Santos" if dia == 1 else ""
        fila["dia_semana_sin"] = float(np.sin(2 * np.pi * fila["dia_semana"] / 7))
        fila["dia_semana_cos"] = float(np.cos(2 * np.pi * fila["dia_semana"] / 7))
        fila["mes_sin"] = float(np.sin(2 * np.pi * (11 - 1) / 12))
        fila["mes_cos"] = float(np.cos(2 * np.pi * (11 - 1) / 12))
        fila["es_temporada_navidad"] = bool(
            (fecha.month == 12 and fecha.day >= 15)
            or (fecha.month == 1 and fecha.day <= 6)
        )
        registros.append(fila)

    return pd.DataFrame(registros).reset_index(drop=True)


def preparar_producto(producto_df, descuento_pct, escenario_competencia):
    """Aplica el ajuste de descuento y competencia para un producto concreto."""
    producto_df = producto_df.sort_values("fecha").reset_index(drop=True).copy()

    # Recalcula el precio de venta con el descuento elegido por el usuario.
    producto_df["descuento_porcentaje"] = float(descuento_pct)
    producto_df["precio_venta"] = producto_df["precio_base"] * (1 - descuento_pct / 100)

    # Ajusta precio de competencia según el escenario para los tres competidores.
    escenario_pct = {"Actual (0%)": 0.0, "Competencia -5%": -0.05, "Competencia +5%": 0.05}[escenario_competencia]
    if {"Amazon", "Decathlon", "Deporvillage"}.issubset(producto_df.columns):
        producto_df[["Amazon", "Decathlon", "Deporvillage"]] = (
            producto_df[["Amazon", "Decathlon", "Deporvillage"]] * (1 + escenario_pct)
        )
        producto_df["precio_competencia"] = (
            producto_df[["Amazon", "Decathlon", "Deporvillage"]].mean(axis=1)
        )
    else:
        producto_df["precio_competencia"] = producto_df["precio_competencia"] * (1 + escenario_pct)

    producto_df["ratio_precio"] = producto_df["precio_venta"] / producto_df["precio_competencia"]
    return producto_df


def simular_prediccion_recursiva(producto_df, modelo, descuento_pct, escenario_competencia):
    """Genera predicciones dia a dia actualizando lags y media movil recursivamente."""
    producto_df = preparar_producto(producto_df, descuento_pct, escenario_competencia)
    columnas_lag = obtener_columnas_lag()

    # Compatibilidad con la nomenclatura actual del dataset y la que se usa en la app.
    if "media_movil_7_dias" not in producto_df.columns and "unidades_vendidas_ma7" in producto_df.columns:
        producto_df["media_movil_7_dias"] = producto_df["unidades_vendidas_ma7"]
    if "unidades_vendidas_ma7" not in producto_df.columns and "media_movil_7_dias" in producto_df.columns:
        producto_df["unidades_vendidas_ma7"] = producto_df["media_movil_7_dias"]

    resultado = producto_df[["fecha", "producto_id", "nombre", "dia_mes", "dia_semana", "precio_base", "precio_venta", "precio_competencia", "descuento_porcentaje", "ratio_precio", "es_black_friday", "es_festivo"]].copy()
    resultado["unidades_prediccion"] = np.nan
    resultado["ingresos_prediccion"] = np.nan

    # Se usan los lagsque ya vienen en el archivo para el primer dia y luego se actualizan recursivamente.
    lag_actual = {columna: float(producto_df.iloc[0][columna]) for columna in columnas_lag}
    ma7_actual = float(producto_df.iloc[0]["media_movil_7_dias"] if "media_movil_7_dias" in producto_df.columns else producto_df.iloc[0]["unidades_vendidas_ma7"])
    predicciones_previas = []

    columnas_modelo = list(modelo.feature_names_in_)

    for indice, fila in producto_df.iterrows():
        fila_modelo = fila.copy()

        # Recalcula las columnas de lag y media movil antes de cada prediccion.
        if indice > 0:
            lag_actual = {"unidades_vendidas_lag_1": float(predicciones_previas[-1])}
            lag_actual["unidades_vendidas_lag_2"] = float(lag_actual_anterior["unidades_vendidas_lag_1"])
            lag_actual["unidades_vendidas_lag_3"] = float(lag_actual_anterior["unidades_vendidas_lag_2"])
            lag_actual["unidades_vendidas_lag_4"] = float(lag_actual_anterior["unidades_vendidas_lag_3"])
            lag_actual["unidades_vendidas_lag_5"] = float(lag_actual_anterior["unidades_vendidas_lag_4"])
            lag_actual["unidades_vendidas_lag_6"] = float(lag_actual_anterior["unidades_vendidas_lag_5"])
            lag_actual["unidades_vendidas_lag_7"] = float(lag_actual_anterior["unidades_vendidas_lag_6"])
            ma7_actual = float(np.mean(list(lag_actual.values())))

        # Mantener los valores del dia 1 tal cual vienen en el archivo.
        if indice == 0:
            lag_actual = {columna: float(fila[columna]) for columna in columnas_lag}
            ma7_actual = float(fila["media_movil_7_dias"] if "media_movil_7_dias" in fila.index else fila["unidades_vendidas_ma7"])

        for columna_lag, valor_lag in lag_actual.items():
            fila_modelo[columna_lag] = float(valor_lag)
        fila_modelo["media_movil_7_dias"] = float(ma7_actual)
        fila_modelo["unidades_vendidas_ma7"] = float(ma7_actual)

        fila_modelo["precio_venta"] = float(fila["precio_venta"])
        fila_modelo["precio_competencia"] = float(fila["precio_competencia"])
        fila_modelo["ratio_precio"] = float(fila["ratio_precio"])
        fila_modelo["descuento_porcentaje"] = float(fila["descuento_porcentaje"])

        features = pd.DataFrame([fila_modelo[columnas_modelo]], columns=columnas_modelo)
        prediccion = float(modelo.predict(features)[0])
        predicciones_previas.append(prediccion)
        lag_actual_anterior = lag_actual.copy()

        resultado.at[indice, "unidades_prediccion"] = prediccion
        resultado.at[indice, "ingresos_prediccion"] = float(fila["precio_venta"]) * prediccion

    resultado["fecha"] = pd.to_datetime(resultado["fecha"])
    resultado["dia_semana"] = resultado["fecha"].dt.day_name()
    resultado["es_black_friday"] = resultado["fecha"].dt.day.eq(28)
    resultado["highlight_black_friday"] = resultado["es_black_friday"].map({True: "🔴", False: ""})
    resultado["ingresos_prediccion"] = resultado["unidades_prediccion"] * resultado["precio_venta"]
    return resultado


def construir_grafico_prediccion(df_resultado):
    """Grafico de prediccion diaria con marca visual para Black Friday."""
    fig, ax = plt.subplots(figsize=(11, 5))
    sns.set_theme(style="whitegrid")

    sns.lineplot(
        data=df_resultado,
        x="fecha",
        y="unidades_prediccion",
        color="#667eea",
        linewidth=2.5,
        ax=ax,
    )

    black_friday = df_resultado[df_resultado["es_black_friday"]].iloc[0]
    ax.axvline(
        x=pd.Timestamp(black_friday["fecha"]),
        color="#ff4d4d",
        linestyle="--",
        linewidth=2,
        alpha=0.9,
    )
    ax.scatter(
        [pd.Timestamp(black_friday["fecha"])],
        [black_friday["unidades_prediccion"]],
        color="#ff4d4d",
        s=80,
        zorder=5,
    )
    ax.annotate(
        "Black Friday",
        xy=(pd.Timestamp(black_friday["fecha"]), black_friday["unidades_prediccion"]),
        xytext=(10, -28),
        textcoords="offset points",
        color="#ff4d4d",
        fontsize=10,
        fontweight="bold",
    )

    ax.set_title("Prediccion diaria de ventas en noviembre 2025", fontsize=14, fontweight="bold")
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Unidades vendidas")
    ax.grid(True, alpha=0.25)
    fig.autofmt_xdate(rotation=30)
    fig.tight_layout()
    return fig


def calcular_escenarios(producto_df, modelo, descuento_pct):
    """Calcula la comparativa de escenarios de competencia."""
    escenarios = {
        "Actual (0%)": 0.0,
        "Competencia -5%": -0.05,
        "Competencia +5%": 0.05,
    }
    resultados = []
    for nombre_escenario, valor_escenario in escenarios.items():
        df_escenario = simular_prediccion_recursiva(producto_df, modelo, descuento_pct, nombre_escenario)
        resultados.append(
            {
                "scenario": nombre_escenario,
                "unidades_totales": float(df_escenario["unidades_prediccion"].sum()),
                "ingresos_totales": float(df_escenario["ingresos_prediccion"].sum()),
                "precio_promedio": float(df_escenario["precio_venta"].mean()),
            }
        )
    return pd.DataFrame(resultados)


def main():
    st.set_page_config(page_title="Forecast de ventas", page_icon="📈", layout="wide")
    st.markdown(
        """
        <style>
            .main { background: linear-gradient(135deg, #f5f7ff 0%, #eef2ff 100%); }
            .stApp { background: linear-gradient(135deg, #f5f7ff 0%, #eef2ff 100%); }
            div[data-testid="stSidebar"] { background: linear-gradient(180deg, #1f1b3a 0%, #2f2158 100%); color: white; }
            .stMetric { background: rgba(255,255,255,0.9); border-radius: 12px; padding: 10px; border: 1px solid rgba(102,126,234,0.15); }
            .block-container { padding-top: 1rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    try:
        modelo, datos = cargar_modelo_y_datos()
    except Exception as error:
        st.error(f"No se pudo cargar el modelo o los datos: {error}")
        st.stop()

    producto_options = sorted(datos["nombre"].drop_duplicates().tolist())

    with st.sidebar:
        st.title("Controles de Simulación")
        producto_seleccionado = st.selectbox("Producto", producto_options, index=0)
        descuento_usuario = st.slider("Ajuste de descuento", min_value=-50, max_value=50, value=0, step=5, format="%d%%")
        escenario_competencia = st.radio("Escenario de competencia", ["Actual (0%)", "Competencia -5%", "Competencia +5%"], index=0)
        boton_simular = st.button("Simular Ventas", type="primary")

    df_producto = datos[datos["nombre"].eq(producto_seleccionado)].copy()
    if df_producto.empty:
        st.warning("No hay filas para el producto seleccionado.")
        st.stop()
    df_producto = expandir_horizonte_noviembre(df_producto)

    clave_simulacion = (
        producto_seleccionado,
        float(descuento_usuario),
        escenario_competencia,
    )
    if (
        boton_simular
        or st.session_state.get("clave_simulacion") != clave_simulacion
    ):
        with st.spinner("Calculando predicción recursiva día a día para noviembre 2025..."):
            resultado_simulacion = simular_prediccion_recursiva(
                df_producto,
                modelo,
                float(descuento_usuario),
                escenario_competencia,
            )
        st.session_state["resultado_simulacion"] = resultado_simulacion
        st.session_state["clave_simulacion"] = clave_simulacion
    else:
        resultado_simulacion = st.session_state["resultado_simulacion"]

    resultado_simulacion = resultado_simulacion.sort_values("fecha").reset_index(drop=True)
    resultado_simulacion["ingresos_prediccion"] = resultado_simulacion["unidades_prediccion"] * resultado_simulacion["precio_venta"]

    st.header(f"Prototipo de IA para proyección de ventas · noviembre 2025 · {producto_seleccionado}")
    st.caption(
        "Ejercicio demostrativo de ciencia de datos e IA para explorar escenarios de ventas; "
        "las proyecciones no son una recomendación operativa ni una garantía de resultados."
    )

    kpis = st.columns(4)
    kpis[0].metric("Unidades totales proyectadas", f"{resultado_simulacion['unidades_prediccion'].sum():,.0f}")
    kpis[1].metric("Ingresos proyectados", f"€ {resultado_simulacion['ingresos_prediccion'].sum():,.2f}")
    kpis[2].metric("Precio promedio de venta", f"€ {resultado_simulacion['precio_venta'].mean():,.2f}")
    kpis[3].metric("Descuento promedio", f"{resultado_simulacion['descuento_porcentaje'].mean():,.2f}%")

    st.subheader("Predicción diaria")
    fig = construir_grafico_prediccion(resultado_simulacion)
    st.pyplot(fig)

    st.subheader("Detalle por día")
    tabla = resultado_simulacion[["fecha", "dia_semana", "precio_venta", "precio_competencia", "descuento_porcentaje", "unidades_prediccion", "ingresos_prediccion"]].copy()
    tabla.columns = ["Fecha", "Día de la semana", "Precio venta", "Precio competencia", "Descuento aplicado", "Unidades predichas", "Ingresos proyectados"]

    def marcar_black_friday(row):
        fecha_objeto = pd.Timestamp(row["Fecha"])
        if fecha_objeto.day == 28:
            return ["background-color: #ffe3e3; color: #7a1f1f; font-weight: bold"] * len(row)
        return ["" for _ in row]

    tabla_formateada = tabla.style.apply(marcar_black_friday, axis=1).format(
        {
            "Fecha": lambda valor: valor.strftime("%d/%m/%Y"),
            "Precio venta": "€{:.2f}".format,
            "Precio competencia": "€{:.2f}".format,
            "Descuento aplicado": "{:.2f}%".format,
            "Unidades predichas": "{:.0f}".format,
            "Ingresos proyectados": "€{:.2f}".format,
        }
    )

    st.dataframe(
        tabla_formateada,
        use_container_width=True,
        height=420,
    )

    st.markdown("---")
    st.subheader("Comparativa de escenarios de competencia")
    escenarios_df = calcular_escenarios(df_producto, modelo, float(descuento_usuario))
    cols = st.columns(3)
    for columna, fila in zip(cols, escenarios_df.itertuples(index=False)):
        columna.metric(
            label=f"{fila.scenario}",
            value=f"{fila.unidades_totales:,.0f} uds",
            delta=f"€ {fila.ingresos_totales:,.2f}",
        )

    st.caption("El escenario actual mantiene el precio de la competencia y solo se ajusta el descuento elegido por el usuario.")


if __name__ == "__main__":
    main()
