# Forecast de ventas

Proyecto de machine learning para preparar datos, entrenar modelos y consultar predicciones desde Streamlit.

## Estructura

- `data/raw/`: datos originales.
- `data/external/`: datos externos o de referencia.
- `data/processed/`: datos listos para análisis o entrenamiento.
- `notebooks/`: exploración y experimentos.
- `src/`: código reutilizable de preparación y modelado.
- `models/`: artefactos entrenados localmente.
- `app/`: aplicación Streamlit.
- `docs/`: documentación del proyecto.

Los archivos de datos y modelos locales están excluidos de Git por defecto.

## Inicio rápido

```bash
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```
