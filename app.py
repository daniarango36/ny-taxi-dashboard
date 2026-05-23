import streamlit as st

# Configuración de la página principal (debe ser la primera directiva)
st.set_page_config(
    page_title="NY Taxi Demand | Home",
    page_icon="🚖",
    layout="wide"
)

st.title("🚖 Dashboard Predictivo: Demanda de Taxis en Nueva York")
st.markdown("---")

st.markdown("""
### Arquitectura y Modelado
Bienvenido al sistema de inteligencia de negocio y predicción espacial para la red de taxis de NY.

**Pipeline de Datos:**
1. **ETL Geoespacial:** Unificamos transacciones agregadas (`.parquet`) con un maestro de coordenadas (`.csv`). Se construye un `GeoDataFrame` proyectado bajo el estándar **EPSG:4326** para garantizar renderizado 3D de alta performance.
2. **Modelo Predictivo:** Una **Red Neuronal (MLP Regressor)** de Scikit-Learn. Cuenta con 5 capas ocultas de arquitectura densa y funciones de activación `ReLU`. El modelo optimiza hiperparámetros mediante `GridSearchCV` para predecir las próximas 3 semanas de demanda.

**Cómo navegar:**
* 📊 **Resumen y KPIs:** Analiza el cumplimiento (Real vs Predicción) mediante visualizaciones de Gauge e Histogramas.
* 🗺️ **Análisis Geoespacial:** Mapas de calor interactivos usando `PyDeck` con agregación hexagonal dinámica en GPU.

👈 **Selecciona una página en la barra lateral para comenzar.**
""")

st.info("Nota técnica: Toda la ingesta y entrenamiento corre sobre una capa de caché (@st.cache_data) para mantener el dashboard veloz.")