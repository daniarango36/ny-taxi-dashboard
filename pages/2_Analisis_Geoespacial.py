import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import pydeck as pdk
from src.data_processing import load_and_preprocess_data

# Configuración de diseño panorámico para optimizar los mapas
st.set_page_config(page_title="Análisis Geoespacial", layout="wide")

st.title("🗺️ Análisis Geoespacial y Temporal de Demanda Diario")
st.markdown("---")

# 1. Carga de datos base
df_raw = load_and_preprocess_data('data/resultado_agregado_total.parquet', 'data/taxi_zone_lookup_coordinates.csv')

if df_raw.empty:
    st.stop()

# --- FORZAR FORMATO DE FECHA ESTRICTO (YYYY-MM-DD) SIN HORA ---
# Se convierte a datetime y luego a string formateado para limpiar cualquier residuo de horas en la lectura
df_raw['fecha'] = pd.to_datetime(df_raw['fecha']).dt.strftime('%Y-%m-%d')

# --- SECCIÓN DE FILTROS (PANEL LATERAL) ---
st.sidebar.header("Filtros de Análisis")

# 1. Filtro de Flujo (Unifica toda la página)
indicador = st.sidebar.radio("Tipo de Viaje", options=["Ida (Origen / Pick-Up)", "Vuelta (Destino / Drop-Off)"])

# TRANSFORMACIÓN ÚNICA: Renombrar columnas según el flujo para tener un DataFrame estándar
if indicador == "Ida (Origen / Pick-Up)":
    df_base = df_raw.rename(columns={'PU_Zone': 'Zona', 'PU_lat': 'lat', 'PU_lon': 'lon'})
else:
    df_base = df_raw.rename(columns={'DO_Zone': 'Zona', 'DO_lat': 'lat', 'DO_lon': 'lon'})

# 2. Filtro de Fechas (Garantizado como texto plano en formato YYYY-MM-DD)
fechas = sorted(df_base['fecha'].dropna().unique())
fechas_sel = st.sidebar.multiselect("Fecha", options=fechas, default=[fechas[-1]] if fechas else [])

# 3. Filtro de Rangos de Hora
rangos_hora = sorted(df_base['rango_hora'].dropna().unique())
rangos_sel = st.sidebar.multiselect("Rango de Hora", options=rangos_hora, default=[])

# 4. Filtro de Zonas
todas_zonas = sorted(df_base['Zona'].dropna().unique())
zonas_sel = st.sidebar.multiselect("Zona Operativa", options=todas_zonas, default=[])

# --- APLICACIÓN DINÁMICA DE FILTROS ---
df_f = df_base.copy()

if fechas_sel:
    df_f = df_f[df_f['fecha'].isin(fechas_sel)]
if rangos_sel:
    df_f = df_f[df_f['rango_hora'].isin(rangos_sel)]
if zonas_sel:
    df_f = df_f[df_f['Zona'].isin(zonas_sel)]


# ==============================================================================
# GRAFICO 1 (ARRIBA): MAPA DE CALOR TEMPORAL (ESCALA DE ROJOS TRANSLÚCIDOS)
# ==============================================================================
st.subheader(f"🗓️ Matriz de Intensidad Temporal: {indicador}")

# Agrupación por la fecha limpia en formato string
df_time = df_f.groupby(['fecha', 'rango_hora'])['conteo'].sum().reset_index()

if not df_time.empty:
    # Pivotar los datos para armar la matriz (Filas: Horas, Columnas: Fechas)
    df_matrix = df_time.pivot(index='rango_hora', columns='fecha', values='conteo').fillna(0)
    
    # Mapa de calor en escala de rojos (transparente a sólido)
    fig_heat = go.Figure(data=go.Heatmap(
        z=df_matrix.values,
        x=df_matrix.columns,
        y=df_matrix.index,
        colorscale=[
            [0.0, 'rgba(255, 255, 255, 0)'],     # Transparente para valor 0
            [0.2, 'rgba(254, 224, 210, 0.4)'],   # Rojo muy suave
            [0.6, 'rgba(251, 106, 74, 0.8)'],    # Rojo medio
            [1.0, 'rgba(165, 15, 21, 1.0)']      # Rojo intenso
        ],
        colorbar=dict(title="Viajes")
    ))
    
    fig_heat.update_layout(
        xaxis_title="Fecha de Operación",
        yaxis_title="Bloque Horario",
        xaxis=dict(type='category'), # Forzar tipo categoría evita que Plotly auto-complete con horas
        height=380,
        margin=dict(t=20, b=20, l=10, r=10)
    )
    st.plotly_chart(fig_heat, use_container_width=True)
else:
    st.info("No hay registros que coincidan con los filtros seleccionados.")

st.markdown("---")


# ==============================================================================
# GRAFICO 2 (ABAJO): MAPA GEOESPACIAL DE POLÍGONOS (PYDECK)
# ==============================================================================
st.subheader("📍 Densidad Espacial por Zonas")

# Agrupar coordenadas usando la transformación estándar
df_geo = df_f.groupby(['Zona', 'lat', 'lon'])['conteo'].sum().reset_index()

if not df_geo.empty:
    max_c = df_geo['conteo'].max()
    min_c = df_geo['conteo'].min()
    rng = (max_c - min_c) if max_c != min_c else 1

    # Construcción del polígono alrededor de la coordenada central
    def generar_cuadrante_zona(row):
        delta = 0.0055  
        lat, lon = row['lat'], row['lon']
        return [
            [lon - delta, lat - delta],
            [lon + delta, lat - delta],
            [lon + delta, lat + delta],
            [lon - delta, lat + delta]
        ]

    df_geo['polygon'] = df_geo.apply(generar_cuadrante_zona, axis=1)

    # Asignación de intensidad de rojo según el volumen relativo de la zona
    def calcular_color_rojo(conteo):
        norm = (conteo - min_c) / rng
        r = 220
        g = int(40 * (1 - norm))
        b = int(40 * (1 - norm))
        a = int(60 + 195 * norm)  
        return [r, g, b, a]

    df_geo['fill_color'] = df_geo['conteo'].apply(calcular_color_rojo)
    
    # Pre-formatear el texto del Tooltip en Pandas
    df_geo['tooltip_viajes'] = df_geo['conteo'].apply(lambda x: f"{x:,.0f}")

    capa_poligonos = pdk.Layer(
        "PolygonLayer",
        df_geo,
        get_polygon="polygon",
        get_fill_color="fill_color",
        get_line_color=[255, 255, 255, 80], 
        line_width_min_pixels=1,
        pickable=True,
        extruded=False
    )

    # Centrado inicial 
    estado_vista = pdk.ViewState(
        latitude=40.7128,
        longitude=-74.0060,
        zoom=10,
        pitch=0
    )

    st.pydeck_chart(pdk.Deck(
        layers=[capa_poligonos],
        initial_view_state=estado_vista,
        map_style="mapbox://styles/mapbox/dark-v10",  
        tooltip={
            "html": "<b>Zona Operativa:</b> {Zona} <br/> <b>Servicios:</b> {tooltip_viajes}",
            "style": {"backgroundColor": "steelblue", "color": "white"}
        }
    ))
else:
    st.info("No hay datos geográficos para la selección actual.")