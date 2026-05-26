import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import geopandas as gpd
from shapely.geometry import Polygon
from src.data_processing import obtener_datos_mapa

st.set_page_config(page_title="Análisis Geoespacial", layout="wide")

st.title("🗺️ Análisis Geoespacial y Temporal de Demanda Diario")
st.markdown("---")

# Carga directa de datos espaciales estables
df_raw = obtener_datos_mapa()
if df_raw.empty:
    st.error("No se pudieron inicializar las variables geográficas.")
    st.stop()

# --- CONTROLADORES DEL MENÚ SIDEBAR ---
st.sidebar.header("Filtros de Análisis")

indicador = st.sidebar.radio("Tipo de Viaje", options=["Ida (Origen / Pick-Up)", "Vuelta (Destino / Drop-Off)"])

if indicador == "Ida (Origen / Pick-Up)":
    df_base = df_raw.rename(columns={'PU_Zone': 'Zona', 'PU_lat': 'lat', 'PU_lon': 'lon'})
else:
    df_base = df_raw.rename(columns={'DO_Zone': 'Zona', 'DO_lat': 'lat', 'DO_lon': 'lon'})

# Lógica: Seleccionar los últimos 5 días por defecto
fechas = sorted(df_base['fecha'].dropna().unique())
dias_por_defecto = fechas[-5:] if len(fechas) >= 5 else fechas
fechas_sel = st.sidebar.multiselect("Fecha", options=fechas, default=dias_por_defecto)

rangos_hora = sorted(df_base['rango_hora'].dropna().unique())
rangos_sel = st.sidebar.multiselect("Rango de Hora", options=rangos_hora, default=[])

todas_zonas = sorted(df_base['Zona'].dropna().unique())
zonas_sel = st.sidebar.multiselect("Zona Operativa", options=todas_zonas, default=[])

# --- SEGMENTACIÓN DINÁMICA ---
df_f = df_base.copy()

if fechas_sel:
    df_f = df_f[df_f['fecha'].isin(fechas_sel)]
if rangos_sel:
    df_f = df_f[df_f['rango_hora'].isin(rangos_sel)]
if zonas_sel:
    df_f = df_f[df_f['Zona'].isin(zonas_sel)]

# --- RENDERS GRÁFICOS: MATRIZ TEMPORAL ---
st.subheader(f"🗓️ Matriz de Intensidad Temporal: {indicador}")

df_time = df_f.groupby(['fecha', 'rango_hora'])['conteo'].sum().reset_index()

if not df_time.empty:
    df_matrix = df_time.pivot(index='rango_hora', columns='fecha', values='conteo').fillna(0)
    
    fig_heat = go.Figure(data=go.Heatmap(
        z=df_matrix.values,
        x=df_matrix.columns,
        y=df_matrix.index,
        colorscale=[
            [0.0, 'rgba(255, 255, 255, 0)'],
            [0.2, 'rgba(254, 224, 210, 0.4)'],
            [0.6, 'rgba(251, 106, 74, 0.8)'],
            [1.0, 'rgba(165, 15, 21, 1.0)']
        ],
        colorbar=dict(title="Viajes")
    ))
    
    fig_heat.update_layout(
        xaxis_title="Fecha de Operación",
        yaxis_title="Bloque Horario",
        xaxis=dict(type='category'),
        height=380,
        margin=dict(t=20, b=20, l=10, r=10)
    )
    st.plotly_chart(fig_heat, use_container_width=True)
else:
    st.info("No hay registros que coincidan con los filtros seleccionados.")

st.markdown("---")

# --- RENDERS GRÁFICOS: DENSIDAD GEOPANDAS + PLOTLY ---
st.subheader("📍 Densidad Espacial por Zonas")

df_geo = df_f.groupby(['Zona', 'lat', 'lon'])['conteo'].sum().reset_index()

if not df_geo.empty:
    # 1. Crear polígonos matemáticos ligeros para no sobrecargar la RAM
    def generar_poligono(row):
        delta = 0.0055  
        lat, lon = row['lat'], row['lon']
        return Polygon([
            (lon - delta, lat - delta),
            (lon + delta, lat - delta),
            (lon + delta, lat + delta),
            (lon - delta, lat + delta)
        ])

    df_geo['geometry'] = df_geo.apply(generar_poligono, axis=1)
    
    # 2. Convertir el DataFrame normal a un GeoDataFrame oficial
    gdf = gpd.GeoDataFrame(df_geo, geometry='geometry', crs="EPSG:4326")
    
    # 3. Renderizar sobre un mapa base gratuito usando Plotly Express
    fig_mapa = px.choropleth_mapbox(
        gdf,
        geojson=gdf.geometry,
        locations=gdf.index,
        color='conteo',
        color_continuous_scale="Reds",
        mapbox_style="carto-darkmatter", # Render de ciudad estable y sin API Key
        zoom=9.5,
        center={"lat": 40.7128, "lon": -73.9560},
        opacity=0.6,
        hover_name="Zona",
        hover_data={"conteo": True, "lat": False, "lon": False}
    )
    
    fig_mapa.update_layout(
        margin={"r":0,"t":0,"l":0,"b":0},
        coloraxis_colorbar=dict(title="Volumen de Viajes"),
        height=500
    )
    
    st.plotly_chart(fig_mapa, use_container_width=True)
else:
    st.info("No hay datos geográficos para la selección actual.")