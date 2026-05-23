import pandas as pd
import geopandas as gpd
import streamlit as st
import os

@st.cache_data(show_spinner="Cargando y procesando datos...")
def load_and_preprocess_data(parquet_path: str, csv_path: str) -> pd.DataFrame:
    """
    Carga el archivo parquet y el CSV, realiza limpieza temporal y cruces básicos.
    """
    if not os.path.exists(parquet_path) or not os.path.exists(csv_path):
        st.error(f"Faltan archivos de datos. Verifica: {parquet_path} o {csv_path}")
        return pd.DataFrame()

    # 1. Cargar datos
    df = pd.read_parquet(parquet_path)
    df_coords = pd.read_csv(csv_path)

    # 2. Preparación Temporal
    # Parsear a datetime
    df['fecha'] = pd.to_datetime(df['fecha'])
    
    # Asegurar que rango_hora sea ordinal (ej: '00:00-03:59' -> 1)
    # Se asume que viene como string, lo limpiamos y extraemos la hora inicial para ordenar
    df['rango_hora_ordinal'] = df['rango_hora'].apply(lambda x: int(str(x).split('-')[0].split(':')[0]) if isinstance(x, str) else x)

    # 3. Preparación Geoespacial (Merge)
    # Hacemos merge para Origen (PULocationID)
    df = df.merge(
        df_coords, 
        left_on='PULocationID', 
        right_on='LocationID', 
        how='left'
    ).rename(columns={'longitude': 'PU_lon', 'latitude': 'PU_lat', 'Zone': 'PU_Zone'})

    # Hacemos merge para Destino (DOLocationID)
    df = df.merge(
        df_coords, 
        left_on='DOLocationID', 
        right_on='LocationID', 
        how='left'
    ).rename(columns={'longitude': 'DO_lon', 'latitude': 'DO_lat', 'Zone': 'DO_Zone'})

    return df

@st.cache_data(show_spinner="Generando geometrías (EPSG:4326)...")
def create_geodataframe(df: pd.DataFrame, lon_col: str, lat_col: str) -> gpd.GeoDataFrame:
    """
    Convierte un DataFrame de pandas a un GeoDataFrame de Geopandas.
    """
    # =====================================================================
    # MEJORES PRÁCTICAS PARA GEO-PREPARACIÓN EN STREAMLIT
    # =====================================================================
    # 1. Limpieza de nulos: gpd.points_from_xy fallará si hay coordenadas nulas.
    df_clean = df.dropna(subset=[lon_col, lat_col]).copy()
    
    # 2. Creación de la columna 'geometry':
    # Convertimos los pares de longitud y latitud en objetos 'Point' de Shapely.
    # geopandas optimiza esto internamente con arrays vectorizados.
    geometries = gpd.points_from_xy(df_clean[lon_col], df_clean[lat_col])
    
    # 3. Inicialización del GeoDataFrame
    gdf = gpd.GeoDataFrame(df_clean, geometry=geometries)
    
    # 4. ASIGNACIÓN DEL CRS (Sistema de Referencia de Coordenadas)
    # epsg=4326 corresponde a WGS 84 (World Geodetic System 1984), que es el 
    # estándar global para GPS y mapas web (incluyendo PyDeck, Folium, Google Maps).
    # Sin este paso, las librerías de mapeo no sabrán cómo proyectar los puntos
    # sobre el mapa base tridimensional y aparecerán en medio del océano o darán error.
    gdf.set_crs(epsg=4326, inplace=True)
    # =====================================================================
    
    return gdf