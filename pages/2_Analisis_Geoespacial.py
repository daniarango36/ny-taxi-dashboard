import streamlit as st
import pydeck as pdk
import plotly.express as px
from src.data_processing import load_and_preprocess_data, create_geodataframe

st.set_page_config(page_title="Análisis Geoespacial", page_icon="🗺️", layout="wide")

st.title("🗺️ Análisis Geoespacial de Alta Performance")

# Carga
df_raw = load_and_preprocess_data('data/resultado_agregado_total.parquet', 'data/taxi_zone_lookup_coordinates.csv')
if df_raw.empty:
    st.stop()

# Filtros Sidebar
st.sidebar.header("Filtros Espaciales")
fechas = sorted(df_raw['fecha'].dt.date.unique())
fecha_sel = st.sidebar.select_slider("Fecha", options=fechas, value=fechas[-1])

indicador = st.sidebar.selectbox("Indicador Espacial", ["Origen (Llegada)", "Destino (Salida)"])

# Filtrar dataset base
df_map = df_raw[df_raw['fecha'].dt.date == fecha_sel].copy()

# Configurar variables según el indicador seleccionado
if indicador == "Origen (Llegada)":
    lon_col, lat_col, label_col = 'PU_lon', 'PU_lat', 'PU_Zone'
else:
    lon_col, lat_col, label_col = 'DO_lon', 'DO_lat', 'DO_Zone'

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Mapa de Calor Temporal")
    # Matriz fecha/rango hora
    # Para visualizar un periodo más amplio en el heatmap, tomamos los últimos 7 días
    df_heatmap = df_raw[(df_raw['fecha'].dt.date <= fecha_sel) & 
                        (df_raw['fecha'].dt.date >= fecha_sel - pd.Timedelta(days=7))]
    
    # Agrupar por fecha y rango hora ordinal
    heat_data = df_heatmap.groupby([df_heatmap['fecha'].dt.date, 'rango_hora'])['conteo'].sum().reset_index()
    
    fig_heat = px.density_heatmap(
        heat_data, 
        x="fecha", 
        y="rango_hora", 
        z="conteo",
        color_continuous_scale="Viridis",
        title="Volumen por Día y Rango Horario"
    )
    st.plotly_chart(fig_heat, use_container_width=True)

with col2:
    st.subheader(f"Densidad de Taxis: {indicador}")
    
    # Crear GeoDataFrame
    gdf = create_geodataframe(df_map, lon_col, lat_col)
    
    # =====================================================================
    # CONFIGURACIÓN DE PYDECK: HEXAGON LAYER
    # =====================================================================
    # ¿Por qué HexagonLayer en lugar de ScatterplotLayer?
    # Cuando tenemos cientos de miles de puntos de taxis, dibujarlos individualmente
    # satura el navegador. HexagonLayer delega la agregación espacial a la GPU (WebGL).
    # Agrupa los puntos en "bins" hexagonales; la altura y el color del hexágono 
    # representan automáticamente el conteo (o la suma de una métrica) en esa zona.
    
    layer = pdk.Layer(
        "HexagonLayer",
        data=gdf,
        get_position=["geometry.x", "geometry.y"], # Usa la longitud (x) y latitud (y) de Geopandas
        auto_highlight=True,
        elevation_scale=50,      # Multiplicador para la altura de los hexágonos 3D
        pickable=True,           # Habilita interactividad (tooltips)
        elevation_range=[0, 3000], 
        extruded=True,           # True = Renderiza en 3D
        coverage=1,
        radius=200,              # Radio de cobertura de cada hexágono en metros
        get_color_weight="conteo", # Variable a sumar dentro del hexágono
    )

    # Configuración de la cámara inicial
    # Se centra explícitamente en Manhattan/NY y se aplica un pitch (inclinación) 
    # para que la extrusión 3D de los hexágonos sea visible.
    view_state = pdk.ViewState(
        longitude=-74.0060,
        latitude=40.7128,
        zoom=10,
        min_zoom=5,
        max_zoom=15,
        pitch=40.5,
        bearing=-27.36
    )

    # Renderizar el mapa en Streamlit
    st.pydeck_chart(pdk.Deck(
        layers=[layer],
        initial_view_state=view_state,
        map_style='mapbox://styles/mapbox/dark-v10',
        tooltip={"text": "Densidad de viajes en esta área: {elevationValue}"}
    ))