import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from src.data_processing import load_and_preprocess_data
from src.ml_model import train_and_predict_demand

st.set_page_config(page_title="KPI Dashboard", layout="wide")

st.title("📊 Resumen y KPIs de Control Operativo")
st.markdown("---")

# Carga de Datos y Modelo
df_raw = load_and_preprocess_data('data/resultado_agregado_total.parquet', 'data/taxi_zone_lookup_coordinates.csv')
if df_raw.empty:
    st.stop()

df_pred = train_and_predict_demand(df_raw)
df_coords = pd.read_csv('data/taxi_zone_lookup_coordinates.csv')

df_pred = df_pred.merge(df_coords[['LocationID', 'Zone']], left_on='PULocationID', right_on='LocationID', how='left').rename(columns={'Zone': 'PU_Zone'})
df_pred = df_pred.merge(df_coords[['LocationID', 'Zone']], left_on='DOLocationID', right_on='LocationID', how='left').rename(columns={'Zone': 'DO_Zone'})

# --- FILTROS GLOBALES ---
st.sidebar.header("Filtros Globales")

semanas_disponibles = sorted(df_pred['numero_semana'].unique())
ultima_semana_real = df_pred[df_pred['tipo'] == 'Real']['numero_semana'].max()

semanas_seleccionadas = st.sidebar.multiselect(
    "Número de Semana",
    options=semanas_disponibles,
    default=[ultima_semana_real] if pd.notna(ultima_semana_real) else []
)

lista_lugares = sorted(list(set(df_pred['PU_Zone'].dropna().unique()) | set(df_pred['DO_Zone'].dropna().unique())))
lugares_seleccionados = st.sidebar.multiselect(
    "Filtrar por Zona / Lugar",
    options=lista_lugares,
    default=[]
)

if semanas_seleccionadas:
    df_filtrado = df_pred[df_pred['numero_semana'].isin(semanas_seleccionadas)]
else:
    st.warning("⚠️ Por favor, selecciona al menos una semana en el panel lateral.")
    st.stop()

if lugares_seleccionados:
    df_filtrado = df_filtrado[
        (df_filtrado['PU_Zone'].isin(lugares_seleccionados)) | 
        (df_filtrado['DO_Zone'].isin(lugares_seleccionados))
    ]

# --- MÉTRICAS GLOBALES ---
st.subheader("Métricas Globales de Cumplimiento")

total_real = df_filtrado[df_filtrado['tipo'] == 'Real']['conteo'].sum()
total_predictivo = df_filtrado[df_filtrado['tipo'] == 'Predictivo']['conteo'].sum()
porcentaje_cumplimiento = (total_real / total_predictivo) * 100 if total_predictivo > 0 else 0

col1, col2 = st.columns([1, 2])

with col1:
    indicador = "👍 Excelente Desempeño" if porcentaje_cumplimiento >= 90 else "👎 Ajustar Planificación"
    st.metric(
        label="Porcentaje de Cumplimiento Total", 
        value=f"{porcentaje_cumplimiento:.1f}%",
        delta=indicador,
        delta_color="normal" if porcentaje_cumplimiento >= 90 else "inverse"
    )
    st.write(f"**Viajes Reales:** {total_real:,.0f}")
    st.write(f"**Viajes Proyectados:** {total_predictivo:,.0f}")

with col2:
    fig_gauge = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = porcentaje_cumplimiento,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Progreso de Cumplimiento Global", 'font': {'size': 16}},
        gauge = {
            'axis': {'range': [None, 150], 'tickwidth': 1},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 85], 'color': "#FF9999"},
                {'range': [85, 95], 'color': "#FFFF99"},
                {'range': [95, 150], 'color': "#99FF99"}
            ],
            'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 100}
        }
    ))
    fig_gauge.update_layout(height=250, margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig_gauge, use_container_width=True)

st.markdown("---")

# --- GRÁFICOS POR ZONA ---
def generar_bloque_graficos(tipo_str, col_zona):
    st.header(f"Sección Análisis por {tipo_str}")
    
    real_counts = df_filtrado[df_filtrado['tipo'] == 'Real'].groupby(col_zona)['conteo'].sum()
    pred_counts = df_filtrado[df_filtrado['tipo'] == 'Predictivo'].groupby(col_zona)['conteo'].sum()
    
    top_zones = pd.DataFrame({'Real': real_counts, 'Predictivo': pred_counts}).fillna(0).reset_index()
    top_zones['cumplimiento'] = (top_zones['Real'] / top_zones['Predictivo']) * 100
    top_zones.fillna({'cumplimiento': 0}, inplace=True)
    
    top_zones = top_zones.sort_values(by='Real', ascending=False).head(5)
    top_zones['color'] = top_zones['cumplimiento'].apply(lambda x: '#2E7D32' if x >= 90 else '#C62828')
    
    c1, c2 = st.columns(2)
    
    with c1:
        st.subheader(f"Top 5 Zonas (Volumen Real vs Proyección)")
        fig_hist = px.bar(
            top_zones, 
            x=col_zona, 
            y='Real',
            text=top_zones['cumplimiento'].apply(lambda x: f"{x:.1f}%"),
            labels={'Real': 'Viajes Reales', col_zona: 'Zona'},
        )
        fig_hist.update_traces(marker_color=top_zones['color'], textposition='outside')
        fig_hist.update_layout(height=350, margin=dict(t=20, b=20))
        st.plotly_chart(fig_hist, use_container_width=True)
        
    with c2:
        st.subheader(f"Tendencia de Demanda Histórica y Proyección Futura")
        
        # Filtrar el histórico completo únicamente para las top 5 zonas
        df_historico = df_pred[df_pred[col_zona].isin(top_zones[col_zona])]
        df_linea = df_historico.groupby(['numero_semana', col_zona, 'tipo'])['conteo'].sum().reset_index()
        
        # Determinar la frontera entre el pasado real y el futuro predictivo
        max_semana_real = df_linea[df_linea['tipo'] == 'Real']['numero_semana'].max()
        
        # 1. Tramo Histórico: Mantener estrictamente el 'Real'
        df_real_part = df_linea[df_linea['tipo'] == 'Real'].copy()
        
        # 2. Tramo Futuro: Mantener el 'Predictivo' solo para las semanas nuevas
        df_pred_part = df_linea[(df_linea['tipo'] == 'Predictivo') & (df_linea['numero_semana'] > max_semana_real)].copy()
        
        # 3. Punto de Conexión: Clonar el último punto real como predictivo para evitar quiebres en la gráfica
        puntos_conexion = []
        for zona in df_real_part[col_zona].unique():
            df_zona_real = df_real_part[df_real_part[col_zona] == zona]
            if not df_zona_real.empty:
                ultimo_punto = df_zona_real.loc[df_zona_real['numero_semana'].idxmax()].copy()
                ultimo_punto['tipo'] = 'Predictivo'  # Lo etiquetamos para que empalme con el tramo futuro
                puntos_conexion.append(ultimo_punto)
                
        if puntos_conexion:
            df_pred_part = pd.concat([df_pred_part, pd.DataFrame(puntos_conexion)], ignore_index=True)
            
        # Unificar tramos ordenando cronológicamente
        df_grafico_linea = pd.concat([df_real_part, df_pred_part], ignore_index=True).sort_values(by=['numero_semana'])
        
        # Graficar controlando explícitamente el estilo de la línea según el 'tipo'
        fig_line = px.line(
            df_grafico_linea, 
            x='numero_semana', 
            y='conteo', 
            color=col_zona,
            line_dash='tipo',
            line_dash_map={'Real': 'solid', 'Predictivo': 'dash'}, # Real = Continua | Predictivo = Punteada
            markers=True,
            labels={'conteo': 'Volumen de Viajes', 'numero_semana': 'Semana', 'tipo': 'Métrica'}
        )
        
        fig_line.update_layout(
            height=350, 
            margin=dict(t=20, b=20),
            xaxis=dict(tickmode='linear', dtick=1) # Asegura que se vean todas las semanas enteras
        )
        st.plotly_chart(fig_line, use_container_width=True)

generar_bloque_graficos("Origen (Pick-Up)", "PU_Zone")
st.markdown("---")
generar_bloque_graficos("Destino (Drop-Off)", "DO_Zone")