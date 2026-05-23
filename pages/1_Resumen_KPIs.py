import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from src.data_processing import load_and_preprocess_data
from src.ml_model import train_and_predict_demand
import os

st.set_page_config(page_title="Resumen y KPIs", page_icon="📊", layout="wide")

st.title("📊 Resumen de KPIs y Predicción")

# Carga de datos
df_raw = load_and_preprocess_data('data/resultado_agregado_total.parquet', 'data/taxi_zone_lookup_coordinates.csv')
if df_raw.empty:
    st.stop()

df_pred = train_and_predict_demand(df_raw)

# Sidebar: Filtros
st.sidebar.header("Filtros")
semanas_disponibles = sorted(df_pred['numero_semana'].unique())
ultima_semana_real = df_pred[df_pred['tipo'] == 'Real']['numero_semana'].max()

semana_sel = st.sidebar.multiselect(
    "Número de Semana", 
    options=semanas_disponibles, 
    default=[ultima_semana_real]
)

# Filtrado
df_filtered = df_pred[df_pred['numero_semana'].isin(semana_sel)] if semana_sel else df_pred

# Cálculos KPI Global
total_real = df_filtered[df_filtered['tipo'] == 'Real']['conteo'].sum()
total_pred = df_filtered[df_filtered['tipo'] == 'Predictivo']['conteo'].sum()

# Evitar división por cero
cumplimiento = (total_real / total_pred * 100) if total_pred > 0 else 0

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Métricas de Cumplimiento")
    st.metric(label="Total Viajes (Reales)", value=f"{total_real:,.0f}")
    
    # Semáforo de Gauge Plotly
    fig_gauge = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = cumplimiento,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "% Cumplimiento vs Predicción"},
        gauge = {
            'axis': {'range': [None, 150]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 80], 'color': "lightcoral"},
                {'range': [80, 100], 'color': "lightyellow"},
                {'range': [100, 150], 'color': "lightgreen"}
            ],
            'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 100}
        }
    ))
    fig_gauge.update_layout(height=300, margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig_gauge, use_container_width=True)
    
    st.markdown("### Estado Visual: " + ("👍 Óptimo" if cumplimiento >= 90 else "👎 Bajo Expectativa"))

with col2:
    st.subheader("Top 5 Zonas (Origen)")
    top_5 = df_filtered[df_filtered['tipo'] == 'Real'].groupby('PULocationID')['conteo'].sum().nlargest(5).reset_index()
    top_5['PULocationID'] = top_5['PULocationID'].astype(str)
    
    # Asignar color verde/rojo de forma ilustrativa (ejemplo basado en media)
    media_viajes = top_5['conteo'].mean()
    top_5['Color'] = top_5['conteo'].apply(lambda x: 'green' if x >= media_viajes else 'red')
    
    fig_bar = px.bar(top_5, x='PULocationID', y='conteo', color='Color', 
                     color_discrete_map={'green': '#2ecc71', 'red': '#e74c3c'},
                     title="Volumen Real Top 5 Orígenes")
    st.plotly_chart(fig_bar, use_container_width=True)

# Serie de tiempo
st.markdown("---")
st.subheader("Evolución Temporal (Últimas 8 semanas)")
# Filtramos las últimas 8 semanas de todo el dataframe (real + predictivo)
top_5_ids = top_5['PULocationID'].astype(int).tolist()
df_ts = df_pred[(df_pred['PULocationID'].isin(top_5_ids)) & 
                (df_pred['numero_semana'] >= ultima_semana_real - 7)]

fig_line = px.line(df_ts, x='numero_semana', y='conteo', color='PULocationID', 
                   line_dash='tipo', markers=True,
                   title="Serie de Tiempo: Real vs Proyección")
st.plotly_chart(fig_line, use_container_width=True)