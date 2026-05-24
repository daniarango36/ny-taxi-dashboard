import os
import pandas as pd
import streamlit as st
from src.ml_model import train_and_predict_demand

def load_and_preprocess_data(ruta_parquet, ruta_coords):
    """
    Carga el archivo parquet base y cruza las coordenadas de latitud y longitud
    tanto para el origen (Pick-Up) como para el destino (Drop-Off).
    """
    if not os.path.exists(ruta_parquet) or not os.path.exists(ruta_coords):
        print(f"Error: No se encontraron los archivos base en {ruta_parquet} o {ruta_coords}")
        return pd.DataFrame()
        
    df = pd.read_parquet(ruta_parquet)
    df_coords = pd.read_csv(ruta_coords)
    
    # Cruce de datos para Origen (PU)
    df_pu = df_coords.rename(columns={
        'LocationID': 'PULocationID', 
        'latitude': 'PU_lat', 
        'longitude': 'PU_lon',
        'Zone': 'PU_Zone'
    })
    df = df.merge(df_pu[['PULocationID', 'PU_Zone', 'PU_lat', 'PU_lon']], on='PULocationID', how='left')
    
    # Cruce de datos para Destino (DO)
    df_do = df_coords.rename(columns={
        'LocationID': 'DOLocationID', 
        'latitude': 'DO_lat', 
        'longitude': 'DO_lon',
        'Zone': 'DO_Zone'
    })
    df = df.merge(df_do[['DOLocationID', 'DO_Zone', 'DO_lat', 'DO_lon']], on='DOLocationID', how='left')
    
    return df

@st.cache_data(show_spinner=False)
def obtener_datos_kpi():
    """Genera localmente o lee en la nube los datos predictivos para la Página 1"""
    ruta_final = 'data/kpi_listo.parquet'
    
    # MODO NUBE: Si ya existe el archivo preprocesado, lo lee directamente
    if os.path.exists(ruta_final):
        return pd.read_parquet(ruta_final)
    
    # MODO LOCAL: Si no existe, realiza el procesamiento y entrenamiento pesado
    print("Preprocesando datos y entrenando modelos de Machine Learning...")
    df_raw = load_and_preprocess_data('data/resultado_agregado_total.parquet', 'data/taxi_zone_lookup_coordinates.csv')
    if df_raw.empty:
        return pd.DataFrame()
        
    df_pred = train_and_predict_demand(df_raw)
    df_coords = pd.read_csv('data/taxi_zone_lookup_coordinates.csv')

    # Garantizar que las zonas queden mapeadas en el dataframe final
    if 'PU_Zone' not in df_pred.columns:
        df_pred = df_pred.merge(df_coords[['LocationID', 'Zone']], left_on='PULocationID', right_on='LocationID', how='left').rename(columns={'Zone': 'PU_Zone'})
    if 'DO_Zone' not in df_pred.columns:
        df_pred = df_pred.merge(df_coords[['LocationID', 'Zone']], left_on='DOLocationID', right_on='LocationID', how='left').rename(columns={'Zone': 'DO_Zone'})
    
    # Guardar el archivo listo para distribución
    os.makedirs('data', exist_ok=True)
    df_pred.to_parquet(ruta_final, index=False)
    print(f"¡Archivo definitivo generado con éxito en: {ruta_final}!")
    return df_pred

@st.cache_data(show_spinner=False)
def obtener_datos_mapa():
    """Genera localmente o lee en la nube los datos geoespaciales para la Página 2"""
    ruta_final = 'data/mapa_listo.parquet'
    
    # MODO NUBE: Lectura directa instantánea
    if os.path.exists(ruta_final):
        return pd.read_parquet(ruta_final)
    
    # MODO LOCAL: Formateo y consolidación geográfica
    print("Generando matriz geoespacial optimizada...")
    df_raw = load_and_preprocess_data('data/resultado_agregado_total.parquet', 'data/taxi_zone_lookup_coordinates.csv')
    if df_raw.empty:
        return pd.DataFrame()
        
    # Estandarizar formato de fechas sin horas remanentes
    df_raw['fecha'] = pd.to_datetime(df_raw['fecha']).dt.strftime('%Y-%m-%d')
    
    os.makedirs('data', exist_ok=True)
    df_raw.to_parquet(ruta_final, index=False)
    print(f"¡Archivo definitivo generado con éxito en: {ruta_final}!")
    return df_raw