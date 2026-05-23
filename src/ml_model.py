import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
import streamlit as st

@st.cache_data(show_spinner="Entrenando redes neuronales (MLPRegressor)...")
def train_and_predict_demand(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pipeline de Machine Learning para predecir la demanda por zona.
    """
    if df.empty:
        return pd.DataFrame()

    # Agrupar por semana y location para crear el dataset de entrenamiento
    df_agg = df.groupby(['numero_semana', 'PULocationID', 'DOLocationID'])['conteo'].sum().reset_index()
    
    # Separamos en X e y
    X = df_agg[['numero_semana', 'PULocationID', 'DOLocationID']]
    y = df_agg['conteo']

    # Escalado de features (Crítico para Redes Neuronales)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split Train/Test
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # 1. Configuración de la Red Neuronal (5 capas ocultas, activación ReLU)
    # Mantenemos las neuronas relativamente bajas para evitar timeouts en el entorno web
    mlp = MLPRegressor(
        hidden_layer_sizes=(64, 64, 64, 64, 64), 
        activation='relu', 
        max_iter=500,
        random_state=42
    )

    # 2. GridSearch de Hiperparámetros (reducido para Streamlit)
    param_grid = {
        'learning_rate_init': [0.001, 0.01],
        'alpha': [0.0001, 0.001]
    }
    
    grid_search = GridSearchCV(mlp, param_grid, cv=3, n_jobs=-1, scoring='r2')
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_

    # 3. Construir el dataset consolidado (Últimas 5 semanas reales)
    max_semana = df_agg['numero_semana'].max()
    df_reales = df_agg[df_agg['numero_semana'] > max_semana - 5].copy()
    df_reales['tipo'] = 'Real'

    # 4. Proyección de 3 semanas hacia el futuro
    future_weeks = [max_semana + 1, max_semana + 2, max_semana + 3]
    unique_PUs = df_agg['PULocationID'].unique()
    unique_DOs = df_agg['DOLocationID'].unique()
    
    # Creamos un grid básico de combinaciones frecuentes para predecir (para evitar explosión de datos)
    top_routes = df_agg.groupby(['PULocationID', 'DOLocationID']).size().nlargest(100).index
    
    future_records = []
    for w in future_weeks:
        for pu, do in top_routes:
            future_records.append({'numero_semana': w, 'PULocationID': pu, 'DOLocationID': do})
            
    df_future = pd.DataFrame(future_records)
    
    # Predecir
    X_future_scaled = scaler.transform(df_future[['numero_semana', 'PULocationID', 'DOLocationID']])
    predictions = best_model.predict(X_future_scaled)
    
    # Evitar conteos negativos (ReLU a la salida o max)
    df_future['conteo'] = np.maximum(0, predictions).astype(int)
    df_future['tipo'] = 'Predictivo'

    # 5. Unificar
    df_final = pd.concat([df_reales, df_future], ignore_index=True)
    # Asegurar una sola fila por combinación
    df_final = df_final.groupby(['numero_semana', 'PULocationID', 'DOLocationID', 'tipo'])['conteo'].sum().reset_index()
    
    return df_final