import pandas as pd
import numpy as np
import os

# Rutas del proyecto (ejecución local)
DATA_DIR      = "data/raw/"
PROCESSED_DIR = "data/processed/"
OUTPUT_DIR    = "outputs/"

def rmsle(y_true, y_pred):
    """Calcula Root Mean Squared Logarithmic Error"""
    y_pred = np.clip(y_pred, 0, None)
    return np.sqrt(np.mean((np.log1p(y_true) - np.log1p(y_pred))**2))

def run_baseline():
    print("1. Cargando base_features.parquet...")
    df = pd.read_parquet(os.path.join(PROCESSED_DIR, 'base_features.parquet'))
    
    # Filtrar solo la parte de train original (ignorar kaggle test set por ahora)
    # y asegurarnos de que la fecha sea tipo datetime
    df['date'] = pd.to_datetime(df['date'])
    df = df[df['is_test'] == 0].copy()
    
    # Definir Validation Split (últimos 15 días: 2017-08-01 a 2017-08-15)
    val_start = pd.to_datetime('2017-08-01')
    
    train_data = df[df['date'] < val_start]
    val_data = df[df['date'] >= val_start].copy()
    
    print(f"   Train: hasta {train_data['date'].max().date()}")
    print(f"   Val: {val_data['date'].min().date()} a {val_data['date'].max().date()}")
    
    # Modelo Baseline: "Seasonal Naive Móvil"
    # Tomaremos el promedio de ventas por store, family y DÍA DE LA SEMANA de los últimos 28 días
    print("\n2. Entrenando Seasonal Naive Baseline (Media de las últimas 4 semanas por día de la semana)...")
    train_28 = train_data[train_data['date'] >= (val_start - pd.Timedelta(days=28))]
    
    baseline_model = train_28.groupby(['store_nbr', 'family', 'day_of_week'])['sales'].mean().reset_index()
    baseline_model = baseline_model.rename(columns={'sales': 'pred_sales'})
    
    # Generar predicciones en la data de validación cruzando la lógica generada
    val_pred = val_data.merge(baseline_model, on=['store_nbr', 'family', 'day_of_week'], how='left')
    
    # Llenar NaNs de combinaciones sin historia reciente con 0
    val_pred['pred_sales'] = val_pred['pred_sales'].fillna(0)
    
    # Series Muertas = 0
    val_pred.loc[val_pred['is_dead_series'] == 1, 'pred_sales'] = 0
    
    print("\n3. Calculando métricas de evaluación...")
    error = rmsle(val_pred['sales'], val_pred['pred_sales'])
    mae = np.mean(np.abs(val_pred['sales'] - val_pred['pred_sales']))
    
    print(f"   ==> Baseline RMSLE: {error:.4f}")
    print(f"   ==> Baseline MAE:   {mae:.2f}")
    
    # Guardar predicciones base para inspección
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, 'pred_baseline_val.csv')
    val_pred[['date', 'store_nbr', 'family', 'sales', 'pred_sales']].to_csv(out_path, index=False)
    print(f"\nPredicciones de validación guardadas en {out_path}")

if __name__ == "__main__":
    run_baseline()
