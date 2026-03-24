import pandas as pd
import numpy as np
import lightgbm as lgb
import os
import warnings
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

# Rutas del proyecto (ejecución local)
DATA_DIR      = "data/raw/"
PROCESSED_DIR = "data/processed/"
OUTPUT_DIR    = "outputs/"

def rmsle(y_true, y_pred):
    """Calcula Root Mean Squared Logarithmic Error nativo en ventas reales"""
    y_pred = np.clip(y_pred, 0, None)
    return np.sqrt(np.mean((np.log1p(y_true) - np.log1p(y_pred))**2))

def create_lag_features(df):
    print("Generando lags temporales y estadísticos móviles...")
    # Asegurar el orden crónico correcto
    df = df.sort_values(['store_nbr', 'family', 'date'])
    
    # Agrupación por unidad mínima lógica
    grouped = df.groupby(['store_nbr', 'family'])['log_sales']
    
    # El horizonte de test real de este proyecto es de 16 días (16 al 31 de agosto).
    # Para poder hacer un Direct Forecast y no un Iterative Forecast que acumula errores,
    # nuestro primer lag será de 16 días.
    df['lag_16'] = grouped.shift(16)
    df['lag_21'] = grouped.shift(21)
    df['lag_28'] = grouped.shift(28)
    df['lag_364'] = grouped.shift(364) # Anual (52 semanas x 7 = 364 exactas)
    
    # Ventanas móviles sobre el primer lag válido
    df['rmean_7_lag_16'] = df.groupby(['store_nbr', 'family'])['lag_16'].transform(lambda x: x.rolling(7).mean())
    df['rmean_28_lag_16'] = df.groupby(['store_nbr', 'family'])['lag_16'].transform(lambda x: x.rolling(28).mean())
    
    return df

def run_ml_pipeline():
    print("1. Cargando base_features.parquet...")
    df = pd.read_parquet(os.path.join(PROCESSED_DIR, 'base_features.parquet'))
    df['date'] = pd.to_datetime(df['date'])
    
    # Función para crear histórico
    df = create_lag_features(df)
    
    print("2. Codificando variables categóricas...")
    cat_cols = ['family', 'type', 'city', 'state', 'holiday_type']
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].fillna("Unknown").astype(str))
        df[col] = df[col].astype('category')  # A LightGBM le encanta la clase pandas 'category'
        
    df['cluster'] = df['cluster'].astype('category')
    df['store_nbr'] = df['store_nbr'].astype('category')
    
    print("3. Preparando validación y features...")
    # Para evitar un mar de NaN por culpa del 'lag_364', cortamos el inicio de datos.
    # Empezar en mayo 2016 nos quita los NaN y además ignora un poco del pico distractor del terremoto
    train_start = pd.to_datetime('2016-05-01') 
    val_start = pd.to_datetime('2017-08-01')
    val_end = pd.to_datetime('2017-08-15')
    
    df = df[df['date'] >= train_start]
    
    features = [
        'store_nbr', 'family', 'onpromotion', 'dcoilwtico', 
        'cluster', 'type', 'holiday_type', 'is_payday', 'earthquake_effect',
        'day_of_week', 'month', 'day_of_month', 'year',
        'lag_16', 'lag_21', 'lag_28', 'lag_364', 
        'rmean_7_lag_16', 'rmean_28_lag_16',
        'te_store_fam', 'te_store_fam_dow'
    ]
    
    # Train
    train_df = df[(df['date'] < val_start) & (df['is_test'] == 0)].copy()
    train_df = train_df[train_df['is_dead_series'] == 0] # 🚀 No ensuciar el optimizador con valores muertos
    
    # Validation
    val_df = df[(df['date'] >= val_start) & (df['date'] <= val_end) & (df['is_test'] == 0)].copy()
    
    X_train = train_df[features]
    y_train = train_df['log_sales']
    X_val = val_df[features]
    y_val = val_df['log_sales']
    
    print(f"   Shape Entrenamiento: {X_train.shape}")
    print(f"   Shape Validación: {X_val.shape}")
    
    print("\n4. Entrenando modelo LightGBM ML con hyper-tuning...")
    params = {
        'objective': 'rmse',
        'metric': 'rmse',
        'learning_rate': 0.05274424664118348,
        'num_leaves': 97,
        'max_depth': 7,
        'feature_fraction': 0.8384419786911796,
        'bagging_fraction': 0.717779887115876,
        'bagging_freq': 5,
        'min_data_in_leaf': 21,
        'n_estimators': 1500,
        'verbose': -1
    }
    
    model = lgb.LGBMRegressor(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=100)]
    )
    
    print("\n5. Validando contra RMSLE nativo...")
    # Predicción está en formato log(1 + x)
    val_pred_log = model.predict(X_val)
    
    # Convertir métrica devuelta a ventas reales usando exponencial
    val_df['pred_sales'] = np.expm1(val_pred_log)
    val_df['pred_sales'] = np.clip(val_df['pred_sales'], 0, None)
    
    # Las series muertas se les asigna 0.
    val_df.loc[val_df['is_dead_series'] == 1, 'pred_sales'] = 0
    
    # Cálculo final de performance
    error = rmsle(val_df['sales'], val_df['pred_sales'])
    mae = np.mean(np.abs(val_df['sales'] - val_df['pred_sales']))
    
    print(f"   ==> LightGBM RMSLE Final: {error:.4f} 🔥")
    print(f"   ==> LightGBM MAE Final:   {mae:.2f}")
    
    importance = pd.DataFrame({
        'Feature': features,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    print("\n[ Top 5 Variables de peso ]")
    print(importance.head(5).to_string(index=False))

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    val_df[['date', 'store_nbr', 'family', 'sales', 'pred_sales']].to_csv(os.path.join(OUTPUT_DIR, 'pred_lgbm_val.csv'), index=False)

if __name__ == "__main__":
    run_ml_pipeline()
