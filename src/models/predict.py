import pandas as pd
import numpy as np
import lightgbm as lgb
import os
import warnings
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

DATA_DIR      = "data/raw/"
PROCESSED_DIR = "data/processed/"
OUTPUT_DIR    = "outputs/"

def create_lag_features(df):
    print("   -> Generando lags temporales...")
    df = df.sort_values(['store_nbr', 'family', 'date'])
    grouped = df.groupby(['store_nbr', 'family'])['log_sales']
    
    # Mantenemos los mismos lags que validamos exitosamente
    df['lag_16'] = grouped.shift(16)
    df['lag_21'] = grouped.shift(21)
    df['lag_28'] = grouped.shift(28)
    df['lag_364'] = grouped.shift(364)
    
    df['rmean_7_lag_16'] = df.groupby(['store_nbr', 'family'])['lag_16'].transform(lambda x: x.rolling(7).mean())
    df['rmean_28_lag_16'] = df.groupby(['store_nbr', 'family'])['lag_16'].transform(lambda x: x.rolling(28).mean())
    return df

def generate_submission():
    print("1. Cargando base_features.parquet...")
    df = pd.read_parquet(os.path.join(PROCESSED_DIR, 'base_features.parquet'))
    df['date'] = pd.to_datetime(df['date'])
    
    # Construcción de Lags Dinámicos
    df = create_lag_features(df)
    
    # Codificaciones
    print("2. Codificando variables...")
    cat_cols = ['family', 'type', 'city', 'state', 'holiday_type']
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].fillna("Unknown").astype(str))
        df[col] = df[col].astype('category')
        
    df['cluster'] = df['cluster'].astype('category')
    df['store_nbr'] = df['store_nbr'].astype('category')
    
    # Definición limpia de todo el entrenamiento (Train va hasta el 15 Agosto)
    # y los datos de la competencia están marcados como is_test=1.
    train_start = pd.to_datetime('2016-05-01') 
    
    df_model = df[df['date'] >= train_start].copy()
    
    features = [
        'store_nbr', 'family', 'onpromotion', 'dcoilwtico', 
        'cluster', 'type', 'holiday_type', 'is_payday', 'earthquake_effect',
        'day_of_week', 'month', 'day_of_month', 'year',
        'lag_16', 'lag_21', 'lag_28', 'lag_364', 
        'rmean_7_lag_16', 'rmean_28_lag_16',
        'te_store_fam', 'te_store_fam_dow'
    ]
    
    # Separación final Train y Test de Kaggle
    train_df = df_model[(df_model['is_test'] == 0) & (df_model['is_dead_series'] == 0)].copy()
    test_df = df_model[df_model['is_test'] == 1].copy()
    
    X_train = train_df[features]
    y_train = train_df['log_sales']
    
    X_test = test_df[features]
    
    print("\n3. Entrenando Modelo LightGBM Final con Toda la Data Histórica...")
    # Usamos la hiper-parametrización ganadora de Optuna
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
        'n_estimators': 210,
        'verbose': -1
    }
    
    model = lgb.LGBMRegressor(**params)
    model.fit(X_train, y_train)
    
    print("\n4. Generando Predicciones Futuras (Submisión)...")
    preds_log = model.predict(X_test)
    preds_real = np.expm1(preds_log)
    preds_real = np.clip(preds_real, 0, None)
    
    # Inyectar reglas duras
    test_df['pred_sales'] = preds_real
    test_df.loc[test_df['is_dead_series'] == 1, 'pred_sales'] = 0
    
    # Formatear archivo submisión (usando 'id' directamente)
    submission = test_df[['id', 'pred_sales']].rename(columns={'pred_sales': 'sales'})
    submission['id'] = submission['id'].astype(int)
    
    # Manejar posibles NaNs forzando 0
    submission['sales'] = submission['sales'].fillna(0)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_file = os.path.join(OUTPUT_DIR, 'submission_final.csv')
    submission.to_csv(out_file, index=False)
    
    print(f"\n[ EXITOSO ] Archivo de predicción guardado con éxito en: {out_file}")
    print("Listo para ser enviado o evaluado externamente. \N{ROCKET}")

if __name__ == "__main__":
    generate_submission()
