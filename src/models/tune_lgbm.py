import pandas as pd
import numpy as np
import lightgbm as lgb
import os
import warnings
import optuna
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

DATA_DIR      = "data/raw/"
PROCESSED_DIR = "data/processed/"
OUTPUT_DIR    = "outputs/"

def rmsle(y_true, y_pred):
    y_pred = np.clip(y_pred, 0, None)
    return np.sqrt(np.mean((np.log1p(y_true) - np.log1p(y_pred))**2))

def create_lag_features(df):
    df = df.sort_values(['store_nbr', 'family', 'date'])
    grouped = df.groupby(['store_nbr', 'family'])['log_sales']
    
    df['lag_16'] = grouped.shift(16)
    df['lag_21'] = grouped.shift(21)
    df['lag_28'] = grouped.shift(28)
    df['lag_364'] = grouped.shift(364)
    
    df['rmean_7_lag_16'] = df.groupby(['store_nbr', 'family'])['lag_16'].transform(lambda x: x.rolling(7).mean())
    df['rmean_28_lag_16'] = df.groupby(['store_nbr', 'family'])['lag_16'].transform(lambda x: x.rolling(28).mean())
    
    return df

def prepare_data():
    df = pd.read_parquet(os.path.join(PROCESSED_DIR, 'base_features.parquet'))
    df['date'] = pd.to_datetime(df['date'])
    df = create_lag_features(df)
    
    cat_cols = ['family', 'type', 'city', 'state', 'holiday_type']
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].fillna("Unknown").astype(str))
        df[col] = df[col].astype('category')
        
    df['cluster'] = df['cluster'].astype('category')
    df['store_nbr'] = df['store_nbr'].astype('category')
    
    train_start = pd.to_datetime('2016-05-01') 
    val_start = pd.to_datetime('2017-08-01')
    val_end = pd.to_datetime('2017-08-15')
    
    df = df[df['date'] >= train_start]
    
    features = [
        'store_nbr', 'family', 'onpromotion', 'dcoilwtico', 
        'cluster', 'type', 'holiday_type', 'is_payday', 'earthquake_effect',
        'day_of_week', 'month', 'day_of_month', 'year',
        'lag_16', 'lag_21', 'lag_28', 'lag_364', 
        'rmean_7_lag_16', 'rmean_28_lag_16'
    ]
    
    train_df = df[(df['date'] < val_start) & (df['is_test'] == 0)].copy()
    train_df = train_df[train_df['is_dead_series'] == 0]
    val_df = df[(df['date'] >= val_start) & (df['date'] <= val_end) & (df['is_test'] == 0)].copy()
    
    return train_df, val_df, features

def objective(trial, train_df, val_df, features):
    X_train = train_df[features]
    y_train = train_df['log_sales']
    X_val = val_df[features]
    y_val = val_df['log_sales']

    # Optimización de hiperparámetros
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'verbosity': -1,
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 20, 150),
        'max_depth': trial.suggest_int('max_depth', 5, 15),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 0.9),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 0.9),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 20, 100)
    }

    model = lgb.LGBMRegressor(**params, n_estimators=1000)
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)],
    )

    preds_log = model.predict(X_val)
    preds_real = np.expm1(preds_log)
    preds_real = np.clip(preds_real, 0, None)
    
    # Clampear series muertas
    preds_real[val_df['is_dead_series'] == 1] = 0
    
    # Calcular RMSLE local (nuestra métrica base real)
    error = rmsle(val_df['sales'], preds_real)
    return error

if __name__ == "__main__":
    import tempfile
    optuna.logging.set_verbosity(optuna.logging.INFO)
    
    print("1. Preparando datos para Tuning...")
    train_df, val_df, features = prepare_data()
    
    print("2. Iniciando Optuna (Buscaremos los mejores hiperparámetros - 15 trials temporales)...")
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, train_df, val_df, features), n_trials=15)
    
    print("\n[ MEJOR MODELO ENCONTRADO ]")
    print(f"Mejor RMSLE: {study.best_value:.4f}")
    print("Mejores Hiperparámetros:")
    for key, value in study.best_params.items():
        print(f"    {key}: {value}")
