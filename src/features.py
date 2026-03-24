import pandas as pd
import numpy as np
import warnings
import os

# Rutas del proyecto (ejecución local)
DATA_DIR      = "data/raw/"
PROCESSED_DIR = "data/processed/"
OUTPUT_DIR    = "outputs/"

# Elimina advertencias pandas
warnings.filterwarnings("ignore")

def load_data():
    """Carga los datasets base."""
    train = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'), parse_dates=['date'])
    test = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'), parse_dates=['date'])
    stores = pd.read_csv(os.path.join(DATA_DIR, 'stores.csv'))
    oil = pd.read_csv(os.path.join(DATA_DIR, 'oil.csv'), parse_dates=['date'])
    holidays = pd.read_csv(os.path.join(DATA_DIR, 'holidays_events.csv'), parse_dates=['date'])
    return train, test, stores, oil, holidays

def process_oil(oil_df, all_dates):
    """Interpola los NaNs de oil y rellena fines de semana."""
    oil_df = oil_df.set_index('date').reindex(all_dates)
    # Interpolar hacia adelante y llenar últimos vacíos iniciales hacia atrás
    oil_df['dcoilwtico'] = oil_df['dcoilwtico'].interpolate(method='linear', limit_direction='both')
    return oil_df.reset_index().rename(columns={'index': 'date'})

def process_holidays(holidays_df):
    """Limpia los feriados basándose en la regla lógica de Transferidos."""
    # Los que fueron transferidos a otro día, operan como día normal
    holidays_real = holidays_df[holidays_df['transferred'] == False].copy()
    
    # Manejamos múltiples feriados en la misma fecha tomando el primero o más importante
    # (Regional y National superan a Local).
    # Simplificación: Tomaremos solo el tipo de feriado y la descripción
    holidays_real = holidays_real.drop_duplicates(subset=['date'], keep='first')
    
    holidays_real = holidays_real[['date', 'type']].rename(columns={'type': 'holiday_type'})
    return holidays_real

def create_time_features(df):
    """Características temporales y eventos estáticos."""
    df['day_of_week'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df['day_of_month'] = df['date'].dt.day
    
    # Quincena flag: Ecuador paga sector público el 15 y fin de mes
    df['is_payday'] = df['day_of_month'].isin([14, 15, 30, 31]).astype(int)
    
    # Terremoto (abril y mayo 2016)
    df['earthquake_effect'] = ((df['date'] >= '2016-04-16') & (df['date'] <= '2016-05-16')).astype(int)
    return df

def identify_dead_series(train_df):
    """Busca las series (store + family) que tengan > 99% de ventas 0."""
    total_days = train_df['date'].nunique()
    zeros = train_df[train_df['sales'] == 0].groupby(['store_nbr', 'family']).size().reset_index(name='zero_days')
    zeros['zero_pct'] = zeros['zero_days'] / total_days
    
    dead_series = zeros[zeros['zero_pct'] >= 0.99][['store_nbr', 'family']]
    dead_series['is_dead_series'] = 1
    return dead_series

def build_features():
    print("1. Cargando datos...")
    train, test, stores, oil, holidays = load_data()
    
    # Marcar train/test
    train['is_test'] = 0
    test['is_test'] = 1
    test['sales'] = np.nan
    
    # Concatenar para procesar características en conjunto
    df = pd.concat([train, test], ignore_index=True)
    all_dates = pd.date_range(start=df['date'].min(), end=df['date'].max())
    
    print("2. Procesando Exógenas (Oil, Feriados, Identificadores muertos)...")
    oil_clean = process_oil(oil, all_dates)
    holidays_clean = process_holidays(holidays)
    dead_series = identify_dead_series(train)
    
    print("3. Ejecutando Merges...")
    # Merge de oil, holidays y stores
    df = df.merge(oil_clean, on='date', how='left')
    df = df.merge(holidays_clean, on='date', how='left')
    df = df.merge(stores, on='store_nbr', how='left')
    
    # Merge de dead series
    df = df.merge(dead_series, on=['store_nbr', 'family'], how='left')
    df['is_dead_series'] = df['is_dead_series'].fillna(0).astype(int)
    
    print("4. Extrayendo variables de Tiempo y Calendario...")
    df = create_time_features(df)
    df['holiday_type'] = df['holiday_type'].fillna('WorkDay')
    
    # Transformación RMSLE natural
    df['log_sales'] = np.log1p(df['sales'].clip(lower=0))

    print("4.5. Creando Target Encodings Históricos...")
    # Calcular promedios históricos excluyendo la data de test del futuro
    train_for_te = df[df['is_test'] == 0]
    
    # 1. Promedio histórico por tienda y familia
    te_store_fam = train_for_te.groupby(['store_nbr', 'family'])['log_sales'].mean().reset_index(name='te_store_fam')
    
    # 2. Promedio histórico por tienda, familia y dia de la semana
    te_store_fam_dow = train_for_te.groupby(['store_nbr', 'family', 'day_of_week'])['log_sales'].mean().reset_index(name='te_store_fam_dow')
    
    # Printear ambas al DF principal
    df = df.merge(te_store_fam, on=['store_nbr', 'family'], how='left')
    df = df.merge(te_store_fam_dow, on=['store_nbr', 'family', 'day_of_week'], how='left')
    
    print("5. Guardando dataset maestro...")
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    df.to_parquet(os.path.join(PROCESSED_DIR, 'base_features.parquet'), index=False)
    print("Dataset de base completado. Dimensiones:", df.shape)

if __name__ == "__main__":
    build_features()
