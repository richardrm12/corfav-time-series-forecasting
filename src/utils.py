import pandas as pd
import os
import streamlit as st

DATA_DIR = "data/raw/"
OUTPUT_DIR = "outputs/"

@st.cache_data
def load_data():
    """Carga y re-estructura los datos necesarios (TODO EL HISTORIAL)."""
    train = pd.read_csv(os.path.join(DATA_DIR, 'train.csv.gz'), parse_dates=['date'])
    test = pd.read_csv(os.path.join(DATA_DIR, 'test.csv.gz'), parse_dates=['date'])
    
    try:
        preds = pd.read_csv(os.path.join(OUTPUT_DIR, 'submission_final.csv.gz'))
        test_preds = test.merge(preds, on='id', how='left')
    except FileNotFoundError:
        try:
            preds = pd.read_csv(os.path.join(OUTPUT_DIR, 'validation_final.csv.gz'))
            test_preds = test.merge(preds, on='id', how='left')
        except FileNotFoundError:
            st.error("No se encontró validation_final.csv.gz ni submission_final.csv.gz")
            test_preds = test.copy()
            test_preds['sales'] = 0

    return train, test_preds

def calculate_insights(train_filtro):
    """Calcula los insights dinámicos de fin de semana y promoción."""
    df_expl = train_filtro.copy()
    if not df_expl.empty:
        df_expl['is_weekend'] = df_expl['date'].dt.dayofweek >= 5
        df_expl['has_promo'] = df_expl['onpromotion'] > 0
        
        # 1. Fin de semana
        mean_weekend = df_expl[df_expl['is_weekend']]['sales'].mean()
        mean_weekday = df_expl[~df_expl['is_weekend']]['sales'].mean()
        
        if pd.isna(mean_weekend) or pd.isna(mean_weekday) or mean_weekday == 0:
            txt_finde = "Faltan datos para medir el efecto de fin de semana adecuadamente."
        else:
            var_finde = ((mean_weekend - mean_weekday) / mean_weekday) * 100
            if var_finde > 5:
                txt_finde = f"Se detecta un aumento promedio del {var_finde:.0f}% en el volumen de ventas durante los sábados y domingos."
            elif var_finde < -5:
                txt_finde = f"Se observa una disminución del {abs(var_finde):.0f}% en las ventas durante los fines de semana en esta categoría."
            else:
                txt_finde = "El consumo en fin de semana es muy similar al de lunes a viernes (distribución plana)."
                
        # 2. Promociones
        mean_promo = df_expl[df_expl['has_promo']]['sales'].mean()
        mean_no_promo = df_expl[~df_expl['has_promo']]['sales'].mean()
        
        if pd.isna(mean_promo) or pd.isna(mean_no_promo) or mean_no_promo == 0:
            txt_promo = "Históricamente, no hay registros de ventas conjuntas con promociones suficientes para medir impacto."
        else:
            multiplicador = mean_promo / mean_no_promo
            if multiplicador >= 1.1:
                txt_promo = f"Las promociones impulsan fuertemente la demanda, multiplicando las ventas habituales por x{multiplicador:.1f} en días de oferta."
            elif multiplicador <= 0.9:
                txt_promo = f"Estar en promoción curiosamente coincide con menos ventas promedio (x{multiplicador:.1f})."
            else:
                txt_promo = f"Para esta selección, poner el producto en promoción no altera dramáticamente la demanda base (x{multiplicador:.1f})."
    else:
        txt_finde = "Sin datos históricos suficientes."
        txt_promo = "Sin datos históricos suficientes."
        
    return txt_finde, txt_promo
