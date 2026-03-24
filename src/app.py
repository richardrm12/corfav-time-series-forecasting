import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import os
from utils import load_data, calculate_insights

# --- PAGE CONFIG ---
st.set_page_config(page_title="Dashboard Predictivo: IA en Retail", layout="wide", page_icon="📈")

# --- CUSTOM CSS BASADO EN TAILWIND/LIGHT MODE REFERENCIA ---
def load_local_css(file_name):
    with open(file_name, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_local_css("src/css/style.css")

# --- CARGA ---
with st.spinner("Cargando motor de datos completo..."):
    train_data, preds_data = load_data()

# --- SIDEBAR (Menú lateral y Filtros ajustado) ---
with st.sidebar:
    st.markdown("""
        <h2 style="font-size: 1.125rem; font-weight: 900; font-family: 'Manrope'; margin-bottom:0; color:#0f172a;">Project Alpha</h2>
        <p style="font-size: 0.75rem; font-weight: 600; color: #64748b; text-transform: uppercase;">Store Sales - Time Series Forecasting</p>
        <hr style="margin-top:0.75rem; border-color:#e2e8f0;">
        <p style="font-size: 0.65rem; font-weight: 700; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.1em;">Filtros</p>
    """, unsafe_allow_html=True)
    
    lista_tiendas = sorted(train_data['store_nbr'].unique())
    tienda_seleccionada = st.selectbox("Número de Tienda", lista_tiendas, index=0, help="Busca escribiendo el número de la sucursal")

    # Filtro inteligente: Mostrar solo las familias que realmente tuvieron ventas en esta tienda si se desea, 
    # o seguir mostrando todas pero re-evaluando el índice. Aquí mantenemos el filtro dependiente de la tienda.
    familias_por_tienda = train_data[train_data['store_nbr'] == tienda_seleccionada]['family'].unique()
    lista_familias = sorted(familias_por_tienda)
    default_idx = lista_familias.index('GROCERY I') if 'GROCERY I' in lista_familias else 0
    familia_seleccionada = st.selectbox("Familia de Productos (Categoria)", lista_familias, index=default_idx, help="Muestra familias disponibles en esta tienda")
    
    st.markdown("""
        <div style="margin-top: 1rem; padding: 1rem; background-color: rgba(255, 255, 255, 0.7); border-radius: 0.75rem; border: 1px solid #e2e8f0;">
            <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.5rem;">
                <span class="material-symbols-outlined" style="color: #b81120; font-size: 1rem;">model_training</span>
                <span style="font-size: 0.85rem; font-weight: 700; color: #0f172a;">Info del Modelo</span>
            </div>
            <p style="font-size: 0.7rem; color: #475569; margin: 0; line-height: 1.4;">Optuna Optimized LightGBM. Error residual aprox: 0.41 RMSLE.</p>
        </div>
    """, unsafe_allow_html=True)


# --- PROCESAMIENTO FILTROS ---
train_filtro = train_data[(train_data['store_nbr'] == tienda_seleccionada) & (train_data['family'] == familia_seleccionada)]
preds_filtro = preds_data[(preds_data['store_nbr'] == tienda_seleccionada) & (preds_data['family'] == familia_seleccionada)]

ventas_hist_recientes = train_filtro[train_filtro['date'] >= '2017-08-01']['sales'].sum()
ventas_futuras_pred = preds_filtro['sales'].sum()
variacion = 0
if ventas_hist_recientes > 0:
    variacion = ((ventas_futuras_pred - ventas_hist_recientes) / ventas_hist_recientes) * 100

es_serie_muerta = (ventas_futuras_pred == 0 and train_filtro['sales'].sum() == 0)


# --- PANEL PRINCIPAL ---
st.markdown("""
<div style="margin-bottom: 2.5rem;">
    <h1 style="font-size: 2.75rem; font-weight: 800; color: #0f172a; line-height: 1.2; margin-bottom: 0.5rem;">Dashboard Predictivo: IA en Retail</h1>
    <p style="font-size: 1.125rem; color: #475569; font-weight: 500; margin-top: 0; max-width: 800px;">
        Visualización avanzada de la demanda para Corporación Favorita. Proyección analítica de ventas para los próximos 15 días mediante gradiente boost.
    </p>
</div>
""", unsafe_allow_html=True)

# METRICS GRID
col1, col2, col3 = st.columns(3)

# Metric 1
html_metric_1 = f"""
<div class="bento-card">
    <div class="bento-title">Ventas reales (Últimos 15 días)</div>
    <div style="display: flex; align-items: baseline; gap: 0.5rem;">
        <span class="bento-value">{ventas_hist_recientes:,.0f}</span>
        <span style="font-size: 1rem; font-weight: 600; color: #64748b;">u.</span>
    </div>
</div>
"""
col1.markdown(html_metric_1, unsafe_allow_html=True)

# Metric 2 
color_var_text = "#15803d" if variacion >= 0 else "#b91c1c"
color_var_bg = "#f0fdf4" if variacion >= 0 else "#fef2f2"
color_var_border = "#dcfce7" if variacion >= 0 else "#fee2e2"
icono_var = "trending_up" if variacion >= 0 else "trending_down"
signo = "+" if variacion >= 0 else ""

html_metric_2 = f"""
<div class="bento-card">
    <div class="bento-title">Pronóstico (Próximos 15 días)</div>
    <div style="display: flex; align-items: center; justify-content: space-between;">
        <div style="display: flex; align-items: baseline; gap: 0.5rem;">
            <span class="bento-value">{ventas_futuras_pred:,.0f}</span>
            <span style="font-size: 1rem; font-weight: 600; color: #64748b;">u.</span>
        </div>
        <div style="display: flex; align-items: center; gap: 0.25rem; padding: 0.25rem 0.5rem; background-color: {color_var_bg}; color: {color_var_text}; border: 1px solid {color_var_border}; border-radius: 9999px; font-size: 0.8rem; font-weight: 700;">
            <span class="material-symbols-outlined" style="font-size: 1rem;">{icono_var}</span>
            {signo}{variacion:.1f}%
        </div>
    </div>
</div>
"""
col2.markdown(html_metric_2, unsafe_allow_html=True)

# Metric 3
html_estado = "Serie Inactiva" if es_serie_muerta else "Serie Activa"
color_estado = "#ef4444" if es_serie_muerta else "#10b981" # emerald-500 / red-500

html_metric_3 = f"""
<div class="bento-card" style="border-left: 4px solid {color_estado}; --hover-color: {color_estado};">
    <div class="bento-title">Estado de la Serie</div>
    <div style="display: flex; align-items: center; gap: 0.75rem; margin-top: 0.5rem;">
        <div style="width: 0.75rem; height: 0.75rem; border-radius: 50%; background-color: {color_estado}; {'animation: pulse-green 2s infinite;' if not es_serie_muerta else ''}"></div>
        <span style="font-size: 1.125rem; font-weight: 700; color: #0f172a;">{html_estado}</span>
    </div>
</div>
"""
col3.markdown(html_metric_3, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# --- GRÁFICO PRINCIPAL ---
st.markdown("""
<div style="background-color: white; padding: 2rem 2rem 0.5rem 2rem; border-radius: 0.75rem 0.75rem 0 0; border: 1px solid #e2e8f0; border-bottom: none;">
    <h3 style="font-size: 1.25rem; font-weight: 700; color: #0f172a; margin-bottom: 0.25rem;">Histórico vs. Predicción Futura</h3>
    <div style="display: flex; gap: 1rem; align-items: center;">
        <p style="font-size: 0.85rem; color: #64748b; margin: 0;">Unidades vendidas agregadas por día</p>
        <span style="background-color: #f8fafc; padding: 0.1rem 0.5rem; border-radius: 0.25rem; font-size: 0.7rem; font-weight: 600; color: #475569; border: 1px solid #e2e8f0;">
            INFO: Arrastra la barra inferior para ver la historia completa.
        </span>
    </div>
</div>
""", unsafe_allow_html=True)

fig = go.Figure()

# Traza histórica completa
fig.add_trace(go.Scatter(
    x=train_filtro['date'], 
    y=train_filtro['sales'], 
    name='Histórico (Real)',
    line=dict(color='#0f172a', width=1.5),
    fill='tozeroy',
    fillcolor='rgba(15, 23, 42, 0.05)'
))

# Traza predictiva
fig.add_trace(go.Scatter(
    x=preds_filtro['date'], 
    y=preds_filtro['sales'], 
    name='Predicción AI',
    line=dict(color='#b81120', width=2.5, dash='dash')
))

# Ajustes de UI en fondo blanco
fig.update_layout(
    height=450,
    margin=dict(l=0, r=0, t=10, b=0),
    hovermode="x unified",
    template="plotly_white",
    plot_bgcolor="white", 
    paper_bgcolor="white",
    font=dict(color="#0f172a", size=13),
    legend=dict(
        orientation="h", 
        yanchor="bottom", 
        y=1.02, 
        xanchor="center", 
        x=0.5, 
        font=dict(color="#0f172a", size=14, family="Inter")
    ),
    xaxis=dict(
        rangeslider=dict(visible=True, thickness=0.08, bgcolor="#f1f5f9"),
        range=['2017-01-01', '2017-08-31'], 
        type="date",
        gridcolor="#e2e8f0",
        tickfont=dict(color="#475569", weight="bold"),
        title_font=dict(color="#475569", size=14)
    ),
    yaxis=dict(
        title="Unidades Vendidas",
        gridcolor="#e2e8f0",
        tickfont=dict(color="#475569", weight="bold"),
        title_font=dict(color="#475569", size=14)
    )
)

fig.add_vline(x='2017-08-15', line_width=2, line_dash="dash", line_color="#64748b")
fig.add_annotation(x='2017-08-15', y=0.95, yref="paper", text="INICIO DE LA PREDICCIÓN", showarrow=False, font=dict(color="#64748b", size=12, weight="bold"), bgcolor="white", borderpad=4)

st.plotly_chart(fig, use_container_width=True)


# --- CÁLCULOS DINÁMICOS PARA LA EXPLICACIÓN ---
txt_finde, txt_promo = calculate_insights(train_filtro)


# --- EXPLICACIÓN Y ARQUITECTURA ---
st.markdown("<br>", unsafe_allow_html=True)

st.markdown(f"""
<div style="background-color: #ffffff; padding: 2rem; border-radius: 0.75rem; border: 1px solid #e2e8f0; margin-bottom: 2rem;">
    <h3 style="font-size: 1.5rem; font-weight: 800; color: #0f172a; margin-bottom: 2rem; margin-top: 0;">¿Por qué la IA predice esto?</h3>
    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 1.5rem;">
        <div>
            <div class="icon-box">
                <span class="material-symbols-outlined" style="color: #b81120;">calendar_view_week</span>
            </div>
            <h4 style="font-size: 0.875rem; font-weight: 700; color: #0f172a; margin-bottom: 0.5rem;">Efecto Fin de Semana</h4>
            <p style="font-size: 0.75rem; color: #475569; line-height: 1.5; margin:0;">{txt_finde}</p>
        </div>
        <div>
            <div class="icon-box">
                <span class="material-symbols-outlined" style="color: #b81120;">sell</span>
            </div>
            <h4 style="font-size: 0.875rem; font-weight: 700; color: #0f172a; margin-bottom: 0.5rem;">Sensibilidad Promocional</h4>
            <p style="font-size: 0.75rem; color: #475569; line-height: 1.5; margin:0;">{txt_promo}</p>
        </div>
        <div>
            <div class="icon-box">
                <span class="material-symbols-outlined" style="color: #b81120;">history</span>
            </div>
            <h4 style="font-size: 0.875rem; font-weight: 700; color: #0f172a; margin-bottom: 0.5rem;">Memoria Temporal (Lags)</h4>
            <p style="font-size: 0.75rem; color: #475569; line-height: 1.5; margin:0;">La proyección asimila automáticamente caídas después de quincenas y estacionalidades de ventas de 7/14 y 28 días.</p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


st.markdown("""
<div class="dark-card" style="box-shadow: none;">
    <span style="font-size: 0.625rem; font-weight: 800; text-transform: uppercase; letter-spacing: 0.2em; opacity: 0.6; display: block; margin-bottom: 0.75rem;">Arquitectura del Modelo & Configuración</span>
    <h4 style="font-size: 1.5rem; font-weight: 800; margin-top: 0; margin-bottom: 1.5rem; color: white;">LightGBM Regressor Optimizated</h4>
    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 2rem;">
        <div>
            <h5 style="font-size: 0.85rem; font-weight: 700; color: #94a3b8; margin-top: 0; margin-bottom: 0.75rem; border-bottom: 1px solid #334155; padding-bottom: 0.5rem;">Especificaciones Base</h5>
            <ul style="list-style: none; padding: 0; margin: 0; font-size: 0.85rem; opacity: 0.9; color: white;">
                <li style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.5rem;">
                    <span class="material-symbols-outlined" style="color: #ffb3ae; font-size: 1.1rem;">settings</span> Estrategia: Gradient Boosting
                </li>
                <li style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.5rem;">
                    <span class="material-symbols-outlined" style="color: #ffb3ae; font-size: 1.1rem;">model_training</span> Función Objetivo: Regression
                </li>
                <li style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.5rem;">
                    <span class="material-symbols-outlined" style="color: #ffb3ae; font-size: 1.1rem;">account_tree</span> Optimización: Optuna (Bayesiana)
                </li>
            </ul>
        </div>
        <div>
            <h5 style="font-size: 0.85rem; font-weight: 700; color: #94a3b8; margin-top: 0; margin-bottom: 0.75rem; border-bottom: 1px solid #334155; padding-bottom: 0.5rem;">Feature Engineering</h5>
            <ul style="list-style: none; padding: 0; margin: 0; font-size: 0.85rem; opacity: 0.9; color: white;">
                <li style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.5rem;">
                    <span class="material-symbols-outlined" style="color: #ffb3ae; font-size: 1.1rem;">history</span> Lags temporales: 16, 30, 60 días
                </li>
                <li style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.5rem;">
                    <span class="material-symbols-outlined" style="color: #ffb3ae; font-size: 1.1rem;">moving</span> Rolling Means: 7, 14, 28 días
                </li>
                <li style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.5rem;">
                    <span class="material-symbols-outlined" style="color: #ffb3ae; font-size: 1.1rem;">calendar_month</span> Variables de Feriados Nacionales
                </li>
            </ul>
        </div>
        <div>
            <h5 style="font-size: 0.85rem; font-weight: 700; color: #94a3b8; margin-top: 0; margin-bottom: 0.75rem; border-bottom: 1px solid #334155; padding-bottom: 0.5rem;">Rendimiento & Parámetros</h5>
            <ul style="list-style: none; padding: 0; margin: 0; font-size: 0.85rem; opacity: 0.9; color: white;">
                <li style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.5rem;">
                    <span class="material-symbols-outlined" style="color: #ffb3ae; font-size: 1.1rem;">score</span> Validación RMSLE: ~0.41
                </li>
                <li style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.5rem;">
                    <span class="material-symbols-outlined" style="color: #ffb3ae; font-size: 1.1rem;">tune</span> Learning Rate: 0.05
                </li>
                <li style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.5rem;">
                    <span class="material-symbols-outlined" style="color: #ffb3ae; font-size: 1.1rem;">park</span> Max Leaves: 64
                </li>
            </ul>
        </div>
    </div>
</div>
    """, unsafe_allow_html=True)

# --- FOOTER ---
st.markdown("""
<div style="text-align: center; margin-top: 3rem; padding-top: 1.5rem; padding-bottom: 0.5rem; border-top: 1px solid #e2e8f0;">
    <p style="font-size: 0.875rem; color: #64748b; font-family: 'Inter', sans-serif; margin: 0;">
        Desarrollado por: <strong style="color: #0f172a;">Richard Ramirez</strong>
    </p>
    <p style="font-size: 0.8rem; color: #64748b; font-family: 'Inter', sans-serif; margin-top: 0.5rem;">
        <a href="https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data" target="_blank" style="color: #b81120; text-decoration: none; display: inline-flex; align-items: center; justify-content: center; gap: 0.25rem; font-weight: 600;">
            <span class="material-symbols-outlined" style="font-size: 1rem;">database</span>
            Ver dataset original (Kaggle: Store Sales Forecasting)
        </a>
    </p>
</div>
""", unsafe_allow_html=True)
