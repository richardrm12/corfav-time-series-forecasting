# 📈 Store Sales - Time Series Forecasting
**Project Alpha**: Predicción de ventas para Corporación Favorita.

Este proyecto aplica técnicas de Machine Learning para predecir las ventas diarias en miles de combinaciones de tiendas e ítems, respondiendo a la histórica competencia de Kaggle.

## 🚀 Características Principales
* **Desarrollo de Modelos:** Entrenamiento de modelos predictivos empleando **LightGBM Regressor** (Validación local RMSLE: ~0.41).
* **Hyperparameter Tuning:** Búsqueda Bayesiana de hiperparámetros automatizada mediante **Optuna**.
* **Feature Engineering:** Ingeniería de variantes que incluye rezagos (lags) y medias móviles temporales, además de análisis del efecto del calendario de festividades y precios del crudo.
* **Dashboard Interactivo:** Una aplicación integral construida con **Streamlit**, que despliega métricas de negocio en tiempo real, predicciones visualizadas con **Plotly**, e información en vivo respecto al impacto del fin de semana y promociones.

## 🛠️ Stack Tecnológico
* **Lenguaje:** Python 3.10+
* **Data Science & ML:** `pandas`, `numpy`, `lightgbm`, `optuna`, `scikit-learn`
* **Visualización Web:** `streamlit`, `plotly`

## 📁 Estructura del Proyecto
```text
store-sales/
├── data/
│   ├── raw/                 # Sets de datos crudos de Kaggle (.gitignore)
│   └── processed/           # Datos procesados/Feature engineering (.gitignore)
├── notebooks/               # Archivos experimentales y EDA interactivo
├── outputs/                 # Gráficas, predicciones y artefactos del modelo (.gitignore)
├── src/                     # Código fuente de producción
│   ├── app.py               # Aplicación interactiva de Streamlit (Dashboard)
│   ├── utils.py             # Funciones lógicas auxiliares (carga y análisis)
│   └── css/                 # Estilos modularizados (UI/UX)
├── .gitignore               # Exclusión de credenciales y datos masivos
└── README.md                # Documentación del proyecto
```

## ⚙️ Cómo iniciar el proyecto

1. **Clonar este repositorio:**
    ```bash
    git clone https://github.com/richardrm12/corfav-time-series-forecasting
    cd store-sales
    ```

2. **Crear e inicializar un entorno virtual (opcional pero recomendado):**
    ```bash
    python -m venv .venv
    # Activar entorno (Windows)
    .venv\Scripts\activate
    ```

3. **Instalar dependencias necesarias:**
    ```bash
    pip install pandas numpy lightgbm optuna streamlit plotly scikit-learn
    ```

4. **Descargar los datos:**
   Debes descargar los archivos crudos de [Kaggle](https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data) y almacenarlos en el directorio `data/raw/` (ej: `train.csv`, `test.csv`, `oil.csv`, `holidays_events.csv`, etc.).

5. **Ejecutar el Dashboard:**
    ```bash
    streamlit run src/app.py
    ```

---
*Desarrollado por Richard Ramirez.*
