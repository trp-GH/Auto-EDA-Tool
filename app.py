import matplotlib
matplotlib.use('Agg') # CRITICAL: Prevents GUI thread crashing on Linux/Streamlit Cloud servers

import streamlit as st
import pandas as pd
from pygwalker.api.streamlit import StreamlitRenderer
from prophet import Prophet
import matplotlib.pyplot as plt

# ====================================================================
# PAGE SETUP & BRANDING (Enterprise Executive Layout)
# ====================================================================
st.set_page_config(
    page_title="DataSense AI - Automated Insight Engine", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling for Polished Enterprise Aesthetics
st.markdown("""
    <style>
    .big-title { font-size:38px !important; font-weight: 700; color: #1E3A8A; letter-spacing: -0.5px; }
    .subtitle { font-size:16px !important; color: #4B5563; margin-bottom: 30px; }
    .report-card { background-color: #F9FAFB; padding: 24px; border-radius: 8px; border-left: 5px solid #10B981; margin-bottom: 25px; box-shadow: 0 1px 3px rgba(0,0,0,0.05); }
    </style>
""", unsafe_allow_html=True)

# Sidebar System Configuration & Branding
st.sidebar.title("🎯 DataSense AI")
st.sidebar.markdown("### System Engineer:\n**Tusar Ranjan Panda**")
st.sidebar.markdown("*Data Analyst & Web Developer*")
st.sidebar.markdown("---")
st.sidebar.info("Executive-level automated system facilitating instant Exploratory Data Analysis (EDA) alongside Advanced Machine Learning Time-Series Forecasting.")

# Main Application Headers
st.markdown('<p class="big-title">🚀 DataSense AI: Automated Exploratory & Predictive Analytics Engine</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Upload operational files (CSV/XLSX) to isolate critical historical metrics via <b>Auto-EDA Workspace</b>, or execute predictive workflows using the native <b>Data Science Pipeline</b>.</p>', unsafe_allow_html=True)

# ====================================================================
# SECURE FILE INGESTION & STRUCTURAL INTEGRITY VALIDATION
# ====================================================================
uploaded_file = st.file_uploader("📂 Select Source Dataset (Supported Formats: CSV, XLSX)", type=['csv', 'xlsx'])

if uploaded_file is not None:
    try:
        # Load Dataset cleanly
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
            
        st.success(f"✅ Data Engine Synchronized Successfully. (Records Matrix: {df.shape[0]} Rows | {df.shape[1]} Columns)")
        
        # Compartmentalize workflows into distinct analytical profiles
        tab1, tab2 = st.tabs(["📊 Past Analytics (Auto-EDA Dashboard)", "🔮 Future Predictions (Data Science Report)"])
        
        # ================================================================
        # TAB 1: PAST ANALYTICS (Optimized Multi-Threaded PyGWalker Canvas)
        # ================================================================
        with tab1:
            st.header("📈 Drag & Drop Interactive Canvas")
            st.write("Construct dynamic analytical charts by dropping dimensional schema attributes onto the chart layout matrices.")
            
            # Isolated speculative local spec mapping to guarantee pipeline caching consistency
            @st.cache_resource(ttl="1h")
            def get_pyg_renderer(dataframe):
                return StreamlitRenderer(dataframe, spec_io_mode="local")
                
            try:
                renderer = get_pyg_renderer(df)
                renderer.explorer()
            except Exception as pyg_err:
                st.error(f"Canvas Render Pipeline Error: {pyg_err}")
            
        # ================================================================
        # TAB 2: FUTURE PREDICTIONS (Industrial Meta Prophet Forecast Core)
        # ================================================================
        with tab2:
            st.header("🤖 Automated Machine Learning Pipeline")
            st.write("Configure targeting vectors to map long-range timelines against explicit numeric historical variables.")
            
            # Form Control architecture mapped out symmetrically
            col1, col2 = st.columns(2)
            with col1:
                date_col = st.selectbox("📆 Temporal Metric (Date Index Variable):", df.columns, key="ds_date_idx")
            with col2:
                target_col = st.selectbox("🎯 Target Objective (Value to Predict):", df.columns, key="ds_target_idx")
                
            periods = st.slider("🔮 Target Forecast Horizon (Days Outward):", min_value=7, max_value=365, value=30)
            
            if st.button("Execute Predictive Pipeline 🚀"):
                with st.spinner("Processing advanced trend-lines and matrix multi-seasonality curves... Please wait."):
                    try:
                        # Data Engineering Guardrails: Type Casting & Isolation
                        df_prophet = df[[date_col, target_col]].dropna().copy()
                        df_prophet.columns = ['ds', 'y']
                        
                        df_prophet['ds'] = pd.to_datetime(df_prophet['ds'], errors='coerce')
                        df_prophet['y'] = pd.to_numeric(df_prophet['y'], errors='coerce')
                        
                        # Purge structural corruptions seamlessly
                        df_prophet = df_prophet.dropna()
                        
                        if df_prophet.empty:
                            st.error("⚠️ Pipeline Exception: The selected columns do not contain valid or sufficient historical Datetime format / Numerical observations.")
                            st.stop()
                        
                        # Model Architecture Fit (Adjusted for production enterprise scale)
                        model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
                        model.fit(df_prophet)
                        
                        # Forecast Vector Matrix Computations
                        future = model.make_future_dataframe(periods=periods)
                        forecast = model.predict(future)
                        
                        # ====================================================
                        # SYSTEM REPORT GENERATION METRICS
                        # ====================================================
                        st.markdown("---")
                        st.markdown('<div class="report-card"><h3>📋 Executive Data Science Insights Summary</h3>'
                                    f'• <b>Historical Baseline Bounds:</b> {df_prophet["ds"].min().strftime("%Y-%m-%d")} through {df_prophet["ds"].max().strftime("%Y-%m-%d")}<br>'
                                    f'• <b>Projected Forward Horizon:</b> Extended calculations out <b>{periods} continuous operational days</b>.<br>'
                                    f'• <b>Mathematical Trend Extraction:</b> Machine Learning models have fully parsed and registered the historical seasonal variations (weekly and yearly cycles).</div>', 
                                    unsafe_allow_html=True)
                        
                        # Thread-Safe Matplotlib Object Oriented Asset Generation
                        st.subheader(f"📊 Projected System Trend Variance (Next {periods} Days)")
                        fig1 = model.plot(forecast)
                        ax = fig1.gca()
                        ax.set_title(f"Predictive Vector Evaluation: Mapping {target_col}", fontsize=12, fontweight='bold')
                        ax.set_xlabel("Time Domain Timeline")
                        ax.set_ylabel(target_col)
                        st.pyplot(fig1)
                        
                        # Prediction High-Resolution Matrix Table Display
                        st.subheader("🔮 Calculated Output Projection Matrix")
                        predicted_table = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(periods)
                        predicted_table.columns = ['Date', 'Point Prediction (Expected)', 'Optimistic Ceiling (Upper)', 'Pessimistic Floor (Lower)']
                        
                        # Precision Formatter
                        formatted_table = predicted_table.style.format({
                            'Point Prediction (Expected)': '{:.2f}', 'Optimistic Ceiling (Upper)': '{:.2f}', 'Pessimistic Floor (Lower)': '{:.2f}'
                        })
                        st.dataframe(formatted_table, use_container_width=True)
                        
                        # Structured Flat-File Download Interface
                        csv_data = predicted_table.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="📥 Export Prediction Dataset (.CSV Flat File)",
                            data=csv_data,
                            file_name=f"DataSense_Forecast_Report_{target_col}.csv",
                            mime="text/csv"
                        )
                        
                    except Exception as model_err:
                        st.error(f"⚠️ Computational Engine Regression Error: {model_err}")
                        st.info("💡 Solution Strategy: Confirm that your chosen Date column is purely configured as a standardized temporal value, and the Target metrics contain only raw integers/floats.")

    except Exception as e:
        st.error(f"⚠️ System IO Pipeline Interruption: Unable to securely parse the object instance. Profile logs: {e}")

else:
    # Minimalist Ingestion State
    st.info("💡 Ingestion Queue Empty: Please upload a structured file payload above to initialize automated pipelines.")
