import matplotlib
matplotlib.use('Agg') # CRITICAL: Prevents GUI thread crashing on Linux/Codespaces environments

import streamlit as st
import pandas as pd
from pygwalker.api.streamlit import StreamlitRenderer
from prophet import Prophet
import matplotlib.pyplot as plt

# Force clean wide page layout state
st.set_page_config(page_title="DataSense AI", layout="wide")

st.markdown('<h2 style="color: #1E3A8A;">🚀 DataSense AI: Automated Insight Engine</h2>', unsafe_allow_html=True)
st.write("Logged Engineer: **Tusar Ranjan Panda**")

uploaded_file = st.file_uploader("📂 Select Source Dataset (CSV or XLSX)", type=['csv', 'xlsx'])

if uploaded_file is not None:
    try:
        # Isolated parsing layer to protect memory initialization
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            excel_object = pd.ExcelFile(uploaded_file)
            sheet_names = excel_object.sheet_names
            valid_df_found = False
            
            # Scan sheets sequentially for a real population footprint
            for sheet in sheet_names:
                temp_df = pd.read_excel(uploaded_file, sheet_name=sheet)
                if temp_df is not None and not temp_df.empty and temp_df.shape[0] > 0 and temp_df.shape[1] > 0:
                    df = temp_df.copy()
                    valid_df_found = True
                    break
            
            if not valid_df_found:
                df = pd.DataFrame() # Fallback to empty context if all sheets fail

        # Clean structural system anomalies (stripping completely empty rows/columns)
        df = df.dropna(how='all', axis=1)
        df = df.dropna(how='all', axis=0)
            
        # Explicitly block empty dataframes before sending to rendering engines
        if df.empty or df.shape[0] == 0 or df.shape[1] == 0:
            st.error("⚠️ Ingestion Error: The uploaded file has no valid data footprints or rows. Verify your Excel worksheets are populated.")
            st.stop()
            
        st.success(f"✅ Data Engine Synchronized. (Matrix Footprint: {df.shape[0]} Rows | {df.shape[1]} Columns)")
        
        tab1, tab2 = st.tabs(["📊 Auto-EDA Dashboard", "🔮 Time-Series Forecast"])
        
        with tab1:
            # Reconfigured Renderer invocation without local state leakage
            try:
                renderer = StreamlitRenderer(df, spec_io_mode="local")
                renderer.explorer()
            except ZeroDivisionError:
                st.error("⚠️ Visual Layout Engine Mismatch: The selected data layout contains zero-variance attributes. Try plotting numerical dimensions manually.")
            except Exception as canvas_err:
                st.error(f"❌ Canvas Error: {canvas_err}")
            
        with tab2:
            st.subheader("🤖 Predictive Modeling Pipeline")
            date_col = st.selectbox("📆 Select Temporal Axis (Date Column):", df.columns)
            target_col = st.selectbox("🎯 Select Objective Target (Value to Predict):", df.columns)
            periods = st.slider("Forecast Horizon Window (Days Forward):", 7, 365, 30)
            
            if st.button("Execute Predictive Modeling"):
                with st.spinner("Crunching historical seasonality patterns..."):
                    df_p = df[[date_col, target_col]].dropna().copy()
                    df_p.columns = ['ds', 'y']
                    
                    # Convert formats safely
                    df_p['ds'] = pd.to_datetime(df_p['ds'], errors='coerce')
                    df_p['y'] = pd.to_numeric(df_p['y'], errors='coerce')
                    df_p = df_p.dropna()
                    
                    if df_p.empty or len(df_p) < 2:
                        st.error("❌ Process Halting: Insufficient target rows or invalid date profiles match those selected variables.")
                    else:
                        model = Prophet(yearly_seasonality=True, weekly_seasonality=True)
                        model.fit(df_p)
                        
                        future = model.make_future_dataframe(periods=periods)
                        forecast = model.predict(future)
                        
                        fig = model.plot(forecast)
                        st.pyplot(fig)
                        st.dataframe(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(periods), use_container_width=True)

    except Exception as e:
        st.error(f"❌ System Pipeline Exception: {e}")
else:
    st.info("💡 Application deployment operational. Awaiting structured dataset payload injection.")
