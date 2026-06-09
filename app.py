import matplotlib
matplotlib.use('Agg') # CRITICAL: Prevents GUI thread crashing on Linux/Codespaces environments

import streamlit as st
import pandas as pd
from pygwalker.api.streamlit import StreamlitRenderer
from prophet import Prophet
import matplotlib.pyplot as plt
import io

# Page Configuration
st.set_page_config(page_title="DataSense AI", layout="wide")

st.markdown('<h2 style="color: #1E3A8A;">🚀 DataSense AI: Automated Insight Engine</h2>', unsafe_allow_html=True)
st.write("Logged Engineer: **Tusar Ranjan Panda**")

uploaded_file = st.file_uploader("📂 Select Source Dataset (CSV or XLSX)", type=['csv', 'xlsx'])

if uploaded_file is not None:
    try:
        # Read file into memory bytes to prevent large file stream exhaustion (14.4MB handling)
        file_bytes = uploaded_file.read()
        
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(io.BytesIO(file_bytes))
        else:
            # Smart Excel Sheet Scanner: Locates the first worksheet containing data
            excel_file = pd.ExcelFile(io.BytesIO(file_bytes))
            df = pd.DataFrame() # Default fallback empty container
            
            for sheet in excel_file.sheet_names:
                temp_df = pd.read_excel(excel_file, sheet_name=sheet)
                if temp_df is not None and not temp_df.empty and temp_df.shape[0] > 0:
                    df = temp_df.copy()
                    break # Stop scanning once data is found

        # Clean structural system anomalies (stripping completely empty rows/columns)
        df = df.dropna(how='all', axis=1)
        df = df.dropna(how='all', axis=0)

        # Safety Check: Stop execution if dataframe is still parsed as empty
        if df.empty or df.shape[0] == 0:
            st.error("⚠️ Ingestion Error: The uploaded sheet contains no readable data rows. Please verify your Excel file layout.")
            st.stop()
            
        st.success(f"✅ Data Engine Synchronized. (Matrix Footprint: {df.shape[0]} Rows | {df.shape[1]} Columns)")
        
        # Application Tab Navigation
        tab1, tab2 = st.tabs(["📊 Auto-EDA Dashboard", "🔮 Time-Series Forecast"])
        
        with tab1:
            st.subheader("📈 Interactive Visualization Engine")
            # Instantiate the Drag & Drop Canvas
            renderer = StreamlitRenderer(df, spec_io_mode="local")
            renderer.explorer()
            
        with tab2:
            st.subheader("🤖 Predictive Modeling Pipeline")
            date_col = st.selectbox("📆 Select Date Column:", df.columns, key="predict_date")
            target_col = st.selectbox("🎯 Select Value to Predict:", df.columns, key="predict_target")
            periods = st.slider("Forecast Days Forward:", 7, 365, 30)
            
            if st.button("Execute Predictive Modeling"):
                with st.spinner("Calculating future trends..."):
                    df_p = df[[date_col, target_col]].dropna().copy()
                    df_p.columns = ['ds', 'y']
                    df_p['ds'] = pd.to_datetime(df_p['ds'], errors='coerce')
                    df_p['y'] = pd.to_numeric(df_p['y'], errors='coerce')
                    df_p = df_p.dropna()
                    
                    if len(df_p) < 2:
                        st.error("❌ Process Halting: Not enough valid date/numeric data points available to fit trends.")
                    else:
                        model = Prophet(yearly_seasonality=True, weekly_seasonality=True)
                        model.fit(df_p)
                        future = model.make_future_dataframe(periods=periods)
                        forecast = model.predict(future)
                        
                        fig = model.plot(forecast)
                        st.pyplot(fig)
                        st.dataframe(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(periods), use_container_width=True)

    except Exception as e:
        st.error(f"❌ System Error: {e}")
else:
    st.info("💡 Awaiting dataset payload injection. Please upload your file above.")
