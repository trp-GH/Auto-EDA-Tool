import streamlit as st
import pandas as pd
import pygwalker as pyg
import streamlit.components.v1 as components
from prophet import Prophet
import matplotlib.pyplot as plt

# Page Setup (Full width for better dashboard view)
st.set_page_config(page_title="Advanced Auto-EDA & Predictor", layout="wide")

st.title("🚀 DataSense: Past Analytics & Future Prediction Platform")
st.markdown("Upload your CSV/Excel file. Use **PyGWalker** for exploring the past, and **AI Forecasting** to predict the future.")

# --- FILE UPLOAD SECTION ---
uploaded_file = st.file_uploader("Upload Data File (CSV or Excel)", type=['csv', 'xlsx'])

if uploaded_file is not None:
    # Read the file dynamically
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
            
        st.success(f"✅ Data Loaded Successfully! (Rows: {df.shape[0]}, Columns: {df.shape[1]})")
        
        # --- CREATE TABS FOR PAST & FUTURE ---
        tab1, tab2 = st.tabs(["📊 Past Analytics (PyGWalker)", "🔮 Future Predictions (AI)"])
        
        # TAB 1: PAST ANALYTICS
        with tab1:
            st.header("Interactive Drag & Drop Dashboard")
            st.write("Drag and drop variables onto the X and Y axes to create instant visualizations.")
            
            # PyGWalker HTML component
            pyg_html = pyg.walk(df, return_html=True)
            components.html(pyg_html, height=800, scrolling=True)
            
        # TAB 2: FUTURE PREDICTIONS
        with tab2:
            st.header("📈 Time-Series AI Forecasting")
            st.write("Select a Date column and a Target/Value column to forecast future trends.")
            
            col1, col2 = st.columns(2)
            with col1:
                date_col = st.selectbox("Select Date Column (Time axis):", df.columns)
            with col2:
                target_col = st.selectbox("Select Target Column (What to predict):", df.columns)
                
            periods = st.slider("How many days into the future would you like to predict?", min_value=7, max_value=365, value=30)
            
            if st.button("Predict Future 🚀"):
                with st.spinner("AI is analyzing past patterns to predict the future... Please wait."):
                    try:
                        # Prophet Model Data Prep
                        df_prophet = df[[date_col, target_col]].dropna()
                        df_prophet.columns = ['ds', 'y']
                        df_prophet['ds'] = pd.to_datetime(df_prophet['ds'])
                        
                        # Train Model
                        model = Prophet(yearly_seasonality=True, weekly_seasonality=True)
                        model.fit(df_prophet)
                        
                        # Predict
                        future = model.make_future_dataframe(periods=periods)
                        forecast = model.predict(future)
                        
                        st.subheader(f"Forecast Graph for next {periods} days")
                        
                        # Plot
                        fig1 = model.plot(forecast)
                        plt.title("Past Data with Future Forecast")
                        plt.xlabel("Date")
                        plt.ylabel("Value")
                        st.pyplot(fig1)
                        
                        # Data Table
                        st.subheader("🔮 Predicted Data Table")
                        predicted_table = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(periods)
                        predicted_table.columns = ['Date', 'Predicted Value', 'Lowest Estimate', 'Highest Estimate']
                        st.dataframe(predicted_table)
                        
                    except Exception as e:
                        st.error(f"⚠️ Error in Prediction: {e}")
                        st.info("Tip: Ensure that your Date column contains actual dates and Target column contains numerical values.")

    # YAHAN ERROR THA: Ye naya block add kiya hai file read ko handle karne ke liye
    except Exception as e:
        st.error(f"⚠️ Error loading file: {e}")

else:
    st.info("👆 Please upload a dataset to begin the analysis.")
