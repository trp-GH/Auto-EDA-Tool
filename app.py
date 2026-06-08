import streamlit as st
import pandas as pd
from pygwalker.api.streamlit import StreamlitRenderer
from prophet import Prophet
import matplotlib.pyplot as plt

# ==========================================
# PAGE SETUP & PORTFOLIO BRANDING
# ==========================================
st.set_page_config(page_title="Advanced Auto-EDA & Predictor", layout="wide")

st.sidebar.title("DataSense")
st.sidebar.markdown("Developed by **Tusar Ranjan Panda**")
st.sidebar.markdown("*Data Analyst & Web Developer*")
st.sidebar.markdown("---")
st.sidebar.info("This professional tool performs automated Exploratory Data Analysis (EDA) and Time-Series AI Forecasting instantly.")

st.title("🚀 DataSense: Past Analytics & Future Prediction")
st.markdown("Upload your CSV/Excel file. Use **PyGWalker** for exploring the past, and **AI Forecasting** to predict the future.")

# ==========================================
# FILE UPLOAD SECTION
# ==========================================
uploaded_file = st.file_uploader("Upload Data File (CSV or Excel)", type=['csv', 'xlsx'])

if uploaded_file is not None:
    try:
        # Load Data
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
            
        st.success(f"✅ Data Loaded Successfully! (Rows: {df.shape[0]}, Columns: {df.shape[1]})")
        
        # Create Tabs
        tab1, tab2 = st.tabs(["📊 Past Analytics (PyGWalker)", "🔮 Future Predictions (AI)"])
        
        # ==========================================
        # TAB 1: PAST ANALYTICS (Optimized for Large Data)
        # ==========================================
        with tab1:
            st.header("Interactive Drag & Drop Dashboard")
            st.write("Drag and drop variables onto the X and Y axes to create instant visualizations.")
            
            # Using the stable StreamlitRenderer to prevent HTML overflow/crashes
            @st.cache_resource
            def get_pyg_renderer(dataframe):
                return StreamlitRenderer(dataframe, explorer_default=True)
                
            renderer = get_pyg_renderer(df)
            renderer.explorer()
            
        # ==========================================
        # TAB 2: FUTURE PREDICTIONS (Prophet AI)
        # ==========================================
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
                        # Clean and prepare data for Prophet
                        df_prophet = df[[date_col, target_col]].dropna()
                        df_prophet.columns = ['ds', 'y']
                        df_prophet['ds'] = pd.to_datetime(df_prophet['ds'])
                        
                        # Train the Time-Series Model
                        model = Prophet(yearly_seasonality=True, weekly_seasonality=True)
                        model.fit(df_prophet)
                        
                        # Generate Future Forecast
                        future = model.make_future_dataframe(periods=periods)
                        forecast = model.predict(future)
                        
                        st.subheader(f"Forecast Graph for next {periods} days")
                        
                        # Render Plotly-like matplotlib graph
                        fig1 = model.plot(forecast)
                        plt.title("Past Data with Future Forecast")
                        plt.xlabel("Date")
                        plt.ylabel("Value")
                        st.pyplot(fig1)
                        
                        # Show raw prediction table
                        st.subheader("🔮 Predicted Data Table")
                        predicted_table = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(periods)
                        predicted_table.columns = ['Date', 'Predicted Value', 'Lowest Estimate', 'Highest Estimate']
                        st.dataframe(predicted_table)
                        
                    except Exception as e:
                        st.error(f"⚠️ Error in Prediction: {e}")
                        st.info("Tip: Ensure your Date column contains actual dates (e.g., YYYY-MM-DD) and your Target column contains numbers.")

    except Exception as e:
        st.error(f"⚠️ Error processing the uploaded file: {e}")

else:
    st.info("👆 Please upload a dataset to begin the analysis.")
