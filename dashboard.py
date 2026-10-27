import streamlit as st
import pandas as pd
from engine import FloodPredictionEngine

st.set_page_config(page_title="Pune Flood Forecast", page_icon="🌧️", layout="wide")

st.title("🌧️ Pune Flood Forecasting System")
st.markdown("This system queries a live numerical weather model, engineers 10 features, and runs an XGBoost classifier to predict flood risk for the next 7 days.")

# --- SIDEBAR CONTROLS ---
st.sidebar.header("🎛️ What-If Analysis")
simulate = st.sidebar.checkbox("Enable Custom Weather Simulation")
st.sidebar.markdown("Simulate a 3-day storm event:")

# Add sliders for Rain, Wind, and Temp
sim_rain = st.sidebar.slider("Rainfall (mm/day)", 0.0, 200.0, 0.0, step=10.0, disabled=not simulate)
sim_wind = st.sidebar.slider("Wind Speed (km/h)", 0.0, 100.0, 15.0, step=5.0, disabled=not simulate)
sim_temp = st.sidebar.slider("Temperature (°C)", 10.0, 40.0, 30.0, step=1.0, disabled=not simulate)

# Pass the simulated values to the backend if the checkbox is ticked
r_val = sim_rain if simulate else None
w_val = sim_wind if simulate else None
t_val = sim_temp if simulate else None

try:
    engine = FloodPredictionEngine()
    
    # The backend does the heavy lifting here
    with st.spinner("Processing data through AI model..."):
        data = engine.fetch_and_predict(sim_rain=r_val, sim_wind=w_val, sim_temp=t_val)
    
    status = data["status"]
    max_risk = data["max_risk_7day"]
    today_risk = data["today_risk"]
    forecast_df = data["forecast"]
    features = data["today_features"]

    # --- MAIN UI ---
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("7-Day Outlook")
        if simulate:
            st.info(f"Simulation Active:\n{r_val}mm Rain, {w_val}km/h Wind, {t_val}°C Temp")
        
        if status == "VERY HIGH":
            st.error(f"🚨 **PUNE FLOOD RISK: {max_risk:.1f}%**\n\nCondition: VERY HIGH")
        elif status == "MODERATE":
            st.warning(f"⚠️ **PUNE FLOOD RISK: {max_risk:.1f}%**\n\nCondition: MODERATE")
        else:
            st.success(f"✅ **PUNE FLOOD RISK: {max_risk:.1f}%**\n\nCondition: LOW")
            
        st.metric(label="Today's Specific Risk", value=f"{today_risk:.2f}%")

    with col2:
        st.subheader("Daily Forecast Trend")
        chart_data = forecast_df.set_index('date')[['risk']]
        chart_data.index = chart_data.index.strftime('%a, %b %d')
        st.bar_chart(chart_data, color="#ef4444")

    st.markdown("---")
    
    st.subheader("Backend Data Processing: Features Used Today")
    st.markdown("These are the exact 10 values the XGBoost backend received to make today's prediction:")
    
    feature_df = pd.DataFrame([features]).T.reset_index()
    feature_df.columns = ['Feature', 'Live Value']
    
    if simulate:
        st.warning("Notice how the precipitation, wind, and temperature values have changed based on your sliders!")
        
    st.dataframe(feature_df, use_container_width=True, hide_index=True)

except Exception as e:
    st.error(f"Backend Error: {e}")
