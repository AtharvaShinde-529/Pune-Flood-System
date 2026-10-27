import requests
import pandas as pd
import numpy as np
import joblib
import os

# 1. Load the trained XGBoost model
if not os.path.exists('xgb_pune_flood_model.pkl'):
    print("❌ Error: Model file 'xgb_pune_flood_model.pkl' not found in this folder!")
    exit()

print("Loading AI Model...")
model = joblib.load('xgb_pune_flood_model.pkl')

# 2. Fetch Live Weather Data for Pune (Lat: 18.52, Lon: 73.85)
print("Fetching live weather data for Pune from Open-Meteo...")
url = "https://api.open-meteo.com/v1/forecast"
params = {
    "latitude": 18.5204,
    "longitude": 73.8567,
    "daily": ["precipitation_sum", "temperature_2m_max", "temperature_2m_min", "wind_speed_10m_max", "wind_gusts_10m_max"],
    "past_days": 7, # We need past 7 days to calculate rolling rain features
    "forecast_days": 1,
    "timezone": "Asia/Kolkata"
}

response = requests.get(url, params=params).json()

# 3. Process the Data
daily_data = response['daily']
df = pd.DataFrame({
    'date': daily_data['time'],
    'precipitation_sum': daily_data['precipitation_sum'],
    'temperature_2m_max': daily_data['temperature_2m_max'],
    'temperature_2m_min': daily_data['temperature_2m_min'],
    'wind_speed_10m_max': daily_data['wind_speed_10m_max'],
    'wind_gusts_10m_max': daily_data['wind_gusts_10m_max']
})

# Create the same rolling features we used in training
df['rain_3day'] = df['precipitation_sum'].rolling(window=3).sum()
df['rain_7day'] = df['precipitation_sum'].rolling(window=7).sum()

# Fill NaNs (in case the API returns missing data)
df = df.fillna(0)

# Get today's data (the last row)
today = df.iloc[-1]

# 4. Prepare features for the model
features = [
    'precipitation_sum', 'rain_3day', 'rain_7day', 
    'temperature_2m_max', 'temperature_2m_min', 
    'wind_speed_10m_max', 'wind_gusts_10m_max'
]
X_live = today[features].values.reshape(1, -1)

# 5. Predict
probability = model.predict_proba(X_live)[0][1] * 100

# 6. Print the Results
print("\n" + "="*40)
print(f"🌧️  LIVE FLOOD RISK FOR PUNE - {today['date']}")
print("="*40)
print(f"🌡️  Max Temp: {today['temperature_2m_max']}°C")
print(f"🌧️  Rain Today: {today['precipitation_sum']} mm")
print(f"🌧️  Rain Last 3 Days: {today['rain_3day']} mm")
print(f"🌧️  Rain Last 7 Days: {today['rain_7day']} mm")
print(f"💨  Max Wind Speed: {today['wind_speed_10m_max']} km/h")
print("-" * 40)

if probability > 50:
    print(f"🚨 FLOOD RISK: {probability:.1f}% - VERY HIGH")
    print("Action: Prepare for potential flooding!")
elif probability > 20:
    print(f"⚠️  FLOOD RISK: {probability:.1f}% - MODERATE")
    print("Action: Keep monitoring the weather.")
else:
    print(f"✅ FLOOD RISK: {probability:.1f}% - LOW")
    print("Action: No immediate threat.")
print("="*40)
