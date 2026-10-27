import requests
import pandas as pd
import numpy as np
import joblib
import os

class FloodPredictionEngine:
    def __init__(self, model_path='xgb_pune_flood_model_v3.pkl'):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model {model_path} not found.")
        self.model = joblib.load(model_path)
        self.features = [
            'precipitation_sum', 'rain_3day', 'rain_7day', 'rain_14day', 
            'apparent_temperature_max', 'apparent_temperature_min', 
            'rain_sum', 'wind_speed_10m_max', 'wind_gusts_10m_max', 'wind_direction_10m_dominant'
        ]

    def fetch_and_predict(self, lat=18.5204, lon=73.8567, sim_rain=None, sim_wind=None, sim_temp=None):
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": lat, "longitude": lon,
            "daily": ["precipitation_sum", "rain_sum", "apparent_temperature_max", "apparent_temperature_min", "wind_speed_10m_max", "wind_gusts_10m_max", "wind_direction_10m_dominant"],
            "past_days": 14, "forecast_days": 7, "timezone": "Asia/Kolkata"
        }
        response = requests.get(url, params=params).json()
        daily = response['daily']
        
        df = pd.DataFrame({
            'date': pd.to_datetime(daily['time']),
            'precipitation_sum': daily['precipitation_sum'],
            'rain_sum': daily['rain_sum'],
            'apparent_temperature_max': daily['apparent_temperature_max'],
            'apparent_temperature_min': daily['apparent_temperature_min'],
            'wind_speed_10m_max': daily['wind_speed_10m_max'],
            'wind_gusts_10m_max': daily['wind_gusts_10m_max'],
            'wind_direction_10m_dominant': daily['wind_direction_10m_dominant']
        })

        # --- REALISTIC CUSTOM SIMULATION ---
        # If any simulation parameter is provided, override the forecast for 3 days
        if sim_rain is not None or sim_wind is not None or sim_temp is not None:
            for i in range(14, 17): # Apply to next 3 days
                if sim_rain is not None:
                    df.at[i, 'precipitation_sum'] = sim_rain
                    df.at[i, 'rain_sum'] = sim_rain
                if sim_wind is not None:
                    df.at[i, 'wind_speed_10m_max'] = sim_wind
                    df.at[i, 'wind_gusts_10m_max'] = sim_wind * 1.5 # Gusts are higher
                if sim_temp is not None:
                    df.at[i, 'apparent_temperature_max'] = sim_temp
                    df.at[i, 'apparent_temperature_min'] = sim_temp - 5
                
                # Keep storm wind direction fixed from the west
                df.at[i, 'wind_direction_10m_dominant'] = 270 
            
            # Dry out after the 3-day storm
            for i in range(17, 21):
                df.at[i, 'precipitation_sum'] = 0
                df.at[i, 'rain_sum'] = 0

        df['rain_3day'] = df['precipitation_sum'].rolling(window=3).sum()
        df['rain_7day'] = df['precipitation_sum'].rolling(window=7).sum()
        df['rain_14day'] = df['precipitation_sum'].rolling(window=14).sum()
        df = df.fillna(0)

        past_df = df.iloc[:14].copy()
        forecast_df = df.iloc[14:].copy()

        risks = []
        for _, row in forecast_df.iterrows():
            X = row[self.features].values.reshape(1, -1)
            prob = self.model.predict_proba(X)[0][1] * 100
            risks.append(prob)
        
        forecast_df['risk'] = risks
        max_risk = forecast_df['risk'].max()
        
        if max_risk > 50:
            overall_status = "VERY HIGH"
        elif max_risk > 20:
            overall_status = "MODERATE"
        else:
            overall_status = "LOW"

        return {
            "past": past_df, "forecast": forecast_df,
            "today_features": forecast_df.iloc[0][self.features].to_dict(),
            "today_risk": float(forecast_df.iloc[0]['risk']),
            "max_risk_7day": float(max_risk), "status": overall_status
        }
