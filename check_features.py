import joblib
import matplotlib.pyplot as plt
import numpy as np

# Load the V3 model
model = joblib.load('xgb_pune_flood_model_v3.pkl')
features = [
    'precipitation_sum', 'rain_3day', 'rain_7day', 'rain_14day', 
    'apparent_temperature_max', 'apparent_temperature_min', 
    'rain_sum', 'wind_speed_10m_max', 'wind_gusts_10m_max', 'wind_direction_10m_dominant'
]

# Get importances
importances = model.feature_importances_
indices = np.argsort(importances)

# Plot
plt.figure(figsize=(10, 6))
plt.title('Which Features Matter Most to the Pune Flood AI?')
plt.barh(range(len(indices)), importances[indices], color='#ef4444', align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance')
plt.tight_layout()
plt.show()
