import requests
import os
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("OPENWEATHER_API_KEY")

LAT = 24.8607
LON = 67.0011

PM25_BREAKPOINTS = [
    (0.0, 9.0, 0, 50),
    (9.1, 35.4, 51, 100),
    (35.5, 55.4, 101, 150),
    (55.5, 125.4, 151, 200),
    (125.5, 225.4, 201, 300),
    (225.5, 325.4, 301, 500),
]

PM10_BREAKPOINTS = [
    (0, 54, 0, 50),
    (55, 154, 51, 100),
    (155, 254, 101, 150),
    (255, 354, 151, 200),
    (355, 424, 201, 300),
    (425, 604, 301, 500),
]

def calculate_sub_index(concentration, breakpoints):
    for bp_lo, bp_hi, i_lo, i_hi in breakpoints:
        if bp_lo <= concentration <= bp_hi:
            return ((i_hi - i_lo) / (bp_hi - bp_lo)) * (concentration - bp_lo) + i_lo
    return None

def calculate_aqi(pm2_5, pm10):
    pm25_index = calculate_sub_index(pm2_5, PM25_BREAKPOINTS)
    pm10_index = calculate_sub_index(pm10, PM10_BREAKPOINTS)
    aqi = max(pm25_index, pm10_index)
    dominant = "PM2.5" if pm25_index >= pm10_index else "PM10"
    return round(aqi), dominant


url = "https://api.openweathermap.org/data/2.5/air_pollution"
params = {
    "lat": LAT,
    "lon": LON,
    "appid": API_KEY
}

response = requests.get(url, params=params)
data = response.json()

print(data)

components = data["list"][0]["components"]
aqi, dominant = calculate_aqi(components["pm2_5"], components["pm10"])
print(f"Calculated AQI: {aqi} (dominant: {dominant})")