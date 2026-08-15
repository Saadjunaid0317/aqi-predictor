import requests
import os
from datetime import datetime, timezone, timedelta
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
    if concentration is None:
        return None

    concentration = max(concentration, 0)   # clamp negative model noise to 0

    for bp_lo, bp_hi, i_lo, i_hi in breakpoints:
        if concentration <= bp_hi:
            return ((i_hi - i_lo) / (bp_hi - bp_lo)) * (concentration - bp_lo) + i_lo

    return 500   # worse than every row -> treat as max hazardous

def calculate_aqi(pm2_5, pm10):
    pm25_index = calculate_sub_index(pm2_5, PM25_BREAKPOINTS)
    pm10_index = calculate_sub_index(pm10, PM10_BREAKPOINTS)
    aqi = max(pm25_index, pm10_index)
    dominant = "PM2.5" if pm25_index >= pm10_index else "PM10"
    return round(aqi), dominant

def extract_time_features(dt_unix, timezone_offset_seconds):
    utc_time = datetime.fromtimestamp(dt_unix, tz=timezone.utc)
    local_time = utc_time + timedelta(seconds=timezone_offset_seconds)
    return {
        "hour": local_time.hour,
        "day": local_time.day,
        "month": local_time.month,
        "day_of_week": local_time.weekday()
    }

def fetch_weather():
    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {"lat": LAT, "lon": LON, "appid": API_KEY, "units": "metric"}
    response = requests.get(url, params=params)
    return response.json()

def fetch_pollution():
    url = "https://api.openweathermap.org/data/2.5/air_pollution"
    params = {"lat": LAT, "lon": LON, "appid": API_KEY}
    response = requests.get(url, params=params)
    return response.json()

def build_feature_record():
    weather = fetch_weather()
    pollution = fetch_pollution()

    time_feats = extract_time_features(weather["dt"], weather["timezone"])
    components = pollution["list"][0]["components"]
    aqi, dominant = calculate_aqi(components["pm2_5"], components["pm10"])

    record = {
        "timestamp": weather["dt"],
        "hour": time_feats["hour"],
        "day": time_feats["day"],
        "month": time_feats["month"],
        "day_of_week": time_feats["day_of_week"],
        "temp": float(weather["main"]["temp"]),
        "humidity": float(weather["main"]["humidity"]),
        "wind_speed": float(weather["wind"]["speed"]),
        "pm2_5": float(components["pm2_5"]),
        "pm10": float(components["pm10"]),
        "co": float(components["co"]),
        "no2": float(components["no2"]),
        "so2": float(components["so2"]),
        "o3": float(components["o3"]),
        "aqi": aqi,
        "dominant_pollutant": dominant
        }
    return record

if __name__ == "__main__":
    record = build_feature_record()
    print(record)