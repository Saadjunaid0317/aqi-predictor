import requests
from datetime import datetime

from feature_pipeline import calculate_aqi   # reuse the AQI function we already built and trust

LAT = 24.8607
LON = 67.0011

def fetch_historical_weather(start_date, end_date):
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": LAT, "longitude": LON,
        "start_date": start_date, "end_date": end_date,
        "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m",
        "timezone": "Asia/Karachi"
    }
    return requests.get(url, params=params).json()

def fetch_historical_air_quality(start_date, end_date):
    url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    params = {
        "latitude": LAT, "longitude": LON,
        "start_date": start_date, "end_date": end_date,
        "hourly": "pm2_5,pm10,carbon_monoxide,nitrogen_dioxide,sulphur_dioxide,ozone",
        "timezone": "Asia/Karachi"
    }
    return requests.get(url, params=params).json()

def extract_time_features_from_iso(iso_string):
    # Open-Meteo already hands back LOCAL time as text (e.g. "2026-08-01T00:00"),
    # unlike OpenWeather's raw UTC timestamp — no timezone math needed this time.
    dt = datetime.fromisoformat(iso_string)
    return {"hour": dt.hour, "day": dt.day, "month": dt.month, "day_of_week": dt.weekday()}

def build_historical_records(start_date, end_date):
    weather = fetch_historical_weather(start_date, end_date)
    air = fetch_historical_air_quality(start_date, end_date)

    times = weather["hourly"]["time"]
    records = []

    for i, timestamp_str in enumerate(times):
        pm2_5 = air["hourly"]["pm2_5"][i]
        pm10 = air["hourly"]["pm10"][i]

        if pm2_5 is None or pm10 is None:
            continue   # some hours can have gaps in the data — skip rather than crash

        time_feats = extract_time_features_from_iso(timestamp_str)
        aqi, dominant = calculate_aqi(pm2_5, pm10)

        record = {
            "timestamp": int(datetime.fromisoformat(timestamp_str).timestamp()),
            "hour": time_feats["hour"],
            "day": time_feats["day"],
            "month": time_feats["month"],
            "day_of_week": time_feats["day_of_week"],
            "temp": weather["hourly"]["temperature_2m"][i],
            "humidity": weather["hourly"]["relative_humidity_2m"][i],
            "wind_speed": weather["hourly"]["wind_speed_10m"][i],
            "pm2_5": pm2_5,
            "pm10": pm10,
            "co": air["hourly"]["carbon_monoxide"][i],
            "no2": air["hourly"]["nitrogen_dioxide"][i],
            "so2": air["hourly"]["sulphur_dioxide"][i],
            "o3": air["hourly"]["ozone"][i],
            "aqi": aqi,
            "dominant_pollutant": dominant
        }
        records.append(record)

    return records




import os
import pandas as pd
from dotenv import load_dotenv
import hopsworks

if __name__ == "__main__":
    records = build_historical_records("2023-08-11", "2026-08-10")
    print(f"Built {len(records)} records")

    df = pd.DataFrame(records)    # turn the WHOLE list into one multi-row table at once

    load_dotenv()
    HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")

    project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY)
    fs = project.get_feature_store()

    fg = fs.get_or_create_feature_group(
        name="aqi_features",
        version=1,
        description="Weather, pollution, calculated AQI, and time features for Karachi",
        primary_key=["timestamp"],
        event_time="timestamp",
        time_travel_format="HUDI"
    )

    fg.insert(df)
    print(f"Inserted {len(df)} historical records into Hopsworks.")