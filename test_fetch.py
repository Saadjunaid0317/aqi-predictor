import requests
import os
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("OPENWEATHER_API_KEY")

LAT = 24.8607
LON = 67.0011


from datetime import datetime, timezone, timedelta   # tools for working with dates/times

def extract_time_features(dt_unix, timezone_offset_seconds):
    utc_time = datetime.fromtimestamp(dt_unix, tz=timezone.utc)   # the raw timestamp, in UTC
    local_time = utc_time + timedelta(seconds=timezone_offset_seconds)  # shift it to Karachi's local time

    return {
        "hour": local_time.hour,             # 0-23, Karachi local hour
        "day": local_time.day,               # day of the month
        "month": local_time.month,           # 1-12
        "day_of_week": local_time.weekday()  # 0=Monday ... 6=Sunday
    }

url = "https://api.openweathermap.org/data/2.5/weather"
params = {
    "lat": LAT,
    "lon": LON,
    "appid": API_KEY,
    "units": "metric"
}

response = requests.get(url, params=params)
data = response.json()

print(data)

time_features = extract_time_features(data["dt"], data["timezone"])
print(time_features)