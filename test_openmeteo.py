import requests

LAT = 24.8607
LON = 67.0011

# Historical WEATHER (separate endpoint from air quality)
weather_url = "https://archive-api.open-meteo.com/v1/archive"
weather_params = {
    "latitude": LAT,
    "longitude": LON,
    "start_date": "2026-08-01",
    "end_date": "2026-08-07",
    "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m",
    "timezone": "Asia/Karachi"     # ask Open-Meteo to return times already in Karachi's local time
}
weather_response = requests.get(weather_url, params=weather_params)
weather_data = weather_response.json()

# Historical AIR QUALITY (a separate endpoint entirely)
air_url = "https://air-quality-api.open-meteo.com/v1/air-quality"
air_params = {
    "latitude": LAT,
    "longitude": LON,
    "start_date": "2026-08-01",
    "end_date": "2026-08-07",
    "hourly": "pm2_5,pm10,carbon_monoxide,nitrogen_dioxide,sulphur_dioxide,ozone",
    "timezone": "Asia/Karachi"
}
air_response = requests.get(air_url, params=air_params)
air_data = air_response.json()

print("--- WEATHER SAMPLE ---")
print(weather_data["hourly"]["time"][:5])          # first 5 timestamps
print(weather_data["hourly"]["temperature_2m"][:5]) # first 5 temperature readings

print("--- AIR QUALITY SAMPLE ---")
print(air_data["hourly"]["time"][:5])
print(air_data["hourly"]["pm2_5"][:5])