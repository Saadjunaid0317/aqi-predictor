# Pearls AQI Predictor — Progress Cheat Sheet

**Project:** 10Pearls SHINE Internship, Data Sciences track
**Goal:** A serverless system that forecasts Karachi's AQI 3 days ahead, end to end — data collection, training, automation, and a dashboard.

---

## 1. The Big Picture (Architecture)

Four connected pieces:

```
Weather & Pollution API → Feature Pipeline ──┐
                                              ▼
                          Training Pipeline ↔ Feature Store & Model Registry
                                              ▲
                                    Web App (dashboard) ──┘
```

| Piece | Job |
|---|---|
| **Weather API** | External source of raw numbers (temperature, PM2.5, etc.) |
| **Feature pipeline** | Turns raw numbers into structured, model-ready inputs |
| **Feature store & model registry** | Shared storage — write once, read from many places (like a vector store in RAG) |
| **Training pipeline** | Learns patterns from historical features, saves the best model |
| **Web app** | Loads the latest features + model, shows a 3-day forecast |
| **Automation (CI/CD)** | Runs the feature script hourly and training script daily, on its own — no manual runs |

**Key idea to remember:** this isn't "train a model once." It's a system designed to keep collecting data and re-predicting *by itself*, forever, without you touching it after setup.

---

## 2. Environment Setup — Done ✅

- **Editor:** VS Code, using an Anaconda Python interpreter underneath.
- **Project folder:** `D:\AQI-Predictor`
- **Isolated conda environment** created specifically for this project (keeps its packages separate from everything else on the system):

```bash
conda create -n aqi-predictor python=3.11 -y
conda activate aqi-predictor
```

- **Windows-specific fix applied:** VS Code's terminal (PowerShell) didn't know about `conda` by default. Fixed once, permanently, by running this from a separate **Anaconda Prompt** window:
```bash
conda init powershell
```
  (then opened a *fresh* VS Code terminal for it to take effect)

- **Packages installed** (inside the `aqi-predictor` environment):
```bash
pip install requests        # for calling APIs
pip install python-dotenv   # for reading the API key safely
```

---

## 3. API Key — Secured Properly ✅

**Data source chosen:** OpenWeather (Current Weather API + Air Pollution API — one free key covers both).

Deliberately **not** hardcoded in the script, because this project will eventually go on GitHub, and anyone could see a key sitting in plain code.

- `.env` file (never uploaded to GitHub) holds the real key:
```
OPENWEATHER_API_KEY=your_actual_key_here
```
- `.gitignore` file tells git to always ignore it:
```
.env
```
- Script reads it like this:
```python
import os
from dotenv import load_dotenv
load_dotenv()
API_KEY = os.getenv("OPENWEATHER_API_KEY")
```

---

## 4. Scripts Written So Far

### `test_fetch.py` — raw weather for Karachi
```python
import requests                    # library for calling web APIs over the internet
import os                          # lets us read values from the environment
from dotenv import load_dotenv     # tool for loading secrets out of a .env file

load_dotenv()                      # reads .env and makes its contents available
API_KEY = os.getenv("OPENWEATHER_API_KEY")   # pulls the key out by its name

LAT = 24.8607                      # Karachi's latitude
LON = 67.0011                      # Karachi's longitude

url = "https://api.openweathermap.org/data/2.5/weather"   # fixed address of the "current weather" endpoint
params = {
    "lat": LAT,
    "lon": LON,
    "appid": API_KEY,              # this is what proves the request is authorized
    "units": "metric"              # asks for Celsius instead of the API's default, Kelvin
}

response = requests.get(url, params=params)   # sends the request, waits for OpenWeather's reply
data = response.json()             # the reply arrives as raw text; this turns it into a Python dict
print(data)                        # show us what came back
```

**Line-by-line, what's actually happening:**
- `import requests` / `os` / `load_dotenv` — these three lines just load the tools the rest of the script needs. Nothing runs yet, we're just gathering equipment.
- `load_dotenv()` + `os.getenv(...)` — this is the secure key-reading pair from Section 3 above: open `.env`, find the line named `OPENWEATHER_API_KEY`, hand back its value.
- `LAT` / `LON` — plain variables holding two numbers. We use coordinates instead of typing "Karachi" because city names can be ambiguous (multiple cities share names); coordinates are exact.
- `url` — the one fixed web address for this specific feature. Every request to "current weather" goes to this same place.
- `params` — a dictionary of extra info to attach to the request (where, who's asking, what units). `requests` automatically turns this into the `?lat=...&lon=...` part of the web address for you — you never have to build that string by hand.
- `requests.get(url, params=params)` — the actual network call. This is the line that leaves your computer, goes to OpenWeather's server, and waits for an answer.
- `.json()` — converts the server's raw text reply into a Python dictionary, so you can access pieces of it like `data["main"]["temp"]` instead of parsing text yourself.
- `print(data)` — just so a human can see it. In the real feature pipeline later, we won't print this — we'll extract specific fields from it instead.

**Confirmed working** — returned real Karachi weather (`'cod': 200`, `'name': 'Karachi'`).

### `test_pollution.py` — raw pollution + real AQI calculation
```python
import requests
import os
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("OPENWEATHER_API_KEY")

LAT = 24.8607
LON = 67.0011

# --- Official EPA breakpoint tables ---
# Each row = (concentration_low, concentration_high, aqi_low, aqi_high)
# These are the fixed government cutoffs. If a reading falls between
# concentration_low and concentration_high, its AQI falls somewhere
# between aqi_low and aqi_high — exactly where, we calculate below.
# (PM2.5 table reflects the EPA's 2024 update — older tutorials online
# still show the pre-2024 numbers, which are now outdated.)
PM25_BREAKPOINTS = [
    (0.0, 9.0, 0, 50),         # Good
    (9.1, 35.4, 51, 100),      # Moderate
    (35.5, 55.4, 101, 150),    # Unhealthy for Sensitive Groups
    (55.5, 125.4, 151, 200),   # Unhealthy
    (125.5, 225.4, 201, 300),  # Very Unhealthy
    (225.5, 325.4, 301, 500),  # Hazardous
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
    # Walk through each row until we find the one this concentration fits inside
    for bp_lo, bp_hi, i_lo, i_hi in breakpoints:
        if bp_lo <= concentration <= bp_hi:
            # Official EPA formula: find how far concentration sits between
            # bp_lo and bp_hi (as a fraction), then apply that same fraction
            # to the matching AQI range (i_lo to i_hi).
            return ((i_hi - i_lo) / (bp_hi - bp_lo)) * (concentration - bp_lo) + i_lo
    return None   # concentration was outside every table row (shouldn't normally happen)

def calculate_aqi(pm2_5, pm10):
    pm25_index = calculate_sub_index(pm2_5, PM25_BREAKPOINTS)   # AQI if PM2.5 were the only pollutant
    pm10_index = calculate_sub_index(pm10, PM10_BREAKPOINTS)    # AQI if PM10 were the only pollutant
    aqi = max(pm25_index, pm10_index)    # official rule: the WORST pollutant wins, never an average
    dominant = "PM2.5" if pm25_index >= pm10_index else "PM10"  # which pollutant "caused" this AQI
    return round(aqi), dominant

url = "https://api.openweathermap.org/data/2.5/air_pollution"   # pollution-specific endpoint (different from weather)
params = {"lat": LAT, "lon": LON, "appid": API_KEY}

response = requests.get(url, params=params)
data = response.json()
print(data)   # raw reply: pollutant concentrations + OpenWeather's own 1-5 index (not what we want)

components = data["list"][0]["components"]   # dig into the nested dict to reach the pollutant numbers
aqi, dominant = calculate_aqi(components["pm2_5"], components["pm10"])
print(f"Calculated AQI: {aqi} (dominant: {dominant})")
```

**Line-by-line, what's actually happening:**
- Setup lines (`import`, `load_dotenv`, `LAT`/`LON`) — identical purpose to the weather script above.
- `PM25_BREAKPOINTS` / `PM10_BREAKPOINTS` — these aren't calculated, they're just typed-in copies of the government's official conversion table. One row per health category.
- `calculate_sub_index(concentration, breakpoints)` — a reusable function that works for *any* pollutant's table, not just PM2.5. Given a concentration and a table, it finds the matching row, then does straight-line interpolation to pinpoint the exact AQI value (not just "somewhere between 51 and 100," but the precise number, e.g. 77).
- Why interpolation is needed at all: a table can only give exact answers at its listed boundaries (9.0 → exactly 50, 35.4 → exactly 100). Since a real reading like 22.97 falls *between* those points, we calculate proportionally where it lands — that's what the formula on that `return` line is doing.
- `calculate_aqi(pm2_5, pm10)` — runs both pollutants through their own tables using the function above, then keeps whichever result is *higher* (worse air quality) — because the real AQI is defined as "your single worst pollutant," never a blend of all of them.
- `components = data["list"][0]["components"]` — the pollutant numbers are nested a few levels deep inside OpenWeather's response (a list containing a dict containing another dict). This line just digs down to reach them.
- Final two lines — call our function with the real numbers from the API, then print a human-readable result.

**Confirmed working** — returned `Calculated AQI: 77 (dominant: PM2.5)`, cross-checked against independent live AQI sites for Karachi (which showed 67–102, all "Moderate") — consistent.

---

## 5. Key Concepts Learned (Glossary)

- **AQI** — a single 0–500 number summarizing air pollution; higher = worse. Built from the *worst-scoring* individual pollutant, not an average.
- **Why we calculate AQI ourselves** — OpenWeather's built-in `aqi` field is its own 1–5 scale, *not* the standard 0–500 AQI people actually recognize. We convert raw pollutant concentrations (PM2.5, PM10) into real AQI using the official EPA breakpoint formula (piecewise linear interpolation between table values).
- **Feature** — a raw value transformed into something a model can learn from (e.g., turning a timestamp into "hour of day").
- **Feature store** — shared storage so features are computed once and reused by multiple pipelines, instead of every script recomputing them from scratch (same idea as a vector store in RAG).
- **Model registry** — where trained models are saved so other parts of the system (like the web app) can load the latest one without retraining.
- **Isolated environment (`conda create -n ...`)** — a separate, self-contained Python + package set for this project only, avoiding version conflicts with other projects.
- **Sanity-checking API data** — three checks: (1) did the request succeed (`cod: 200`), (2) did it return data for the right location, (3) is the value plausible compared to an independent source.

---

## 6. Decisions Made & Why

| Decision | Reasoning |
|---|---|
| OpenWeather over AQICN | One key covers both weather + pollution; has a real historical endpoint for backfill |
| OpenWeather over Open-Meteo | Deliberately chose to practice handling a real API key — a skill needed almost everywhere in production APIs |
| Coordinates (lat/lon) instead of city name | City names can be ambiguous; coordinates are exact |
| `.env` + `.gitignore` for the key | Keeps secrets out of the GitHub repo once this is pushed |
| Calculate AQI ourselves (not use OpenWeather's built-in index) | OpenWeather's AQI is a 1–5 scale, not the real 0–500 scale we need to predict and display |

---

## 7. What's Next

- Add **time-based features**: hour of day, day of week, month — since AQI follows daily/seasonal patterns a model needs to see explicitly.
- Combine weather + pollution + calculated AQI + time features into one unified feature record.
- Set up the actual **Feature Store** (Hopsworks) to save these records instead of just printing them.
- Then: historical backfill → training pipeline → automation → web app.
