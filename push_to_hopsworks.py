import os
import pandas as pd
from dotenv import load_dotenv
import hopsworks

from feature_pipeline import build_feature_record

load_dotenv()
HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")

record = build_feature_record()
df = pd.DataFrame([record])

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

print("Inserted record into Hopsworks feature group.")