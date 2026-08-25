import os
from datetime import datetime
from dotenv import load_dotenv
import hopsworks

from prepare_training_data import read_feature_group_in_chunks

load_dotenv()
HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")

project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY)
fs = project.get_feature_store()
fg = fs.get_feature_group(name="aqi_features", version=1)

df = read_feature_group_in_chunks(fg, "2026-08-16", "2026-08-26", chunk_days=10)
print(f"\nTotal rows found for Aug 16-26: {len(df)}")
if len(df) > 0:
    df = df.sort_values("timestamp")
    print(f"Earliest: {datetime.fromtimestamp(df['timestamp'].iloc[0])}")
    print(f"Latest:   {datetime.fromtimestamp(df['timestamp'].iloc[-1])}")