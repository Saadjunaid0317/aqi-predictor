import os
from dotenv import load_dotenv
import hopsworks

load_dotenv()
HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")

project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY)
fs = project.get_feature_store()

fg = fs.get_feature_group(name="aqi_features", version=1)
print("Feature group found:", fg.name)

df = fg.read()
print("Row count:", len(df))
print(df.tail(3))