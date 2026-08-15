import os
from dotenv import load_dotenv
import hopsworks

load_dotenv()
HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")

project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY)
fs = project.get_feature_store()
fg = fs.get_feature_group(name="aqi_features", version=1)

df = fg.show(5)   # ask for just 5 rows, not the whole dataset
print(df)