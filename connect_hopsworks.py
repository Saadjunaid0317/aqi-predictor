import os
from dotenv import load_dotenv
import hopsworks

load_dotenv()
HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")

project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY)
fs = project.get_feature_store()

print("Connected to project:", project.name)
print("Feature store:", fs.name)