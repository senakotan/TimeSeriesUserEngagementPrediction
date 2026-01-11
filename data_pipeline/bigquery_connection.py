import os
from google.cloud import bigquery
from google.oauth2 import service_account
from dotenv import load_dotenv

load_dotenv()

KEY_PATH = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

credentials = service_account.Credentials.from_service_account_file(KEY_PATH)

client = bigquery.Client(
    credentials=credentials,
    project="tabii-469409"
)

query = "SELECT CURRENT_DATE() as today"
df = client.query(query).to_dataframe()
print(df)
