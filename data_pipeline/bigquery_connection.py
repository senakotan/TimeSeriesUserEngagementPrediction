import os
from google.cloud import bigquery
from google.oauth2 import service_account
from dotenv import load_dotenv

load_dotenv()

#BigQuery tarafında oluşturduğumuz service account json dosyasının yolu
KEY_PATH = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

#JSON dosyasını okuyup bir credentials objesi oluşturuyoruz. Giriş anahtarı gibi düşünülebilir.
credentials = service_account.Credentials.from_service_account_file(KEY_PATH)

#Verdiğimiz kimliği kullanarak BigQuery ile haberleşecek, sorgu gönderebileceğimiz istemci   
client = bigquery.Client(credentials=credentials, project="tabii-469409") 

#Bağlantı çalışıyor mu diye test amaçlı, sunumdan önce silinebilir.
query = "SELECT CURRENT_DATE() as today"
df = client.query(query).to_dataframe()
print(df)
