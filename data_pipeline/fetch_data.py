import pandas as pd
from .bigquery_connection import client


PROJECT_ID = "tabii-469409"
DATASET_ID = "anormaly_detection"
TABLE_FQN = f"{PROJECT_ID}.{DATASET_ID}.tabii_engagement_data"
LOCATION = "US"

SQL = f"""
WITH base AS (
  SELECT
    DATE(event_date) AS event_date,
    view_id,
    TIMESTAMP_DIFF(TIMESTAMP(view_end), TIMESTAMP(view_start), SECOND) / 60.0 AS izleme_suresi_dk
  FROM `{TABLE_FQN}`
  WHERE event_date >= '2024-08-01'
)
SELECT
  event_date,
  COUNT(DISTINCT view_id) AS oturum_sayisi,
  SUM(izleme_suresi_dk) AS toplam_izleme_suresi_dk
FROM base
GROUP BY event_date
ORDER BY event_date
"""


def get_daily_engagements():
    job = client.query(SQL, location=LOCATION)
    df = job.result().to_dataframe(create_bqstorage_client=True)

    df["event_date"] = pd.to_datetime(df["event_date"])
    df["oturum_sayisi"] = pd.to_numeric(df["oturum_sayisi"])
    df["toplam_izleme_suresi_dk"] = pd.to_numeric(df["toplam_izleme_suresi_dk"])

    return df


if __name__ == "__main__":
    df = get_daily_engagements()
    print("Satır:", len(df))
    print(df.head())

    output_path = "data/daily_engagements.csv"
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
