import os
import numpy as np
import pandas as pd
from .fetch_data import get_daily_engagements


def _flag_outliers_iqr(s, k=1.5):
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - k * iqr, q3 + k * iqr
    return (s < lower) | (s > upper)


def _flag_outliers_z(s, thresh=3.0):
    s = pd.to_numeric(s, errors="coerce")
    mean = s.mean()
    std = s.std(ddof=0)

    if std == 0 or np.isnan(std):
        return pd.Series(False, index=s.index)

    z = (s - mean) / std
    return z.abs() > thresh


def preprocess_daily(df, start_date=None, end_date=None, fill_method="zero", save_csv_path=None):
    df = df.copy()
    df["event_date"] = pd.to_datetime(df["event_date"])
    df["oturum_sayisi"] = pd.to_numeric(df["oturum_sayisi"], errors="coerce")
    df["toplam_izleme_suresi_dk"] = pd.to_numeric(df["toplam_izleme_suresi_dk"], errors="coerce")

    df = df.sort_values("event_date")

    if start_date:
        df = df[df["event_date"] >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df["event_date"] <= pd.to_datetime(end_date)]

    full_idx = pd.date_range(df["event_date"].min(), df["event_date"].max(), freq="D")
    df = df.set_index("event_date").reindex(full_idx)
    df.index.name = "event_date"

    if fill_method == "zero":
        df[["oturum_sayisi", "toplam_izleme_suresi_dk"]] = \
            df[["oturum_sayisi", "toplam_izleme_suresi_dk"]].fillna(0)

    elif fill_method == "ffill":
        df[["oturum_sayisi", "toplam_izleme_suresi_dk"]] = \
            df[["oturum_sayisi", "toplam_izleme_suresi_dk"]].ffill().fillna(0)

    elif fill_method == "bfill":
        df[["oturum_sayisi", "toplam_izleme_suresi_dk"]] = \
            df[["oturum_sayisi", "toplam_izleme_suresi_dk"]].bfill().fillna(0)

    elif fill_method == "median":
        df[["oturum_sayisi", "toplam_izleme_suresi_dk"]] = \
            df[["oturum_sayisi", "toplam_izleme_suresi_dk"]].apply(
                lambda s: s.fillna(s.median())
            )

    elif fill_method == "mode":
        for col in ["oturum_sayisi", "toplam_izleme_suresi_dk"]:
            mode_val = df[col].mode(dropna=True)
            fill_val = mode_val.iloc[0] if len(mode_val) > 0 else 0
            df[col] = df[col].fillna(fill_val)
    else:
        raise ValueError("fill_method 'zero', 'ffill' , 'bfill' ,'median' veya 'mode' olmalı")

    df["ortalama_sure"] = np.where(
        df["oturum_sayisi"] > 0,
        df["toplam_izleme_suresi_dk"] / df["oturum_sayisi"],
        0.0,
    )

    df["yil"] = df.index.year
    df["ay"] = df.index.month
    df["hafta_gunu"] = df.index.weekday
    df["gun_adi"] = df.index.day_name(locale="en_US")
    df["ay_baslangic"] = (df.index.is_month_start).astype(int)
    df["ay_sonu"] = (df.index.is_month_end).astype(int)

    df["outlier_oturum_iqr"] = _flag_outliers_iqr(df["oturum_sayisi"]).astype(int)
    df["outlier_sure_iqr"] = _flag_outliers_iqr(df["toplam_izleme_suresi_dk"]).astype(int)

    df["outlier_oturum_z"] = _flag_outliers_z(df["oturum_sayisi"]).astype(int)
    df["outlier_sure_z"] = _flag_outliers_z(df["toplam_izleme_suresi_dk"]).astype(int)

    max_sure_day = df["toplam_izleme_suresi_dk"].idxmax()
    min_sure_day = df["toplam_izleme_suresi_dk"].idxmin()
    max_otr_day = df["oturum_sayisi"].idxmax()
    min_otr_day = df["oturum_sayisi"].idxmin()

    print("\nPREPROCESS SONRASI VERİ")
    print("Satır sayısı:", len(df))
    print("\nEksik değerler:")
    print(df[["oturum_sayisi", "toplam_izleme_suresi_dk", "ortalama_sure"]].isna().sum())

    print("\nMax-Min Günleri:")
    print("Max izleme süresi:", max_sure_day, "-", int(df.loc[max_sure_day, "toplam_izleme_suresi_dk"]))
    print("Min izleme süresi:", min_sure_day, "-", int(df.loc[min_sure_day, "toplam_izleme_suresi_dk"]))
    print("Max oturum sayısı:", max_otr_day, "-", int(df.loc[max_otr_day, "oturum_sayisi"]))
    print("Min oturum sayısı:", min_otr_day, "-", int(df.loc[min_otr_day, "oturum_sayisi"]))

    if save_csv_path:
        os.makedirs(os.path.dirname(save_csv_path), exist_ok=True)
        df.to_csv(save_csv_path, index=True, encoding="utf-8-sig")

    return df


if __name__ == "__main__":
    raw_df = get_daily_engagements()

    print("\nPREPROCESS ÖNCESİ VERİ")
    print(raw_df.head())
    print(raw_df.info())
    print()
    print(raw_df[["event_date", "oturum_sayisi", "toplam_izleme_suresi_dk"]].describe())

    clean_df = preprocess_daily(
        raw_df,
        start_date="2024-08-01",
        end_date="2025-08-17",
        fill_method="zero",
        save_csv_path="data/daily_engagements_process.csv",
    )

    print("\nİSTATİSTİKLER ")
    print(clean_df[["oturum_sayisi", "toplam_izleme_suresi_dk", "ortalama_sure"]].describe())
