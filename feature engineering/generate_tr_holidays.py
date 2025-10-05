# generate_tr_holidays.py
import os
import sys
import pandas as pd

try:
    from holidays import Turkey
except Exception:
    from holidays.countries import Turkey

INP = "data/daily_engagements_fe.csv"
OUT = "data/prophet_holidays_tr.csv"
DATE_COL = "event_date"

START = pd.Timestamp("2024-08-02")
END   = pd.Timestamp("2025-08-17")

def create_tr_holidays_range(start=START, end=END):
    years = list(range(start.year, end.year + 1))
    tr = Turkey(years=years)
    dates = pd.to_datetime(list(tr.keys()))
    mask = (dates >= start) & (dates <= end)
    dates = dates[mask]
    rows = [{"ds": d, "holiday": tr.get(d.date())} for d in dates]
    return pd.DataFrame(rows).sort_values("ds").reset_index(drop=True)

def weekend_effect_from_df(df, date_col=DATE_COL, start=START, end=END):
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col]).dt.normalize()
    mask_range = (out[date_col] >= start) & (out[date_col] <= end)
    wknd_dates = (
        out.loc[mask_range & out[date_col].dt.weekday.isin([5, 6]), date_col]
        .drop_duplicates()
        .sort_values()
        .reset_index(drop=True)
    )
    rows = [{"ds": d, "holiday": "weekend"} for d in wknd_dates]
    return pd.DataFrame(rows)

def main():
    if not os.path.exists(INP):
        print(f"[HATA] Girdi dosyası bulunamadı: {INP}")
        sys.exit(1)

    df = pd.read_csv(INP, parse_dates=[DATE_COL])

    tr_h = create_tr_holidays_range(START, END)
    wk_h = weekend_effect_from_df(df, DATE_COL, START, END)

    holidays_df = (
        pd.concat([tr_h, wk_h], ignore_index=True)
        .drop_duplicates(subset=["ds", "holiday"])
        .sort_values("ds")
        .reset_index(drop=True)
    )

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    holidays_df.to_csv(OUT, index=False)

    print(f"[OK] Çıktı: {OUT} → shape={holidays_df.shape}")
    print(holidays_df.head(10))

if __name__ == "__main__":
    main()
