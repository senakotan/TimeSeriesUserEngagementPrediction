import pandas as pd
import numpy as np
import os 

def _ensure_datetime(df, date_col="event_date"):
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
    return out.sort_values(date_col).reset_index(drop=True)

def add_cyclical_if_missing(df, day_col="hafta_gunu", month_col="ay"):
    out = df.copy()
    if day_col in out.columns:
        if "dow_sin" not in out.columns:
            out["dow_sin"] = np.sin(2*np.pi*out[day_col]/7.0)
        if "dow_cos" not in out.columns:
            out["dow_cos"] = np.cos(2*np.pi*out[day_col]/7.0)
    if month_col in out.columns:
        if "month_sin" not in out.columns:
            out["month_sin"] = np.sin(2*np.pi*out[month_col]/12.0)
        if "month_cos" not in out.columns:
            out["month_cos"] = np.cos(2*np.pi*out[month_col]/12.0)
    return out

def add_lags_if_missing(df, cols, lags=(1, 2, 7, 14, 28)):
    out = df.copy()
    for c in cols:
        if c not in out.columns: 
            continue
        for L in lags:
            name = f"{c}_lag{L}"
            if name not in out.columns:
                out[name] = out[c].shift(L)
    return out

def add_rollings_if_missing(df, cols, windows=(7, 14, 28)):
    out = df.copy()
    for c in cols:
        if c not in out.columns: 
            continue
        shifted = out[c].shift(1)  
        for w in windows:
            mname = f"{c}_roll{w}_mean"
            sname = f"{c}_roll{w}_std"
            if mname not in out.columns:
                out[mname] = shifted.rolling(w).mean()
            if sname not in out.columns:
                out[sname] = shifted.rolling(w).std()
    return out

def add_diffs_if_missing(df, cols, periods=(1, 7)):
    out = df.copy()
    for c in cols:
        if c not in out.columns: 
            continue
        for p in periods:
            name = f"{c}_diff{p}"
            if name not in out.columns:
                out[name] = out[c].diff(p)
    return out

def add_outlier_carry_if_missing(df, outlier_cols=("outlier_oturum", "outlier_sure"), horizons=(1, 2, 3)):
    out = df.copy()
    for o in outlier_cols:  
        if o not in out.columns: 
            continue
        for h in horizons:
            name = f"after_{o}_{h}d"
            if name not in out.columns:
                out[name] = out[o].shift(h).fillna(0).astype(int)
    return out

def engineer_features(df, date_col="event_date"):
    out = _ensure_datetime(df, date_col)
    target_cols = [c for c in ["oturum_sayisi", "toplam_izleme_suresi_dk", "ortalama_sure"] if c in out.columns]
    out = add_cyclical_if_missing(out, day_col="hafta_gunu", month_col="ay")
    out = add_lags_if_missing(out, cols=target_cols)
    out = add_rollings_if_missing(out, cols=target_cols)
    out = add_diffs_if_missing(out, cols=target_cols)
    out = add_outlier_carry_if_missing(out)
    return out

if __name__ == "__main__":
    input_file = os.path.join("data", "daily_engagements_clean.csv") 
    output_file = os.path.join("data", "daily_engagements_fe2.csv")    
    df = pd.read_csv(input_file)  
    out = engineer_features(df)
    out.to_csv(output_file, index=False)
    print(f"Feature engineering tamamlandı → {output_file} (shape={out.shape})")
