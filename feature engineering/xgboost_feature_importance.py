from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

HERE = Path(__file__).resolve().parent
CSV_PATH = HERE / "data" / "daily_engagements_final.csv"
OUT_DIR = HERE / "_outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"CSV okunuyor: {CSV_PATH}")
assert CSV_PATH.exists(), f"CSV bulunamadı: {CSV_PATH}"

df = pd.read_csv(CSV_PATH)

if "event_date" in df.columns:
    df["event_date"] = pd.to_datetime(df["event_date"], errors="coerce")
    df = df.sort_values("event_date").reset_index(drop=True)

target = "ortalama_sure"
exclude = {"event_date", "oturum_sayisi", "toplam_izleme_suresi_dk", "ortalama_sure"}
non_numeric_cols = set(df.select_dtypes(exclude=[np.number]).columns.tolist())
exclude = exclude.union(non_numeric_cols)

feature_cols = [c for c in df.columns if c not in exclude]
X_all = df[feature_cols].copy()
y_all = df[target].copy()

valid_idx = X_all.dropna().index.intersection(y_all.dropna().index)
X_all = X_all.loc[valid_idx].reset_index(drop=True)
y_all = y_all.loc[valid_idx].reset_index(drop=True)

assert all(np.issubdtype(dt, np.number) for dt in X_all.dtypes), "X içinde numerik olmayan kolon var."

print(f" Kullanılan özellik sayısı: {X_all.shape[1]}")
print(f" Toplam örnek: {X_all.shape[0]}")

split_idx = int(len(X_all) * 0.80)
X_train, X_test = X_all.iloc[:split_idx], X_all.iloc[split_idx:]
y_train, y_test = y_all.iloc[:split_idx], y_all.iloc[split_idx:]

print(f"Train: {X_train.shape}, Test: {X_test.shape}")

model = XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.9,
    colsample_bytree=0.9,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1,
    eval_metric="rmse"
)

model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = float(np.sqrt(mse))
mae = mean_absolute_error(y_test, y_pred)
print(f"Test RMSE: {rmse:.4f} | MAE: {mae:.4f}")

booster = model.get_booster()
gain_dict = booster.get_score(importance_type='gain')

name_map = {f"f{i}": col for i, col in enumerate(X_train.columns)}
rows = [{"feature": name_map.get(f_name, f_name), "gain_importance": val}
        for f_name, val in gain_dict.items()]
imp_df = pd.DataFrame(rows).sort_values("gain_importance", ascending=False)

out_csv = OUT_DIR / "xgb_importance_ortalama_sure.csv"
imp_df.to_csv(out_csv, index=False)
print(f"Önem tablosu: {out_csv}")

if len(imp_df):
    topk = min(20, len(imp_df))
    plot_df = imp_df.head(topk).iloc[::-1]
    plt.figure(figsize=(10, 7))
    plt.barh(plot_df["feature"], plot_df["gain_importance"])
    plt.title("Top 20 Features for 'ortalama_sure' (XGBoost - Gain)")
    plt.xlabel("Gain Importance")
    plt.tight_layout()
    out_png = OUT_DIR / "xgb_importance_ortalama_sure_top20.png"
    plt.savefig(out_png, dpi=150)
    plt.show()
    print(f"Grafik: {out_png}")

