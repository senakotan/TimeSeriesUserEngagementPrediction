import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

df = pd.read_csv('data/daily_engagements_final.csv')
holidays_df = pd.read_csv('data/prophet_holidays_tr.csv')

df['event_date'] = pd.to_datetime(df['event_date'])
holidays_df['ds'] = pd.to_datetime(holidays_df['ds'])

regressors = [
    'ortalama_sure_lag1','ortalama_sure_roll7_mean', 'toplam_izleme_suresi_dk_roll14_std',
    'ortalama_sure_roll14_mean', 'month_sin', 'ortalama_sure_roll28_mean',
    'toplam_izleme_suresi_dk_lag1', 'ortalama_sure_lag7'
]

yeni_hedef_degisken = 'ortalama_sure'

df_prophet = df.rename(columns={'event_date': 'ds', yeni_hedef_degisken: 'y'})
df_prophet = df_prophet[['ds', 'y'] + regressors]
df_prophet.dropna(inplace=True)

test_gun_sayisi = 40
train_df = df_prophet.iloc[:-test_gun_sayisi].copy()
test_df = df_prophet.iloc[-test_gun_sayisi:].copy()

scaler = StandardScaler()
train_df[regressors] = scaler.fit_transform(train_df[regressors])
test_df[regressors] = scaler.transform(test_df[regressors])

model = Prophet(
    holidays=holidays_df,
    yearly_seasonality=False,
    weekly_seasonality=False
)

for regressor in regressors:
    model.add_regressor(regressor)

model.fit(train_df)

future_df = pd.concat([train_df, test_df])[['ds'] + regressors]
forecast = model.predict(future_df)

y_true = test_df['y'].values
y_pred = forecast['yhat'][-test_gun_sayisi:].values

mae = mean_absolute_error(y_true, y_pred)
ortalama_deger = pd.concat([train_df, test_df])['y'].mean()
ortalama_yuzde_hata = (mae / ortalama_deger) * 100
dogruluk_orani = 100 - ortalama_yuzde_hata

print("- TAHMİN SONUÇLARI -")
print(f'-Ortalama Mutlak Hata (MAE): {mae:.2f} dakika')
print(f'-Ortalama Yüzdesel Hata: %{ortalama_yuzde_hata:.2f}')
print('-----------------------------------------')
print(f'-YAKLAŞIK DOĞRULUK ORANI: %{dogruluk_orani:.2f} ')
print('-----------------------------------------\n')


full_actual_data = pd.concat([train_df, test_df])

fig, ax = plt.subplots(figsize=(15, 7))
ax.plot(full_actual_data['ds'].dt.to_pydatetime(), full_actual_data['y'], label='Gerçek Değerler', color='dodgerblue', linewidth=2)
ax.plot(forecast['ds'].dt.to_pydatetime(), forecast['yhat'], label='Tahmin Edilen Değerler', color='red', linestyle='--')

split_date = test_df['ds'].iloc[0]
ax.axvline(x=split_date, color='green', linestyle=':', linewidth=2, label='Eğitim/Test Ayrımı')

ax.set_title('Ortalama Sure | Gerçek ve Tahmin Edilen Değerlerin Karşılaştırılması ')
ax.set_xlabel('Tarih')
ax.set_ylabel('Ortalama Süre (dk)')
ax.legend()
ax.grid(True)
fig.autofmt_xdate()
plt.tight_layout()
plt.show()


