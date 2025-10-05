import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from statsmodels.tsa.statespace.sarimax import SARIMAX

df = pd.read_csv('data/daily_engagements_final.csv')
holidays_df = pd.read_csv('data/prophet_holidays_tr.csv')

df['event_date'] = pd.to_datetime(df['event_date'])
holidays_df['ds'] = pd.to_datetime(holidays_df['ds'])

regressors = [
    'ortalama_sure_lag1','ortalama_sure_roll7_mean','toplam_izleme_suresi_dk_roll14_std',
    'ortalama_sure_roll14_mean','month_sin','ortalama_sure_roll28_mean',
    'toplam_izleme_suresi_dk_lag1','ortalama_sure_lag7'
]

yeni_hedef_degisken = 'ortalama_sure'

df_arima = df.rename(columns={'event_date': 'ds', yeni_hedef_degisken: 'y'})
df_arima = df_arima[['ds', 'y'] + regressors].copy()

holiday_set = set(holidays_df['ds'].dt.normalize())
df_arima['holiday_dummy'] = df_arima['ds'].dt.normalize().isin(holiday_set).astype(int)

df_arima.dropna(inplace=True)

test_gun_sayisi = 40
train_df = df_arima.iloc[:-test_gun_sayisi].copy()
test_df  = df_arima.iloc[-test_gun_sayisi:].copy()

train_df = train_df.set_index('ds')
test_df  = test_df.set_index('ds')

scaler = StandardScaler()
exog_cols = regressors + ['holiday_dummy']

train_exog = scaler.fit_transform(train_df[exog_cols])
test_exog  = scaler.transform(test_df[exog_cols])

train_exog = pd.DataFrame(train_exog, index=train_df.index, columns=exog_cols)
test_exog  = pd.DataFrame(test_exog,  index=test_df.index,  columns=exog_cols)

candidate_p = [0,1,2,3]
candidate_d = [0,1]
candidate_q = [0,1,2,3]

best_order = None
best_aic = np.inf
y_train = train_df['y']

for p in candidate_p:
    for d in candidate_d:
        for q in candidate_q:
            try:
                model_tmp = SARIMAX(
                    y_train,
                    order=(p,d,q),
                    seasonal_order=(0,0,0,0),
                    exog=train_exog,
                    enforce_stationarity=False,
                    enforce_invertibility=False
                )
                res_tmp = model_tmp.fit(disp=False)
                if res_tmp.aic < best_aic:
                    best_aic = res_tmp.aic
                    best_order = (p,d,q)
            except Exception:
                continue


model = SARIMAX(
    y_train,
    order=best_order,
    seasonal_order=(0,0,0,0),
    exog=train_exog,
    enforce_stationarity=False,
    enforce_invertibility=False
)
model_fit = model.fit(disp=False)


fitted_in_sample = model_fit.fittedvalues
steps = len(test_df)
fc_res = model_fit.get_forecast(steps=steps, exog=test_exog)
yhat_oos = fc_res.predicted_mean

forecast = pd.DataFrame({
    'ds': pd.Index(train_df.index.tolist() + test_df.index.tolist(), name='ds'),
    'yhat': np.concatenate([fitted_in_sample.values, yhat_oos.values], axis=0)
}).set_index('ds')

y_true = test_df['y'].values
y_pred = forecast['yhat'].loc[test_df.index].values

mae = mean_absolute_error(y_true, y_pred)
ortalama_deger = pd.concat([train_df['y'], test_df['y']]).mean()
ortalama_yuzde_hata = (mae / ortalama_deger) * 100
dogruluk_orani = 100 - ortalama_yuzde_hata

print("- TAHMİN SONUÇLARI -")
print(f'-Ortalama Mutlak Hata (MAE): {mae:.2f} dakika')
print(f'-Ortalama Yüzdesel Hata: %{ortalama_yuzde_hata:.2f}')
print('-----------------------------------------')
print(f'-YAKLAŞIK DOĞRULIK ORANI: %{dogruluk_orani:.2f} ')

full_actual_data = pd.concat([train_df, test_df])

fig, ax = plt.subplots(figsize=(15, 7))
ax.plot(full_actual_data.index.to_pydatetime(), full_actual_data['y'], label='Gerçek Değerler', linewidth=2)
ax.plot(forecast.index.to_pydatetime(), forecast['yhat'], label='Tahmin Edilen Değerler', linestyle='--')

split_date = test_df.index[0]
ax.axvline(x=split_date, color='green', linestyle=':', linewidth=2, label='Eğitim/Test Ayrımı')

ax.set_title('Ortalama Süre | Gerçek ve ARIMA (SARIMAX) Tahminlerinin Karşılaştırılması')
ax.set_xlabel('Tarih')
ax.set_ylabel('Ortalama Süre (dk)')
ax.legend()
ax.grid(True)
fig.autofmt_xdate()
plt.tight_layout()
plt.show()
