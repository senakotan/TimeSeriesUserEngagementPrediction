import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.metrics import mean_absolute_error
import plotly.graph_objects as go
import base64
from pathlib import Path

st.set_page_config(page_title="Prophet Forecast Dashboard", layout="wide")

def inject_base_css():
    st.markdown("""
    <style>
    #root .block-container{ padding-top: 2.2rem; }
    .metric-card{
        background:#ffffff; border:1px solid #e6f2ec; border-radius:14px; padding:14px 16px;
        box-shadow:0 1px 8px rgba(46,139,87,0.05);
    }
    .metric-card .metric-label{ color:#2b2b2b; font-size:12px; }
    .metric-card .metric-value{ font-weight:700; }
    .logo-container{ position:absolute; top:14px; right:24px; z-index:1000; }
    .logo-container img{ width:200px; }
    @media (max-width: 1200px){ .logo-container img{ width:160px; } }
    @media (max-width: 992px){ .logo-container{ top:10px; right:14px; } .logo-container img{ width:130px; } }
    .js-plotly-plot .rangeslider-range,
    .js-plotly-plot .rangeslider-mask{ background:rgba(46,139,87,0.08) !important; }
    </style>
    """, unsafe_allow_html=True)

def inject_dark_css():
    st.markdown("""
    <style>
    html, body, [data-testid="stAppViewContainer"]{
        background:#0b1220; color:#e6edf3;
    }
    .metric-card{
        background:#0f172a; border:1px solid #1f2a44; border-radius:14px; padding:14px 16px;
        box-shadow:0 1px 10px rgba(0,0,0,0.20);
    }
    .metric-card .metric-label{ color:#c9d1d9; font-size:12px; }
    .metric-card .metric-value{ color:#ffffff; font-weight:700; }
    .logo-container{ position:absolute; top:14px; right:24px; z-index:1000; }
    .logo-container img{ filter: drop-shadow(0 1px 2px rgba(0,0,0,.35)); }
    .js-plotly-plot .rangeslider-range,
    .js-plotly-plot .rangeslider-mask{ background:rgba(46,139,87,0.20) !important; }
    </style>
    """, unsafe_allow_html=True)

inject_base_css()

@st.cache_data
def get_image_as_base64(file: str):
    try:
        path = Path(file)
        with open(path, "rb") as f:
            data = f.read()
        return base64.b64encode(data).decode()
    except FileNotFoundError:
        return None

logo_b64 = get_image_as_base64("tabii_logo.png")
if logo_b64:
    st.markdown(f'<div class="logo-container"><img src="data:image/png;base64,{logo_b64}"></div>', unsafe_allow_html=True)

st.markdown("<h3>Zaman Serisi Kullanıcı Etkileşimleri Tahmin Servisi (Prophet)</h3>", unsafe_allow_html=True)

st.sidebar.header("Ayarlar")
test_gun_sayisi = st.sidebar.slider("Test gün sayısı", 7, 60, 30, 1)
use_uploads = st.sidebar.checkbox("CSV'leri yükleyerek kullan", value=False)

st.sidebar.subheader("Görselleştirme")
show_ci = st.sidebar.checkbox("Güven bandını göster (95% CI)", value=True)
show_legend = st.sidebar.checkbox("Lejandı göster", value=True)
ma_window = st.sidebar.slider("Yumuşatma (MA pencere, gün)", 1, 14, 1, 1)

st.sidebar.subheader("Görünüm")
tema = st.sidebar.selectbox("Tema", ["Ferah", "Koyu"], index=0)
if tema == "Koyu":
    inject_dark_css()
    plotly_template = "plotly_dark"
else:
    plotly_template = "plotly_white"

if use_uploads:
    df_file = st.sidebar.file_uploader("daily_engagements_final.csv", type=["csv"])
    holidays_file = st.sidebar.file_uploader("prophet_holidays_tr.csv", type=["csv"])
    if df_file is None or holidays_file is None:
        st.info("Lütfen her iki CSV’yi de yükleyin veya yükleme seçeneğini kapatın.")
        st.stop()
    df = pd.read_csv(df_file)
    holidays_df = pd.read_csv(holidays_file)
else:
    try:
        df = pd.read_csv("data/daily_engagements_final.csv")
        holidays_df = pd.read_csv("data/prophet_holidays_tr.csv")
    except FileNotFoundError:
        st.error("Lokal veri dosyaları bulunamadı. Lütfen 'CSV'leri yükleyerek kullan' seçeneğini aktif hale getirin.")
        st.stop()

df["event_date"] = pd.to_datetime(df["event_date"])
holidays_df["ds"] = pd.to_datetime(holidays_df["ds"])

regressors = [
    "ortalama_sure_lag1", "ortalama_sure_roll7_mean", "toplam_izleme_suresi_dk_roll14_std",
    "ortalama_sure_roll14_mean", "month_sin", "toplam_izleme_suresi_dk_lag1", "ortalama_sure_lag7"
]
target = "ortalama_sure"

df_prophet = df.rename(columns={"event_date":"ds", target:"y"})[["ds","y"]+regressors].copy()
df_prophet.dropna(inplace=True)

if len(df_prophet) <= test_gun_sayisi:
    st.error("Veri satırı test gün sayısından az/eşit. Lütfen test gün sayısını düşürün.")
    st.stop()

train_df = df_prophet.iloc[:-test_gun_sayisi].copy()
test_df  = df_prophet.iloc[-test_gun_sayisi:].copy()

model = Prophet(holidays=holidays_df, yearly_seasonality=False, weekly_seasonality=False)
for r in regressors:
    model.add_regressor(r)

with st.spinner("Model eğitiliyor..."):
    model.fit(train_df)

future_df = pd.concat([train_df, test_df], ignore_index=True)[["ds"] + regressors]
forecast = model.predict(future_df)

y_true = test_df["y"].values
y_pred = forecast["yhat"].iloc[-test_gun_sayisi:].values
mae = mean_absolute_error(y_true, y_pred)
ortalama_deger = df_prophet["y"].mean()
ortalama_yuzde_hata = (mae / ortalama_deger) * 100 if ortalama_deger != 0 else np.nan
dogruluk_orani = 100 - ortalama_yuzde_hata if not np.isnan(ortalama_yuzde_hata) else np.nan

c1, c2, c3, c4 = st.columns(4)
with c1: st.markdown(f'<div class="metric-card"><div class="metric-label">MAE (dk)</div><div class="metric-value" style="font-size:22px">{mae:.2f}</div></div>', unsafe_allow_html=True)
with c2: st.markdown(f'<div class="metric-card"><div class="metric-label">Ortalama Yüzdesel Hata</div><div class="metric-value" style="font-size:22px">%{ortalama_yuzde_hata:.2f}</div></div>', unsafe_allow_html=True)
with c3: st.markdown(f'<div class="metric-card"><div class="metric-label">Yaklaşık Doğruluk</div><div class="metric-value" style="font-size:22px">%{dogruluk_orani:.2f}</div></div>', unsafe_allow_html=True)
with c4: st.markdown(f'<div class="metric-card"><div class="metric-label">Test Gün Sayısı</div><div class="metric-value" style="font-size:22px">{test_gun_sayisi}</div></div>', unsafe_allow_html=True)

full_actual_data = pd.concat([train_df, test_df], ignore_index=True)
split_date_dt = pd.to_datetime(test_df["ds"].iloc[0])

plot_actual_y = full_actual_data["y"].rolling(ma_window, min_periods=1).mean() if ma_window > 1 else full_actual_data["y"]
plot_pred_y   = forecast["yhat"].rolling(ma_window, min_periods=1).mean() if ma_window > 1 else forecast["yhat"]

fig = go.Figure()

if show_ci:
    fig.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat_upper"], mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(
        x=forecast["ds"], y=forecast["yhat_lower"], mode="lines", line=dict(width=0),
        fill="tonexty", fillcolor="rgba(46,139,87,0.15)", name="95% CI",
        hovertemplate="Tarih: %{x|%d %b %Y}<br>Alt-Üst: %{y:.2f}-%{customdata:.2f}",
        customdata=forecast["yhat_upper"]
    ))

fig.add_trace(go.Scatter(
    x=forecast["ds"], y=plot_pred_y, name="Tahmin (yhat)",
    line=dict(dash="dash", color="seagreen"),
    hovertemplate="Tarih: %{x|%d %b %Y}<br>Tahmin: %{y:.2f} dk"
))
fig.add_trace(go.Scatter(
    x=full_actual_data["ds"], y=plot_actual_y, name="Gerçek",
    line=dict(color="lightslategray"),
    hovertemplate="Tarih: %{x|%d %b %Y}<br>Gerçek: %{y:.2f} dk"
))

fig.add_shape(type="line", x0=split_date_dt, x1=split_date_dt, y0=0, y1=1, xref="x", yref="paper",
              line=dict(color="darkgreen", width=2, dash="dot"))
fig.add_annotation(x=split_date_dt, y=1, xref="x", yref="paper", text="Eğitim/Test",
                   showarrow=False, yshift=10, font=dict(color="darkgreen"))

fig.update_layout(
    height=460, template=plotly_template, title="Ortalama Süre | Gerçek ve Tahmin",
    xaxis_title="Tarih", yaxis_title="Ortalama Süre (dk)",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                bgcolor="rgba(255,255,255,0.6)" if tema=="Ferah" else "rgba(15,23,42,0.6)"),
    margin=dict(l=40, r=30, t=50, b=20), showlegend=show_legend
)
fig.update_xaxes(
    rangeslider=dict(visible=True, bgcolor="rgba(46,139,87,0.08)", bordercolor="rgba(46,139,87,0.35)", thickness=0.12)
)
st.plotly_chart(fig, use_container_width=True)

with st.expander("Veri önizlemeleri"):
    st.markdown("<p style='font-size:16px; font-weight:bold;'> Eğitim Verileri (Train)</p>", unsafe_allow_html=True)
    st.dataframe(train_df.head(), use_container_width=True)
    st.markdown("<p style='font-size:16px; font-weight:bold;'> Test Verileri (Test)</p>", unsafe_allow_html=True)
    st.dataframe(test_df.head(), use_container_width=True)
    st.markdown("<p style='font-size:16px; font-weight:bold;'> Tahmin Sonuçları (Forecast)</p>", unsafe_allow_html=True)
    st.dataframe(
        forecast.tail(test_gun_sayisi)[["ds","yhat","yhat_lower","yhat_upper"]],
        use_container_width=True
    )

with st.expander("Tatil günleri önizleme"):
    cols_to_show = [c for c in ["ds", "holiday"] if c in holidays_df.columns]
    st.dataframe(holidays_df[cols_to_show].sort_values("ds").head(10), use_container_width=True)
