import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import base64
from pathlib import Path
import sys, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)
from model_experiments.prophet.prophet_prediction import run_prophet_model

# Genel sayfa ayarları
st.set_page_config(
    page_title="Zaman Serisi Kullanıcı Etkileşimleri Tahmin Servisi",
    layout="wide"
)

# CSS stilleri
def inject_base_css():
    """
    Uygulamanın genel tipografi, padding, sidebar ve metric kart stillerini
    tanımlayan temel (light) tema CSS'i.
    """
    st.markdown("""
    <style>
    /* Genel layout */
    html, body, [data-testid="stAppViewContainer"] {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }

    #root .block-container{
        padding-top: 1.6rem;
        padding-left: 2.2rem;
        padding-right: 2.2rem;
    }

    /* Light tema sidebar */
    [data-testid="stSidebar"]{
        background:#f8fafc !important;
        border-right:1px solid #e2e8f0;
    }

    /* Üst header */
    .app-header{
        display:flex;
        align-items:center;
        justify-content:space-between;
        margin-bottom:1.1rem;
    }

    .app-title h2{
        margin:0;
        font-size:1.6rem;
        font-weight:700;
        color:#0f172a;
    }
    .app-title p{
        margin:4px 0 0 0;
        font-size:0.88rem;
        color:#64748b;
    }

    /* Metric kartları */
    .metric-card{
        background:#ffffff;
        border-radius:16px;
        padding:10px 14px;
        border:1px solid #e2f3ec;
        box-shadow:0 4px 14px rgba(0,0,0,0.03);
    }
    .metric-card .metric-label{
        color:#64748b;
        font-size:11px;
        text-transform:uppercase;
        letter-spacing:0.04em;
        display:flex;
        align-items:center;
        gap:6px;
    }
    .metric-card .metric-value{
        font-weight:700;
        font-size:22px;
        margin-top:6px;
        color:#0f172a;
    }

    .metric-icon{
        font-size:14px;
    }

    /* Logo */
    .logo-container{
        position:absolute;
        top:16px;
        right:24px;
        z-index:1000;
    }
    .logo-container img{
        width:200px;
    }
    @media (max-width: 1200px){
        .logo-container img{ width:160px; }
    }
    @media (max-width: 992px){
        .logo-container{ top:10px; right:14px; }
        .logo-container img{ width:130px; }
    }

    /* Plotly range slider */
    .js-plotly-plot .rangeslider-range,
    .js-plotly-plot .rangeslider-mask{
        background:rgba(0,146,95,0.10) !important;
    }
    </style>
    """, unsafe_allow_html=True)

# Koyu tema
def inject_dark_css():
    """
    Koyu (dark) tema seçildiğinde uygulanan arka plan, kart ve tablo stilleri.
    """
    st.markdown("""
    <style>
    /* Arka plan – modern koyu */
    html, body, [data-testid="stAppViewContainer"] {
        background: #0b1120;
        color: #e3e8ef;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: #020617 !important;
        border-right: 1px solid #111827;
    }

    /* Header */
    .app-title h2{
        color:#e5e7eb;
    }
    .app-title p{
        color:#9ca3af;
    }

    /* Metric Card Tasarımı */
    .metric-card {
        background: #020617;
        border: 1px solid #1f2937;
        border-radius: 16px;
        padding: 10px 14px;
        box-shadow: 0 6px 18px rgba(0,0,0,0.6);
    }

    .metric-card .metric-label {
        color: #9ca3af;
    }

    .metric-card .metric-value {
        color: #f9fafb;
    }

    /* Logo */
    .logo-container img {
        filter: drop-shadow(0 2px 4px rgba(0,0,0,.7));
    }

    /* Plotly range slider stili */
    .js-plotly-plot .rangeslider-range,
    .js-plotly-plot .rangeslider-mask {
        background: rgba(0,146,95,0.20) !important;
    }

    /* Veri tabloları */
    .stDataFrame { 
        background-color:#020617 !important;
        color:#e3e8ef !important;
        border-radius:8px;
    }
    </style>
    """, unsafe_allow_html=True)


inject_base_css()


# Logonun ekrana basılması
@st.cache_data
def get_image_as_base64(file: str):
    """
    Verilen dosya yolundaki resmi base64 formatına çevirir.
    Resim bulunamazsa None döner.
    """
    try:
        path = Path(file)
        with open(path, "rb") as f:
            data = f.read()
        return base64.b64encode(data).decode()
    except FileNotFoundError:
        return None


# Logo dosyasını base64 olarak oku ve sayfanın sağ üstüne yerleştirelim
logo_b64 = get_image_as_base64("figures/tabii_logo.png")
if logo_b64:
    st.markdown(
        f'<div class="logo-container"><img src="data:image/png;base64,{logo_b64}"></div>',
        unsafe_allow_html=True
    )


# Üst başlık
st.markdown("""
<div class="app-header">
  <div class="app-title">
    <h2>Zaman Serisi Kullanıcı Etkileşimleri Tahmin Servisi (Prophet)</h2>
    <p>Tabii platformu için günlük ortalama izleme süresi tahmini</p>
  </div>
</div>
""", unsafe_allow_html=True)



# Sidebar ayarları
st.sidebar.header("Ayarlar")

# Test dönemi uzunluğu – kullanıcı slider ile seçebiliyor
test_gun_sayisi = st.sidebar.slider("Test gün sayısı", 7, 60, 30, 1)

# Kullanıcı isterse CSV dosyalarını kendi yükleyebiliyor
use_uploads = st.sidebar.checkbox("CSV'leri yükleyerek kullan", value=False)

st.sidebar.subheader("Görselleştirme")

# Güven bandı göster/gizle secenegi
show_ci = st.sidebar.checkbox("Güven bandını göster", value=True)

# Lejant göster/gizle secenegi
show_legend = st.sidebar.checkbox("Lejandı göster", value=True)

# Hareketli ortalama (MA) ile yumuşatma penceresi
ma_window = st.sidebar.slider("Yumuşatma (MA pencere, gün)", 1, 14, 1, 1)

# Tema seçimi (açık / koyu)
st.sidebar.subheader("Görünüm")
tema = st.sidebar.selectbox("Tema", ["Ferah", "Koyu"], index=0)
if tema == "Koyu":
    inject_dark_css()
    plotly_template = "plotly_dark"
else:
    plotly_template = "plotly_white"


# Veri yükleme

if use_uploads:
    # Kullanıcıdan veri ve tatil günleri dosyalarını alıyoruz
    df_file = st.sidebar.file_uploader("daily_engagements_final.csv", type=["csv"])
    holidays_file = st.sidebar.file_uploader("prophet_holidays_tr.csv", type=["csv"])

    # Dosyalar gelmediyse uygulamayı durdurup ve bilgilendiriyoruz
    if df_file is None or holidays_file is None:
        st.info("Lütfen her iki CSV’yi de yükleyin veya yükleme seçeneğini kapatın.")
        st.stop()

    df = pd.read_csv(df_file)
    holidays_df = pd.read_csv(holidays_file)
else:
    # Lokal klasörden hazır CSV dosyalarını okumaya çalışıyor
    try:
        df = pd.read_csv("data/daily_engagements_final.csv")
        holidays_df = pd.read_csv("data/prophet_holidays_tr.csv")
    except FileNotFoundError:
        st.error("Lokal veri dosyaları bulunamadı. Lütfen 'CSV'leri yükleyerek kullan' seçeneğini aktif hale getirin.")
        st.stop()

# Tarih kolonlarını datetime tipine dönüştürüyoruz
df["event_date"] = pd.to_datetime(df["event_date"])
holidays_df["ds"] = pd.to_datetime(holidays_df["ds"])

# Prophet modeli kurulur (method import edilir)
try:
    forecast, metrics, train_df, test_df, df_prophet = run_prophet_model(
        df=df,
        holidays_df=holidays_df,
        test_gun_sayisi=test_gun_sayisi
        )
except ValueError as e:
    st.error(str(e))
    st.stop()

mae = metrics["mae"]
ortalama_yuzde_hata = metrics["mape"]
dogruluk_orani = metrics["accuracy"]

# Metrik kartları

c1, c2, c3, c4 = st.columns(4)
# MAE kartı
with c1:
    st.markdown(
        f'<div class="metric-card">'
        f'<div class="metric-label"><span class="metric-icon"></span>MAE (dk)</div>'
        f'<div class="metric-value">{mae:.2f}</div>'
        f'</div>',
        unsafe_allow_html=True
    )

# Ortalama yüzdesel hata kartı
with c2:
    st.markdown(
        f'<div class="metric-card">'
        f'<div class="metric-label"><span class="metric-icon"></span>Ortalama Yüzdesel Hata</div>'
        f'<div class="metric-value">%{ortalama_yuzde_hata:.2f}</div>'
        f'</div>',
        unsafe_allow_html=True
    )

# Yaklaşık doğruluk kartı
with c3:
    st.markdown(
        f'<div class="metric-card">'
        f'<div class="metric-label"><span class="metric-icon"></span>Yaklaşık Doğruluk</div>'
        f'<div class="metric-value">%{dogruluk_orani:.2f}</div>'
        f'</div>',
        unsafe_allow_html=True
    )

# Seçilen test gün sayısı kartı
with c4:
    st.markdown(
        f'<div class="metric-card">'
        f'<div class="metric-label"><span class="metric-icon"></span>Test Gün Sayısı</div>'
        f'<div class="metric-value">{test_gun_sayisi}</div>'
        f'</div>',
        unsafe_allow_html=True
    )

# Plotly ile zaman serisi grafiği çizdirme kısmı (gerçek+tahmin)
# Grafik için full (train+test) gerçek seri
full_actual_data = pd.concat([train_df, test_df], ignore_index=True)

# Eğitim/test ayrımının olduğu tarih
split_date_dt = pd.to_datetime(test_df["ds"].iloc[0])

# MA penceresi > 1 ise hareketli ortalama ile yumuşatma uygulanıyor
if ma_window > 1:
    plot_actual_y = full_actual_data["y"].rolling(ma_window, min_periods=1).mean()
    plot_pred_y = forecast["yhat"].rolling(ma_window, min_periods=1).mean()
else:
    plot_actual_y = full_actual_data["y"]
    plot_pred_y = forecast["yhat"]

fig = go.Figure()

tabii_green = "#00925F"
ci_color = "rgba(0,146,95,0.18)"

# Güven bandı (Prophet'in yhat_lower / yhat_upper'ı)
if show_ci and "yhat_lower" in forecast.columns and "yhat_upper" in forecast.columns:
    # Üst sınır
    fig.add_trace(go.Scatter(
        x=forecast["ds"],
        y=forecast["yhat_upper"],
        mode="lines",
        line=dict(width=0),
        showlegend=False,
        hoverinfo="skip"
    ))
    # Alt sınır + alan doldurma
    fig.add_trace(go.Scatter(
        x=forecast["ds"],
        y=forecast["yhat_lower"],
        mode="lines",
        line=dict(width=0),
        fill="tonexty",
        fillcolor=ci_color,
        name="Güven Bandı",
        hovertemplate="Tarih: %{x|%d %b %Y}<br>Alt-Üst: %{y:.2f} - %{customdata:.2f}",
        customdata=forecast["yhat_upper"]
    ))

# Tahmin serisi
fig.add_trace(go.Scatter(
    x=forecast["ds"],
    y=plot_pred_y,
    name="Tahmin (yhat)",
    line=dict(dash="dash", color=tabii_green, width=2),
    hovertemplate="Tarih: %{x|%d %b %Y}<br>Tahmin: %{y:.2f} dk"
))

# Gerçek seri
fig.add_trace(go.Scatter(
    x=full_actual_data["ds"],
    y=plot_actual_y,
    name="Gerçek",
    line=dict(color="lightslategray", width=2),
    hovertemplate="Tarih: %{x|%d %b %Y}<br>Gerçek: %{y:.2f} dk"
))

# Eğitim/test ayrım çizgisi
fig.add_shape(
    type="line",
    x0=split_date_dt, x1=split_date_dt, y0=0, y1=1,
    xref="x", yref="paper",
    line=dict(color="#065f46", width=2, dash="dot")
)

# Ayrım annotation
fig.add_annotation(
    x=split_date_dt, y=1,
    xref="x", yref="paper",
    text="Eğitim/Test",
    showarrow=False, yshift=10,
    font=dict(color="#065f46")
)

# Genel grafik ayarları
fig.update_layout(
    height=460,
    template=plotly_template,
    title="Ortalama Süre | Gerçek ve Tahmin",
    xaxis_title="Tarih",
    yaxis_title="Ortalama Süre (dk)",
    legend=dict(
        orientation="h", yanchor="bottom", y=1.02,
        xanchor="right", x=1,
        bgcolor="rgba(255,255,255,0.6)" if tema == "Ferah" else "rgba(15,23,42,0.7)"
    ),
    margin=dict(l=40, r=30, t=50, b=20),
    showlegend=show_legend
)

fig.update_xaxes(
    rangeslider=dict(
        visible=True,
        bgcolor="rgba(0,146,95,0.10)",
        bordercolor="rgba(0,146,95,0.35)",
        thickness=0.12
    )
)

st.plotly_chart(fig, use_container_width=True)

# Veri ve tahmin önizlemeleri
with st.expander("Veri önizlemeleri"):
    st.markdown("<p style='font-size:16px; font-weight:bold;'>Eğitim Verileri (Train)</p>", unsafe_allow_html=True)
    st.dataframe(train_df.head(), use_container_width=True)

    st.markdown("<p style='font-size:16px; font-weight:bold;'>Test Verileri (Test)</p>", unsafe_allow_html=True)
    st.dataframe(test_df.head(), use_container_width=True)

    st.markdown("<p style='font-size:16px; font-weight:bold;'>Tahmin Sonuçları (Forecast)</p>", unsafe_allow_html=True)
    st.dataframe(
        forecast.tail(test_gun_sayisi)[["ds", "yhat", "yhat_lower", "yhat_upper"]],
        use_container_width=True
    )

# Tatil özel gün önizlemeleri
with st.expander("Tatil günleri önizleme"):
    cols_to_show = [c for c in ["ds", "holiday"] if c in holidays_df.columns]
    st.dataframe(holidays_df[cols_to_show].sort_values("ds").head(10), use_container_width=True)
