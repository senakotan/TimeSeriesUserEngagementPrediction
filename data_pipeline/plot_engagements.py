import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import matplotlib.dates as mdates


def _fmt_mn(x, _pos):
    return f"{x/1e6:.0f} Mn" if abs(x) >= 1e6 else f"{x:.0f}"


def plot_dual_axis(df, start_date, end_date):
    df = df[
        (df["event_date"] >= pd.to_datetime(start_date)) &
        (df["event_date"] <= pd.to_datetime(end_date))
    ].copy()

    fig, ax1 = plt.subplots(figsize=(14, 6), dpi=140)

    line1, = ax1.plot(
        df["event_date"],
        df["toplam_izleme_suresi_dk"],
        label="Toplam İzleme Süresi (dk)",
        color="tab:blue",
        linewidth=1.6,
    )
    ax1.set_ylabel("Toplam İzleme Süresi (Mn dk)", color="tab:blue")
    ax1.yaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v/1e6:.0f} Mn"))

    ax2 = ax1.twinx()
    line2, = ax2.plot(
        df["event_date"],
        df["oturum_sayisi"],
        label="Oturum Sayısı",
        color="tab:orange",
        alpha=0.85,
        linewidth=1.2,
    )
    ax2.set_ylabel("Oturum Sayısı (Mn)", color="tab:orange")
    ax2.yaxis.set_major_formatter(FuncFormatter(_fmt_mn))

    locator = mdates.AutoDateLocator(minticks=8, maxticks=14)
    ax1.xaxis.set_major_locator(locator)
    ax1.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    ax1.set_xlim(pd.to_datetime(start_date), pd.to_datetime(end_date))

    ax1.grid(True, which="major", linestyle="--", alpha=0.35)
    plt.title("Tarih Bazlı İzleme Süresi ve Oturum Sayısı")

    lines = [line1, line2]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper right")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    df = pd.read_csv(
        "data/daily_engagements.csv",
        parse_dates=["event_date"]
    )

    plot_dual_axis(df, start_date="2024-08-01", end_date="2025-08-17")
