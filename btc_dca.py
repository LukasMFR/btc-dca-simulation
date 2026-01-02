#!/usr/bin/env python3
import os
import sys
import math
import json
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# =========================
# CONFIG (modifie ici)
# =========================
START_DATE = "2020-12-30"
END_DATE   = "2025-12-30"
DAILY_EUR  = 10.0
FEE_RATE   = 0.0   # ex: 0.002 = 0.2% par achat, 0.0 = sans frais
OUT_DIR    = "out"
DATA_DIR   = "data"
# Source CSV (BTC/EUR daily OHLC)
STOOQ_CSV_URL = "https://stooq.com/q/d/l/?s=btceur&i=d"
# =========================


def ensure_dirs():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)

def download_csv(url: str, dest_path: str) -> None:
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    with open(dest_path, "wb") as f:
        f.write(r.content)

def load_prices_stooq(csv_path: str) -> pd.Series:
    """
    Stooq CSV columns: Date, Open, High, Low, Close
    We use Close as daily price in EUR.
    """
    df = pd.read_csv(csv_path)
    if "Date" not in df.columns or "Close" not in df.columns:
        raise ValueError(f"CSV inattendu. Colonnes trouvées: {list(df.columns)}")
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").set_index("Date")
    # Build daily calendar and forward-fill (weekends/holidays)
    full = pd.date_range(df.index.min(), df.index.max(), freq="D")
    close = df["Close"].reindex(full).ffill()
    close.name = "Close_EUR"
    return close

def xirr(dates, cashflows) -> float:
    """
    XIRR annualisé (TRI) avec bisection + recherche de bracket.
    dates: array-like datetime
    cashflows: array-like floats
    """
    dates = pd.to_datetime(dates)
    cfs = np.array(cashflows, dtype=float)
    t0 = dates.min()
    years = (dates - t0).days.values / 365.0

    def npv(r):
        return np.sum(cfs / np.power(1.0 + r, years))

    # Try to find an interval [low, high] with sign change
    grid = np.concatenate([
        np.linspace(-0.99, -0.5, 200),
        np.linspace(-0.5, 0.0, 200),
        np.linspace(0.0, 5.0, 800),
        np.linspace(5.0, 50.0, 200),
    ])
    vals = np.array([npv(r) for r in grid])

    idx = np.where(np.sign(vals[:-1]) * np.sign(vals[1:]) < 0)[0]
    if len(idx) == 0:
        raise ValueError("Impossible de calculer le TRI (pas de changement de signe NPV).")
    i = idx[0]
    low, high = grid[i], grid[i+1]
    f_low, f_high = vals[i], vals[i+1]

    # Bisection
    for _ in range(200):
        mid = (low + high) / 2
        f_mid = npv(mid)
        if abs(f_mid) < 1e-10:
            return mid
        if f_low * f_mid < 0:
            high, f_high = mid, f_mid
        else:
            low, f_low = mid, f_mid
    return (low + high) / 2

def simulate_dca(prices: pd.Series, start: str, end: str, daily_eur: float, fee_rate: float):
    idx = pd.date_range(pd.to_datetime(start), pd.to_datetime(end), freq="D")
    p = prices.reindex(idx).ffill()

    net = daily_eur * (1.0 - fee_rate)
    btc_bought = (net / p).fillna(0.0)
    btc_total = btc_bought.cumsum()

    invested = daily_eur * np.arange(1, len(idx)+1)
    value = btc_total.values * p.values

    df = pd.DataFrame({
        "price_eur": p.values,
        "btc": btc_total.values,
        "invested_eur": invested,
        "value_eur": value
    }, index=idx)

    # Drawdown
    peak = df["value_eur"].cummax()
    dd = (df["value_eur"] / peak) - 1.0
    max_dd = float(dd.min())

    # TRI (cashflows: -daily each day, + liquidation at end)
    cfs = np.full(len(idx), -daily_eur, dtype=float)
    cfs[-1] += df["value_eur"].iloc[-1]
    tri = float(xirr(idx, cfs))

    stats = {
        "period_start": str(idx[0].date()),
        "period_end": str(idx[-1].date()),
        "days": int(len(idx)),
        "daily_eur": float(daily_eur),
        "fee_rate": float(fee_rate),
        "invested_eur": float(df["invested_eur"].iloc[-1]),
        "final_value_eur": float(df["value_eur"].iloc[-1]),
        "profit_eur": float(df["value_eur"].iloc[-1] - df["invested_eur"].iloc[-1]),
        "multiple": float(df["value_eur"].iloc[-1] / df["invested_eur"].iloc[-1]),
        "tri_annualized": tri,
        "max_drawdown": max_dd
    }
    return df, stats

def save_outputs(sim_df: pd.DataFrame, stats: dict):
    ensure_dirs()
    # Save stats JSON
    with open(os.path.join(OUT_DIR, "stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    # Save simulation CSV
    sim_df.to_csv(os.path.join(OUT_DIR, "simulation.csv"), index_label="date")

    # Graph 1: Price
    plt.figure(figsize=(10,4))
    plt.plot(sim_df.index, sim_df["price_eur"])
    plt.title("BTC/EUR — prix (close) quotidien")
    plt.xlabel("Date"); plt.ylabel("€")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "01_price.png"), dpi=160)
    plt.close()

    # Graph 2: Value vs Invested
    plt.figure(figsize=(10,4))
    plt.plot(sim_df.index, sim_df["invested_eur"], label="Total investi")
    plt.plot(sim_df.index, sim_df["value_eur"], label="Valeur portefeuille (DCA)")
    plt.title("DCA BTC — valeur vs investi")
    plt.xlabel("Date"); plt.ylabel("€")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "02_value_vs_invested.png"), dpi=160)
    plt.close()

    # Graph 3: Drawdown
    peak = sim_df["value_eur"].cummax()
    dd = (sim_df["value_eur"] / peak) - 1.0
    plt.figure(figsize=(10,4))
    plt.plot(sim_df.index, dd)
    plt.title("DCA BTC — drawdown (recul depuis le plus haut)")
    plt.xlabel("Date"); plt.ylabel("Drawdown")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "03_drawdown.png"), dpi=160)
    plt.close()

def main():
    ensure_dirs()
    csv_path = os.path.join(DATA_DIR, "btceur_stooq_daily.csv")

    # Download once (or redownload if you want fresh data)
    if not os.path.exists(csv_path):
        print(f"⬇️ Téléchargement du CSV BTC/EUR depuis Stooq…\n{STOOQ_CSV_URL}")
        download_csv(STOOQ_CSV_URL, csv_path)
        print(f"✅ Sauvé: {csv_path}")
    else:
        print(f"✅ CSV déjà présent: {csv_path}")

    prices = load_prices_stooq(csv_path)

    sim_df, stats = simulate_dca(
        prices=prices,
        start=START_DATE,
        end=END_DATE,
        daily_eur=DAILY_EUR,
        fee_rate=FEE_RATE
    )

    save_outputs(sim_df, stats)

    # Pretty print
    print("\n===== RÉSULTATS DCA BTC/EUR =====")
    print(f"Période           : {stats['period_start']} → {stats['period_end']} ({stats['days']} jours)")
    print(f"DCA               : {stats['daily_eur']} €/jour   (frais: {stats['fee_rate']*100:.2f}%)")
    print(f"Total investi      : {stats['invested_eur']:.0f} €")
    print(f"Valeur finale      : {stats['final_value_eur']:.0f} €")
    print(f"Gain               : {stats['profit_eur']:.0f} €")
    print(f"Multiple           : x{stats['multiple']:.2f}")
    print(f"TRI (annualisé)    : {stats['tri_annualized']*100:.1f} % / an")
    print(f"Pire drawdown      : {stats['max_drawdown']*100:.1f} %")
    print("\n📁 Fichiers générés dans ./out :")
    print(" - stats.json")
    print(" - simulation.csv")
    print(" - 01_price.png")
    print(" - 02_value_vs_invested.png")
    print(" - 03_drawdown.png")

if __name__ == "__main__":
    main()
