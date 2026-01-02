#!/usr/bin/env python3
import os
import json
import math
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, date

# ========= SOURCE DATA =========
STOOQ_CSV_URL = "https://stooq.com/q/d/l/?s=btceur&i=d"  # BTC/EUR daily OHLC CSV
DATA_DIR = "data"
CACHE_CSV = os.path.join(DATA_DIR, "btceur_stooq_daily.csv")

# ========= OUTPUT =========
OUT_ROOT = "out"

# ========= DEFAULTS =========
DEFAULT_DCA_EUR = 10.0
DEFAULT_FEE_PCT = 0.0  # percent (e.g. 0.2 for 0.2%)
DEFAULT_PERIOD_CHOICE = "5"  # 1y/3y/5y/YTD/X/range -> default to 5y
DEFAULT_FREQ = "D"  # D/W/M
DEFAULT_EVERY = 1   # every N units

# ========= French date formatting (no locale dependency) =========
FR_MONTHS = ["janv.", "févr.", "mars", "avr.", "mai", "juin",
             "juil.", "août", "sept.", "oct.", "nov.", "déc."]

def fr_date(d: date) -> str:
    return f"{d.day:02d} {FR_MONTHS[d.month-1]} {d.year}"

def iso(d: date) -> str:
    return d.strftime("%Y-%m-%d")

def now_ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def ensure_dirs():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(OUT_ROOT, exist_ok=True)

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
        raise ValueError(f"CSV inattendu. Colonnes: {list(df.columns)}")
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").set_index("Date")

    # Create a full daily calendar and forward-fill missing (weekends/holidays)
    full = pd.date_range(df.index.min(), df.index.max(), freq="D")
    close = df["Close"].reindex(full).ffill()
    close.name = "Close_EUR"
    return close

# ---------- XIRR (TRI annualisé) ----------
def xirr(dates, cashflows) -> float:
    """
    XIRR annualisé (TRI) via bisection, avec recherche de bracket.
    dates: array-like datetime
    cashflows: array-like float
    """
    dates = pd.to_datetime(dates)
    cfs = np.array(cashflows, dtype=float)

    t0 = dates.min()
    years = (dates - t0).days.values / 365.0

    def npv(r):
        return np.sum(cfs / np.power(1.0 + r, years))

    # Grid search to find sign change
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

# ---------- Date selection (CLI + optional GUI) ----------
def try_gui_date_range(default_start: date, default_end: date):
    """
    Optional GUI date picker using tkcalendar if installed.
    Returns (start_date, end_date) as date objects, or None if not available/cancel.
    """
    try:
        import tkinter as tk
        from tkcalendar import DateEntry
    except Exception:
        return None

    result = {"ok": False, "start": default_start, "end": default_end}

    def on_ok():
        try:
            s = start_cal.get_date()
            e = end_cal.get_date()
            if s > e:
                msg.config(text="⚠️ La date de début doit être <= date de fin.")
                return
            result["start"] = s
            result["end"] = e
            result["ok"] = True
            root.destroy()
        except Exception as ex:
            msg.config(text=f"Erreur: {ex}")

    def on_cancel():
        root.destroy()

    root = tk.Tk()
    root.title("Sélection de dates (DCA BTC)")
    root.geometry("420x180")
    root.resizable(False, False)

    tk.Label(root, text="Date de début").grid(row=0, column=0, padx=10, pady=10, sticky="w")
    start_cal = DateEntry(root, width=16, date_pattern="yyyy-mm-dd")
    start_cal.set_date(default_start)
    start_cal.grid(row=0, column=1, padx=10, pady=10)

    tk.Label(root, text="Date de fin").grid(row=1, column=0, padx=10, pady=10, sticky="w")
    end_cal = DateEntry(root, width=16, date_pattern="yyyy-mm-dd")
    end_cal.set_date(default_end)
    end_cal.grid(row=1, column=1, padx=10, pady=10)

    msg = tk.Label(root, text="", fg="red")
    msg.grid(row=2, column=0, columnspan=2, padx=10)

    btns = tk.Frame(root)
    btns.grid(row=3, column=0, columnspan=2, pady=10)

    tk.Button(btns, text="OK", width=10, command=on_ok).pack(side="left", padx=10)
    tk.Button(btns, text="Annuler", width=10, command=on_cancel).pack(side="left", padx=10)

    root.mainloop()

    if result["ok"]:
        return result["start"], result["end"]
    return None

def input_float(prompt: str, default: float = None, allow_empty_default=True) -> float:
    while True:
        raw = input(prompt).strip().replace(",", ".")
        if raw == "" and default is not None and allow_empty_default:
            return default
        try:
            return float(raw)
        except:
            print("❌ Entrée invalide. Exemple: 10 ou 10.5")

def input_int(prompt: str, default: int = None) -> int:
    while True:
        raw = input(prompt).strip()
        if raw == "" and default is not None:
            return default
        try:
            return int(raw)
        except:
            print("❌ Entier invalide.")

def input_date_cli(prompt: str, default: date = None) -> date:
    while True:
        raw = input(prompt).strip()
        if raw == "" and default is not None:
            return default
        try:
            return datetime.strptime(raw, "%Y-%m-%d").date()
        except:
            print("❌ Format attendu: YYYY-MM-DD")

def choose_period(today: date):
    """
    Returns (start_date, end_date) as date objects.
    """
    print("\n=== Choix de la période ===")
    print("1) 1 an")
    print("3) 3 ans")
    print("5) 5 ans")
    print("Y) YTD (depuis le 1er janvier)")
    print("X) X années (tu choisis)")
    print("R) Range custom (dates)")
    print("G) Range custom avec GUI (si dispo)")

    choice = input(f"Choix [défaut {DEFAULT_PERIOD_CHOICE}]: ").strip().upper()
    if choice == "":
        choice = DEFAULT_PERIOD_CHOICE.upper()

    end = today

    if choice == "1":
        start = (pd.Timestamp(end) - pd.DateOffset(years=1)).date()
        return start, end
    if choice == "3":
        start = (pd.Timestamp(end) - pd.DateOffset(years=3)).date()
        return start, end
    if choice == "5":
        start = (pd.Timestamp(end) - pd.DateOffset(years=5)).date()
        return start, end
    if choice == "Y":
        start = date(end.year, 1, 1)
        return start, end
    if choice == "X":
        x = input_int("Combien d'années ? (ex: 7) : ", default=5)
        start = (pd.Timestamp(end) - pd.DateOffset(years=x)).date()
        return start, end
    if choice == "G":
        # default range = 5y
        default_start = (pd.Timestamp(end) - pd.DateOffset(years=5)).date()
        picked = try_gui_date_range(default_start, end)
        if picked is not None:
            return picked[0], picked[1]
        print("⚠️ GUI non disponible (ou annulé). On passe en mode texte.")
        choice = "R"  # fallthrough to CLI range
    if choice == "R":
        default_start = (pd.Timestamp(end) - pd.DateOffset(years=5)).date()
        print(f"Dates au format YYYY-MM-DD (ex: 2020-12-30).")
        start = input_date_cli(f"Date de début [défaut {iso(default_start)}] : ", default=default_start)
        end2  = input_date_cli(f"Date de fin   [défaut {iso(end)}] : ", default=end)
        if start > end2:
            print("⚠️ Début > fin, j'inverse automatiquement.")
            start, end2 = end2, start
        return start, end2

    print("⚠️ Choix inconnu → 5 ans par défaut.")
    start = (pd.Timestamp(end) - pd.DateOffset(years=5)).date()
    return start, end

def choose_dca_params():
    print("\n=== Paramètres DCA ===")
    amount = input_float(f"Montant par achat en € [défaut {DEFAULT_DCA_EUR}] : ", default=DEFAULT_DCA_EUR)

    print("\nFréquence :")
    print("D) Journalier")
    print("W) Hebdomadaire")
    print("M) Mensuel")
    freq = input(f"Choix fréquence [défaut {DEFAULT_FREQ}] : ").strip().upper()
    if freq == "":
        freq = DEFAULT_FREQ
    if freq not in ("D","W","M"):
        print("⚠️ Fréquence invalide → D (journalier).")
        freq = "D"

    every = input_int(f"Toutes les combien d'unités ? [défaut {DEFAULT_EVERY}] : ", default=DEFAULT_EVERY)
    if every <= 0:
        every = 1

    fee_pct = input_float(f"Frais (%) par achat [Entrée = {DEFAULT_FEE_PCT}] : ", default=DEFAULT_FEE_PCT)
    fee_rate = max(0.0, fee_pct) / 100.0

    return amount, freq, every, fee_rate

def build_buy_dates(start: date, end: date, freq: str, every: int) -> pd.DatetimeIndex:
    """
    Returns a DatetimeIndex of buy dates from start to end, inclusive.
    freq: D/W/M, every: N
    Strategy:
    - D: every N days
    - W: every N weeks (from start date)
    - M: every N months (same day-of-month as start when possible)
    """
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)

    if freq == "D":
        return pd.date_range(start_ts, end_ts, freq=pd.DateOffset(days=every))
    if freq == "W":
        return pd.date_range(start_ts, end_ts, freq=pd.DateOffset(weeks=every))
    if freq == "M":
        return pd.date_range(start_ts, end_ts, freq=pd.DateOffset(months=every))

    # fallback daily
    return pd.date_range(start_ts, end_ts, freq="D")

def simulate_dca(prices: pd.Series, start: date, end: date, amount_eur: float, freq: str, every: int, fee_rate: float):
    # daily index (for valuation)
    idx = pd.date_range(pd.Timestamp(start), pd.Timestamp(end), freq="D")
    p = prices.reindex(idx).ffill()

    buy_dates = build_buy_dates(start, end, freq, every)
    buy_dates = buy_dates[(buy_dates >= idx[0]) & (buy_dates <= idx[-1])]

    buy_prices = p.reindex(buy_dates).ffill()

    net = amount_eur * (1.0 - fee_rate)
    btc_bought = (net / buy_prices).fillna(0.0)

    # align buys on daily index
    btc_buys_daily = pd.Series(0.0, index=idx)
    btc_buys_daily.loc[btc_bought.index] = btc_bought.values

    btc = btc_buys_daily.cumsum()
    invested_daily = pd.Series(0.0, index=idx)
    invested_daily.loc[btc_bought.index] = amount_eur
    invested = invested_daily.cumsum()

    value = btc.values * p.values

    df = pd.DataFrame({
        "price_eur": p.values,
        "btc": btc.values,
        "invested_eur": invested.values,
        "value_eur": value
    }, index=idx)

    # drawdown
    peak = df["value_eur"].cummax()
    dd = (df["value_eur"] / peak) - 1.0
    max_dd = float(dd.min())

    # XIRR cashflows: -amount on each buy date, + liquidation at end date
    cfs_dates = list(btc_bought.index.to_pydatetime())
    cfs = list((-amount_eur) * np.ones(len(cfs_dates)))
    # add final liquidation on end date
    cfs_dates.append(pd.Timestamp(end).to_pydatetime())
    cfs.append(float(df["value_eur"].iloc[-1]))
    tri = float(xirr(cfs_dates, cfs))

    stats = {
        "period_start": iso(start),
        "period_end": iso(end),
        "period_start_fr": fr_date(start),
        "period_end_fr": fr_date(end),
        "buy_count": int(len(btc_bought)),
        "amount_eur_per_buy": float(amount_eur),
        "frequency": freq,
        "every": int(every),
        "fee_rate": float(fee_rate),
        "invested_eur": float(df["invested_eur"].iloc[-1]),
        "final_value_eur": float(df["value_eur"].iloc[-1]),
        "profit_eur": float(df["value_eur"].iloc[-1] - df["invested_eur"].iloc[-1]),
        "multiple": float(df["value_eur"].iloc[-1] / df["invested_eur"].iloc[-1]) if df["invested_eur"].iloc[-1] > 0 else float("nan"),
        "tri_annualized": tri,
        "max_drawdown": max_dd
    }
    return df, stats

def save_outputs(run_dir: str, sim_df: pd.DataFrame, stats: dict, source_csv_path: str):
    os.makedirs(run_dir, exist_ok=True)

    # Save inputs / stats
    with open(os.path.join(run_dir, "stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    sim_df.to_csv(os.path.join(run_dir, "simulation.csv"), index_label="date")

    # Save a copy of source CSV used (repro)
    try:
        import shutil
        shutil.copy2(source_csv_path, os.path.join(run_dir, "source_btceur.csv"))
    except Exception:
        pass

    # Graph 1: price
    plt.figure(figsize=(10,4))
    plt.plot(sim_df.index, sim_df["price_eur"])
    plt.title("BTC/EUR — prix (close) quotidien")
    plt.xlabel("Date"); plt.ylabel("€")
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "01_price.png"), dpi=160)
    plt.close()

    # Graph 2: value vs invested
    plt.figure(figsize=(10,4))
    plt.plot(sim_df.index, sim_df["invested_eur"], label="Total investi")
    plt.plot(sim_df.index, sim_df["value_eur"], label="Valeur portefeuille (DCA)")
    plt.title("DCA BTC — valeur vs investi")
    plt.xlabel("Date"); plt.ylabel("€")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "02_value_vs_invested.png"), dpi=160)
    plt.close()

    # Graph 3: drawdown
    peak = sim_df["value_eur"].cummax()
    dd = (sim_df["value_eur"] / peak) - 1.0
    plt.figure(figsize=(10,4))
    plt.plot(sim_df.index, dd)
    plt.title("DCA BTC — drawdown (recul depuis le plus haut)")
    plt.xlabel("Date"); plt.ylabel("Drawdown")
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "03_drawdown.png"), dpi=160)
    plt.close()

def fmt_money(x: float) -> str:
    return f"{x:,.0f}".replace(",", " ").replace("\xa0"," ")

def main():
    ensure_dirs()

    print("=== DCA BTC/EUR — simulateur interactif ===")
    today = date.today()

    # CSV download choice
    print("\n=== Données BTC/EUR ===")
    if os.path.exists(CACHE_CSV):
        print(f"✅ CSV en cache trouvé: {CACHE_CSV}")
        refresh = input("Re-télécharger pour mettre à jour ? (y/N) : ").strip().lower() == "y"
    else:
        refresh = True

    if refresh:
        print(f"⬇️ Téléchargement Stooq: {STOOQ_CSV_URL}")
        download_csv(STOOQ_CSV_URL, CACHE_CSV)
        print("✅ Téléchargement terminé.")

    prices = load_prices_stooq(CACHE_CSV)

    # Choose period
    start_d, end_d = choose_period(today)

    # Clamp to available data
    min_available = prices.index.min().date()
    max_available = prices.index.max().date()
    if start_d < min_available:
        print(f"⚠️ Début ajusté au min dispo: {iso(min_available)}")
        start_d = min_available
    if end_d > max_available:
        print(f"⚠️ Fin ajustée au max dispo: {iso(max_available)}")
        end_d = max_available
    if start_d > end_d:
        start_d, end_d = end_d, start_d

    # Choose DCA parameters
    amount, freq, every, fee_rate = choose_dca_params()

    # Run simulation
    sim_df, stats = simulate_dca(prices, start_d, end_d, amount, freq, every, fee_rate)

    # Output directory per run
    run_dir = os.path.join(OUT_ROOT, f"run_{now_ts()}")
    save_outputs(run_dir, sim_df, stats, CACHE_CSV)

    # Print results
    freq_label = {"D":"jour(s)", "W":"semaine(s)", "M":"mois"}[freq]
    print("\n===== RÉSULTATS =====")
    print(f"Période           : {stats['period_start_fr']} → {stats['period_end_fr']}")
    print(f"Achats            : {stats['buy_count']} × {stats['amount_eur_per_buy']:.2f} €  (tous les {every} {freq_label})")
    print(f"Frais             : {stats['fee_rate']*100:.2f}% / achat")
    print(f"Total investi     : {fmt_money(stats['invested_eur'])} €")
    print(f"Valeur finale     : {fmt_money(stats['final_value_eur'])} €")
    print(f"Gain              : {fmt_money(stats['profit_eur'])} €")
    print(f"Multiple          : x{stats['multiple']:.2f}")
    print(f"TRI (annualisé)   : {stats['tri_annualized']*100:.1f}% / an")
    print(f"Pire drawdown     : {stats['max_drawdown']*100:.1f}%")

    print(f"\n📁 Outputs: {run_dir}")
    print(" - stats.json")
    print(" - simulation.csv")
    print(" - source_btceur.csv")
    print(" - 01_price.png / 02_value_vs_invested.png / 03_drawdown.png")

if __name__ == "__main__":
    main()
