# Bitcoin DCA Simulation

An **interactive Python tool** to simulate **Dollar-Cost Averaging (DCA) strategies on Bitcoin** using **real BTC/EUR historical data**.  
The tool computes key performance metrics such as **ROI, XIRR (IRR), drawdowns**, and generates **charts** for analysis.

> ⚠️ **Note**:  
> The **code and documentation are written in English**, but the **interactive CLI interface is currently in French**.

---

## Table of Contents
- [Features](#features)
- [Data Source](#data-source)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Simulation Options](#simulation-options)
- [Outputs](#outputs)
- [Project Structure](#project-structure)
- [Methodology](#methodology)
- [Limitations](#limitations)
- [Disclaimer](#disclaimer)
- [License](#license)

---

## Features
- Interactive **CLI-based simulator**
- Uses **real BTC/EUR daily historical prices**
- Supports multiple **time ranges**:
  - 1 year, 3 years, 5 years
  - Year-To-Date (YTD)
  - Custom number of years
  - Custom date range (CLI or optional GUI)
- Fully configurable **DCA parameters**:
  - Amount per purchase
  - Frequency (daily / weekly / monthly)
  - Custom interval (every N days/weeks/months)
  - Transaction fees
- Computes advanced metrics:
  - Total invested amount
  - Final portfolio value
  - Profit / Loss
  - Portfolio multiple
  - **XIRR (annualized internal rate of return)**
  - Maximum drawdown
- Generates **CSV outputs** and **charts**
- Each run is saved in a **timestamped output directory** for reproducibility

---

## Data Source
- **Stooq** — BTC/EUR daily OHLC data  
  - Public and freely accessible
  - Used price: **daily close (EUR)**

Source URL:

[https://stooq.com/q/d/l/?s=btceur&i=d](https://stooq.com/q/d/l/?s=btceur&i=d)

---

## Requirements
- Python **3.9+** recommended

Python dependencies:
- `pandas`
- `numpy`
- `matplotlib`
- `requests`
- `tkcalendar` *(optional, for GUI date picker)*

---

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/your-username/btc-dca-simulation.git
cd btc-dca-simulation
```

### 2. Install dependencies

```bash
python3 -m pip install -r requirements.txt
```

If you want the **GUI date picker**:

```bash
python3 -m pip install tkcalendar
```

---

## Usage

Run the simulator:

```bash
python3 btc_dca.py
```

You will be guided step by step through:

1. Data update (optional CSV refresh)
2. Period selection
3. DCA configuration
4. Fee configuration
5. Simulation execution

---

## Simulation Options

### Time Period Selection

* 1 year
* 3 years
* 5 years
* Year-To-Date (YTD)
* Custom number of years
* Custom date range:

  * CLI input (`YYYY-MM-DD`)
  * Optional GUI calendar picker (if available)

### DCA Parameters

* Amount per purchase (EUR)
* Frequency:

  * Daily
  * Weekly
  * Monthly
* Interval:

  * Every N days / weeks / months
* Fees:

  * Percentage per transaction
  * Press Enter for 0%

---

## Outputs

Each simulation creates a **unique timestamped folder**:

```
out/
└── run_YYYYMMDD_HHMMSS/
    ├── stats.json
    ├── simulation.csv
    ├── source_btceur.csv
    ├── 01_price.png
    ├── 02_value_vs_invested.png
    └── 03_drawdown.png
```

### Output Files

* **stats.json**
  Summary of all computed metrics
* **simulation.csv**
  Daily portfolio evolution
* **source_btceur.csv**
  Snapshot of the price data used (reproducibility)
* **PNG charts**

  * BTC/EUR price
  * Portfolio value vs invested capital
  * Drawdown curve

---

## Project Structure

```
btc-dca-simulation/
├── btc_dca.py
├── requirements.txt
├── README.md
├── .gitignore
├── data/
│   └── btceur_stooq_daily.csv
└── out/
    └── run_*/
```

---

## Methodology

* Prices used: **daily close BTC/EUR**
* Missing days (weekends/holidays) are **forward-filled**
* DCA purchases occur exactly on the selected schedule
* Portfolio value is computed daily
* **XIRR (TRI)** is calculated using real cash-flow dates
* Drawdown is measured from portfolio equity peaks

---

## Limitations

* Past performance does **not** predict future results
* No slippage modeling
* Fees are applied as a simple percentage
* Does not include taxation
* Not a real-time trading tool

---

## Disclaimer

This project is provided for **educational and research purposes only**.

* This is **not financial advice**
* Cryptocurrency investments are highly volatile
* Always do your own research and risk assessment

---

## License

This project is released under the **MIT License**.

Feel free to use, modify, and share.
