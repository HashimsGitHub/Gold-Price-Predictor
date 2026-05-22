# 🥇 Gold Price Predictor

A machine learning pipeline that forecasts gold prices using multi-asset correlation analysis, GPU-accelerated XGBoost, and Supabase for live data persistence and tracking.

---

## 📌 Overview

This notebook fetches 10 years of monthly gold spot price data alongside correlated macro assets (USD Index, S&P 500, Silver) from Yahoo Finance, engineers time-series features, trains an XGBoost regression model, and stores predictions back into Supabase — refreshing actuals on every run for ongoing monitoring.

---
<img width="1563" height="766" alt="image" src="https://github.com/user-attachments/assets/23867677-20af-43ed-a7bb-3bf1db4b9f71" />

## ✨ Features

| Feature | Description |
|---|---|
| 📈 10-Year Historical Chart | Monthly gold spot prices visualised over a decade |
| 🔗 Multi-Asset Correlation | USD Index, S&P 500, and Silver as predictive lag features |
| ⚡ GPU-Accelerated XGBoost | NVIDIA CUDA training with automatic CPU fallback |
| 🔄 Stationary Returns Pipeline | Percentage return engineering to eliminate extrapolation bias |
| 🛠️ Hyperparameter Tuning | Grid Search CV with TimeSeriesSplit to preserve chronological order |
| 🔮 3-Month Recursive Forecast | Autoregressive forward projection with annotated price targets |
| 🗄️ Supabase Integration | Live upsert of actuals and predictions on every notebook execution |
| 🔁 Auto Price Refresh | Latest gold and macro prices pulled from Yahoo Finance on each run |

---

## 🗂️ Notebook Structure

| Cell | Pipeline Stage |
|---|---|
| 1 | Install libraries, connect to Supabase, initialise DB tables |
| 2 | Fetch 10-year gold history, upsert to `gold_monthly_prices`, RF baseline model, historical chart |
| 3 | Multi-asset download (Gold, USD, S&P 500, Silver), XGBoost training, predictions saved to `gold_correlated_macro` |
| 4 | Price refresh — upserts latest actuals into both Supabase tables on every run |
| 5 | Stationary percentage returns pipeline — reduces non-stationarity bias |
| 6 | Hyperparameter tuning via Grid Search |
| 7 | 3-month recursive forecast with annotated chart |

---

## 🗄️ Database Tables (Supabase)

| Table | Purpose |
|---|---|
| `gold_monthly_prices` | Raw monthly gold spot prices (date, price) |
| `gold_correlated_macro` | Multi-asset prices + XGBoost predicted prices by month |

---

## 🛠️ Tech Stack

- **Python** 3.12
- **XGBoost** — gradient boosted regression with GPU acceleration
- **Scikit-learn** — model evaluation, Grid Search, TimeSeriesSplit
- **Pandas / NumPy** — data wrangling and feature engineering
- **Seaborn / Matplotlib** — visualisations
- **yfinance** — Yahoo Finance API for market data
- **psycopg2** — PostgreSQL connection to Supabase
- **python-dotenv** — secure credential loading from `.env`
- **PyTorch** — CUDA availability detection for GPU routing

---

## ⚙️ Setup

### 1. Clone the repository
```bash
git clone https://github.com/HashimsGitHub/GoldPricePredictor.git
cd GoldPricePredictor
```

### 2. Create and activate a virtual environment
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install psycopg2-binary pandas numpy scikit-learn seaborn matplotlib \
            python-dotenv requests yfinance xgboost jupyter ipykernel
```

Install PyTorch with CUDA support (CUDA 12.8 nightly — compatible with driver 581.95+):
```bash
pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128
```

### 4. Configure environment variables

Create a `.env` file in the project root:
```
DATABASE_URL=postgresql://postgres:<password>@<host>:<port>/<database>
```
> ⚠️ Never commit this file. It is excluded via `.gitignore`.

### 5. Launch Jupyter
```bash
jupyter notebook
```

Or inside VSCode with WSL Remote — select the `GoldPredictor (venv)` kernel from the top-right kernel picker.

---

## 🖥️ Environment

| Component | Detail |
|---|---|
| GPU | NVIDIA GeForce RTX 3050 Laptop (4GB VRAM) |
| CUDA | 13.0 |
| Driver | 581.95 |
| OS | WSL2 Ubuntu on Windows |
| Python | 3.12.3 |

---

## 📁 Project Structure

```
GoldPricePredictor/
│
├── GoldPrice_SupaBase.ipynb   # Main notebook
├── README.md                  # This file
├── .env                       # Database credentials (not committed)
├── .gitignore                 # Excludes .env, venv, checkpoints
└── venv/                      # Virtual environment (not committed)
```

---

## 📝 Notes

- The notebook is designed to be **re-run monthly** — each execution refreshes the latest gold and macro prices in Supabase automatically
- XGBoost detects CUDA at runtime and switches to GPU training automatically; falls back to CPU (`device="hist"`) if unavailable
- All time-series splits are **chronological** (no shuffle) to prevent data leakage from future prices into training

---

## 📄 License

MIT License — free to use and modify.
