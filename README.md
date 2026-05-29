# 🥇 Gold Price Predictor

A machine learning pipeline that forecasts gold prices using multi-asset correlation analysis, GPU-accelerated XGBoost, and Supabase for live data persistence and tracking.

---

## 📌 Overview

This notebook fetches 10 years of monthly gold spot price data alongside correlated macro assets (USD Index, S&P 500, Silver) from Yahoo Finance, engineers time-series lag features, trains both a baseline Random Forest and a GPU-accelerated XGBoost regression model, and stores all predictions back into Supabase — automatically refreshing actuals on every run for ongoing accuracy monitoring.

The pipeline concludes with a 3-month recursive forward forecast and a live accuracy dashboard rendered directly in the notebook.

---

<img width="1397" height="704" alt="image" src="https://github.com/user-attachments/assets/9da9af46-cf15-44f9-8851-9c4240648ee8" />

<img width="827" height="713" alt="image" src="https://github.com/user-attachments/assets/d8d0caf7-1f35-48ed-9faa-c463dcbc0a06" />


## ✨ Features

| Feature | Description |
|---|---|
| 📈 10-Year Historical Chart | Monthly gold spot prices visualised over a decade |
| 🔗 Multi-Asset Correlation | USD Index, S&P 500, and Silver as predictive lag features |
| ⚡ GPU-Accelerated XGBoost | NVIDIA T4 CUDA training on Google Colab with automatic CPU fallback |
| 🔄 Stationary Returns Pipeline | Percentage return engineering to eliminate extrapolation bias |
| 🛠️ Hyperparameter Tuning | Grid Search CV with TimeSeriesSplit to preserve chronological order |
| 🔮 3-Month Recursive Forecast | Autoregressive forward projection with annotated price targets |
| 🗄️ Supabase Integration | Live upsert of actuals and predictions on every notebook execution |
| 🔁 Auto Price Refresh | Latest gold and macro prices pulled from Yahoo Finance on each run |
| 📊 Accuracy Dashboard | HTML backtest table with historical predictions, today's run, and future forecasts |

---

## 🗂️ Notebook Structure

| Cell | Pipeline Stage |
|---|---|
| 0 | Environment setup — Colab guard, package installation, GPU/TPU detection, execution timer start |
| 1 | Import libraries, Supabase connection helper, DB table initialisation |
| 2 | Fetch 10-year gold history, upsert to `gold_monthly_prices`, Random Forest baseline model, historical chart |
| 3 | Multi-asset download (Gold, USD Index, S&P 500, Silver), correlation matrix, XGBoost training, predictions saved to `gold_correlated_macro` |
| 4 | Price refresh — upserts latest actuals into both Supabase tables on every run |
| 5 | Stationary percentage returns pipeline — reduces non-stationarity bias |
| 6 | Hyperparameter tuning via Grid Search CV |
| 7 | 3-month recursive forecast with annotated chart |
| 8 | Accuracy dashboard — HTML table showing backtest history, today's run, and future forecast rows with total runtime |

---

## 🗄️ Database Tables (Supabase)

| Table | Purpose |
|---|---|
| `gold_monthly_prices` | Raw monthly gold spot prices (date, price) |
| `gold_correlated_macro` | Multi-asset prices + XGBoost predicted prices by month |
| `gold_forecast_log` | Per-run forecast log with predicted vs actual prices for accuracy tracking |

---

## ⏱️ Average Runtime Estimates (T4 GPU)

| Stage | Estimated Time |
|-------|---------------|
| Package installation | ~60 seconds |
| Data ingestion & Supabase sync | ~20 seconds |
| Random Forest training | ~5 seconds |
| Multi-asset XGBoost training | ~3 seconds |
| Hyperparameter grid search | ~3–4 minutes |
| Full notebook end-to-end | **~4.44 minutes** |

> Runtimes on CPU will be significantly longer, particularly for the grid search cell.

---

## 🛠️ Tech Stack

- **Python** 3.x (Google Colab runtime)
- **XGBoost** — gradient boosted regression with GPU acceleration
- **Scikit-learn** — model evaluation, Grid Search, TimeSeriesSplit
- **Pandas / NumPy** — data wrangling and feature engineering
- **Seaborn / Matplotlib** — visualisations
- **yfinance** — Yahoo Finance API for market data
- **psycopg2** — PostgreSQL connection to Supabase
- **PyTorch** — CUDA availability detection for XGBoost device routing

---

## ⚙️ Setup

### 1. Open in Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HashimsGitHub/Gold-Price-Predictor/blob/main/GoldPrice_SupaBase.ipynb)

> ⚠️ This notebook is designed to run **exclusively on Google Colab with a T4 GPU**.

To enable hardware acceleration before running:
```
Runtime → Change runtime type → Hardware accelerator → GPU (T4)
```

### 2. Create a Supabase project

1. Go to [https://supabase.com](https://supabase.com) and sign up for a free account
2. Click **New Project** and fill in a name, password, and region
3. Once ready, navigate to:
   ```
   Project Settings → Database → Connection string → URI
   ```
4. Copy your connection string — it will look like:
   ```
   postgresql://postgres:<password>@<host>:<port>/<database>
   ```

### 3. Store the connection string in Colab Secrets

> **Never paste credentials directly into a notebook cell.** Use Colab Secrets instead.

1. In Colab, click the 🔑 **Secrets** icon in the left sidebar
2. Click **+ Add new secret**
3. Set **Name** exactly as:
   ```
   DATABASE_URL
   ```
4. Paste your full Supabase connection string as the **Value**
5. Toggle **Notebook access** to **ON**

The notebook reads `DATABASE_URL` automatically on every run — no `.env` file or local configuration needed.

### 4. Run all cells

```
Runtime → Run all
```

---

## 📁 Project Structure

```
GoldPricePredictor/
│
├── GoldPrice_SupaBase.ipynb   # Main notebook (run in Google Colab)
└── README.md                  # This file
```

---

## 📝 Notes

- The notebook is designed to be **re-run monthly** — each execution refreshes the latest gold and macro prices in Supabase automatically
- XGBoost detects CUDA at runtime and switches to GPU training automatically; falls back to CPU (`device="cpu"`) if no GPU is available
- All time-series splits are **chronological** (no shuffle) to prevent data leakage from future prices into training
- The accuracy dashboard backfills prior forecast rows with real prices as each month closes, enabling ongoing model monitoring

---

## 📄 License

MIT License — free to use and modify.
