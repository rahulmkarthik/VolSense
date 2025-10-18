# ⚡ **VolSense: Explainable Volatility Forecasting**

VolSense is an **explainable AI-powered volatility forecaster** that blends traditional econometric models (like GARCH) with **deep learning architectures** (LSTM/TCN) to predict and interpret market volatility across global assets.

It aims to answer a critical quant research question:  
> _Can deep learning models outperform classical GARCH-style baselines while remaining interpretable to traders and risk managers?_

---

## 🚀 **Key Features**

### 🧩 Model Architecture
- **GlobalVolForecaster** — multi-ticker LSTM forecaster with:
  - Residual + multi-horizon heads  
  - EMA smoothing, cosine LR scheduling  
  - LayerNorm and variational dropout regularization  
  - Configurable horizon weighting and dynamic window jitter  
- Supports multiple validation regimes (causal / holdout slice).  
- Fully modular `TrainConfig`, checkpointing, and bundle serialization.

---

### 📈 Data & Feature Engineering
- Automated data ingestion via **yfinance** for equities, ETFs, and crypto.  
- Realized volatility computed from rolling returns with multi-scale features:
  - `vol_3d`, `vol_10d`, `vol_ratio`, `vol_vol`, `vol_chg`
  - `ewma_vol_10d`, `market_stress`, `skew_5d`
  - `day_of_week`, `month_sin`, `month_cos`, `abs_return`
- Unified `build_features()` ensures training/inference feature parity.

---

### 🧠 Explainability
- SHAP-based feature attribution per horizon  
- Sensitivity analysis comparing learned weights vs GARCH parameters  
- Multi-ticker volatility decomposition and interpretability plots  

---

### ⚙️ Evaluation
- RMSE / MAE on realized volatility forecasts  
- Out-of-sample horizon testing (`1d`, `5d`, `10d`)  
- VaR exceedance and volatility clustering diagnostics  

---

### 💡 Deployment
- **Inference engine (`volsense_inference`)** with:
  - `model_loader.py` for loading serialized `.pth` or `.pkl` bundles  
  - `predictor.py` for robust multi-ticker forecasting  
  - `forecast_engine.py` for unified CLI / programmatic inference  
  - Real-time plotting vs realized vol, supporting trader-facing dashboards  
- CLI access via:
  ```bash
  volsense-train
  volsense-forecast

VolSense/
│
├── data/                     # raw & processed datasets
│   ├── raw/
│   └── processed/
│
├── notebooks/                # exploration & experiments
│   ├── 01_data_exploration.ipynb
│   ├── 02_garch_baseline.ipynb
│   ├── 03_lstm_forecast.ipynb
│   ├── 04_explainability.ipynb
│   └── volsense_cache/
│
├── volsense_core/            # main training package
│   ├── forecasters/
│   ├── models/
│   ├── data_fetching/
│   └── utils/
│
├── volsense_inference/       # inference & deployment pipeline
│   ├── forecast_engine.py
│   ├── predictor.py
│   ├── model_loader.py
│   ├── feature_builder.py
│   └── cli/
│
├── checkpoints/              # trained model weights & bundles
├── dist/                     # build artifacts
└── README.md
