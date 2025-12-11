"""
Streamlit dashboard for VolSense.

Provides a small interactive UI to run forecasts, inspect per-ticker plots,
sector heatmaps and cross-sectional signals.
---------------------------------------

This module exposes the Streamlit application and several helper functions:

- :func:`_parse_tickers` - parse user comma-separated tickers
- :func:`_safe_style_format` - style numeric columns for display
- :func:`_detect_horizons` - detect forecast horizons from dataframe columns
- :func:`_load_forecast_model` - cached model loader
- :func:`run_volsense` - run forecasts + signals (cached)
- :func:`export_csv_button` - download button for DataFrame CSV export

Typical usage
~~~~~~~~~~~~~
Run the Streamlit app:

.. code-block:: bash

   streamlit run volsense_inference/dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import date
from typing import List, Optional
import sys
from pathlib import Path

# Plotly for interactive visualizations
import plotly.express as px
import plotly.graph_objects as go

# ──────────────────────────────────────────────────────────────────────────────
# Ensure project root is in Python path (for Streamlit Cloud deployment)
# ──────────────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# VolSense imports
from volsense_inference.forecast_engine import Forecast  # runs models + features  📦
from volsense_inference.signal_engine import SignalEngine  # sector-aware signals
from volsense_inference.sector_mapping import get_sector_map, get_color
from volsense_inference.persistence import get_daily_cache

# ──────────────────────────────────────────────────────────────────────────────
# Streamlit page config
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="VolSense — Volatility Insights",
    page_icon="📈",
    layout="wide",
)

# ──────────────────────────────────────────────────────────────────────────────
# Full V507 Universe
# ──────────────────────────────────────────────────────────────────────────────
V507_UNIVERSE = list(get_sector_map("v507").keys())

# ─────────────────────────────────────────────────────────────
# ⚙️ Session state setup
# ─────────────────────────────────────────────────────────────
if "forecast_data" not in st.session_state:
    st.session_state.forecast_data = None
if "universe_hydrated" not in st.session_state:
    st.session_state.universe_hydrated = False
if "active_tab" not in st.session_state:
    st.session_state.active_tab = "Overview"
if "tabs_list" not in st.session_state:
    st.session_state.tabs_list = [
        "Overview",
        "Ticker Analytics",
        "Sector View",
        "Signal Table",
    ]


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
def _parse_tickers(s: str) -> List[str]:
    """
    Parse a comma-separated ticker string into a sorted list of unique tickers.

    :param s: Comma-separated tickers entered by the user.
    :type s: str
    :return: Sorted list of uppercase, de-duplicated ticker symbols.
    :rtype: List[str]
    """
    return sorted({t.strip().upper() for t in s.split(",") if t.strip()})


def _safe_style_format(df: pd.DataFrame, formatter):
    """
    Apply a number formatter to numeric columns of a DataFrame and return a Styler.

    This avoids formatting attempts on non-numeric columns (which would raise).

    :param df: DataFrame to format.
    :type df: pandas.DataFrame
    :param formatter: Callable or format string to apply to numeric columns.
    :return: pandas Styler with applied formatting.
    :rtype: pandas.io.formats.style.Styler
    """
    num_cols = df.select_dtypes(include=[np.number]).columns
    format_dict = {c: formatter for c in num_cols}
    return df.style.format(format_dict)


def _detect_horizons(df: pd.DataFrame) -> List[int]:
    """
    Detect integer horizon values from column names like 'pred_vol_5'.

    :param df: Forecast DataFrame containing horizon columns.
    :type df: pandas.DataFrame
    :return: Sorted list of unique horizon integers discovered.
    :rtype: List[int]
    """
    cols = [c for c in df.columns if c.startswith("pred_vol_")]
    hs = []
    for c in cols:
        try:
            hs.append(int(c.split("_")[-1].replace("d", "")))
        except Exception:
            pass
    return sorted(set(hs))


@st.cache_resource(show_spinner=False)
def _load_forecast_model(model_version: str, checkpoints_dir: str):
    """
    Cached loader for Forecast object.

    Uses Streamlit's cache_resource to avoid re-loading heavy models on each interaction.

    :param model_version: Model version/stem to load (e.g., "v109").
    :type model_version: str
    :param checkpoints_dir: Directory containing model artifacts.
    :type checkpoints_dir: str
    :return: Instantiated Forecast object ready to run.
    :rtype: volsense_inference.forecast_engine.Forecast
    """
    fcast = Forecast(
        model_version=model_version, checkpoints_dir=checkpoints_dir, start="2005-01-01"
    )
    return fcast


@st.cache_data(show_spinner=False, ttl=60 * 30)
def run_volsense(
    model_version: str, checkpoints_dir: str, start: str, tickers: List[str]
):
    """
    Run VolSense forecasts and compute analytics and signals.

    This function is cached for 30 minutes to avoid repeated expensive runs.

    :param model_version: Model version to use for forecasting.
    :type model_version: str
    :param checkpoints_dir: Directory with model artifacts.
    :type checkpoints_dir: str
    :param start: Start date string for feature fetch.
    :type start: str
    :param tickers: List of tickers to forecast.
    :type tickers: List[str]
    :return: Tuple (Forecast instance, predictions DataFrame, analytics object,
             analytics summary DataFrame, SignalEngine instance, signals DataFrame)
    :rtype: tuple
    """
    fcast = _load_forecast_model(model_version, checkpoints_dir)
    preds = fcast.run(
        tickers
    )  # attaches fcast.signals (Analytics) internally  :contentReference[oaicite:4]{index=4}
    analytics = (
        fcast.signals
    )  # Analytics(preds).compute() already called        :contentReference[oaicite:5]{index=5}
    ae_summary = (
        analytics.summary(horizon="pred_vol_5")
        if "pred_vol_5" in preds.columns
        else analytics.summary()
    )

    # Signal engine (sector-aware)
    se = SignalEngine(model_version=model_version)
    se.set_data(
        preds
    )  # coerce wide → long if needed  :contentReference[oaicite:6]{index=6}
    sig_df = se.compute_signals(enrich_with_sectors=True)

    return fcast, preds, analytics, ae_summary, se, sig_df


# ──────────────────────────────────────────────────────────────────────────────
# Universe Hydration & Cache Loading
# ──────────────────────────────────────────────────────────────────────────────
def hydrate_universe(model_version: str, checkpoints_dir: str) -> tuple:
    """
    Run forecasts on the FULL V507 universe and cache results.
    
    This provides meaningful cross-sectional signals since z-scores
    are computed across the entire universe.
    
    :param model_version: Model version to use.
    :param checkpoints_dir: Directory with model checkpoints.
    :return: Tuple (fcast, preds, analytics, ae_summary, se, sig_df)
    """
    cache = get_daily_cache()
    
    print(f"🌊 Hydrating full universe ({len(V507_UNIVERSE)} tickers)...")
    
    # Run full universe inference
    fcast = _load_forecast_model(model_version, checkpoints_dir)
    preds = fcast.run(V507_UNIVERSE)
    
    analytics = fcast.signals
    ae_summary = (
        analytics.summary(horizon="pred_vol_5")
        if "pred_vol_5" in preds.columns
        else analytics.summary()
    )
    
    # Compute signals across full universe (use v507 sector map)
    se = SignalEngine(model_version="v507")
    se.set_data(preds)
    sig_df = se.compute_signals(enrich_with_sectors=True)
    
    # Store each ticker's data to daily cache
    for ticker in V507_UNIVERSE:
        ticker_preds = preds[preds["ticker"] == ticker].to_dict(orient="records")
        ticker_signals = sig_df[sig_df["ticker"] == ticker].to_dict(orient="records")
        
        if ticker_preds:
            cache.store_entry(ticker, {
                "predictions": ticker_preds[0] if len(ticker_preds) == 1 else ticker_preds,
                "signals": ticker_signals,
            })
    
    print(f"💾 Cached {len(cache)} tickers to daily cache")
    
    return fcast, preds, analytics, ae_summary, se, sig_df


def load_universe_from_cache(model_version: str) -> Optional[tuple]:
    """
    Load cached universe data if available.
    
    :param model_version: Model version for SignalEngine.
    :return: Tuple (None, preds, None, None, se, sig_df) or None if cache empty.
    """
    cache = get_daily_cache()
    
    if len(cache) < 10:  # Require at least some cached data
        return None
    
    print(f"✅ Loading {len(cache)} tickers from daily cache...")
    
    # Reconstruct DataFrames from cache
    pred_rows = []
    signal_rows = []
    
    for ticker, entry in cache.get_all_entries().items():
        if "predictions" in entry:
            p = entry["predictions"]
            if isinstance(p, dict):
                pred_rows.append(p)
            elif isinstance(p, list):
                pred_rows.extend(p)
        
        if "signals" in entry:
            signal_rows.extend(entry["signals"])
    
    if not pred_rows or not signal_rows:
        return None
    
    preds = pd.DataFrame(pred_rows)
    sig_df = pd.DataFrame(signal_rows)
    
    # Convert date columns back to datetime
    for df in [preds, sig_df]:
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
    
    # Create SignalEngine (use v507 sector map)
    se = SignalEngine(model_version="v507")
    se.set_data(preds)
    
    return None, preds, None, None, se, sig_df


def _pretty_number(x, nd=3):
    """
    Human-friendly numeric formatter used by the UI.

    :param x: Numeric value (or NaN).
    :param nd: Number of decimal places to format.
    :type nd: int
    :return: Formatted string or empty string for NaN.
    :rtype: str
    """
    if pd.isna(x):
        return ""
    return f"{x:.{nd}f}"


# ──────────────────────────────────────────────────────────────────────────────
# CSV Export helper
# ──────────────────────────────────────────────────────────────────────────────


def export_csv_button(df: pd.DataFrame, filename: str, label: str = "📥 Export CSV"):
    """
    Create a Streamlit download button that exports a DataFrame as CSV.

    :param df: DataFrame to export.
    :type df: pandas.DataFrame
    :param filename: Suggested filename for the downloaded CSV.
    :type filename: str
    :param label: Button label text.
    :type label: str
    :return: None
    """
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label=label,
        data=csv,
        file_name=filename,
        mime="text/csv",
        width="stretch",
    )


# ──────────────────────────────────────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────────────────────────────────────
st.sidebar.title("VolSense Dashboard")
st.sidebar.caption("Trader-facing Volatility Analytics & Signals")

TICKERS = [
    # 100 cross-sector tickers to initialize the dashboard with
    # ----- Index / ETF -----
    "SPY",
    "QQQ",
    "DIA",
    "IWM",
    "GLD",
    "SLV",
    "TLT",
    "HYG",
    "EEM",
    # ----- Technology (17) -----
    "AAPL",
    "MSFT",
    "GOOG",
    "GOOGL",
    "AMZN",
    "META",
    "NVDA",
    "AVGO",
    "AMD",
    "INTC",
    "ORCL",
    "TXN",
    "QCOM",
    "ADBE",
    "CSCO",
    "NOW",
    "INTU",
    # ----- Financials (13) -----
    "JPM",
    "BAC",
    "C",
    "WFC",
    "GS",
    "MS",
    "V",
    "MA",
    "BLK",
    "PNC",
    "USB",
    "TFC",
    "COF",
    # ----- Healthcare (13) -----
    "JNJ",
    "PFE",
    "MRK",
    "UNH",
    "ABBV",
    "ABT",
    "LLY",
    "BMY",
    "TMO",
    "CVS",
    "AMGN",
    "REGN",
    "MDT",
    # ----- Energy / Materials (12) -----
    "XOM",
    "CVX",
    "COP",
    "SLB",
    "HAL",
    "EOG",
    "BHP",
    "RIO",
    "FCX",
    "LIN",
    "APD",
    "NUE",
    # ----- Consumer Discretionary (11) -----
    "TSLA",
    "HD",
    "MCD",
    "NKE",
    "SBUX",
    "TGT",
    "BKNG",
    "CMG",
    "LOW",
    "MAR",
    "EBAY",
    # ----- Industrials (8) -----
    "CAT",
    "BA",
    "HON",
    "UPS",
    "FDX",
    "LMT",
    "GE",
    "DE",
    # ----- Consumer Staples (8) -----
    "PG",
    "KO",
    "PEP",
    "COST",
    "WMT",
    "MDLZ",
    "CL",
    "KHC",
    # ----- Communication Services (9) -----
    "NFLX",
    "DIS",
    "T",
    "VZ",
    "TMUS",
    "CMCSA",
    "CHTR",
    "EA",
    "TTWO",
]

with st.sidebar:
    # Model settings
    model_version = st.text_input(
        "Model version", value="volnetx"
    )
    checkpoints_dir = st.text_input("Checkpoints directory", value="models")
    
    st.divider()
    
    # Primary action: Refresh Market (full universe)
    st.subheader("🌊 Market Hydration")
    st.caption(f"Universe: {len(V507_UNIVERSE)} tickers")
    
    cache = get_daily_cache()
    cache_status = f"📦 {len(cache)} tickers cached ({cache._today_str()})"
    st.caption(cache_status)
    
    hydrate_btn = st.button("Refresh Market", type="primary", width="stretch")
    
    st.divider()
    st.caption("💡 Refresh Market runs inference on the full V507 universe for meaningful cross-sectional signals.")

# ──────────────────────────────────────────────────────────────────────────────
# Header
# ──────────────────────────────────────────────────────────────────────────────
st.title("📈 VolSense — Sector & Ticker Volatility Insights")
st.markdown(
    "Cross-sectional volatility analytics and sector-aware signals across the V507 universe. "
    "Click **Refresh Market** to hydrate with latest inference."
)

# ──────────────────────────────────────────────────────────────────────────────
# Main workflow
# ──────────────────────────────────────────────────────────────────────────────

# Try to load from cache on startup
if st.session_state.forecast_data is None and not hydrate_btn:
    cached_result = load_universe_from_cache(model_version)
    if cached_result is not None:
        fcast, preds, analytics, ae_summary, se, sig_df = cached_result
        st.session_state.forecast_data = {
            "fcast": fcast,
            "preds": preds,
            "analytics": analytics,
            "ae_summary": ae_summary,
            "se": se,
            "sig_df": sig_df,
        }
        st.session_state.universe_hydrated = True

if hydrate_btn:
    try:
        with st.spinner(f"🌊 Hydrating full universe ({len(V507_UNIVERSE)} tickers)... This may take a few minutes."):
            fcast, preds, analytics, ae_summary, se, sig_df = hydrate_universe(
                model_version=model_version,
                checkpoints_dir=checkpoints_dir,
            )
        st.session_state.forecast_data = {
            "fcast": fcast,
            "preds": preds,
            "analytics": analytics,
            "ae_summary": ae_summary,
            "se": se,
            "sig_df": sig_df,
        }
        st.session_state.universe_hydrated = True
        st.rerun()
    except Exception as e:
        st.error(f"❌ Hydration failed: {e}")
        st.stop()

if st.session_state.forecast_data is not None:
    try:
        fcast = st.session_state.forecast_data["fcast"]
        preds = st.session_state.forecast_data["preds"]
        analytics = st.session_state.forecast_data["analytics"]
        ae_summary = st.session_state.forecast_data["ae_summary"]
        se = st.session_state.forecast_data["se"]
        sig_df = st.session_state.forecast_data["sig_df"]

        st.success("✅ Forecasts & signals ready!")

        # 2️⃣ Remember which tab is active between reruns
        tabs = st.session_state.tabs_list
        selected_tab = st.session_state.active_tab
        tab_objs = st.tabs(tabs)
        selected_index = tabs.index(selected_tab) if selected_tab in tabs else 0

        # 3️⃣ Map tabs to content
        tab_overview, tab_tickers, tab_sector, tab_signals = tab_objs

        # 4️⃣ After rendering, update active tab
        st.session_state.active_tab = tabs[selected_index]

        # 5️⃣ Tab contents
        with tab_overview:
            st.subheader("Snapshot (Realized vs Forecast)")
            ordered_cols = (
                ["ticker", "realized_vol"]
                + [c for c in preds.columns if c.startswith("pred_vol_")]
                + [c for c in ["vol_diff", "vol_direction"] if c in preds.columns]
            )
            st.dataframe(
                _safe_style_format(preds[ordered_cols], _pretty_number),
                width="stretch",
                height=380,
            )
            # Export CSV button for the overview table
            export_csv_button(
                preds[ordered_cols],
                f"volsense_forecast_{date.today()}.csv",
                "📥 Export Predictions",
            )
            # export_csv_button(preds, f"volsense_forecast_{date.today()}.csv", "Export Predictions")

        with tab_tickers:
            st.subheader("Per-Ticker Forecast vs Realized")
            
            # Text input for ticker (not dropdown)
            col1, col2 = st.columns([2, 1])
            with col1:
                ticker_input = st.text_input(
                    "Enter Ticker Symbol",
                    value="NVDA",
                    key="ticker_ta",
                    help="Enter any ticker from the V507 universe"
                ).upper().strip()
            with col2:
                horizons = _detect_horizons(preds)
                chosen_horizon = st.selectbox("Horizon", horizons, key="horizon_ta")
            
            # Validate ticker is in universe
            if ticker_input not in preds["ticker"].values:
                st.warning(f"⚠️ '{ticker_input}' not found in cached universe. Available: {len(preds['ticker'].unique())} tickers.")
                st.info("💡 Click **Refresh Market** to hydrate latest data, or check ticker spelling.")
            else:
                selected_ticker = ticker_input
                
                # Plot with dark mode styling
                if fcast is not None:
                    try:
                        fig = fcast.plot(selected_ticker, show=False)
                        st.pyplot(fig, width="stretch")
                    except Exception as e:
                        st.error(f"Could not generate plot: {e}")
                else:
                    st.info("Plot unavailable (loaded from cache). Click Refresh Market for full functionality.")
                
                # Analytics description
                if analytics is not None:
                    st.info(analytics.describe(selected_ticker, f"pred_vol_{chosen_horizon}"))
                
                # Show ticker's signal data
                st.markdown("#### 📊 Signal Metrics")
                ticker_signals = sig_df[sig_df["ticker"] == selected_ticker]
                if not ticker_signals.empty:
                    st.dataframe(ticker_signals, hide_index=True, width="stretch")
                else:
                    st.caption("No signal data available for this ticker.")

        # ──────────────────────────────────────────────────────────────────────
        # SECTOR VIEW
        # ──────────────────────────────────────────────────────────────────────
        with tab_sector:
            st.subheader("Sector Heatmap & Leaders")

            # Horizon Toggle (Radio for clearer UX)
            st.markdown("##### 📅 Forecast Horizon")
            horizon_option = st.radio(
                "Select forecast horizon for heatmap",
                ["1-Day", "5-Day", "10-Day"],
                index=1,  # Default to 5-Day
                horizontal=True,
                key="sector_horizon_radio"
            )
            horizon_col_map = {"1-Day": 1, "5-Day": 5, "10-Day": 10}
            h_sel = horizon_col_map[horizon_option]
            
            sector_map = get_sector_map("v507")

            # Build sector data for heatmap
            dsub = sig_df.copy()
            latest_date = dsub["date"].max()
            dsub = dsub[dsub["date"] == latest_date]
            
            # Filter to selected horizon
            dsub_h = dsub[dsub["horizon"] == h_sel].copy()

            # ─────────────────────────────────────────────────
            # Plotly Treemap Heatmap
            # ─────────────────────────────────────────────────
            st.markdown(f"### 🗺️ Volatility Universe Heatmap ({horizon_option} Forecast)")
            
            if len(dsub_h) > 0:
                # Prepare data for treemap
                df_tree = dsub_h.copy()
                # Use absolute z-score for sizing (larger = more extreme)
                df_tree["abs_z"] = df_tree["vol_zscore"].abs().clip(lower=0.1)
                
                fig_tree = px.treemap(
                    df_tree,
                    path=["sector", "ticker"],
                    values="abs_z",
                    color="vol_zscore",
                    color_continuous_scale="RdYlGn",
                    color_continuous_midpoint=0,
                    hover_data=["position", "regime_flag", "vol_spread"],
                    height=500,
                )
                
                fig_tree.update_layout(
                    margin=dict(t=10, l=0, r=0, b=0),
                    paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(size=10, color="white"),
                )
                
                fig_tree.update_traces(
                    hovertemplate="<b>%{label}</b><br>Z-Score: %{color:.2f}<br>Signal: %{customdata[0]}<br>Regime: %{customdata[1]}<extra></extra>",
                    textposition="middle center",
                )
                
                st.plotly_chart(fig_tree, width="stretch")
            else:
                st.info("No tickers match the current filter criteria.")

            st.divider()

            # ─────────────────────────────────────────────────
            # Sector Signal Strength Bar Chart
            # ─────────────────────────────────────────────────
            st.markdown("### 📈 Sector Signal Strength")
            
            if len(dsub_h) > 0:
                sector_strength = dsub_h.groupby("sector")["vol_zscore"].mean().reset_index()
                sector_strength = sector_strength.sort_values("vol_zscore", ascending=False)
                
                colors = ["#00FF00" if x > 0 else "#FF4444" for x in sector_strength["vol_zscore"]]
                
                fig_sector = go.Figure(data=[
                    go.Bar(
                        x=sector_strength["sector"],
                        y=sector_strength["vol_zscore"],
                        marker_color=colors,
                        text=sector_strength["vol_zscore"].round(2),
                        textposition="outside",
                    )
                ])
                
                fig_sector.update_layout(
                    height=400,
                    margin=dict(t=20, l=0, r=0, b=0),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    xaxis_title="Sector",
                    yaxis_title="Avg Z-Score",
                    showlegend=False,
                    xaxis=dict(tickangle=-45, color="white"),
                    yaxis=dict(color="white"),
                    font=dict(color="white"),
                )
                
                st.plotly_chart(fig_sector, width="stretch")
            else:
                st.info("No sector data available.")

            st.divider()
            
            # ─────────────────────────────────────────────────
            # Sector-Specific Tabs
            # ─────────────────────────────────────────────────
            st.markdown("### 📊 Sector Deep Dive")
            
            available_sectors = sorted(dsub_h["sector"].dropna().unique().tolist())
            if available_sectors:
                sector_tabs = st.tabs(available_sectors[:10])  # Limit to 10 sectors for UI
                
                for i, sector_name in enumerate(available_sectors[:10]):
                    with sector_tabs[i]:
                        sector_df = dsub_h[dsub_h["sector"] == sector_name].copy()
                        
                        # Display key metrics
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Tickers", len(sector_df))
                        col2.metric("Avg Z-Score", f"{sector_df['vol_zscore'].mean():.2f}")
                        col3.metric("Avg Vol Spread", f"{sector_df['vol_spread'].mean():.2%}" if "vol_spread" in sector_df else "N/A")
                        
                        # Signal table for this sector
                        display_cols = ["ticker", "position", "vol_zscore", "vol_spread", "regime_flag"]
                        display_cols = [c for c in display_cols if c in sector_df.columns]
                        st.dataframe(
                            sector_df[display_cols].sort_values("vol_zscore", ascending=False),
                            hide_index=True,
                            width="stretch",
                        )

        # ──────────────────────────────────────────────────────────────────────
        # SIGNAL TABLE
        # ──────────────────────────────────────────────────────────────────────
        with tab_signals:
            st.subheader("Cross-Sectional Signal Table")
            # ───────────────────────────────────────────────
            # Filters
            # ───────────────────────────────────────────────
            sectors = ["All"] + sorted(sig_df["sector"].dropna().unique().tolist())
            c1, c2, c3, c4, c5 = st.columns([1, 1, 1, 1, 1.2])
            with c1:
                sector_filter = st.selectbox(
                    "Sector", sectors, index=0, key="sig_sector"
                )
            with c2:
                horizon_filter = st.selectbox(
                    "Horizon",
                    sorted(sig_df["horizon"].unique().tolist()),
                    key="sig_horizon",
                )
            with c3:
                regime = st.selectbox(
                    "Regime",
                    ["All", "calm", "normal", "spike"],
                    index=0,
                    key="sig_regime",
                )
            with c4:
                position_filter = st.selectbox(
                    "Signal",
                    [
                        "All", 
                        "BUY_DIP", 
                        "LONG_EQUITY", 
                        "LONG_VOL_TREND", 
                        "FADE_RALLY", 
                        "SHORT_VOL", 
                        "DEFENSIVE", 
                        "LONG_TAIL_HEDGE", 
                        "NEUTRAL"
                    ],
                    index=0,
                    key="sig_position",
                )
            with c5:
                sort_by = st.selectbox(
                    "Sort by",
                    [
                        "vol_zscore",
                        "vol_spread",
                        "term_spread_10v5",
                        "rank_universe",
                        "rank_sector",
                        "ticker",
                    ],
                    key="sig_sort",
                )

            # ───────────────────────────────────────────────
            # Safe filtering logic
            # ───────────────────────────────────────────────
            table = sig_df.copy()
            if "horizon" in table.columns:
                table["horizon"] = table["horizon"].astype(str)
            horizon_filter = str(horizon_filter)
            if "sector" in table.columns:
                table["sector"] = table["sector"].astype(str)

            if sector_filter != "All" and "sector" in table.columns:
                table = table[table["sector"] == sector_filter]
            if horizon_filter != "All" and "horizon" in table.columns:
                table = table[table["horizon"] == horizon_filter]
            if regime != "All" and "regime_flag" in table.columns:
                table = table[table["regime_flag"] == regime]
            if position_filter != "All" and "position" in table.columns:
                table = table[table["position"] == position_filter]

            # ───────────────────────────────────────────────
            # Subset and display
            # ───────────────────────────────────────────────
            cols = [
                "ticker",
                "sector",
                "horizon",
                "forecast_vol",
                "today_vol",
                "vol_spread",
                "term_spread_10v5",
                "vol_zscore",
                "rank_universe",
                "rank_sector",
                "position",
                "action",
                "regime_flag",
            ]
            cols = [c for c in cols if c in table.columns]
            table = table[cols].sort_values(sort_by, ascending=False)

            if table.empty:
                st.warning("⚠️ No signals available for the selected filters.")
            else:
                st.caption(
                    f"📊 Positions: {table['position'].value_counts().to_dict()}"
                )
                st.dataframe(
                    table.style.format(
                        {
                            "forecast_vol": "{:.4f}".format,
                            "today_vol": "{:.4f}".format,
                            "vol_spread": "{:+.2%}".format,
                            "term_spread_10v5": "{:+.2%}".format,
                            "vol_zscore": "{:+.2f}".format,
                            "rank_universe": lambda x: f"{x:.2f}",
                            "rank_sector": lambda x: f"{x:.2f}",
                        }
                    ),
                    width="stretch",
                    height=520,
                )

    except Exception as e:
        st.error(f"❌ Error: {e}")

else:
    st.info(
        "Set model, tickers and press **Run Forecasts** to generate analytics & signals."
    )