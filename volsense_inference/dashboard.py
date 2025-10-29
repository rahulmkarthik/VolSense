# dashboard.py
import streamlit as st
import pandas as pd
import numpy as np
from datetime import date
from typing import List, Optional

# VolSense imports
from volsense_inference.forecast_engine import Forecast  # runs models + features  📦
from volsense_inference.analytics import Analytics  # cross-sectional analytics
from volsense_inference.signal_engine import SignalEngine  # sector-aware signals
from volsense_inference.sector_mapping import get_sector_map, get_color

# ──────────────────────────────────────────────────────────────────────────────
# Streamlit page config
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="VolSense — Volatility Insights",
    page_icon="📈",
    layout="wide",
)

# ─────────────────────────────────────────────────────────────
# ⚙️ Session state setup
# ─────────────────────────────────────────────────────────────
if "forecast_data" not in st.session_state:
    st.session_state.forecast_data = None
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
    return sorted({t.strip().upper() for t in s.split(",") if t.strip()})


def _safe_style_format(df: pd.DataFrame, formatter):
    """
    Apply formatter only to numeric columns.
    Avoids errors when non-numeric data (e.g., strings) exist.
    """
    num_cols = df.select_dtypes(include=[np.number]).columns
    format_dict = {c: formatter for c in num_cols}
    return df.style.format(format_dict)


def _detect_horizons(df: pd.DataFrame) -> List[int]:
    cols = [c for c in df.columns if c.startswith("pred_vol_")]
    hs = []
    for c in cols:
        try:
            hs.append(int(c.split("_")[-1].replace("d", "")))
        except Exception:
            pass
    return sorted(set(hs))


@st.cache_data(show_spinner=False, ttl=60 * 30)
def run_volsense(
    model_version: str, checkpoints_dir: str, start: str, tickers: List[str]
):
    fcast = Forecast(
        model_version=model_version, checkpoints_dir=checkpoints_dir, start=start
    )
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


def _pretty_number(x, nd=3):
    if pd.isna(x):
        return ""
    return f"{x:.{nd}f}"


# ──────────────────────────────────────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────────────────────────────────────
st.sidebar.title("⚡ VolSense Dashboard")
st.sidebar.caption("Trader-facing volatility analytics & signals")

with st.sidebar:
    model_version = st.text_input(
        "Model version", value="v109"
    )  # matches your checkpoints naming
    checkpoints_dir = st.text_input("Checkpoints directory", value="models")
    default_tickers = "AAPL, MSFT, NVDA, TSLA, JPM, XOM, TLT, GLD, SPY, QQQ"
    tickers_str = st.text_area(
        "Tickers (comma-separated)", value=default_tickers, height=90
    )
    start_date = st.date_input("Start fetch (for features)", value=date(2005, 1, 1))
    run_btn = st.button("🚀 Run Forecasts", type="primary", use_container_width=True)

# ──────────────────────────────────────────────────────────────────────────────
# Header
# ──────────────────────────────────────────────────────────────────────────────
st.title("📈 VolSense — Sector & Ticker Volatility Insights")
st.markdown(
    "Real-time cross-sectional analytics and sector-aware signals. "
    "Use the sidebar to select model/tickers and generate insights."
)

# ──────────────────────────────────────────────────────────────────────────────
# Main workflow
# ──────────────────────────────────────────────────────────────────────────────

if run_btn or st.session_state.forecast_data is not None:
    try:
        # 1️⃣ Load cached or run new forecast
        if not run_btn:
            fcast = st.session_state.forecast_data["fcast"]
            preds = st.session_state.forecast_data["preds"]
            analytics = st.session_state.forecast_data["analytics"]
            ae_summary = st.session_state.forecast_data["ae_summary"]
            se = st.session_state.forecast_data["se"]
            sig_df = st.session_state.forecast_data["sig_df"]
        else:
            tickers = _parse_tickers(tickers_str)
            if not tickers:
                st.warning("Please provide at least one ticker.")
                st.stop()

            with st.spinner("Running VolSense forecasts and computing signals…"):
                fcast, preds, analytics, ae_summary, se, sig_df = run_volsense(
                    model_version=model_version,
                    checkpoints_dir=checkpoints_dir,
                    start=str(start_date),
                    tickers=tickers,
                )
            st.session_state.forecast_data = {
                "fcast": fcast,
                "preds": preds,
                "analytics": analytics,
                "ae_summary": ae_summary,
                "se": se,
                "sig_df": sig_df,
            }

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
                use_container_width=True,
                height=380,
            )

        with tab_tickers:
            st.subheader("Per-Ticker Forecast vs Realized")
            horizons = _detect_horizons(preds)
            selected_ticker = st.selectbox(
                "Ticker", preds["ticker"].unique(), key="ticker_ta"
            )
            chosen_horizon = st.selectbox("Horizon", horizons, key="horizon_ta")

            fig = fcast.plot(selected_ticker, show=False)
            st.pyplot(fig, use_container_width=True)
            st.info(analytics.describe(selected_ticker, f"pred_vol_{chosen_horizon}"))

        # ──────────────────────────────────────────────────────────────────────
        # SECTOR VIEW
        # ──────────────────────────────────────────────────────────────────────
        with tab_sector:
            st.subheader("Sector Heatmap & Leaders")

            # Controls
            all_horizons = sorted(sig_df["horizon"].unique().tolist())
            h_sel = st.select_slider(
                "Horizon", options=all_horizons, value=all_horizons[0]
            )
            sector_map = get_sector_map(model_version)

            # Build sector pivot for heatmap (sector z per horizon)
            dsub = sig_df.copy()
            # latest date snapshot (SignalEngine standardizes date)   :contentReference[oaicite:8]{index=8}
            latest_date = dsub["date"].max()
            dsub = dsub[dsub["date"] == latest_date]

            pivot = (
                dsub.pivot_table(
                    index="sector", columns="horizon", values="sector_z", aggfunc="mean"
                )
                .fillna(0.0)
                .sort_index()
            )

            # Plot with st.pyplot (simple heatmap using matplotlib)
            import matplotlib.pyplot as plt
            import seaborn as sns

            fig_hm, ax = plt.subplots(figsize=(8, 5))
            sns.heatmap(
                pivot,
                cmap="coolwarm",
                center=0,
                annot=True,
                fmt=".2f",
                cbar_kws={"label": "Sector Z-score"},
                ax=ax,
            )
            ax.set_title(f"Sector Volatility Heatmap  —  {latest_date.date()}")
            st.pyplot(fig_hm, use_container_width=True)

            # Top sectors by mean z at selected horizon
            top_n = st.slider("Top sectors to display", 3, 12, 8)
            sec_slice = (
                dsub[dsub["horizon"] == h_sel]
                .groupby("sector")["sector_z"]
                .mean()
                .sort_values(ascending=False)
            )
            top = sec_slice.head(top_n)

            # Horizontal bar with sector colors
            colors = [get_color(s) for s in top.index]
            fig_bar, axb = plt.subplots(figsize=(6, 0.45 * len(top) + 1.5))
            axb.barh(top.index, top.values, color=colors)
            axb.invert_yaxis()
            axb.set_xlabel("Mean Sector Z-score")
            axb.set_title(f"Top {top_n} Sectors (H={h_sel}d)")
            st.pyplot(fig_bar, use_container_width=True)

        # ──────────────────────────────────────────────────────────────────────
        # SIGNAL TABLE
        # ──────────────────────────────────────────────────────────────────────
        with tab_signals:
            st.subheader("Cross-Sectional Signal Table")
            # Filters
            sectors = ["All"] + sorted(sig_df["sector"].dropna().unique().tolist())
            c1, c2, c3, c4 = st.columns([1, 1, 1, 1.4])
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
                sort_by = st.selectbox(
                    "Sort by",
                    [
                        "vol_zscore",
                        "vol_spread",
                        "rank_universe",
                        "rank_sector",
                        "ticker",
                    ],
                    key="sig_sort",
                )

            # ────────────────────────────────────────────────────────
            # Safe filtering logic for Signal Table
            # ────────────────────────────────────────────────────────
            table = sig_df.copy()

            # Normalize types
            if "horizon" in table.columns:
                table["horizon"] = table["horizon"].astype(str)
            horizon_filter = str(horizon_filter)

            if "sector" in table.columns:
                table["sector"] = table["sector"].astype(str)

            # Apply filters safely
            if sector_filter != "All" and "sector" in table.columns:
                table = table[table["sector"] == sector_filter]

            if horizon_filter != "All" and "horizon" in table.columns:
                table = table[table["horizon"] == horizon_filter]

            if regime != "All" and "regime_flag" in table.columns:
                table = table[table["regime_flag"] == regime]

            # Subset and display
            cols = [
                "ticker",
                "sector",
                "horizon",
                "forecast_vol",
                "today_vol",
                "vol_spread",
                "vol_zscore",
                "rank_universe",
                "rank_sector",
                "position",
                "regime_flag",
            ]
            cols = [c for c in cols if c in table.columns]
            table = table[cols].sort_values(sort_by, ascending=False)

            if table.empty:
                st.warning("⚠️ No signals available for the selected filters.")
            else:
                st.dataframe(
                    table.style.format(
                        {
                            "forecast_vol": "{:.4f}".format,
                            "today_vol": "{:.4f}".format,
                            "vol_spread": "{:+.2%}".format,
                            "vol_zscore": "{:+.2f}".format,
                            "rank_universe": lambda x: f"{x:.2f}",
                            "rank_sector": lambda x: f"{x:.2f}",
                        }
                    ),
                    use_container_width=True,
                    height=520,
                )

    except Exception as e:
        st.error(f"❌ Error: {e}")

else:
    st.info(
        "Set model, tickers and press **Run Forecasts** to generate analytics & signals."
    )
