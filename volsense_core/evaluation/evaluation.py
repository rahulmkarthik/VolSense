# ============================================================
# 📈 VolSense Unified Evaluation Framework
# ============================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.api import qqplot
from volsense_core.evaluation.metrics import rmse, mae, mape, r2_score


class ModelEvaluator:
    """
    Unified evaluation object for VolSense models.

    Input: standardized forecast DataFrame with columns:
        ['asof_date','date','ticker','horizon','forecast_vol','realized_vol','model']

    Provides:
        - ticker × horizon metrics
        - horizon-level summary
        - residual diagnostics
        - regime/time-slice performance
        - visualization suite
    """

    def __init__(self, eval_df: pd.DataFrame, model_name="UnknownModel"):
        self.df = eval_df.copy()
        self.model_name = model_name
        self.metrics_df = None
        self.summary_df = None

    # --------------------------------------------------------
    # 🧮 Core Metrics
    # --------------------------------------------------------
    def compute_metrics(self):
        metrics = []
        for (t, h), g in self.df.groupby(["ticker", "horizon"]):
            g = g.dropna(subset=["forecast_vol", "realized_vol"])
            if len(g) < 5:
                continue
            y_true, y_pred = g["realized_vol"].values, g["forecast_vol"].values
            resid = y_true - y_pred

            metrics.append({
                "ticker": t, "horizon": h,
                "RMSE": rmse(y_true, y_pred),
                "MAE": mae(y_true, y_pred),
                "MAPE": mape(y_true, y_pred),
                "R2": r2_score(y_true, y_pred),
                "Corr": np.corrcoef(y_true, y_pred)[0,1],
                "DW": durbin_watson(resid),
                "LjungBox_p": acorr_ljungbox(resid, lags=[10], return_df=True)["lb_pvalue"].iloc[0]
            })
        self.metrics_df = pd.DataFrame(metrics)
        return self.metrics_df

    # --------------------------------------------------------
    # 📊 Summary by Horizon
    # --------------------------------------------------------
    def summarize(self):
        if self.metrics_df is None:
            self.compute_metrics()
        self.summary_df = (
            self.metrics_df.groupby("horizon")
            .agg({"RMSE":"mean","MAE":"mean","MAPE":"mean","R2":"mean","Corr":"mean","DW":"mean"})
            .reset_index()
        )
        print(f"\n📈 Horizon-Level Summary for {self.model_name}")
        display(self.summary_df.round(4))
        return self.summary_df

    # --------------------------------------------------------
    # ⏳ Regime/Time-Slice Evaluation
    # --------------------------------------------------------
    def regime_summary(self, freq="M"):
        """
        Evaluates performance by time slices (e.g., month, quarter).
        """
        df = self.df.copy()
        df["period"] = pd.to_datetime(df["date"]).dt.to_period(freq).dt.to_timestamp()
        slices = []
        for (h, p), g in df.groupby(["horizon","period"]):
            if len(g) < 10: 
                continue
            slices.append({
                "horizon": h,
                "period": p,
                "R2": r2_score(g["realized_vol"], g["forecast_vol"]),
                "Corr": np.corrcoef(g["realized_vol"], g["forecast_vol"])[0,1],
                "RMSE": rmse(g["realized_vol"], g["forecast_vol"])
            })
        regime_df = pd.DataFrame(slices)
        plt.figure(figsize=(10,5))
        sns.lineplot(x="period", y="R2", hue="horizon", data=regime_df, marker="o")
        plt.title(f"{self.model_name}: R² Over Time (Regime Robustness)")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
        return regime_df

    # --------------------------------------------------------
    # 📈 Plotting Utilities
    # --------------------------------------------------------
    def plot_true_vs_pred(self, horizon):
        g = self.df[self.df["horizon"] == horizon].dropna(subset=["forecast_vol", "realized_vol"])
        plt.figure(figsize=(5,5))
        sns.scatterplot(x="realized_vol", y="forecast_vol", data=g, s=15, alpha=0.6)
        lims = [min(g["realized_vol"].min(), g["forecast_vol"].min()),
                max(g["realized_vol"].max(), g["forecast_vol"].max())]
        plt.plot(lims, lims, "r--")
        plt.title(f"True vs Predicted Vol – {horizon}d")
        plt.xlabel("Realized Vol")
        plt.ylabel("Forecast Vol")
        plt.show()

    def plot_residual_distribution(self, horizon):
        g = self.df[self.df["horizon"] == horizon]
        resid = g["forecast_vol"] - g["realized_vol"]
        plt.figure(figsize=(8,4))
        sns.histplot(resid, bins=40, kde=True)
        plt.title(f"Residual Distribution – {horizon}d")
        plt.show()

    def plot_qq(self, horizon):
        g = self.df[self.df["horizon"] == horizon]
        resid = g["forecast_vol"] - g["realized_vol"]
        qqplot(resid, line='45', fit=True)
        plt.title(f"QQ Plot – {horizon}d")
        plt.show()

    def plot_best_worst(self, horizon, top_n=10):
        if self.metrics_df is None:
            self.compute_metrics()
        hdf = self.metrics_df[self.metrics_df["horizon"]==horizon]
        top = hdf.nlargest(top_n, "R2")
        worst = hdf.nsmallest(top_n, "R2")
        fig, ax = plt.subplots(1,2, figsize=(12,4))
        sns.barplot(y="ticker", x="R2", data=top, ax=ax[0], color="green")
        sns.barplot(y="ticker", x="R2", data=worst, ax=ax[1], color="red")
        ax[0].set_title(f"Top {top_n} by R² – {horizon}d")
        ax[1].set_title(f"Worst {top_n} by R² – {horizon}d")
        plt.tight_layout()
        plt.show()

    # --------------------------------------------------------
    # 💾 Save / Export
    # --------------------------------------------------------
    def save_metrics(self, save_path):
        if self.metrics_df is not None:
            self.metrics_df.to_csv(save_path, index=False)
            print(f"💾 Saved tickerwise metrics to {save_path}")
        else:
            print("⚠️ Metrics not yet computed. Run .compute_metrics() first.")


    # --------------------------------------------------------
    # 🚀 Quick Evaluation Workflow
    # --------------------------------------------------------
    def run_full_evaluation(self, save_dir=None):
        print(f"\n🚀 Running full evaluation for {self.model_name}")
        self.compute_metrics()
        self.summarize()
        horizons = sorted(self.df["horizon"].unique())

        for h in horizons:
            self.plot_true_vs_pred(h)
            self.plot_residual_distribution(h)
            self.plot_qq(h)
            self.plot_best_worst(h)

        regime_df = self.regime_summary()
        if save_dir:
            self.save_metrics(f"{save_dir}/{self.model_name}_tickerwise_metrics.csv")
        print("✅ Evaluation complete.")
        return self.metrics_df, self.summary_df, regime_df