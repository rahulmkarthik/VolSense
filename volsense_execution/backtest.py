# ============================================================
# 🎯 VolSense Backtesting Engine
# ============================================================
"""
Historical backtesting framework for VolSense directional signals.

Evaluates trading strategies based on:
  1) Volatility direction predictions (vol up/down)
  2) Position signals from SignalEngine (LONG_VOL, SHORT_VOL, etc.)

Provides:
  - Cumulative P&L curves
  - Risk metrics: Sharpe, Sortino, Max Drawdown
  - Win rate and profit factor
  - Regime-based performance breakdown
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Literal
from datetime import datetime


@dataclass
class BacktestConfig:
    """Configuration for backtesting parameters.
    
    :param initial_capital: Starting capital for the strategy.
    :param position_size: Fraction of capital to allocate per trade (0.0 to 1.0).
    :param transaction_cost_bps: Transaction cost in basis points per trade.
    :param risk_free_rate: Annualized risk-free rate for Sharpe calculation.
    :param vol_proxy_ticker: Ticker to use for volatility trading (e.g., UVXY, VXX).
    :param equity_proxy_ticker: Ticker to use for equity exposure (e.g., SPY).
    """
    initial_capital: float = 100_000.0
    position_size: float = 1.0
    transaction_cost_bps: float = 5.0  # 5 bps per trade
    risk_free_rate: float = 0.04  # 4% annual
    vol_proxy_ticker: str = "UVXY"
    equity_proxy_ticker: str = "SPY"


@dataclass
class BacktestResult:
    """Container for backtest results and metrics.
    
    :param pnl_series: Daily P&L series.
    :param cumulative_returns: Cumulative return series.
    :param metrics: Dictionary of performance metrics.
    :param trades: DataFrame of individual trades.
    :param daily_positions: DataFrame of daily position states.
    """
    pnl_series: pd.Series
    cumulative_returns: pd.Series
    metrics: Dict[str, float]
    trades: pd.DataFrame
    daily_positions: pd.DataFrame
    

class VolatilityDirectionBacktest:
    """
    Backtest volatility direction predictions.
    
    Strategy: 
      - Long volatility (UVXY) when model predicts vol increase
      - Short volatility (SVXY equivalent) when model predicts vol decrease
    
    :param eval_df: Evaluation DataFrame with columns:
        ['date', 'ticker', 'horizon', 'forecast_vol', 'realized_vol', 'today_vol']
    :param config: Backtesting configuration parameters.
    """
    
    def __init__(
        self, 
        eval_df: pd.DataFrame, 
        config: Optional[BacktestConfig] = None
    ):
        self.df = eval_df.copy()
        self.config = config or BacktestConfig()
        self.result: Optional[BacktestResult] = None
        
    def _derive_direction_signals(self, horizon: int = 1) -> pd.DataFrame:
        """
        Derive direction signals from regression predictions.
        
        :param horizon: Forecast horizon to use for direction.
        :return: DataFrame with direction signals.
        """
        df = self.df[self.df["horizon"] == horizon].copy()
        df = df.sort_values("date")
        
        # Use today_vol as baseline if available, else use lagged realized_vol
        if "today_vol" in df.columns and df["today_vol"].notna().any():
            baseline = df["today_vol"]
        else:
            baseline = df.groupby("ticker")["realized_vol"].shift(1)
        
        df["baseline_vol"] = baseline
        df["pred_direction"] = (df["forecast_vol"] > df["baseline_vol"]).astype(int)
        df["true_direction"] = (df["realized_vol"] > df["baseline_vol"]).astype(int)
        
        # Confidence score = magnitude of predicted change
        df["confidence"] = (df["forecast_vol"] - df["baseline_vol"]).abs()
        
        return df.dropna(subset=["baseline_vol", "pred_direction"])
    
    def _compute_vol_proxy_returns(self, direction_df: pd.DataFrame) -> pd.DataFrame:
        """
        Simulate returns from a volatility direction strategy.
        
        Assumes:
          - Long vol proxy (UVXY-like): +1 when pred_direction=1, gains if vol rises
          - Short vol proxy (inverse): -1 when pred_direction=0, gains if vol falls
        
        Proxy return approximation: use realized vol change as return proxy.
        In practice, you'd use actual VIX futures or UVXY returns.
        
        :param direction_df: DataFrame with direction signals.
        :return: DataFrame with simulated strategy returns.
        """
        df = direction_df.copy()
        
        # Simulated "vol return" = % change in realized vol (simplified proxy)
        # Positive when vol increases, negative when it decreases
        df["vol_return"] = (df["realized_vol"] - df["baseline_vol"]) / (df["baseline_vol"] + 1e-8)
        
        # Position: +1 = long vol, -1 = short vol
        df["position"] = df["pred_direction"].replace({1: 1, 0: -1})
        
        # Strategy return = position * vol_return
        # If we predict vol up (+1) and vol goes up (positive return), we profit
        # If we predict vol down (-1) and vol goes down (negative return), we profit
        df["strategy_return"] = df["position"] * df["vol_return"]
        
        # Apply transaction costs when position changes
        df["position_change"] = df["position"].diff().abs()
        cost_per_trade = self.config.transaction_cost_bps / 10000
        df["transaction_cost"] = df["position_change"].fillna(0) * cost_per_trade
        
        df["net_return"] = df["strategy_return"] - df["transaction_cost"]
        
        return df
    
    def run_direction_backtest(
        self, 
        horizon: int = 1,
        aggregation: Literal["mean", "weighted"] = "mean"
    ) -> BacktestResult:
        """
        Run the volatility direction backtest.
        
        :param horizon: Forecast horizon to backtest.
        :param aggregation: How to aggregate across tickers ('mean' or 'weighted' by confidence).
        :return: BacktestResult with P&L, metrics, and trade log.
        """
        print(f"🎯 Running Direction Backtest (Horizon: {horizon}d)")
        
        # Derive signals
        direction_df = self._derive_direction_signals(horizon)
        
        # Compute per-ticker returns
        returns_df = self._compute_vol_proxy_returns(direction_df)
        
        # Aggregate across tickers per day
        if aggregation == "weighted":
            # Weight by confidence (predicted magnitude)
            daily = returns_df.groupby("date").apply(
                lambda g: np.average(g["net_return"], weights=g["confidence"] + 1e-8)
            )
        else:
            daily = returns_df.groupby("date")["net_return"].mean()
        
        daily = daily.sort_index()
        
        # Cumulative returns
        cumulative = (1 + daily).cumprod()
        
        # Compute metrics
        metrics = self._compute_metrics(daily, cumulative)
        
        # Trade log
        trades = returns_df[["date", "ticker", "position", "pred_direction", 
                            "true_direction", "strategy_return", "net_return"]].copy()
        trades["correct"] = (trades["pred_direction"] == trades["true_direction"]).astype(int)
        
        self.result = BacktestResult(
            pnl_series=daily,
            cumulative_returns=cumulative,
            metrics=metrics,
            trades=trades,
            daily_positions=returns_df
        )
        
        return self.result
    
    def _compute_metrics(self, returns: pd.Series, cumulative: pd.Series) -> Dict[str, float]:
        """
        Compute performance metrics.
        
        :param returns: Daily returns series.
        :param cumulative: Cumulative returns series.
        :return: Dictionary of performance metrics.
        """
        # Basic stats
        total_return = cumulative.iloc[-1] - 1 if len(cumulative) > 0 else 0
        n_days = len(returns)
        
        # Annualized return (assume 252 trading days)
        ann_return = (1 + total_return) ** (252 / max(n_days, 1)) - 1
        
        # Volatility
        daily_vol = returns.std()
        ann_vol = daily_vol * np.sqrt(252)
        
        # Sharpe Ratio
        excess_return = ann_return - self.config.risk_free_rate
        sharpe = excess_return / (ann_vol + 1e-8)
        
        # Sortino Ratio (downside deviation)
        downside = returns[returns < 0].std()
        sortino = excess_return / (downside * np.sqrt(252) + 1e-8)
        
        # Max Drawdown
        peak = cumulative.cummax()
        drawdown = (cumulative - peak) / peak
        max_drawdown = drawdown.min()
        
        # Win Rate
        n_winning = (returns > 0).sum()
        n_total = len(returns)
        win_rate = n_winning / max(n_total, 1)
        
        # Profit Factor
        gross_profit = returns[returns > 0].sum()
        gross_loss = returns[returns < 0].abs().sum()
        profit_factor = gross_profit / (gross_loss + 1e-8)
        
        # Calmar Ratio
        calmar = ann_return / (abs(max_drawdown) + 1e-8)
        
        return {
            "total_return": total_return,
            "annualized_return": ann_return,
            "annualized_volatility": ann_vol,
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "max_drawdown": max_drawdown,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "calmar_ratio": calmar,
            "n_trading_days": n_days,
        }
    
    def summary(self) -> pd.DataFrame:
        """
        Print and return a summary of backtest results.
        
        :return: DataFrame with metrics.
        """
        if self.result is None:
            raise RuntimeError("Run run_direction_backtest() first.")
        
        metrics = self.result.metrics
        
        print("\n" + "=" * 60)
        print("📊 VOLATILITY DIRECTION BACKTEST RESULTS")
        print("=" * 60)
        
        summary_data = {
            "Metric": [
                "Total Return",
                "Annualized Return",
                "Annualized Volatility",
                "Sharpe Ratio",
                "Sortino Ratio",
                "Max Drawdown",
                "Win Rate",
                "Profit Factor",
                "Calmar Ratio",
                "Trading Days",
            ],
            "Value": [
                f"{metrics['total_return']:.2%}",
                f"{metrics['annualized_return']:.2%}",
                f"{metrics['annualized_volatility']:.2%}",
                f"{metrics['sharpe_ratio']:.2f}",
                f"{metrics['sortino_ratio']:.2f}",
                f"{metrics['max_drawdown']:.2%}",
                f"{metrics['win_rate']:.2%}",
                f"{metrics['profit_factor']:.2f}",
                f"{metrics['calmar_ratio']:.2f}",
                f"{int(metrics['n_trading_days'])}",
            ]
        }
        
        df = pd.DataFrame(summary_data)
        print(df.to_string(index=False))
        print("=" * 60)
        
        return df
    
    def plot_cumulative_returns(self, benchmark: Optional[pd.Series] = None):
        """
        Plot cumulative returns curve.
        
        :param benchmark: Optional benchmark returns series for comparison.
        """
        if self.result is None:
            raise RuntimeError("Run run_direction_backtest() first.")
        
        plt.figure(figsize=(12, 6))
        
        cum_ret = self.result.cumulative_returns
        plt.plot(cum_ret.index, cum_ret.values, label="Vol Direction Strategy", linewidth=2)
        
        if benchmark is not None:
            plt.plot(benchmark.index, (1 + benchmark).cumprod().values, 
                    label="Benchmark", linewidth=1.5, alpha=0.7)
        
        plt.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
        plt.fill_between(cum_ret.index, 1, cum_ret.values, 
                        where=cum_ret.values >= 1, alpha=0.3, color='green')
        plt.fill_between(cum_ret.index, 1, cum_ret.values, 
                        where=cum_ret.values < 1, alpha=0.3, color='red')
        
        plt.title("Cumulative Returns: Volatility Direction Strategy")
        plt.xlabel("Date")
        plt.ylabel("Cumulative Return")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def plot_drawdown(self):
        """Plot drawdown chart."""
        if self.result is None:
            raise RuntimeError("Run run_direction_backtest() first.")
        
        cum_ret = self.result.cumulative_returns
        peak = cum_ret.cummax()
        drawdown = (cum_ret - peak) / peak
        
        plt.figure(figsize=(12, 4))
        plt.fill_between(drawdown.index, 0, drawdown.values, color='red', alpha=0.5)
        plt.plot(drawdown.index, drawdown.values, color='darkred', linewidth=1)
        plt.title("Drawdown Over Time")
        plt.xlabel("Date")
        plt.ylabel("Drawdown")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def plot_monthly_returns(self):
        """Plot monthly returns heatmap."""
        if self.result is None:
            raise RuntimeError("Run run_direction_backtest() first.")
        
        returns = self.result.pnl_series.copy()
        returns.index = pd.to_datetime(returns.index)
        
        # Resample to monthly
        monthly = returns.resample("M").apply(lambda x: (1 + x).prod() - 1)
        
        # Create year x month matrix
        monthly_df = pd.DataFrame({
            "year": monthly.index.year,
            "month": monthly.index.month,
            "return": monthly.values
        })
        
        pivot = monthly_df.pivot(index="year", columns="month", values="return")
        pivot.columns = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", 
                        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"][:len(pivot.columns)]
        
        plt.figure(figsize=(12, 6))
        sns.heatmap(pivot, annot=True, fmt=".1%", cmap="RdYlGn", center=0,
                   cbar_kws={"label": "Monthly Return"})
        plt.title("Monthly Returns Heatmap")
        plt.tight_layout()
        plt.show()
        
        return pivot


class SignalBasedBacktest:
    """
    Backtest trading signals from SignalEngine.
    
    Converts position classifications (LONG_VOL_TREND, DEFENSIVE, etc.)
    into portfolio positions and evaluates performance.
    
    :param signals_df: Output from SignalEngine.compute_signals()
    :param price_data: DataFrame with columns ['date', 'ticker', 'close'] for P&L calculation.
    :param config: Backtesting configuration.
    """
    
    def __init__(
        self,
        signals_df: pd.DataFrame,
        price_data: Optional[pd.DataFrame] = None,
        config: Optional[BacktestConfig] = None
    ):
        self.signals = signals_df.copy()
        self.price_data = price_data
        self.config = config or BacktestConfig()
        self.result: Optional[BacktestResult] = None
        
    def _signal_to_position(self, signal: str) -> int:
        """
        Map signal classification to position direction.
        
        :param signal: Position classification from SignalEngine.
        :return: Position direction (-1, 0, +1).
        """
        bullish = {"LONG_EQUITY", "BUY_DIP", "LONG_VOL_TREND"}
        bearish = {"DEFENSIVE", "FADE_RALLY", "SHORT_VOL"}
        
        if signal in bullish:
            return 1
        elif signal in bearish:
            return -1
        else:
            return 0  # NEUTRAL, LONG_TAIL_HEDGE
    
    def run_signal_backtest(self, horizon: int = 5) -> BacktestResult:
        """
        Run backtest based on SignalEngine position classifications.
        
        :param horizon: Forecast horizon to use.
        :return: BacktestResult with metrics.
        """
        print(f"🎯 Running Signal-Based Backtest (Horizon: {horizon}d)")
        
        df = self.signals[self.signals["horizon"] == horizon].copy()
        df = df.sort_values("date")
        
        # Convert signals to positions
        df["position_dir"] = df["position"].apply(self._signal_to_position)
        
        # Use vol_spread as proxy for return (simplified)
        # In practice, you'd use actual price returns
        if "vol_spread" in df.columns:
            df["proxy_return"] = df["vol_spread"].clip(-0.5, 0.5) / 10  # Scale down
        else:
            df["proxy_return"] = 0.0
        
        # Strategy return = position * proxy return
        df["strategy_return"] = df["position_dir"] * df["proxy_return"]
        
        # Aggregate by date
        daily = df.groupby("date")["strategy_return"].mean()
        daily = daily.sort_index()
        
        cumulative = (1 + daily).cumprod()
        
        # Compute metrics
        metrics = self._compute_metrics(daily, cumulative)
        
        # Signal distribution
        signal_counts = df["position"].value_counts()
        print(f"\n📊 Signal Distribution:")
        print(signal_counts.to_string())
        
        self.result = BacktestResult(
            pnl_series=daily,
            cumulative_returns=cumulative,
            metrics=metrics,
            trades=df[["date", "ticker", "position", "position_dir", "strategy_return"]],
            daily_positions=df
        )
        
        return self.result
    
    def _compute_metrics(self, returns: pd.Series, cumulative: pd.Series) -> Dict[str, float]:
        """Compute performance metrics (same as VolatilityDirectionBacktest)."""
        total_return = cumulative.iloc[-1] - 1 if len(cumulative) > 0 else 0
        n_days = len(returns)
        
        ann_return = (1 + total_return) ** (252 / max(n_days, 1)) - 1
        daily_vol = returns.std()
        ann_vol = daily_vol * np.sqrt(252)
        
        excess_return = ann_return - self.config.risk_free_rate
        sharpe = excess_return / (ann_vol + 1e-8)
        
        downside = returns[returns < 0].std()
        sortino = excess_return / (downside * np.sqrt(252) + 1e-8)
        
        peak = cumulative.cummax()
        drawdown = (cumulative - peak) / peak
        max_drawdown = drawdown.min()
        
        win_rate = (returns > 0).sum() / max(len(returns), 1)
        
        gross_profit = returns[returns > 0].sum()
        gross_loss = returns[returns < 0].abs().sum()
        profit_factor = gross_profit / (gross_loss + 1e-8)
        
        return {
            "total_return": total_return,
            "annualized_return": ann_return,
            "annualized_volatility": ann_vol,
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "max_drawdown": max_drawdown,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "n_trading_days": n_days,
        }
    
    def summary(self) -> pd.DataFrame:
        """Print and return metrics summary."""
        if self.result is None:
            raise RuntimeError("Run run_signal_backtest() first.")
        
        metrics = self.result.metrics
        
        print("\n" + "=" * 60)
        print("📊 SIGNAL-BASED BACKTEST RESULTS")
        print("=" * 60)
        
        summary_data = {
            "Metric": [
                "Total Return",
                "Annualized Return",
                "Sharpe Ratio",
                "Max Drawdown",
                "Win Rate",
            ],
            "Value": [
                f"{metrics['total_return']:.2%}",
                f"{metrics['annualized_return']:.2%}",
                f"{metrics['sharpe_ratio']:.2f}",
                f"{metrics['max_drawdown']:.2%}",
                f"{metrics['win_rate']:.2%}",
            ]
        }
        
        df = pd.DataFrame(summary_data)
        print(df.to_string(index=False))
        print("=" * 60)
        
        return df


def run_backtest_from_evaluator(eval_df: pd.DataFrame, horizon: int = 1) -> BacktestResult:
    """
    Convenience function to run direction backtest from ModelEvaluator output.
    
    :param eval_df: DataFrame from model evaluation with forecast_vol and realized_vol.
    :param horizon: Forecast horizon to backtest.
    :return: BacktestResult.
    
    Example usage:
        >>> from volsense_core.evaluation.evaluation import ModelEvaluator
        >>> from volsense_execution.backtest import run_backtest_from_evaluator
        >>> 
        >>> evaluator = ModelEvaluator(eval_df, "VolNetX")
        >>> result = run_backtest_from_evaluator(evaluator.df, horizon=1)
        >>> print(result.metrics)
    """
    bt = VolatilityDirectionBacktest(eval_df)
    result = bt.run_direction_backtest(horizon=horizon)
    bt.summary()
    bt.plot_cumulative_returns()
    bt.plot_drawdown()
    return result
