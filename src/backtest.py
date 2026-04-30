import numpy as np
import pandas as pd

from .config import DEFAULT_CLOSE_FEE, DEFAULT_OPEN_FEE


def predictions_to_frame(
    df: pd.DataFrame,
    predictions,
    confidence=None,
    open_fee: float = DEFAULT_OPEN_FEE,
    close_fee: float = DEFAULT_CLOSE_FEE,
) -> pd.DataFrame:
    out = df[["ts", "coin", "pair", "entry_price", "future_price", "future_return", "label"]].copy()
    out["prediction"] = predictions
    out["confidence"] = confidence if confidence is not None else np.nan
    out["gross_pnl"] = out["prediction"] * out["future_return"]
    out.loc[out["prediction"] == 0, "gross_pnl"] = 0.0
    out["fee"] = np.where(out["prediction"] != 0, open_fee + close_fee, 0.0)
    out["net_pnl"] = out["gross_pnl"] - out["fee"]
    out["is_trade"] = out["prediction"] != 0
    out["win"] = out["net_pnl"] > 0
    return out


def summarize_trades(trades: pd.DataFrame) -> dict:
    active = trades[trades["is_trade"]].copy()
    if active.empty:
        return {
            "trades": 0,
            "net_win_rate": 0.0,
            "avg_net_pnl": 0.0,
            "total_net_pnl": 0.0,
            "max_drawdown": 0.0,
            "profit_factor": 0.0,
        }

    equity = active["net_pnl"].cumsum()
    running_max = equity.cummax()
    drawdown = equity - running_max
    gross_profit = active.loc[active["net_pnl"] > 0, "net_pnl"].sum()
    gross_loss = active.loc[active["net_pnl"] < 0, "net_pnl"].sum()

    return {
        "trades": int(len(active)),
        "net_win_rate": float(active["win"].mean()),
        "avg_net_pnl": float(active["net_pnl"].mean()),
        "total_net_pnl": float(active["net_pnl"].sum()),
        "max_drawdown": float(drawdown.min()),
        "profit_factor": float(gross_profit / abs(gross_loss)) if gross_loss != 0 else float("inf"),
    }


def summarize_by_pair(trades: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for pair, group in trades.groupby("pair"):
        row = summarize_trades(group)
        row["pair"] = pair
        rows.append(row)
    return pd.DataFrame(rows).sort_values("total_net_pnl", ascending=False)


def summarize_by_month(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()

    rows = []
    dated = trades.copy()
    dated["month"] = pd.to_datetime(dated["ts"]).dt.to_period("M").astype(str)
    for month, group in dated.groupby("month"):
        row = summarize_trades(group)
        row["month"] = month
        rows.append(row)
    return pd.DataFrame(rows).sort_values("month")


def apply_confidence_threshold(trades: pd.DataFrame, threshold: float) -> pd.DataFrame:
    out = trades.copy()
    if "confidence" not in out.columns:
        out["confidence"] = np.nan
    low_confidence = out["confidence"].fillna(0.0) < threshold
    out.loc[low_confidence, "prediction"] = 0
    out.loc[low_confidence, "gross_pnl"] = 0.0
    out.loc[low_confidence, "fee"] = 0.0
    out.loc[low_confidence, "net_pnl"] = 0.0
    out["is_trade"] = out["prediction"] != 0
    out["win"] = out["net_pnl"] > 0
    return out


def threshold_summaries(trades: pd.DataFrame, thresholds: list[float]) -> pd.DataFrame:
    rows = []
    for threshold in thresholds:
        filtered = apply_confidence_threshold(trades, threshold)
        row = summarize_trades(filtered)
        row["threshold"] = threshold
        rows.append(row)
    return pd.DataFrame(rows)


def select_threshold(
    trades: pd.DataFrame,
    thresholds: list[float],
    min_trades: int = 100,
    metric: str = "total_net_pnl",
) -> tuple[float, pd.Series, pd.DataFrame]:
    summaries = threshold_summaries(trades, thresholds)
    eligible = summaries[summaries["trades"] >= min_trades].copy()
    if eligible.empty:
        eligible = summaries.copy()
    if metric not in eligible.columns:
        raise ValueError(f"Unknown threshold metric: {metric}")

    eligible = eligible.sort_values(
        [metric, "profit_factor", "avg_net_pnl", "trades"],
        ascending=[False, False, False, False],
    )
    best = eligible.iloc[0]
    return float(best["threshold"]), best, summaries
