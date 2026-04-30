import numpy as np
import pandas as pd

from .config import DEFAULT_BUFFER, DEFAULT_CLOSE_FEE, DEFAULT_OPEN_FEE, LABEL_LONG, LABEL_NO_TRADE, LABEL_SHORT


def add_trade_labels(
    df: pd.DataFrame,
    horizon: int = 4,
    open_fee: float = DEFAULT_OPEN_FEE,
    close_fee: float = DEFAULT_CLOSE_FEE,
    buffer: float = DEFAULT_BUFFER,
    label_mode: str = "trade",
) -> pd.DataFrame:
    out = df.sort_values(["pair", "ts"]).copy()
    out["entry_price"] = out["price"]
    out["future_price"] = out.groupby("pair")["price"].shift(-horizon)
    out["future_return"] = out["future_price"] / out["entry_price"] - 1.0

    if label_mode not in {"trade", "direction"}:
        raise ValueError(f"Unknown label_mode: {label_mode}")

    threshold = open_fee + close_fee + buffer if label_mode == "trade" else 0.0
    out["label"] = LABEL_NO_TRADE
    out.loc[out["future_return"] > threshold, "label"] = LABEL_LONG
    out.loc[out["future_return"] < -threshold, "label"] = LABEL_SHORT
    return out.dropna(subset=["future_return"]).reset_index(drop=True)


def label_distribution(df: pd.DataFrame) -> pd.DataFrame:
    counts = df["label"].value_counts(dropna=False).sort_index()
    total = counts.sum()
    return pd.DataFrame({"count": counts, "pct": counts / total if total else np.nan})
