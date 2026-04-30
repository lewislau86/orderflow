import numpy as np
import pandas as pd


BASE_WINDOWS = (3, 5, 10, 20, 50)
PRICE_STATE_WINDOWS = (14, 20, 50)


def _rsi(close: pd.Series, window: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window, min_periods=max(3, window // 3)).mean()
    loss = (-delta.clip(upper=0)).rolling(window, min_periods=max(3, window // 3)).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _add_price_state_features(group: pd.DataFrame) -> pd.DataFrame:
    close = group["price"]
    high = group["high"]
    low = group["low"]
    prev_close = close.shift(1)
    true_range = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    features = {}
    for window in PRICE_STATE_WINDOWS:
        min_periods = max(3, window // 3)
        rolling_close = close.rolling(window, min_periods=min_periods)
        mid = rolling_close.mean()
        std = rolling_close.std()
        upper = mid + 2 * std
        lower = mid - 2 * std
        band_range = (upper - lower).replace(0, np.nan)

        features[f"boll_mid_{window}"] = mid
        features[f"boll_upper_{window}"] = upper
        features[f"boll_lower_{window}"] = lower
        features[f"boll_percent_b_{window}"] = (close - lower) / band_range
        features[f"boll_band_width_{window}"] = band_range / mid.replace(0, np.nan)
        features[f"distance_to_boll_upper_{window}"] = (close - upper) / close.replace(0, np.nan)
        features[f"distance_to_boll_mid_{window}"] = (close - mid) / close.replace(0, np.nan)
        features[f"distance_to_boll_lower_{window}"] = (close - lower) / close.replace(0, np.nan)

        atr = true_range.rolling(window, min_periods=min_periods).mean()
        features[f"atr_{window}"] = atr
        features[f"atr_pct_{window}"] = atr / close.replace(0, np.nan)
        features[f"realized_vol_{window}"] = group["price_return"].rolling(window, min_periods=min_periods).std()

        rolling_high = high.rolling(window, min_periods=min_periods).max()
        rolling_low = low.rolling(window, min_periods=min_periods).min()
        features[f"rolling_range_pct_{window}"] = (rolling_high - rolling_low) / close.replace(0, np.nan)
        features[f"drawdown_from_high_{window}"] = close / rolling_high.replace(0, np.nan) - 1
        features[f"rally_from_low_{window}"] = close / rolling_low.replace(0, np.nan) - 1

        ma = close.rolling(window, min_periods=min_periods).mean()
        ma_slope = ma / ma.shift(window).replace(0, np.nan) - 1
        features[f"ma_{window}"] = ma
        features[f"ma_slope_{window}"] = ma_slope
        features[f"distance_to_ma_{window}"] = close / ma.replace(0, np.nan) - 1
        features[f"trend_strength_{window}"] = ma_slope.abs() / features[f"realized_vol_{window}"].replace(0, np.nan)

        features[f"rsi_{window}"] = _rsi(close, window)
        features[f"stochastic_position_{window}"] = (close - rolling_low) / (rolling_high - rolling_low).replace(0, np.nan)
        features[f"rolling_rank_{window}"] = close.rolling(window, min_periods=min_periods).rank(pct=True)

    return pd.concat([group, pd.DataFrame(features, index=group.index)], axis=1)


def _add_pair_features(group: pd.DataFrame) -> pd.DataFrame:
    group = group.sort_values("ts").copy()
    group["price_return"] = group["price"].pct_change()
    group["oi_return"] = group["oi"].pct_change()
    group["cvd_diff"] = group["cvd"].diff()
    group["cvd_pct_change"] = group["cvd_diff"] / group["cvd"].shift(1).abs().replace(0, np.nan)
    group["taker_imbalance"] = (
        (group["taker_buy_vol"] - group["taker_sell_vol"])
        / (group["taker_buy_vol"] + group["taker_sell_vol"]).replace(0, np.nan)
    )

    group["price_up_cvd_down"] = ((group["price_return"] > 0) & (group["cvd_diff"] < 0)).astype(float)
    group["price_down_cvd_up"] = ((group["price_return"] < 0) & (group["cvd_diff"] > 0)).astype(float)
    group["oi_up_price_up"] = ((group["oi_return"] > 0) & (group["price_return"] > 0)).astype(float)
    group["oi_up_price_down"] = ((group["oi_return"] > 0) & (group["price_return"] < 0)).astype(float)
    group["oi_down_price_up"] = ((group["oi_return"] < 0) & (group["price_return"] > 0)).astype(float)
    group["oi_down_price_down"] = ((group["oi_return"] < 0) & (group["price_return"] < 0)).astype(float)
    group["cvd_acceleration"] = group["cvd_diff"].diff()
    group["oi_acceleration"] = group["oi_return"].diff()

    for col in ["price_return", "oi_return", "cvd_pct_change", "taker_imbalance"]:
        for lag in (1, 2, 3):
            group[f"{col}_lag{lag}"] = group[col].shift(lag)
        for window in BASE_WINDOWS:
            shifted = group[col].shift(1)
            rolling = shifted.rolling(window, min_periods=max(3, window // 3))
            mean = rolling.mean()
            std = rolling.std()
            group[f"{col}_mean_{window}"] = mean
            group[f"{col}_std_{window}"] = std
            group[f"{col}_z_{window}"] = (group[col].shift(1) - mean) / std.replace(0, np.nan)
            group[f"{col}_up_ratio_{window}"] = (shifted > 0).astype(float).rolling(window, min_periods=3).mean()

    group = _add_price_state_features(group)
    return group


def add_market_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    market = (
        df.groupby("ts", as_index=False)
        .agg(
            market_avg_return=("price_return", "mean"),
            market_breadth=("price_return", lambda s: float((s > 0).mean())),
        )
    )
    df = df.merge(market, on="ts", how="left")

    btc = df[df["pair"] == "BTCUSDT"][["ts", "price_return", "oi_return", "cvd_pct_change"]].rename(
        columns={
            "price_return": "btc_return",
            "oi_return": "btc_oi_return",
            "cvd_pct_change": "btc_cvd_pct_change",
        }
    )
    df = df.merge(btc, on="ts", how="left")
    df["relative_return_vs_btc"] = df["price_return"] - df["btc_return"]
    df["relative_oi_vs_btc"] = df["oi_return"] - df["btc_oi_return"]
    df["relative_cvd_vs_btc"] = df["cvd_pct_change"] - df["btc_cvd_pct_change"]
    return df


def build_features(raw: pd.DataFrame) -> pd.DataFrame:
    if raw.empty:
        return raw
    df = raw.groupby("pair", group_keys=False).apply(_add_pair_features)
    df = add_market_features(df)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
    return df.sort_values(["ts", "pair"]).reset_index(drop=True)


def feature_columns(df: pd.DataFrame) -> list[str]:
    excluded = {
        "coin",
        "pair",
        "interval",
        "ts",
        "label",
        "future_return",
        "future_price",
        "entry_price",
    }
    return [
        col
        for col in df.columns
        if col not in excluded and pd.api.types.is_numeric_dtype(df[col])
    ]
