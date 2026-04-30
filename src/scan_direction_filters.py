import argparse

import numpy as np
import pandas as pd

from .config import RESULTS_DIR
from .data_loader import build_all_pairs_dataset
from .feature_engineering import build_features
from .train_hgb import parse_pairs


DEFAULT_FEATURE_PATTERNS = (
    "boll_percent_b",
    "boll_band_width",
    "distance_to_boll",
    "atr_pct",
    "realized_vol",
    "drawdown_from_high",
    "rally_from_low",
    "ma_slope",
    "distance_to_ma",
    "trend_strength",
    "rsi",
    "stochastic_position",
    "rolling_rank",
)


def _candidate_feature_columns(df: pd.DataFrame, patterns: tuple[str, ...]) -> list[str]:
    out = []
    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue
        if any(pattern in col for pattern in patterns):
            out.append(col)
    return out


def _summarize(group: pd.DataFrame) -> dict:
    wins = group["gross_pnl"] > 0
    return {
        "trades": int(len(group)),
        "gross_wins": int(wins.sum()),
        "gross_win_rate": float(wins.mean()) if len(group) else 0.0,
        "avg_gross_pnl": float(group["gross_pnl"].mean()) if len(group) else 0.0,
        "total_gross_pnl": float(group["gross_pnl"].sum()) if len(group) else 0.0,
        "avg_confidence": float(group["confidence"].mean()) if len(group) else 0.0,
    }


def scan(args: argparse.Namespace) -> pd.DataFrame:
    trades = pd.read_csv(args.trades)
    trades["ts"] = pd.to_datetime(trades["ts"])
    trades = trades[trades["is_trade"].astype(str).str.lower().eq("true")].copy()
    trades = trades[trades["prediction"] != 0].copy()

    raw = build_all_pairs_dataset(args.interval, args.bar_rule, args.max_pairs, pairs=parse_pairs(args.pairs))
    features = build_features(raw)
    features["ts"] = pd.to_datetime(features["ts"])

    patterns = tuple(item.strip() for item in args.patterns.split(",") if item.strip())
    feature_cols = _candidate_feature_columns(features, patterns)
    keep = ["ts", "pair"] + feature_cols
    merged = trades.merge(features[keep], on=["ts", "pair"], how="left")
    merged = merged.replace([np.inf, -np.inf], np.nan)

    thresholds = [float(item.strip()) for item in args.conf_thresholds.split(",") if item.strip()]
    quantiles = [float(item.strip()) for item in args.quantiles.split(",") if item.strip()]
    rows = []

    for side_value, side_name in [(-1, "SHORT"), (1, "LONG"), (None, "BOTH")]:
        side_df = merged if side_value is None else merged[merged["prediction"] == side_value]
        if side_df.empty:
            continue
        for conf in thresholds:
            conf_df = side_df[side_df["confidence"] >= conf]
            if len(conf_df) < args.min_trades:
                continue
            base = _summarize(conf_df)
            base.update(
                {
                    "side": side_name,
                    "confidence_threshold": conf,
                    "feature": "__confidence_only__",
                    "condition": "all",
                    "cutoff": np.nan,
                }
            )
            rows.append(base)

            for col in feature_cols:
                series = conf_df[col].dropna()
                if series.nunique() < 5:
                    continue
                for q in quantiles:
                    cutoff = float(series.quantile(q))
                    if not np.isfinite(cutoff):
                        continue

                    low = conf_df[conf_df[col] <= cutoff]
                    if len(low) >= args.min_trades:
                        row = _summarize(low)
                        row.update(
                            {
                                "side": side_name,
                                "confidence_threshold": conf,
                                "feature": col,
                                "condition": f"<=q{q:g}",
                                "cutoff": cutoff,
                            }
                        )
                        rows.append(row)

                    high = conf_df[conf_df[col] >= cutoff]
                    if len(high) >= args.min_trades:
                        row = _summarize(high)
                        row.update(
                            {
                                "side": side_name,
                                "confidence_threshold": conf,
                                "feature": col,
                                "condition": f">=q{q:g}",
                                "cutoff": cutoff,
                            }
                        )
                        rows.append(row)

    result = pd.DataFrame(rows)
    if result.empty:
        return result

    result = result.sort_values(
        ["gross_win_rate", "trades", "avg_gross_pnl"],
        ascending=[False, False, False],
    )
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False, encoding="utf-8-sig")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scan direction-only filters.")
    parser.add_argument("trades")
    parser.add_argument("--interval", default="1h")
    parser.add_argument("--bar-rule", default="1h")
    parser.add_argument("--max-pairs", type=int, default=None)
    parser.add_argument("--pairs", default=None, help="Comma-separated pair list, e.g. BTCUSDT,ETHUSDT")
    parser.add_argument("--min-trades", type=int, default=50)
    parser.add_argument("--conf-thresholds", default="0,0.6,0.65,0.7,0.75,0.8,0.85")
    parser.add_argument("--quantiles", default="0.1,0.2,0.3,0.7,0.8,0.9")
    parser.add_argument("--patterns", default=",".join(DEFAULT_FEATURE_PATTERNS))
    parser.add_argument(
        "--output",
        default=str(RESULTS_DIR / "direction_filter_scan.csv"),
    )
    return parser.parse_args()


def main() -> None:
    result = scan(parse_args())
    if result.empty:
        print("No candidates found.")
        return
    print(result.head(30).to_string(index=False))


if __name__ == "__main__":
    main()
