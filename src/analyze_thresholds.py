import argparse
from pathlib import Path

import pandas as pd

from .backtest import threshold_summaries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze confidence thresholds for a trades CSV.")
    parser.add_argument("trades_csv")
    parser.add_argument("--thresholds", default="0,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path = Path(args.trades_csv)
    thresholds = [float(item) for item in args.thresholds.split(",") if item.strip()]
    trades = pd.read_csv(path)
    summary = threshold_summaries(trades, thresholds)
    out_path = path.with_name(path.stem.replace("_trades", "") + "_thresholds.csv")
    summary.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(summary.to_string(index=False))
    print(f"thresholds: {out_path}")


if __name__ == "__main__":
    main()
