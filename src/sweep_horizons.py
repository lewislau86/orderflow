import argparse

import pandas as pd

from .backtest import threshold_summaries
from .config import DEFAULT_BUFFER, DEFAULT_CLOSE_FEE, DEFAULT_OPEN_FEE, RESULTS_DIR
from .train_hgb import run as run_hgb


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep horizons and confidence thresholds.")
    parser.add_argument("--interval", default="1h")
    parser.add_argument("--bar-rule", default="1h")
    parser.add_argument("--horizons", default="2,4,8,12")
    parser.add_argument("--thresholds", default="0,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8")
    parser.add_argument("--buffer", type=float, default=DEFAULT_BUFFER)
    parser.add_argument("--open-fee", type=float, default=DEFAULT_OPEN_FEE)
    parser.add_argument("--close-fee", type=float, default=DEFAULT_CLOSE_FEE)
    parser.add_argument("--max-pairs", type=int, default=None)
    parser.add_argument("--model", choices=["hgb", "lightgbm", "auto"], default="hgb")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    horizons = [int(item) for item in args.horizons.split(",") if item.strip()]
    thresholds = [float(item) for item in args.thresholds.split(",") if item.strip()]

    rows = []
    for horizon in horizons:
        hgb_args = argparse.Namespace(
            interval=args.interval,
            bar_rule=args.bar_rule,
            horizon=horizon,
            buffer=args.buffer,
            open_fee=args.open_fee,
            close_fee=args.close_fee,
            max_pairs=args.max_pairs,
            model="hist_gradient_boosting" if args.model == "hgb" else args.model,
        )

        summary = run_hgb(hgb_args)
        model_name = summary["model"]
        stem = f"{model_name}_{args.interval}_h{horizon}_pairs{args.max_pairs or 'all'}"
        trades_path = RESULTS_DIR / f"{stem}_trades.csv"
        trades = pd.read_csv(trades_path)
        threshold_frame = threshold_summaries(trades, thresholds)
        threshold_frame["horizon"] = horizon
        threshold_frame["model"] = model_name
        rows.append(threshold_frame)

    out = pd.concat(rows, ignore_index=True)
    out_path = RESULTS_DIR / f"sweep_{args.model}_{args.interval}_horizons_{args.horizons.replace(',', '-')}.csv"
    out.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(out.sort_values("total_net_pnl", ascending=False).head(20).to_string(index=False))
    print(f"sweep: {out_path}")


if __name__ == "__main__":
    main()
