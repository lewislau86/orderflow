import argparse

from .config import DATASET_DIR, DEFAULT_BUFFER, DEFAULT_CLOSE_FEE, DEFAULT_OPEN_FEE
from .data_loader import build_all_pairs_dataset
from .feature_engineering import build_features
from .labeling import add_trade_labels, label_distribution


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build cached OrderFlow ML dataset.")
    parser.add_argument("--interval", default="1h")
    parser.add_argument("--bar-rule", default="1h")
    parser.add_argument("--horizon", type=int, default=4)
    parser.add_argument("--buffer", type=float, default=DEFAULT_BUFFER)
    parser.add_argument("--open-fee", type=float, default=DEFAULT_OPEN_FEE)
    parser.add_argument("--close-fee", type=float, default=DEFAULT_CLOSE_FEE)
    parser.add_argument("--max-pairs", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    DATASET_DIR.mkdir(parents=True, exist_ok=True)

    suffix = f"{args.interval}_h{args.horizon}_pairs{args.max_pairs or 'all'}"
    raw_path = DATASET_DIR / f"raw_{suffix}.parquet"
    features_path = DATASET_DIR / f"features_{suffix}.parquet"
    labeled_path = DATASET_DIR / f"labeled_{suffix}.parquet"

    raw = build_all_pairs_dataset(args.interval, args.bar_rule, args.max_pairs)
    raw.to_parquet(raw_path, index=False)
    print(f"raw: {raw_path} rows={len(raw):,}")

    features = build_features(raw)
    features.to_parquet(features_path, index=False)
    print(f"features: {features_path} rows={len(features):,}")

    labeled = add_trade_labels(
        features,
        horizon=args.horizon,
        open_fee=args.open_fee,
        close_fee=args.close_fee,
        buffer=args.buffer,
    )
    labeled.to_parquet(labeled_path, index=False)
    print(f"labeled: {labeled_path} rows={len(labeled):,}")
    print(label_distribution(labeled).to_string())


if __name__ == "__main__":
    main()
