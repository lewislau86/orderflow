import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .backtest import (
    apply_confidence_threshold,
    predictions_to_frame,
    select_threshold,
    summarize_by_month,
    summarize_by_pair,
    summarize_trades,
)
from .config import DEFAULT_BUFFER, DEFAULT_CLOSE_FEE, DEFAULT_OPEN_FEE, RESULTS_DIR
from .data_loader import build_all_pairs_dataset
from .feature_engineering import build_features, feature_columns
from .labeling import add_trade_labels, label_distribution
from .metrics import classification_outputs, write_json, write_text_report
from .splits import time_split_masks


def _make_model(model_name: str):
    if model_name == "lightgbm":
        try:
            from lightgbm import LGBMClassifier

            return LGBMClassifier(
                objective="multiclass",
                n_estimators=300,
                learning_rate=0.04,
                num_leaves=31,
                subsample=0.9,
                colsample_bytree=0.9,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
                verbose=-1,
            ), "lightgbm"
        except Exception as exc:
            print(f"LightGBM unavailable, falling back to HistGradientBoosting: {exc}")

    return make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        HistGradientBoostingClassifier(
            max_iter=250,
            learning_rate=0.04,
            max_leaf_nodes=31,
            l2_regularization=0.05,
            random_state=42,
        ),
    ), "hist_gradient_boosting"


def _predict_confidence(model, x_test: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    pred = model.predict(x_test)
    if not hasattr(model, "predict_proba"):
        return pred, np.full(len(pred), np.nan)

    proba = model.predict_proba(x_test)
    return pred, np.max(proba, axis=1)


def _parse_thresholds(raw: str) -> list[float]:
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def parse_pairs(raw: str | None) -> list[str] | None:
    if not raw:
        return None
    return [item.strip().upper() for item in raw.split(",") if item.strip()]


def run(args: argparse.Namespace) -> dict:
    print("Loading all-pair dataset...")
    raw = build_all_pairs_dataset(
        interval=args.interval,
        bar_rule=args.bar_rule,
        max_pairs=args.max_pairs,
        pairs=parse_pairs(args.pairs),
    )
    if raw.empty:
        raise RuntimeError("No data loaded.")
    print(f"raw rows: {len(raw):,}, pairs: {raw['pair'].nunique()}")

    print("Building features...")
    features = build_features(raw)
    labeled = add_trade_labels(
        features,
        horizon=args.horizon,
        open_fee=args.open_fee,
        close_fee=args.close_fee,
        buffer=args.buffer,
    )
    labeled = labeled.dropna(subset=["label"]).sort_values(["ts", "pair"]).reset_index(drop=True)
    print(f"labeled rows: {len(labeled):,}")
    print(label_distribution(labeled).to_string())

    train_mask, val_mask, test_mask = time_split_masks(labeled)
    cols = feature_columns(labeled)
    cols = [col for col in cols if labeled.loc[train_mask, col].notna().any()]
    x_train = labeled.loc[train_mask, cols]
    y_train = labeled.loc[train_mask, "label"]
    x_val = labeled.loc[val_mask, cols]
    x_test = labeled.loc[test_mask, cols]
    y_test = labeled.loc[test_mask, "label"]
    val_frame = labeled.loc[val_mask].reset_index(drop=True)
    test_frame = labeled.loc[test_mask].reset_index(drop=True)

    print(f"features: {len(cols)}")
    print(f"train/val/test: {len(x_train):,}/{len(x_val):,}/{len(x_test):,}")

    model, actual_model_name = _make_model(args.model)
    print(f"Training {actual_model_name}...")
    model.fit(x_train, y_train)

    print("Evaluating...")
    val_pred, val_confidence = _predict_confidence(model, x_val)
    val_trades = predictions_to_frame(
        val_frame,
        val_pred,
        confidence=val_confidence,
        open_fee=args.open_fee,
        close_fee=args.close_fee,
    )
    thresholds = _parse_thresholds(args.thresholds)
    selected_threshold, selected_val, val_thresholds = select_threshold(
        val_trades,
        thresholds=thresholds,
        min_trades=args.min_val_trades,
        metric=args.threshold_metric,
    )

    pred, confidence = _predict_confidence(model, x_test)
    raw_trades = predictions_to_frame(
        test_frame,
        pred,
        confidence=confidence,
        open_fee=args.open_fee,
        close_fee=args.close_fee,
    )
    trades = apply_confidence_threshold(raw_trades, selected_threshold)
    summary = summarize_trades(trades)
    summary.update(
        {
            "model": actual_model_name,
            "interval": args.interval,
            "bar_rule": args.bar_rule,
            "horizon": args.horizon,
            "buffer": args.buffer,
            "open_fee": args.open_fee,
            "close_fee": args.close_fee,
            "rows": int(len(labeled)),
            "pairs": int(labeled["pair"].nunique()),
            "features": int(len(cols)),
            "train_rows": int(len(x_train)),
            "val_rows": int(len(x_val)),
            "test_rows": int(len(x_test)),
            "selected_threshold": selected_threshold,
            "threshold_metric": args.threshold_metric,
            "selected_val_trades": int(selected_val["trades"]),
            "selected_val_total_net_pnl": float(selected_val["total_net_pnl"]),
            "selected_val_avg_net_pnl": float(selected_val["avg_net_pnl"]),
            "selected_val_profit_factor": float(selected_val["profit_factor"]),
        }
    )
    class_metrics = classification_outputs(y_test, trades["prediction"].to_numpy())
    by_pair = summarize_by_pair(trades)
    by_month = summarize_by_month(trades)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    pair_tag = args.pairs.replace(",", "-") if args.pairs else args.max_pairs or "all"
    stem = f"{actual_model_name}_{args.interval}_h{args.horizon}_pairs{pair_tag}"
    report_path = RESULTS_DIR / f"{stem}.txt"
    json_path = RESULTS_DIR / f"{stem}.json"
    raw_trades_path = RESULTS_DIR / f"{stem}_raw_trades.csv"
    trades_path = RESULTS_DIR / f"{stem}_trades.csv"
    val_trades_path = RESULTS_DIR / f"{stem}_val_trades.csv"
    val_thresholds_path = RESULTS_DIR / f"{stem}_val_thresholds.csv"
    pair_path = RESULTS_DIR / f"{stem}_by_pair.csv"
    month_path = RESULTS_DIR / f"{stem}_by_month.csv"

    write_text_report(report_path, summary, class_metrics, by_pair)
    write_json(
        json_path,
        {
            "summary": summary,
            "classification": class_metrics,
            "feature_columns": cols,
            "validation_thresholds": val_thresholds.to_dict(orient="records"),
        },
    )
    raw_trades.to_csv(raw_trades_path, index=False, encoding="utf-8-sig")
    trades.to_csv(trades_path, index=False, encoding="utf-8-sig")
    val_trades.to_csv(val_trades_path, index=False, encoding="utf-8-sig")
    val_thresholds.to_csv(val_thresholds_path, index=False, encoding="utf-8-sig")
    by_pair.to_csv(pair_path, index=False, encoding="utf-8-sig")
    by_month.to_csv(month_path, index=False, encoding="utf-8-sig")

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"report: {report_path}")
    print(f"trades: {trades_path}")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train OrderFlow table baseline.")
    parser.add_argument("--interval", default="1h")
    parser.add_argument("--bar-rule", default="1h")
    parser.add_argument("--horizon", type=int, default=4)
    parser.add_argument("--buffer", type=float, default=DEFAULT_BUFFER)
    parser.add_argument("--open-fee", type=float, default=DEFAULT_OPEN_FEE)
    parser.add_argument("--close-fee", type=float, default=DEFAULT_CLOSE_FEE)
    parser.add_argument("--max-pairs", type=int, default=None)
    parser.add_argument("--pairs", default=None, help="Comma-separated pair list, e.g. BTCUSDT,ETHUSDT")
    parser.add_argument("--model", choices=["auto", "lightgbm", "hgb"], default="auto")
    parser.add_argument("--thresholds", default="0,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8")
    parser.add_argument("--min-val-trades", type=int, default=100)
    parser.add_argument("--threshold-metric", default="total_net_pnl")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_name = "lightgbm" if args.model == "auto" else args.model
    if model_name == "hgb":
        model_name = "hist_gradient_boosting"
    args.model = model_name
    run(args)


if __name__ == "__main__":
    main()
