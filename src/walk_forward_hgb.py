import argparse
import json
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import pandas as pd

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
from .metrics import write_json
from .splits import walk_forward_splits
from .train_hgb import _make_model, _parse_thresholds, _predict_confidence, parse_pairs


def _date_text(value) -> str:
    return pd.Timestamp(value).strftime("%Y-%m-%d %H:%M:%S")


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

    folds = walk_forward_splits(
        labeled,
        train_bars=args.train_bars,
        val_bars=args.val_bars,
        test_bars=args.test_bars,
        step_bars=args.step_bars,
        max_folds=args.max_folds,
    )
    if not folds:
        raise RuntimeError("No walk-forward folds. Reduce train/val/test bars.")

    thresholds = _parse_thresholds(args.thresholds)
    all_trades = []
    all_raw_trades = []
    fold_rows = []
    actual_model_name = None

    for fold in folds:
        train_mask = fold["train_mask"]
        val_mask = fold["val_mask"]
        test_mask = fold["test_mask"]
        cols = feature_columns(labeled)
        cols = [col for col in cols if labeled.loc[train_mask, col].notna().any()]

        x_train = labeled.loc[train_mask, cols]
        y_train = labeled.loc[train_mask, "label"]
        x_val = labeled.loc[val_mask, cols]
        x_test = labeled.loc[test_mask, cols]
        val_frame = labeled.loc[val_mask].reset_index(drop=True)
        test_frame = labeled.loc[test_mask].reset_index(drop=True)

        print(
            "fold {fold}: train {train_start} -> {train_end}, "
            "val {val_start} -> {val_end}, test {test_start} -> {test_end}".format(
                fold=fold["fold"],
                train_start=_date_text(fold["train_start"]),
                train_end=_date_text(fold["train_end"]),
                val_start=_date_text(fold["val_start"]),
                val_end=_date_text(fold["val_end"]),
                test_start=_date_text(fold["test_start"]),
                test_end=_date_text(fold["test_end"]),
            )
        )
        print(f"fold {fold['fold']} rows train/val/test: {len(x_train):,}/{len(x_val):,}/{len(x_test):,}")

        model, actual_model_name = _make_model(args.model)
        model.fit(x_train, y_train)

        val_pred, val_confidence = _predict_confidence(model, x_val)
        val_trades = predictions_to_frame(
            val_frame,
            val_pred,
            confidence=val_confidence,
            open_fee=args.open_fee,
            close_fee=args.close_fee,
        )
        selected_threshold, selected_val, val_thresholds = select_threshold(
            val_trades,
            thresholds=thresholds,
            min_trades=args.min_val_trades,
            metric=args.threshold_metric,
        )
        gate_pass = (
            selected_val["trades"] >= args.gate_min_trades
            and selected_val["profit_factor"] >= args.gate_min_profit_factor
            and selected_val["avg_net_pnl"] >= args.gate_min_avg_net_pnl
            and selected_val["total_net_pnl"] >= args.gate_min_total_net_pnl
        )

        test_pred, test_confidence = _predict_confidence(model, x_test)
        raw_trades = predictions_to_frame(
            test_frame,
            test_pred,
            confidence=test_confidence,
            open_fee=args.open_fee,
            close_fee=args.close_fee,
        )
        if gate_pass:
            trades = apply_confidence_threshold(raw_trades, selected_threshold)
        else:
            trades = apply_confidence_threshold(raw_trades, 1.01)
        raw_trades["fold"] = fold["fold"]
        trades["fold"] = fold["fold"]
        trades["gate_pass"] = gate_pass
        all_raw_trades.append(raw_trades)
        all_trades.append(trades)

        row = summarize_trades(trades)
        row.update(
            {
                "fold": fold["fold"],
                "train_start": _date_text(fold["train_start"]),
                "train_end": _date_text(fold["train_end"]),
                "val_start": _date_text(fold["val_start"]),
                "val_end": _date_text(fold["val_end"]),
                "test_start": _date_text(fold["test_start"]),
                "test_end": _date_text(fold["test_end"]),
                "selected_threshold": selected_threshold,
                "selected_val_trades": int(selected_val["trades"]),
                "selected_val_total_net_pnl": float(selected_val["total_net_pnl"]),
                "selected_val_avg_net_pnl": float(selected_val["avg_net_pnl"]),
                "selected_val_profit_factor": float(selected_val["profit_factor"]),
                "gate_pass": bool(gate_pass),
                "gate_min_trades": args.gate_min_trades,
                "gate_min_profit_factor": args.gate_min_profit_factor,
                "gate_min_avg_net_pnl": args.gate_min_avg_net_pnl,
                "gate_min_total_net_pnl": args.gate_min_total_net_pnl,
                "train_rows": int(len(x_train)),
                "val_rows": int(len(x_val)),
                "test_rows": int(len(x_test)),
                "features": int(len(cols)),
            }
        )
        fold_rows.append(row)
        print(json.dumps(row, indent=2, ensure_ascii=False))

        fold_thresholds_path = (
            RESULTS_DIR
            / f"walk_forward_{actual_model_name}_{args.interval}_h{args.horizon}_fold{fold['fold']}_val_thresholds.csv"
        )
        fold_thresholds_path.parent.mkdir(parents=True, exist_ok=True)
        val_thresholds.to_csv(fold_thresholds_path, index=False, encoding="utf-8-sig")

    trades_all = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
    raw_trades_all = pd.concat(all_raw_trades, ignore_index=True) if all_raw_trades else pd.DataFrame()
    fold_summary = pd.DataFrame(fold_rows).sort_values("fold")
    by_pair = summarize_by_pair(trades_all)
    by_month = summarize_by_month(trades_all)
    summary = summarize_trades(trades_all)
    summary.update(
        {
            "model": actual_model_name,
            "interval": args.interval,
            "bar_rule": args.bar_rule,
            "horizon": args.horizon,
            "buffer": args.buffer,
            "open_fee": args.open_fee,
            "close_fee": args.close_fee,
            "folds": int(len(folds)),
            "pairs": int(labeled["pair"].nunique()),
            "rows": int(len(labeled)),
            "train_bars": args.train_bars,
            "val_bars": args.val_bars,
            "test_bars": args.test_bars,
            "step_bars": args.step_bars or args.test_bars,
            "threshold_metric": args.threshold_metric,
            "positive_folds": int((fold_summary["total_net_pnl"] > 0).sum()),
            "gate_pass_folds": int(fold_summary["gate_pass"].sum()),
            "gate_min_trades": args.gate_min_trades,
            "gate_min_profit_factor": args.gate_min_profit_factor,
            "gate_min_avg_net_pnl": args.gate_min_avg_net_pnl,
            "gate_min_total_net_pnl": args.gate_min_total_net_pnl,
        }
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    gate_tag = (
        f"gate_pf{args.gate_min_profit_factor:g}_avg{args.gate_min_avg_net_pnl:g}_"
        f"tot{args.gate_min_total_net_pnl:g}_n{args.gate_min_trades}"
    )
    pair_tag = args.pairs.replace(",", "-") if args.pairs else args.max_pairs or "all"
    stem = (
        f"walk_forward_{actual_model_name}_{args.interval}_h{args.horizon}_"
        f"tr{args.train_bars}_val{args.val_bars}_te{args.test_bars}_{gate_tag}_pairs{pair_tag}"
    )
    write_json(
        RESULTS_DIR / f"{stem}.json",
        {
            "summary": summary,
            "fold_summary": fold_summary.to_dict(orient="records"),
        },
    )
    fold_summary.to_csv(RESULTS_DIR / f"{stem}_folds.csv", index=False, encoding="utf-8-sig")
    trades_all.to_csv(RESULTS_DIR / f"{stem}_trades.csv", index=False, encoding="utf-8-sig")
    raw_trades_all.to_csv(RESULTS_DIR / f"{stem}_raw_trades.csv", index=False, encoding="utf-8-sig")
    by_pair.to_csv(RESULTS_DIR / f"{stem}_by_pair.csv", index=False, encoding="utf-8-sig")
    by_month.to_csv(RESULTS_DIR / f"{stem}_by_month.csv", index=False, encoding="utf-8-sig")

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"folds: {RESULTS_DIR / f'{stem}_folds.csv'}")
    print(f"trades: {RESULTS_DIR / f'{stem}_trades.csv'}")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Walk-forward OrderFlow HGB baseline.")
    parser.add_argument("--interval", default="1h")
    parser.add_argument("--bar-rule", default="1h")
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--buffer", type=float, default=DEFAULT_BUFFER)
    parser.add_argument("--open-fee", type=float, default=DEFAULT_OPEN_FEE)
    parser.add_argument("--close-fee", type=float, default=DEFAULT_CLOSE_FEE)
    parser.add_argument("--max-pairs", type=int, default=None)
    parser.add_argument("--pairs", default=None, help="Comma-separated pair list, e.g. BTCUSDT,ETHUSDT")
    parser.add_argument("--model", choices=["auto", "lightgbm", "hgb"], default="auto")
    parser.add_argument("--train-bars", type=int, default=6000)
    parser.add_argument("--val-bars", type=int, default=1000)
    parser.add_argument("--test-bars", type=int, default=720)
    parser.add_argument("--step-bars", type=int, default=None)
    parser.add_argument("--max-folds", type=int, default=None)
    parser.add_argument("--thresholds", default="0,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8")
    parser.add_argument("--min-val-trades", type=int, default=100)
    parser.add_argument("--threshold-metric", default="total_net_pnl")
    parser.add_argument("--gate-min-trades", type=int, default=100)
    parser.add_argument("--gate-min-profit-factor", type=float, default=1.0)
    parser.add_argument("--gate-min-avg-net-pnl", type=float, default=0.0)
    parser.add_argument("--gate-min-total-net-pnl", type=float, default=0.0)
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
