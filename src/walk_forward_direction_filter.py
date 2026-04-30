import argparse
import json
import os
from dataclasses import dataclass

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import numpy as np
import pandas as pd

from .backtest import predictions_to_frame
from .config import DEFAULT_BUFFER, DEFAULT_CLOSE_FEE, DEFAULT_OPEN_FEE, RESULTS_DIR
from .data_loader import build_all_pairs_dataset
from .feature_engineering import build_features, feature_columns
from .labeling import add_trade_labels, label_distribution
from .splits import walk_forward_splits
from .train_hgb import _make_model, _predict_confidence, parse_pairs


SIDE_TO_VALUE = {"SHORT": -1, "LONG": 1, "BOTH": None}


@dataclass(frozen=True)
class Condition:
    feature: str
    op: str
    quantile: float


@dataclass(frozen=True)
class Candidate:
    name: str
    side: str
    confidence: float
    conditions: tuple[Condition, ...]


def _date_text(value) -> str:
    return pd.Timestamp(value).strftime("%Y-%m-%d %H:%M:%S")


def _parse_candidates(raw: str) -> list[Candidate]:
    candidates = []
    for item in raw.split(";"):
        item = item.strip()
        if not item:
            continue
        parts = [part.strip() for part in item.split(":")]
        if len(parts) < 5 or (len(parts) - 2) % 3 != 0:
            raise ValueError(
                "Candidate format must be side:confidence:feature:op:quantile"
                "[:feature:op:quantile...], "
                f"got {item!r}"
            )
        side, confidence = parts[:2]
        side = side.upper()
        if side not in SIDE_TO_VALUE:
            raise ValueError(f"Unknown side {side!r}")
        conditions = []
        name_parts = [f"{side}_conf{float(confidence):g}"]
        for index in range(2, len(parts), 3):
            feature, op, quantile = parts[index : index + 3]
            if op not in {"<=", ">="}:
                raise ValueError(f"Unknown op {op!r}")
            condition = Condition(feature=feature, op=op, quantile=float(quantile))
            conditions.append(condition)
            name_parts.append(f"{feature}_{op}q{float(quantile):g}")
        name = "_AND_".join(name_parts)
        candidates.append(
            Candidate(
                name=name,
                side=side,
                confidence=float(confidence),
                conditions=tuple(conditions),
            )
        )
    return candidates


def _active_for_candidate(trades: pd.DataFrame, candidate: Candidate) -> pd.DataFrame:
    active = trades[(trades["prediction"] != 0) & (trades["confidence"] >= candidate.confidence)].copy()
    side_value = SIDE_TO_VALUE[candidate.side]
    if side_value is not None:
        active = active[active["prediction"] == side_value]
    return active


def _apply_cutoffs(frame: pd.DataFrame, candidate: Candidate, cutoffs: dict[str, float]) -> pd.DataFrame:
    out = frame
    for condition in candidate.conditions:
        cutoff = cutoffs[condition.feature]
        if condition.op == "<=":
            out = out[out[condition.feature] <= cutoff]
        else:
            out = out[out[condition.feature] >= cutoff]
    return out.copy()


def _summarize_direction(frame: pd.DataFrame) -> dict:
    if frame.empty:
        return {
            "trades": 0,
            "gross_wins": 0,
            "gross_win_rate": 0.0,
            "avg_gross_pnl": 0.0,
            "total_gross_pnl": 0.0,
            "avg_net_pnl": 0.0,
            "total_net_pnl": 0.0,
        }
    wins = frame["gross_pnl"] > 0
    return {
        "trades": int(len(frame)),
        "gross_wins": int(wins.sum()),
        "gross_win_rate": float(wins.mean()),
        "avg_gross_pnl": float(frame["gross_pnl"].mean()),
        "total_gross_pnl": float(frame["gross_pnl"].sum()),
        "avg_net_pnl": float(frame["net_pnl"].mean()),
        "total_net_pnl": float(frame["net_pnl"].sum()),
    }


def _merge_features(raw_trades: pd.DataFrame, frame: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    cols = ["ts", "pair"] + features
    out = raw_trades.merge(frame[cols], on=["ts", "pair"], how="left")
    out = out.replace([np.inf, -np.inf], np.nan)
    return out.dropna(subset=features)


def run(args: argparse.Namespace) -> dict:
    raw_candidates = args.candidates
    if args.candidates_file:
        with open(args.candidates_file, "r", encoding="utf-8") as file:
            raw_candidates = file.read()
    candidates = _parse_candidates(raw_candidates)
    required_features = sorted({condition.feature for candidate in candidates for condition in candidate.conditions})

    print("Loading all-pair dataset...")
    raw = build_all_pairs_dataset(args.interval, args.bar_rule, args.max_pairs, pairs=parse_pairs(args.pairs))
    if raw.empty:
        raise RuntimeError("No data loaded.")

    print("Building features...")
    features = build_features(raw)
    missing = [feature for feature in required_features if feature not in features.columns]
    if missing:
        raise ValueError(f"Missing features: {missing}")

    labeled = add_trade_labels(
        features,
        horizon=args.horizon,
        open_fee=args.open_fee,
        close_fee=args.close_fee,
        buffer=args.buffer,
        label_mode=args.label_mode,
    )
    labeled = labeled.dropna(subset=["label"]).sort_values(["ts", "pair"]).reset_index(drop=True)
    print(f"labeled rows: {len(labeled):,}, pairs: {labeled['pair'].nunique()}")
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

    fold_rows = []
    all_test_rows = []
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
            "fold {fold}: val {val_start} -> {val_end}, test {test_start} -> {test_end}".format(
                fold=fold["fold"],
                val_start=_date_text(fold["val_start"]),
                val_end=_date_text(fold["val_end"]),
                test_start=_date_text(fold["test_start"]),
                test_end=_date_text(fold["test_end"]),
            )
        )

        model, actual_model_name = _make_model(args.model)
        model.fit(x_train, y_train)

        val_pred, val_confidence = _predict_confidence(model, x_val)
        test_pred, test_confidence = _predict_confidence(model, x_test)
        val_trades = predictions_to_frame(
            val_frame,
            val_pred,
            confidence=val_confidence,
            open_fee=args.open_fee,
            close_fee=args.close_fee,
        )
        test_trades = predictions_to_frame(
            test_frame,
            test_pred,
            confidence=test_confidence,
            open_fee=args.open_fee,
            close_fee=args.close_fee,
        )

        for candidate in candidates:
            candidate_features = [condition.feature for condition in candidate.conditions]
            val_with_feature = _merge_features(val_trades, val_frame, candidate_features)
            test_with_feature = _merge_features(test_trades, test_frame, candidate_features)
            val_active = _active_for_candidate(val_with_feature, candidate)
            if len(val_active) < args.min_val_candidates:
                cutoffs = {}
                val_filtered = val_active.iloc[0:0].copy()
                test_filtered = test_with_feature.iloc[0:0].copy()
            else:
                cutoffs = {
                    condition.feature: float(val_active[condition.feature].quantile(condition.quantile))
                    for condition in candidate.conditions
                }
                val_filtered = _apply_cutoffs(val_active, candidate, cutoffs)
                test_active = _active_for_candidate(test_with_feature, candidate)
                test_filtered = _apply_cutoffs(test_active, candidate, cutoffs)

            val_summary = _summarize_direction(val_filtered)
            test_summary = _summarize_direction(test_filtered)
            row = {
                "candidate": candidate.name,
                "fold": fold["fold"],
                "side": candidate.side,
                "confidence": candidate.confidence,
                "feature": "|".join(condition.feature for condition in candidate.conditions),
                "op": "|".join(condition.op for condition in candidate.conditions),
                "quantile": "|".join(f"{condition.quantile:g}" for condition in candidate.conditions),
                "cutoff": json.dumps(cutoffs, ensure_ascii=False, sort_keys=True),
                "val_start": _date_text(fold["val_start"]),
                "val_end": _date_text(fold["val_end"]),
                "test_start": _date_text(fold["test_start"]),
                "test_end": _date_text(fold["test_end"]),
                "val_base_trades": int(len(val_active)),
                "val_trades": val_summary["trades"],
                "val_direction_win_rate": val_summary["gross_win_rate"],
                "val_avg_gross_pnl": val_summary["avg_gross_pnl"],
                "val_total_gross_pnl": val_summary["total_gross_pnl"],
                "val_avg_net_pnl": val_summary["avg_net_pnl"],
                "val_total_net_pnl": val_summary["total_net_pnl"],
                "test_trades": test_summary["trades"],
                "test_direction_wins": test_summary["gross_wins"],
                "test_direction_win_rate": test_summary["gross_win_rate"],
                "test_avg_gross_pnl": test_summary["avg_gross_pnl"],
                "test_total_gross_pnl": test_summary["total_gross_pnl"],
                "test_avg_net_pnl": test_summary["avg_net_pnl"],
                "test_total_net_pnl": test_summary["total_net_pnl"],
            }
            fold_rows.append(row)

            if not test_filtered.empty:
                test_out = test_filtered.copy()
                test_out["candidate"] = candidate.name
                test_out["fold"] = fold["fold"]
                test_out["cutoff"] = json.dumps(cutoffs, ensure_ascii=False, sort_keys=True)
                all_test_rows.append(test_out)

    fold_summary = pd.DataFrame(fold_rows)
    trades = pd.concat(all_test_rows, ignore_index=True) if all_test_rows else pd.DataFrame()
    aggregate_rows = []
    for candidate, group in fold_summary.groupby("candidate"):
        total_trades = int(group["test_trades"].sum())
        total_wins = int(group["test_direction_wins"].sum())
        aggregate_rows.append(
            {
                "candidate": candidate,
                "folds": int(group["fold"].nunique()),
                "active_folds": int((group["test_trades"] > 0).sum()),
                "trades": total_trades,
                "direction_wins": total_wins,
                "direction_win_rate": float(total_wins / total_trades) if total_trades else 0.0,
                "avg_gross_pnl": float(group["test_total_gross_pnl"].sum() / total_trades)
                if total_trades
                else 0.0,
                "total_gross_pnl": float(group["test_total_gross_pnl"].sum()),
                "avg_net_pnl": float(group["test_total_net_pnl"].sum() / total_trades)
                if total_trades
                else 0.0,
                "total_net_pnl": float(group["test_total_net_pnl"].sum()),
                "positive_folds": int((group["test_total_gross_pnl"] > 0).sum()),
            }
        )
    aggregate = pd.DataFrame(aggregate_rows).sort_values(
        ["direction_win_rate", "trades"],
        ascending=[False, False],
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    pair_tag = args.pairs.replace(",", "-") if args.pairs else args.max_pairs or "all"
    stem = (
        f"direction_walk_forward_{actual_model_name}_{args.interval}_h{args.horizon}_{args.label_mode}_"
        f"tr{args.train_bars}_val{args.val_bars}_te{args.test_bars}_pairs{pair_tag}"
    )
    fold_summary.to_csv(RESULTS_DIR / f"{stem}_folds.csv", index=False, encoding="utf-8-sig")
    aggregate.to_csv(RESULTS_DIR / f"{stem}_summary.csv", index=False, encoding="utf-8-sig")
    if not trades.empty:
        trades.to_csv(RESULTS_DIR / f"{stem}_trades.csv", index=False, encoding="utf-8-sig")
    payload = {
        "summary": aggregate.to_dict(orient="records"),
        "args": vars(args),
    }
    with open(RESULTS_DIR / f"{stem}.json", "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)

    print(aggregate.to_string(index=False))
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Walk-forward direction-only filters.")
    parser.add_argument("--interval", default="1h")
    parser.add_argument("--bar-rule", default="1h")
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument(
        "--label-mode",
        choices=["trade", "direction"],
        default="trade",
        help="trade uses fee+buffer label threshold; direction labels pure next-bar direction.",
    )
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
    parser.add_argument("--min-val-candidates", type=int, default=20)
    parser.add_argument(
        "--candidates",
        default=(
            "LONG:0.80:drawdown_from_high_50:<=:0.10;"
            "LONG:0.80:rsi_50:<=:0.20;"
            "LONG:0.85:rsi_50:<=:0.30;"
            "LONG:0.80:distance_to_boll_upper_14:<=:0.10;"
            "LONG:0.80:distance_to_ma_50:<=:0.10"
        ),
    )
    parser.add_argument("--candidates-file", default=None, help="Text file containing semicolon-separated candidates.")
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
