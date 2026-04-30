import argparse
import json

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset

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
from .train_hgb import parse_pairs


LABEL_TO_CLASS = {-1: 0, 0: 1, 1: 2}
CLASS_TO_LABEL = {0: -1, 1: 0, 2: 1}


class SequenceDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray, rows):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        self.rows = rows

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx]


class GRUClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int = 64, num_layers: int = 1, dropout: float = 0.1):
        super().__init__()
        self.gru = nn.GRU(
            input_dim,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden_size, 3)

    def forward(self, x):
        _, hidden = self.gru(x)
        return self.head(hidden[-1])


class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma: float = 2.0):
        super().__init__()
        self.register_buffer("weight", weight if weight is not None else None)
        self.gamma = gamma

    def forward(self, logits, target):
        ce = F.cross_entropy(logits, target, weight=self.weight, reduction="none")
        pt = torch.exp(-ce)
        return ((1.0 - pt) ** self.gamma * ce).mean()


def _parse_thresholds(raw: str) -> list[float]:
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def _class_weights(y: np.ndarray) -> torch.Tensor:
    counts = np.bincount(y, minlength=3).astype(np.float32)
    counts[counts == 0] = 1.0
    weights = counts.sum() / (len(counts) * counts)
    return torch.tensor(weights / weights.mean(), dtype=torch.float32)


def _make_loss(loss_name: str, y_train: np.ndarray, gamma: float, device):
    use_weight = loss_name in {"weighted_ce", "weighted_focal"}
    weight = _class_weights(y_train).to(device) if use_weight else None
    if loss_name in {"focal", "weighted_focal"}:
        return FocalLoss(weight=weight, gamma=gamma)
    return nn.CrossEntropyLoss(weight=weight)


def _make_sequences(df, feature_values: np.ndarray, lookback: int, mask) -> tuple[np.ndarray, np.ndarray, list[int]]:
    xs, ys, rows = [], [], []
    mask_values = mask.to_numpy()
    labels = df["label"].to_numpy()

    for _, group in df.groupby("pair", sort=False):
        idxs = group.index.to_numpy()
        for position in range(lookback - 1, len(idxs)):
            row_idx = idxs[position]
            if not mask_values[row_idx]:
                continue
            seq_idxs = idxs[position - lookback + 1 : position + 1]
            xs.append(feature_values[seq_idxs])
            ys.append(LABEL_TO_CLASS[int(labels[row_idx])])
            rows.append(row_idx)

    return np.asarray(xs), np.asarray(ys), rows


def _predict(model, loader, device):
    model.eval()
    preds, confs = [], []
    softmax = nn.Softmax(dim=1)
    with torch.no_grad():
        for x, _ in loader:
            logits = model(x.to(device))
            probs = softmax(logits).cpu().numpy()
            pred_classes = probs.argmax(axis=1)
            preds.extend([CLASS_TO_LABEL[int(cls)] for cls in pred_classes])
            confs.extend(probs.max(axis=1).tolist())
    return np.asarray(preds), np.asarray(confs)


def run(args: argparse.Namespace) -> dict:
    raw = build_all_pairs_dataset(args.interval, args.bar_rule, args.max_pairs, pairs=parse_pairs(args.pairs))
    if raw.empty:
        raise RuntimeError("No data loaded.")
    features = build_features(raw)
    labeled = add_trade_labels(
        features,
        horizon=args.horizon,
        open_fee=args.open_fee,
        close_fee=args.close_fee,
        buffer=args.buffer,
    ).sort_values(["ts", "pair"]).reset_index(drop=True)
    print(f"labeled rows: {len(labeled):,}, pairs: {labeled['pair'].nunique()}")
    print(label_distribution(labeled).to_string())

    train_mask, val_mask, test_mask = time_split_masks(labeled)
    cols = feature_columns(labeled)
    cols = [col for col in cols if labeled.loc[train_mask, col].notna().any()]

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    train_matrix = imputer.fit_transform(labeled.loc[train_mask, cols])
    scaler.fit(train_matrix)

    all_matrix = imputer.transform(labeled[cols])
    all_matrix = scaler.transform(all_matrix).astype(np.float32)

    x_train, y_train, _ = _make_sequences(labeled, all_matrix, args.lookback, train_mask)
    x_val, y_val, val_rows = _make_sequences(labeled, all_matrix, args.lookback, val_mask)
    x_test, y_test, test_rows = _make_sequences(labeled, all_matrix, args.lookback, test_mask)
    print(f"sequences train/val/test: {len(y_train):,}/{len(y_val):,}/{len(y_test):,}")

    train_loader = DataLoader(SequenceDataset(x_train, y_train, []), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(SequenceDataset(x_val, y_val, val_rows), batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(SequenceDataset(x_test, y_test, test_rows), batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = GRUClassifier(
        input_dim=x_train.shape[-1],
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = _make_loss(args.loss, y_train, args.focal_gamma, device)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for x, y in train_loader:
            optimizer.zero_grad()
            logits = model(x.to(device))
            loss = loss_fn(logits, y.to(device))
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(y)
        print(f"epoch {epoch}/{args.epochs} loss={total_loss / max(1, len(y_train)):.6f}")

    val_pred, val_confidence = _predict(model, val_loader, device)
    val_frame = labeled.loc[val_rows].reset_index(drop=True)
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

    pred, confidence = _predict(model, test_loader, device)
    test_frame = labeled.loc[test_rows].reset_index(drop=True)
    y_true = np.asarray([CLASS_TO_LABEL[int(cls)] for cls in y_test])
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
            "model": "gru",
            "interval": args.interval,
            "bar_rule": args.bar_rule,
            "horizon": args.horizon,
            "lookback": args.lookback,
            "buffer": args.buffer,
            "open_fee": args.open_fee,
            "close_fee": args.close_fee,
            "rows": int(len(labeled)),
            "pairs": int(labeled["pair"].nunique()),
            "features": int(len(cols)),
            "train_sequences": int(len(y_train)),
            "val_sequences": int(len(y_val)),
            "test_sequences": int(len(y_test)),
            "device": str(device),
            "loss": args.loss,
            "focal_gamma": args.focal_gamma,
            "selected_threshold": selected_threshold,
            "threshold_metric": args.threshold_metric,
            "selected_val_trades": int(selected_val["trades"]),
            "selected_val_total_net_pnl": float(selected_val["total_net_pnl"]),
            "selected_val_avg_net_pnl": float(selected_val["avg_net_pnl"]),
            "selected_val_profit_factor": float(selected_val["profit_factor"]),
        }
    )
    class_metrics = classification_outputs(y_true, trades["prediction"].to_numpy())
    by_pair = summarize_by_pair(trades)
    by_month = summarize_by_month(trades)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    loss_tag = args.loss.replace("_", "-")
    pair_tag = args.pairs.replace(",", "-") if args.pairs else args.max_pairs or "all"
    stem = f"gru_{loss_tag}_{args.interval}_h{args.horizon}_lb{args.lookback}_pairs{pair_tag}"
    write_text_report(RESULTS_DIR / f"{stem}.txt", summary, class_metrics, by_pair)
    write_json(
        RESULTS_DIR / f"{stem}.json",
        {
            "summary": summary,
            "classification": class_metrics,
            "feature_columns": cols,
            "validation_thresholds": val_thresholds.to_dict(orient="records"),
        },
    )
    raw_trades.to_csv(RESULTS_DIR / f"{stem}_raw_trades.csv", index=False, encoding="utf-8-sig")
    trades.to_csv(RESULTS_DIR / f"{stem}_trades.csv", index=False, encoding="utf-8-sig")
    val_trades.to_csv(RESULTS_DIR / f"{stem}_val_trades.csv", index=False, encoding="utf-8-sig")
    val_thresholds.to_csv(RESULTS_DIR / f"{stem}_val_thresholds.csv", index=False, encoding="utf-8-sig")
    by_pair.to_csv(RESULTS_DIR / f"{stem}_by_pair.csv", index=False, encoding="utf-8-sig")
    by_month.to_csv(RESULTS_DIR / f"{stem}_by_month.csv", index=False, encoding="utf-8-sig")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train OrderFlow GRU baseline.")
    parser.add_argument("--interval", default="1h")
    parser.add_argument("--bar-rule", default="1h")
    parser.add_argument("--horizon", type=int, default=4)
    parser.add_argument("--lookback", type=int, default=64)
    parser.add_argument("--buffer", type=float, default=DEFAULT_BUFFER)
    parser.add_argument("--open-fee", type=float, default=DEFAULT_OPEN_FEE)
    parser.add_argument("--close-fee", type=float, default=DEFAULT_CLOSE_FEE)
    parser.add_argument("--max-pairs", type=int, default=None)
    parser.add_argument("--pairs", default=None, help="Comma-separated pair list, e.g. BTCUSDT,ETHUSDT")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--loss", choices=["ce", "weighted_ce", "focal", "weighted_focal"], default="weighted_focal")
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--thresholds", default="0,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8")
    parser.add_argument("--min-val-trades", type=int, default=100)
    parser.add_argument("--threshold-metric", default="total_net_pnl")
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
