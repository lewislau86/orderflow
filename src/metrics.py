import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

from .config import LABEL_TO_NAME


def classification_outputs(y_true, y_pred) -> dict:
    labels = [-1, 0, 1]
    names = [LABEL_TO_NAME[label] for label in labels]
    return {
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=labels,
            target_names=names,
            zero_division=0,
            output_dict=True,
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "labels": names,
    }


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)


def write_text_report(path: Path, summary: dict, class_metrics: dict, by_pair: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    lines.append("OrderFlow ML Baseline Report")
    lines.append("=" * 80)
    for key, value in summary.items():
        if isinstance(value, float):
            lines.append(f"{key}: {value:.6f}")
        else:
            lines.append(f"{key}: {value}")
    lines.append("")
    lines.append("Confusion Matrix labels: " + ", ".join(class_metrics["labels"]))
    lines.append(str(class_metrics["confusion_matrix"]))
    lines.append("")
    lines.append("Classification Report")
    lines.append(json.dumps(class_metrics["classification_report"], indent=2, ensure_ascii=False))
    lines.append("")
    lines.append("By Pair")
    lines.append(by_pair.to_string(index=False))
    path.write_text("\n".join(lines), encoding="utf-8")
