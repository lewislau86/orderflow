import pandas as pd


def time_split_masks(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    times = pd.Series(sorted(df["ts"].dropna().unique()))
    train_cut = times.iloc[int(len(times) * train_ratio)]
    val_cut = times.iloc[int(len(times) * (train_ratio + val_ratio))]

    train = df["ts"] < train_cut
    val = (df["ts"] >= train_cut) & (df["ts"] < val_cut)
    test = df["ts"] >= val_cut
    return train, val, test


def walk_forward_splits(
    df: pd.DataFrame,
    train_bars: int,
    val_bars: int,
    test_bars: int,
    step_bars: int | None = None,
    max_folds: int | None = None,
) -> list[dict]:
    times = pd.Series(sorted(df["ts"].dropna().unique()))
    step = step_bars or test_bars
    folds = []
    start = 0
    fold = 1

    while start + train_bars + val_bars + test_bars <= len(times):
        train_start = times.iloc[start]
        train_end = times.iloc[start + train_bars]
        val_end = times.iloc[start + train_bars + val_bars]
        test_end_idx = start + train_bars + val_bars + test_bars
        test_end = times.iloc[test_end_idx - 1]

        train = (df["ts"] >= train_start) & (df["ts"] < train_end)
        val = (df["ts"] >= train_end) & (df["ts"] < val_end)
        test = (df["ts"] >= val_end) & (df["ts"] <= test_end)

        folds.append(
            {
                "fold": fold,
                "train_mask": train,
                "val_mask": val,
                "test_mask": test,
                "train_start": train_start,
                "train_end": times.iloc[start + train_bars - 1],
                "val_start": train_end,
                "val_end": times.iloc[start + train_bars + val_bars - 1],
                "test_start": val_end,
                "test_end": test_end,
            }
        )

        if max_folds is not None and len(folds) >= max_folds:
            break
        start += step
        fold += 1

    return folds
