from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

from .config import DATA_ROOT, REQUIRED_FILES
from .data_catalog import DatasetEntry, discover_datasets, filter_entries


def _read_price(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path / REQUIRED_FILES["price"])
    return df[["ts", "open_num", "high_num", "low_num", "close_num"]].rename(
        columns={
            "open_num": "open",
            "high_num": "high",
            "low_num": "low",
            "close_num": "price",
        }
    )


def _read_oi(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path / REQUIRED_FILES["oi"])
    return df[["ts", "close_num"]].rename(columns={"close_num": "oi"})


def _read_cvd(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path / REQUIRED_FILES["cvd"])
    cols = ["ts", "cum_vol_delta", "taker_buy_vol", "taker_sell_vol"]
    available = [col for col in cols if col in df.columns]
    out = df[available].rename(columns={"cum_vol_delta": "cvd"})
    for col in ["taker_buy_vol", "taker_sell_vol"]:
        if col not in out.columns:
            out[col] = 0.0
    return out


def load_pair_interval(entry: DatasetEntry, bar_rule: Optional[str] = None) -> pd.DataFrame:
    price = _read_price(entry.path)
    oi = _read_oi(entry.path)
    cvd = _read_cvd(entry.path)

    df = price.merge(oi, on="ts", how="outer").merge(cvd, on="ts", how="outer")
    df["ts"] = pd.to_datetime(df["ts"])
    df = df.sort_values("ts").set_index("ts")
    df = df.ffill()

    if bar_rule:
        agg = {
            "open": "first",
            "high": "max",
            "low": "min",
            "price": "last",
            "oi": "last",
            "cvd": "last",
            "taker_buy_vol": "sum",
            "taker_sell_vol": "sum",
        }
        df = df.resample(bar_rule).agg(agg)

    df = df.dropna(subset=["price", "oi", "cvd"]).reset_index()
    df.insert(0, "interval", entry.interval)
    df.insert(0, "pair", entry.pair)
    df.insert(0, "coin", entry.coin)
    return df


def build_all_pairs_dataset(
    interval: str = "1h",
    bar_rule: Optional[str] = "1h",
    max_pairs: Optional[int] = None,
    pairs: Optional[Iterable[str]] = None,
    data_root: Path = DATA_ROOT,
) -> pd.DataFrame:
    entries = filter_entries(discover_datasets(data_root), interval=interval)
    if pairs:
        wanted = {pair.strip().upper() for pair in pairs if pair.strip()}
        entries = [entry for entry in entries if entry.pair.upper() in wanted]
        loaded = {entry.pair.upper() for entry in entries}
        missing = sorted(wanted - loaded)
        if missing:
            print(f"Warning: requested pairs not found for interval={interval}: {', '.join(missing)}")
    if max_pairs:
        entries = entries[:max_pairs]

    frames = []
    for entry in entries:
        frame = load_pair_interval(entry, bar_rule=bar_rule)
        if not frame.empty:
            frames.append(frame)

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True)
    return out.sort_values(["ts", "pair"]).reset_index(drop=True)


def load_entries(entries: Iterable[DatasetEntry], bar_rule: Optional[str] = None) -> pd.DataFrame:
    frames = [load_pair_interval(entry, bar_rule=bar_rule) for entry in entries]
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).sort_values(["ts", "pair"]).reset_index(drop=True)
