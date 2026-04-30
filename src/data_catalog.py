from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd

from .config import DATA_ROOT, REQUIRED_FILES


@dataclass(frozen=True)
class DatasetEntry:
    coin: str
    pair: str
    interval: str
    path: Path
    complete: bool
    missing_files: tuple[str, ...]


def discover_datasets(data_root: Path = DATA_ROOT) -> List[DatasetEntry]:
    entries: List[DatasetEntry] = []
    if not data_root.exists():
        return entries

    for coin_dir in sorted(path for path in data_root.iterdir() if path.is_dir()):
        for pair_dir in sorted(path for path in coin_dir.iterdir() if path.is_dir()):
            for interval_dir in sorted(path for path in pair_dir.iterdir() if path.is_dir()):
                missing = tuple(
                    filename
                    for filename in REQUIRED_FILES.values()
                    if not (interval_dir / filename).exists()
                )
                entries.append(
                    DatasetEntry(
                        coin=coin_dir.name,
                        pair=pair_dir.name,
                        interval=interval_dir.name,
                        path=interval_dir,
                        complete=not missing,
                        missing_files=missing,
                    )
                )
    return entries


def catalog_frame(entries: Iterable[DatasetEntry]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "coin": entry.coin,
                "pair": entry.pair,
                "interval": entry.interval,
                "path": str(entry.path),
                "complete": entry.complete,
                "missing_files": ",".join(entry.missing_files),
            }
            for entry in entries
        ]
    )


def filter_entries(
    entries: Iterable[DatasetEntry],
    interval: Optional[str] = None,
    complete_only: bool = True,
) -> List[DatasetEntry]:
    filtered = []
    for entry in entries:
        if complete_only and not entry.complete:
            continue
        if interval and entry.interval != interval:
            continue
        filtered.append(entry)
    return filtered


def main() -> None:
    entries = discover_datasets()
    frame = catalog_frame(entries)
    if frame.empty:
        print("No datasets found.")
        return

    print(frame.to_string(index=False))
    print()
    print(f"coins: {frame['coin'].nunique()}")
    print(f"pairs: {frame['pair'].nunique()}")
    print(f"interval datasets: {len(frame)}")
    print(f"complete datasets: {int(frame['complete'].sum())}")
    print(f"intervals: {sorted(frame['interval'].unique())}")


if __name__ == "__main__":
    main()
