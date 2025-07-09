# tick_feeder.py  — v0.5  (single‑file interface)
"""Lightweight tick feeder that **guarantees one‑file‑only** when the caller
passes a concrete path.  The old directory + pattern workflow still works for
legacy scripts.
"""
from __future__ import annotations
import os
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

_NS_PER_DAY = 86_400_000_000_000  # 24 h in nanoseconds (24 h)

class TickFeeder:
    def __init__(
        self,
        file_path: str | os.PathLike | None = None,
        *,
        data_directory: str | os.PathLike | None = None,
        file_pattern: str = "*.parquet",
        infer_total_rows_for_tqdm: bool = True,
    ) -> None:
        """Initialise a feeder.

        Parameters
        ----------
        file_path : path‑like | None
            If given, the feeder will read **exactly this file** and stop at EOF.
        data_directory, file_pattern : str | Path
            Legacy mode – discover multiple files; the caller must handle
            roll‑over semantically (not recommended for parallel workers).
        infer_total_rows_for_tqdm : bool
            Compute total rows for external progress bars.
        """
        if file_path is not None:
            self._files = [Path(file_path)]
        else:
            if data_directory is None:
                raise ValueError("Either file_path or data_directory must be provided")
            self._files = sorted(Path(data_directory).glob(file_pattern))
            if not self._files:
                raise FileNotFoundError(f"No parquet files matching {file_pattern} in {data_directory}")

        self._file_idx  = -1  # before first file
        self._row_idx   = 0
        self._arr: np.recarray | None = None
        self._uid_ctr   = 0
        self._new_day   = False

        self.total_rows: int | None = None
        if infer_total_rows_for_tqdm:
            self.total_rows = sum(pq.ParquetFile(f).metadata.num_rows for f in self._files)

        self._load_next_file()

    # ------------------------------------------------------------------
    def get_next_tick(self) -> dict | None:
        if self._arr is None:
            return None

        if self._row_idx == len(self._arr):  # end of current file
            if not self._load_next_file():
                return None

        r = self._arr[self._row_idx]
        self._row_idx += 1

        tick = {
            "datetime":  r.datetime,
            "Last":      r.Last,
            "Bid":       r.Bid,
            "Ask":       r.Ask,
            "Volume":    r.Volume,
            "sec_sm":    r.sec_sm,
            "unique_id": self._uid_ctr,
            "new_day":   self._new_day,
        }
        self._uid_ctr += 1
        self._new_day  = False  # reset after first tick
        return tick

    # ------------------------------------------------------------------
    def _load_next_file(self) -> bool:
        """Load the next parquet file. Returns *False* if none remain."""
        self._file_idx += 1
        if self._file_idx >= len(self._files):
            self._arr = None
            return False

        df = pq.read_table(self._files[self._file_idx]).to_pandas(split_blocks=True)

        df = df.dropna(how="any").reset_index(drop=True) # drop rows with any NaN values, especially every 15 seconds due to candle simulation

        ts_ns = df["datetime"].values.view("int64")
        sec_sm = (ts_ns % _NS_PER_DAY) / 1_000_000_000.0  # convert to float seconds
        df["sec_sm"] = sec_sm.astype("float64")

        cols = ["datetime", "Last", "Bid", "Ask", "Volume", "sec_sm"]
        self._arr = df[cols].to_records(index=False)

        self._row_idx = 0
        self._new_day = True
        return True
