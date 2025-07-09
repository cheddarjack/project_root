#!/usr/bin/env python3
"""
visualize.py ‒ plot the current ring-buffer (Last) plus optional
entry/exit markers, then append the page to a rolling master PDF.

REV-4 2025-06-23
"""
from __future__ import annotations

import uuid
import datetime as _dt
from pathlib import Path
from typing import Any, Iterable, Tuple
 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PyPDF2 import PdfMerger   # pip install PyPDF2

# ────────────────────────────────────────────────────────────────────────────
# config
# ────────────────────────────────────────────────────────────────────────────
OUTPUT_DIR      = Path("C:/Projects/project_root/output/trades/visual_trades")
MASTER_PDF_NAME = "trades_master.pdf"

LINE_COLOR  = (0.00, 0.00, 0.25)
LINE_WIDTH  = 0.5
ENTRY_STYLE = dict(marker="o", markersize=6, linestyle="None", color="green",
                   label="Entry")
EXIT_STYLE  = dict(marker="x", markersize=6, linestyle="None", color="red",
                   label="Exit")


# ────────────────────────────────────────────────────────────────────────────
# helpers
# ────────────────────────────────────────────────────────────────────────────
def _to_ts(val: Any) -> _dt.datetime | None:
    """Best-effort conversion to python datetime (UTC)."""
    if isinstance(val, _dt.datetime):
        return val
    if isinstance(val, pd.Timestamp):
        return val.to_pydatetime()
    if isinstance(val, (np.datetime64,)):
        return pd.to_datetime(val, errors="coerce").to_pydatetime() # type: ignore[return-value]

    if isinstance(val, (int, np.integer, float, np.floating)):
        ival = int(val)
        for div in (1, 1_000, 1_000_000, 1_000_000_000):          # s ms µs ns
            try:
                return _dt.datetime.utcfromtimestamp(ival / div)
            except (OSError, OverflowError):
                continue
        return None
    try:
        return pd.to_datetime(val, errors="coerce").to_pydatetime()
    except Exception:                                             # noqa: BLE001
        return None


def _safe_time_str(val: Any) -> str:
    ts = _to_ts(val)
    return ts.strftime("%Y-%m-%dT%H-%M-%S") if ts else "unknown_time"


# ────────────────────────────────────────────────────────────────────────────
# core plot
# ────────────────────────────────────────────────────────────────────────────
def _plot_ring_buffer(
    df: pd.DataFrame,
    pdf_path: Path,
    entries:   Iterable[Tuple[_dt.datetime, float]] | None = None,
    exits:     Iterable[Tuple[_dt.datetime, float]] | None = None,
    idx:       Iterable[Tuple[_dt.datetime, int]] | None = None,
    max_val:   Iterable[float] | None = None,
    min_val:   Iterable[float] | None = None,
    take_profit: Iterable[float] | None = None,
    stop_loss:   Iterable[float] | None = None,
    low_from_highs: Iterable[float] | None = None,
    high_from_lows: Iterable[float] | None = None,
    pnl: Iterable[float] | None = None
) -> None:
    """
    Draw price trace (equal-step x-axis) + optional entry/exit dots,
    then save to `pdf_path`.
    """
    if df.empty:
        return

    # ensure dtype & chronological order (oldest first)
    if not pd.api.types.is_datetime64_any_dtype(df["datetime"]):
        df = df.copy()
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df.sort_values("datetime", inplace=True, ignore_index=True)

    # build lookup: datetime → index position
    dt_to_idx = {dt: idx for idx, dt in enumerate(df["datetime"])}

    # filter/convert marker coords so they fall inside window
    def _filter_pts(
        pairs: Iterable[Tuple[_dt.datetime, float]] | None
    ) -> Tuple[np.ndarray, np.ndarray, list]:
        if not pairs:
            return np.array([]), np.array([]), []
        xs, ys, dts = [], [], []
        for t, price in pairs:
            idx = dt_to_idx.get(pd.Timestamp(t, tz=None), None)
            if idx is not None:
                xs.append(idx)
                ys.append(price)
                dts.append(pd.Timestamp(t, tz=None))
        return np.asarray(xs), np.asarray(ys), dts

    entry_x, entry_y, entry_dts = _filter_pts(entries)
    exit_x,  exit_y,  _         = _filter_pts(exits)

    # --- determine trim window before plotting ---
    x_pos = None
    trim_start = 0
    if idx is not None:
        idx_list = list(idx)
        if idx_list:
            # Use the datetime from the most recent idx tuple
            idx_dt, idx_offset = idx_list[-1]
            idx_dt = pd.Timestamp(idx_dt, tz=None)
            idx_marker_pos = dt_to_idx.get(idx_dt, None)
            if idx_marker_pos is not None:
                x_pos = idx_marker_pos - idx_offset
                if 0 <= x_pos < len(df):
                    # trim_start = max(int(x_pos) - 100, 0)
                    # df = df.iloc[trim_start:].reset_index(drop=True)
                    # After trimming, x_pos and marker indices must be shifted
                    x_pos = x_pos - trim_start
                    entry_x = entry_x - trim_start
                    exit_x = exit_x - trim_start

    x_idx = np.arange(len(df))
    y_val = df["Last"].astype(float).values

    # --- plotting ----------------------------------------------------------
    fig, ax = plt.subplots(figsize=(16, 6), dpi=150)
    ax.plot(x_idx, y_val, color=LINE_COLOR, linewidth=LINE_WIDTH, label="Last") #type: ignore[arg-type]

    # plot only the most recent entry marker
    if entry_x.size:
        ax.plot(entry_x[-1:], entry_y[-1:], **ENTRY_STYLE) # type: ignore[arg-type]
    # plot only the most recent exit marker
    if exit_x.size:
        ax.plot(exit_x[-1:], exit_y[-1:], **EXIT_STYLE) # type: ignore[arg-type]

    # plot horizontal line for most recent max_val
    if max_val is not None:
        max_val_list = list(max_val)
        if max_val_list:
            ax.axhline(y=max_val_list[-1], color="blue", linestyle="-", linewidth=1.2, label="max_val")

    # plot horizontal line for most recent min_val
    if min_val is not None:
        min_val_list = list(min_val)
        if min_val_list:
            ax.axhline(y=min_val_list[-1], color="blue", linestyle="-", linewidth=1.2, label="min_val")

    # plot horizontal line for most recent take_profit
    if take_profit is not None:
        take_profit_list = list(take_profit)
        if take_profit_list:
            ax.axhline(y=take_profit_list[-1], color="green", linestyle=":", linewidth=0.5, label="take_profit")

    # plot horizontal line for most recent stop_loss
    if stop_loss is not None:
        stop_loss_list = list(stop_loss)
        if stop_loss_list:
            ax.axhline(y=stop_loss_list[-1], color="red", linestyle=":", linewidth=0.5, label="stop_loss")

    # plot horizontal line for most recent low_from_highs
    if low_from_highs is not None:
        low_from_highs_list = list(low_from_highs)
        if low_from_highs_list:
            ax.axhline(y=low_from_highs_list[-1], color="purple", linestyle="-.", linewidth=0.75, label="low_from_highs")

    # plot horizontal line for most recent high_from_lows
    if high_from_lows is not None:
        high_from_lows_list = list(high_from_lows)
        if high_from_lows_list:
            ax.axhline(y=high_from_lows_list[-1], color="purple", linestyle="-.", linewidth=0.75, label="high_from_lows")

    # plot vertical line for idx (most recent only, using its own datetime)
    if x_pos is not None and 0 <= x_pos < len(df):
        ax.axvline(x=x_pos, color="orange", linestyle="--", linewidth=1.5, label="idx marker")

    # --- highlight title color based on pnl ---
    title_color = "black"
    if pnl is not None:
        pnl_list = list(pnl)
        if pnl_list:
            last_pnl = pnl_list[-1]
            if last_pnl < 0:
                title_color = "red"
            elif last_pnl > 0:
                title_color = "green"

    # cosmetics
    ax.set_title(
        f"Ring Buffer snapshot @ {_safe_time_str(df['datetime'].iloc[-1])}",
        color=title_color
    )
    ax.set_xlabel("Step (oldest ➜ newest)")
    ax.set_ylabel("Last")
    ax.grid(True, linestyle=":")
    ax.legend(loc="upper left")

    # pretty x-ticks: just show first & last index
    ax.set_xticks([0, len(df) - 1])
    ax.set_xticklabels(["0", str(len(df) - 1)])

    fig.tight_layout()
    fig.savefig(pdf_path, format="pdf")
    plt.close(fig)


# ────────────────────────────────────────────────────────────────────────────
# public API
# ────────────────────────────────────────────────────────────────────────────
def append_plot_to_master(
    df: pd.DataFrame,
    output_dir : Path = OUTPUT_DIR,
    master_name: str  = MASTER_PDF_NAME,
    *,
    entries: Iterable[Tuple[_dt.datetime, float]] | None = None,
    exits:   Iterable[Tuple[_dt.datetime, float]] | None = None,
    idx:     Iterable[Tuple[_dt.datetime, int]] | None = None,
    max_val: Iterable[float] | None = None,
    min_val: Iterable[float] | None = None,
    take_profit: Iterable[float] | None = None,
    stop_loss: Iterable[float] | None = None,
    low_from_highs: Iterable[float] | None = None,
    high_from_lows: Iterable[float] | None = None,
    pnl: Iterable[float] | None = None
) -> None:
    """
    Append a snapshot page to the rolling master PDF.
    `entries` / `exits` – iterable of (datetime, price) for markers.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    tmp_pdf = output_dir / f"_tmp_{uuid.uuid4().hex}.pdf"

    _plot_ring_buffer(df, tmp_pdf, entries=entries, exits=exits, idx=idx,
                      max_val=max_val, min_val=min_val, take_profit=take_profit, stop_loss=stop_loss,
                        low_from_highs=low_from_highs, high_from_lows=high_from_lows, pnl=pnl)

    if not tmp_pdf.exists():
        return

    master_pdf = output_dir / master_name
    if not master_pdf.exists():
        tmp_pdf.rename(master_pdf)
        return

    # merge master + tmp
    new_master = output_dir / f"_{uuid.uuid4().hex}.pdf"
    merger = PdfMerger()
    with master_pdf.open("rb") as fh_master, tmp_pdf.open("rb") as fh_tmp:
        merger.append(fh_master)
        merger.append(fh_tmp)
        with new_master.open("wb") as fh_out:
            merger.write(fh_out)
    merger.close()

    master_pdf.unlink(missing_ok=True)
    tmp_pdf.unlink(missing_ok=True)
    new_master.rename(master_pdf)
