"""
Parallel back‑test runner – **rev 2025‑07‑01‑k**

Fix: removed accidental duplicate code block that broke `_worker()`
(“`prev_trade_cnt = Nonefp:`” syntax error).  No logic changed.
"""
from __future__ import annotations

import os, sys, shutil, warnings, time, datetime as _dt
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp, threading

import yaml, pandas as pd
from tqdm import tqdm
import pyarrow.parquet as pq

from typing import Optional, Any

import numpy as np

# ── Local project imports ────────────────────────────────────────────────────
# point at /…/project_root, not src/
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

import src.mock_api.tick_feeder as tf  
from src.strategy_engine.ring_buffer import RingBuffer
import src.trade.trade_order as tor
from src.strategy_engine.visualize import append_plot_to_master
from private.oscillator_engine import compute_signal as _public_compute, SignalDict

try:
    # If your private copy exists, override it
    from private.oscillator_engine import compute_signal as _private_compute
    compute_signal = _private_compute
    PRIVATE = True
except ImportError:
    compute_signal = _public_compute
    PRIVATE = False


# ── Configuration ────────────────────────────────────────────────────────────
CONFIG_PATH = "/home/cheddarjackk/Developer/project_root/config/app_config.yaml"
with open(CONFIG_PATH) as fh:
    config = yaml.safe_load(fh)

DATA_DIR   = Path(config["Directory"]["data"]).expanduser().resolve()
PATTERN    = "session_*.parquet"
OUTPUT_DIR = Path(config["Directory"]["results"]).expanduser().resolve()

HITS      = config["Order"]["hit_num"]
RB_SIZE   = config["Main"]["RINGBUFFER"]
MIN_HITS  = config["Main"]["MIN_HITS"]
MAX_HITS  = config["Main"]["MAX_HITS"]
ORDER_TTL = config["Main"]["ORDER_TTL_SECONDS"]
VISUALIZE = config["Main"]["VISUALIZE"]
TDQM_OFF  = config.get("tdqm", {}).get("on_off_1", False)

if HITS < 0:
    RESULTS_DIR = OUTPUT_DIR/"results_long"
if HITS > 0:
    RESULTS_DIR = OUTPUT_DIR/"results_short"

VIS_DIR = OUTPUT_DIR/"visual_trades"

PROG_CHUNK = 1_000
DEDUP_COLS = ["entry_time", "exit_time", "side", "entry_price", "exit_price"]

# ── Helpers ──────────────────────────────────────────────────────────────────

def _wipe(path: Path):
    if path.exists():
        for p in path.iterdir():
            shutil.rmtree(p) if p.is_dir() else p.unlink()

def _ensure_dirs():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    VIS_DIR.mkdir(parents=True, exist_ok=True)

# ── Worker ───────────────────────────────────────────────────────────────────

def _worker(fp: Path, q, chunk=PROG_CHUNK):
    """Process *one* parquet file in an isolated trading engine."""
    from importlib import reload
    import src.trade.routers.sim_router as sim_router
    reload(sim_router)
    tor.router = sim_router.SimRouter()

    feeder = tf.TickFeeder(file_path=fp)
    rb = RingBuffer(max_size=RB_SIZE)

    prev_last = None
    ticks_since_msg = 0
    prev_trade_cnt = None
    accum_volume = 0.0

    vis_dir = VIS_DIR / fp.stem
    if VISUALIZE:
        _wipe(vis_dir); vis_dir.mkdir(parents=True, exist_ok=True)

    while True:
        tick = feeder.get_next_tick()
        if tick is None:
            break

        # accumulate volume per price every unchanging tick
        if prev_last is None or tick["Last"] != prev_last:
            accum_volume = tick["Volume"]
        else:
            accum_volume += tick["Volume"]

        # skip duplicate price: update only the volume in buffer
        if prev_last is not None and tick["Last"] == prev_last:
            rb.update_last_volume(accum_volume)
            ticks_since_msg += 1
            if ticks_since_msg >= chunk:
                q.put(ticks_since_msg); ticks_since_msg = 0
            continue
        prev_last = tick["Last"]

        # reset buffer at start of file
        if tick.get("new_day"):
            rb.reset()

        rb.push(tick["Last"], tick["sec_sm"], tick["unique_id"], tick["datetime"], accum_volume)

        if not rb.is_ready():
            ticks_since_msg += 1
            if ticks_since_msg >= chunk:
                q.put(ticks_since_msg); ticks_since_msg = 0
            continue

        lp, ss, _, _, cumulative_vol = rb.get_last_window()

        osc = compute_signal(lp, ss, cumulative_vol, MIN_HITS, MAX_HITS, config) or {}
        tor.on_tick(
            config=config,
            price=tick["Last"], bid=tick["Bid"], ask=tick["Ask"],
            sec_sm=tick["sec_sm"], dt=tick["datetime"], uid=tick["unique_id"],
            osc_sig=osc,
            order_ttl_seconds=ORDER_TTL
        )

        # ────────── VISUAL SNAPSHOT (only when a trade closes) ──────────
        if VISUALIZE:
            tc = tor.get_trade_counts()["total"]
            if prev_trade_cnt is None or tc != prev_trade_cnt:   # ⇢ a trade just closed
                prev_trade_cnt = tc

                # -------- ring-buffer → DataFrame (oldest-first) --------
                lp, ss, uid, dt_arr, vol = rb.get_last_window()          # newest-first
                vis_df = pd.DataFrame({
                    "Last"      : lp[::-1].astype(float),                 # reverse → oldest-first
                    "sec_sm"    : ss[::-1].astype(float),
                    "unique_id" : uid[::-1],
                    "datetime"  : pd.to_datetime(dt_arr[::-1]
                                                .astype("datetime64[ns]")),
                    "Volume"    : vol[::-1].astype(float)
                })

                # -------- most-recent closed trade markers --------------
                closed = tor.get_closed_trades_df()
                if closed.empty:
                    entries = exits = pnl_val = None
                else:
                    last   = closed.iloc[-1]
                    entries = [(last["entry_time"], last["entry_price"])]
                    exits   = [(last["exit_time"],  last["exit_price"])]
                    pnl_val = [last["pnl"]]        # colour-code title

                # -------- call visualiser -------------------------------
                append_plot_to_master(
                    df          = vis_df,
                    output_dir  = vis_dir,          # …/visual_trades/<stem>/
                    master_name = "trades_master.pdf",
                    entries     = entries,
                    exits       = exits,
                    idx         = [(tick["datetime"], 0)],   # highlight “now”
                    pnl         = pnl_val
                )



        ticks_since_msg += 1
        if ticks_since_msg >= chunk:
            q.put(ticks_since_msg); ticks_since_msg = 0

    if ticks_since_msg:
        q.put(ticks_since_msg)
    q.put(None)

    df = tor.get_closed_trades_df().drop_duplicates(DEDUP_COLS)
    out_path = RESULTS_DIR / f"{fp.stem}_{HITS}_trades.parquet"
    if out_path.exists():
        out_path.unlink()
    if not df.empty:
        df.to_parquet(out_path, index=False)
    wins, losses = (df["pnl"] > 0).sum(), (df["pnl"] < 0).sum()
    return {"file": fp.name, "trades": len(df), "wins": int(wins), "losses": int(losses), "pnl": float(df["pnl"].sum())}

# ── Progress consumer ────────────────────────────────────────────────────────

def _consume(q, bar, workers):
    done = 0
    while done < workers:
        try:
            inc = q.get()
        except EOFError:
            break
        if inc is None:
            done += 1
        else:
            bar.update(inc)

# ── Main orchestrator ────────────────────────────────────────────────────────

def main():
    start = time.time()
    print("===== PARALLEL BACKTEST START =====\n")
    _wipe(VIS_DIR)
    if config["Main"]["WIPE_RESULTS_DIR"]:
        _wipe(RESULTS_DIR)
    _ensure_dirs()

    files = sorted(DATA_DIR.glob(PATTERN))
    if not files:
        print("No parquet files found"); return

    rows = {fp: pq.ParquetFile(fp).metadata.num_rows for fp in files}
    total = sum(rows.values())
    cpus  = min(os.cpu_count() or 1, len(files))
    print(f"Files: {len(files)} | Total ticks: {total:,} | Procs: {cpus}\n")

    manager = mp.Manager(); q = manager.Queue()
    bar = tqdm(total=total, desc="Ticks", unit="tick", dynamic_ncols=True, disable=TDQM_OFF)
    threading.Thread(target=_consume, args=(q, bar, len(files)), daemon=True).start()

    summaries = []
    with ProcessPoolExecutor(max_workers=cpus) as ex:
        futs = {ex.submit(_worker, fp, q): fp for fp in files}
        for fut in as_completed(futs):
            try:
                summaries.append(fut.result())
            except Exception as e:
                print(f"❌ {futs[fut].name} » {e}")
                #print number of trades of the failed file
                failed_df = tor.get_closed_trades_df()
                if not failed_df.empty:
                    print(f"Failed file trades: {len(failed_df)}")

    bar.close()

    total_tr = sum(s["trades"] for s in summaries)
    wins, losses = sum(s["wins"] for s in summaries), sum(s["losses"] for s in summaries)
    pnl = sum(s["pnl"] for s in summaries)

    print("\n===== BACKTEST COMPLETE =====")
    print(f"\nFiles processed : {len(files)}")
    print(f"Hit number      : {HITS}")
    print(f"Trades          : {total_tr}")
    if total_tr:
        print(f"Win ratio       : {wins/total_tr*100:.2f}%")
    print(f"Total PnL       : {pnl:.2f}")
    print(f"Elapsed         : {time.time()-start:.1f}s")
    print(f"Results in      : {RESULTS_DIR}\n")

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
