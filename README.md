# 🏎️ Tick-Sim Pro — High-Speed Futures Back-Testing Sandbox

> **Status:** engineering showcase • indicator logic kept private  
> This project demonstrates a *production-grade* tick simulator / trade engine with
> multi-process back-testing, PDF visual analytics, and Numba-accelerated data
> structures.  
> The **edge indicator** itself is imported from `private.oscillator_engine`
> and is **not** part of this public repo.

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![Numba](https://img.shields.io/badge/Numba-Accelerated-green.svg)](https://numba.pydata.org/)
[![Opt-in GPU](https://img.shields.io/badge/GPU-optional-yellow.svg)]()
![CI](https://img.shields.io/badge/tests-planned-lightgrey)

---

## ✨ What this repo shows off

| Component | Skills & Tech | File(s) |
|-----------|---------------|---------|
| **Parallel back-test runner** | `concurrent.futures`, progress pipes, YAML config fan-out | `backtest_runner.py` |
| **Tick feeder** | PyArrow Parquet → zero-copy NumPy; auto-NaN filter; sec-since-midnight calc | `tick_feeder.py` |
| **Ring buffer (Numba)** | JIT-compiled push/dump kernels; volume accumulation; `O(1)` snapshots | `ring_buffer.py` |
| **Trade simulator** | Limit-order book, stop/TP logic, win/loss tracking | `simulator.py`, `trade_order.py` |
| **Visual debugger** | Matplotlib snapshot → rolling master PDF | `visualize.py` |
| **Plug-in indicator** | `from private.oscillator_engine import compute_signal` stubbed if absent | any script |

---

## 🚀 Quick-start

```bash
# clone
git clone https://github.com/<you>/TickSimPro.git
cd TickSimPro

# env + deps
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt   # pandas, pyarrow, numba, matplotlib, tqdm…

# sample data (1 day of MES ticks ≈ 25 MB)
bash scripts/fetch_sample_day.sh  # or drop your own Parquet file into data/

# run a single-file back-test
python backtest_runner.py --file data/sample_session.parquet
