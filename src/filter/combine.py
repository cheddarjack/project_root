"""
FIRST | Merge the twelve monthly parquet files into one tidy “learning set”
"""



"""
1. Collect paths
"""
import glob, pandas as pd, pyarrow.parquet as pq
results_dir = sorted(glob.glob("/home/cheddarjackk/Developer/project_root/output/results_short/*.parquet"))
combined_file = "/home/cheddarjackk/Developer/project_root/filter/data/oscillator_learning_set.parquet"



"""
2. Load-and-trim each file

Now instead of specifying columns to include, we drop the unwanted columns:
datetime, bid, ask, takeprofit, stoploss, trade_id, entry_time, entry_price, exit_price, reason.
This allows all other columns to be retained.
"""
exclude_cols = [
    "datetime",
    "bid",
    "ask",
    "limit",
    "take_profit",
    "stop_loss",
    "trade_id",
    "entry_time",
    "exit_time",
    "flag_time",
    "entry_price",
    "exit_price",
    "reason",
    "max_i",
    "min_i",
    "low_from_highs",
    "high_from_lows",
    "slice_avg",
    "med_osc"
]
dfs = []
for p in results_dir:
    # Read the entire file and drop columns in exclude_cols if they exist.
    df = pd.read_parquet(p)
    df = df.drop(columns=exclude_cols, errors="ignore")
    dfs.append(df)

"""
3. Concatenate and save
"""

data = pd.concat(dfs, ignore_index=True)
data.to_parquet(combined_file, index=False)

