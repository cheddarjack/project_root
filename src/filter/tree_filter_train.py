#!/usr/bin/env python3
"""
tree_filter_train.py  —  LightGBM-based trade filter trainer

• Reads config from tree_filter_config.yaml
• Uses DataFrame→DataFrame so feature names match (no warnings)
• Chronological split, train/test
• Scans thresholds but ALWAYS keeps ≥ min_kept trades
• Saves {model, threshold, features} via joblib
"""

import warnings
warnings.filterwarnings("ignore", message=".*valid feature names.*")

import yaml
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from sklearn.metrics import roc_auc_score
from tqdm.auto import tqdm
from lightgbm import LGBMClassifier

# ─────────────────────────── CONFIG ────────────────────────────
HERE    = Path(__file__).parent
CFG_PATH= HERE / "tree_filter_config.yaml"
cfg     = yaml.safe_load(CFG_PATH.read_text())

DATA_FILE    = Path(cfg["data_file"])
MODEL_OUT    = Path(cfg["model_out"])
TARGET_COL   = cfg["target_col"]
TEST_SPLIT   = float(cfg["test_split"])
MIN_KEPT     = int(cfg["min_kept"])
THRESHOLDS   = cfg["proba_thresholds"]
LGB_PARAMS   = cfg["lgb"]

# ────────────────────────── LOAD & SPLIT ─────────────────────────
df = pd.read_parquet(DATA_FILE)

# classification target for the model
df["is_win"] = (df[TARGET_COL] > 0).astype(int)

# features = all columns except P/L and is_win
features = [c for c in df.columns if c not in (TARGET_COL, "is_win")]

# chronological train/test split
cut = int(len(df) * (1 - TEST_SPLIT))
df_tr = df.iloc[:cut]
df_te = df.iloc[cut:]

X_tr = df_tr[features]
y_tr = df_tr["is_win"]
pnl_te = df_te[TARGET_COL].to_numpy()
X_te = df_te[features]
y_te = df_te["is_win"]

print(f"Train rows: {len(X_tr)} | Test rows: {len(X_te)}")

# ────────────────────────── TRAIN MODEL ─────────────────────────
print("Training LightGBM …")
model = LGBMClassifier(**LGB_PARAMS)
model.fit(X_tr, y_tr)

proba_te = model.predict_proba(X_te)[:, 1]
auc = roc_auc_score(y_te, proba_te)
print(f"Hold-out ROC-AUC {auc:.3f}")

# ─────────────────────── THRESHOLD SCAN ────────────────────────
best_thr = None
best_pnl = -np.inf
best_hit = 0.0

for thr in THRESHOLDS:
    mask = proba_te >= thr
    if mask.sum() < MIN_KEPT:
        continue
    mp = pnl_te[mask].mean()
    hit = (pnl_te[mask] > 0).mean()
    if mp > best_pnl:
        best_thr, best_pnl, best_hit = thr, mp, hit

# fallback: enforce MIN_KEPT if no grid thr qualified
if best_thr is None:
    # pick the probability that yields exactly MIN_KEPT kept
    sorted_idx = np.argsort(proba_te)
    thr_idx    = len(proba_te) - MIN_KEPT
    best_thr   = float(np.sort(proba_te)[thr_idx])
    mask       = proba_te >= best_thr
    best_pnl   = pnl_te[mask].mean()
    best_hit   = (pnl_te[mask] > 0).mean()

kept = int((proba_te >= best_thr).sum())

print(
    f"Best thr {best_thr:.2f} | hit-rate {best_hit:.1%} | "
    f"mean P/L {best_pnl:.3f} | kept {kept}"
)

# ───────────────────────── SAVE ARTIFACTS ─────────────────────────
MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
joblib.dump(
    {"model": model, "threshold": best_thr, "features": features},
    MODEL_OUT
)
print("Saved →", MODEL_OUT)

# final line = mean P/L (for any wrapper)
print(f"{best_pnl:.6f}")
