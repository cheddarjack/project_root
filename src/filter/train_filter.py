#!/usr/bin/env python3
"""
train_filter.py — self-contained trainer for the trade-filter model.

 • Reads YAML config
 • Accepts JSON overrides via --override
 • Prints mean-P/L as the *last* stdout line (Optuna parses it)
"""
from __future__ import annotations
import argparse, json, copy, sys, warnings, math
from pathlib import Path
from typing import Dict, Any 

import joblib, numpy as np, pandas as pd, yaml
from tqdm.auto import tqdm
from sklearn.metrics import roc_auc_score

# ───────────────────────────────── config ─────────────────────────────────
HERE = Path(__file__).parent
BASE  = yaml.safe_load((HERE / "filter_config.yaml").read_text())

def deep_merge(a:dict,b:dict)->dict:
    out = copy.deepcopy(a)
    for k,v in b.items():
        if isinstance(v,dict) and k in out and isinstance(out[k],dict):
            out[k] = deep_merge(out[k], v)
        else: out[k]=copy.deepcopy(v)
    return out

# ─────────────────────────────── data IO ────────────────────────────────
def load_ds(fp:str|Path, tgt:str):
    df = pd.read_parquet(fp)
    df["is_win"] = (df[tgt]>0).astype(np.uint8)
    X = df.drop(columns=[tgt,"is_win"]).to_numpy()
    y = df["is_win"].to_numpy()
    pnl = df[tgt].to_numpy()
    feats = df.drop(columns=[tgt,"is_win"]).columns.tolist()
    return X,y,pnl,feats

def split(X,y,pnl,ratio):
    cut=int(len(X)*(1-ratio)); return X[:cut],X[cut:],y[:cut],y[cut:],pnl[cut:]

# ────────────────────────── model builders ─────────────────────────────
def build(kind:str, hp:Dict[str,Any]):
    if kind=="lightgbm":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(**hp)
    if kind=="xgboost":
        from xgboost import XGBClassifier
        # guarantee n_jobs=1 so trials can parallelise nicely
        hp = {**hp, "n_jobs": 1, "verbosity": 0}
        return XGBClassifier(**hp)
    if kind=="hist_gbm":
        from sklearn.ensemble import HistGradientBoostingClassifier
        return HistGradientBoostingClassifier(**hp)
    if kind=="extra_trees":
        from sklearn.ensemble import ExtraTreesClassifier
        return ExtraTreesClassifier(**hp)
    if kind=="ada_boost":
        from sklearn.ensemble import AdaBoostClassifier
        return AdaBoostClassifier(**hp)
    if kind=="catboost":
        from catboost import CatBoostClassifier #type: ignore
        return CatBoostClassifier(**hp, verbose=False)
    if kind=="random_forest":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(**hp)
    if kind=="log_reg":
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        return make_pipeline(StandardScaler(with_mean=False),
                             LogisticRegression(**hp))
    raise ValueError(kind)

# ───────────────────────── threshold scan ─────────────────────────────
def opt_thr(prob,pnl,grid,min_kept):
    best_thr,best_pnl,best_hit = 0,-1e9,0
    for thr in grid:
        mask = prob>=thr
        if mask.sum()<min_kept: continue
        mp = pnl[mask].mean(); hit = (pnl[mask]>0).mean()
        if mp>best_pnl:
            best_thr,best_pnl,best_hit = thr,mp,hit
    return best_thr,best_pnl,best_hit

# ──────────────────────────── training  ───────────────────────────────
def train(cfg:Dict[str,Any])->float:
    X,y,pnl,feats = load_ds(cfg["data_file"], cfg["target_pnl"])
    Xtr,Xte,ytr,yte,pnlte = split(X,y,pnl,cfg["test_split"])
    kind  = cfg["model_type"];  hp = cfg["hyper"][kind]

    # progress bar just for XGB long runs
    if kind=="xgboost":
        total = hp["n_estimators"]
        chunk = 200
        model = build(kind,{**hp,"n_estimators":0})   # start at 0
        bar = tqdm(total=total,desc="XGB",unit="tree")
        built=0
        while built<total:
            step=min(chunk,total-built)
            model.set_params(n_estimators=built+step)
            model.fit(Xtr,ytr, xgb_model=model.get_booster() if built else None)
            built+=step; bar.update(step)
        bar.close()
    else:
        # silence LightGBM “no positive gain” spam
        if kind=="lightgbm":
            warnings.filterwarnings("ignore",
                message=".*No further splits with positive gain.*",
                category=UserWarning, module="lightgbm")
        model = build(kind,hp)
        model.fit(Xtr,ytr)

    prob = (model.predict_proba(Xte)[:,1]
            if hasattr(model,"predict_proba")
            else model.decision_function(Xte))
    thr,mpnl,hit = opt_thr(prob,pnlte,cfg["proba_thresholds"],cfg["min_kept"])

    joblib.dump({"model":model,"thr":thr,"feat":feats}, cfg["model_out"])
    print(f"{kind:13s} P/L {mpnl:.3f} hit {hit:.1%} thr {thr:.2f}")
    print(f"{mpnl:.6f}")        # last line for Optuna
    return mpnl

# ──────────────────────────── CLI ───────────────────────────────
if __name__=="__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--override", help="JSON overrides")
    cfg = deep_merge(BASE, json.loads(ap.parse_args().override) if ap.parse_args().override else {})
    train(cfg)
