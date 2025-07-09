#!/usr/bin/env python3
"""
bayes_filter.py – Optuna search with pruning & per-trial timeout.
"""
from __future__ import annotations
import argparse, json, subprocess, os, sys
from pathlib import Path
from datetime import timedelta

import optuna, yaml

HERE   = Path(__file__).parent
TRAIN  = HERE / "train_filter.py"
DB     = HERE / "filter_search.db"
BASE   = yaml.safe_load((HERE / "filter_config.yaml").read_text())

# ─────────── hyper space ───────────
def space(t):
    mdl = t.suggest_categorical("model",
      ["lightgbm","xgboost","hist_gbm","extra_trees",
       "ada_boost","random_forest","linear_svm",
       "naive_bayes","knn","log_reg"])

    if mdl == "lightgbm":
        return {"model_type": mdl, "hyper": {"lightgbm": {
            "n_estimators": t.suggest_int("lgb_n", 400, 4000, step=200),
            "learning_rate": t.suggest_float("lgb_lr", 1e-3, 0.3, log=True),
            "max_depth": t.suggest_int("lgb_md", 3, 12),
            "num_leaves": t.suggest_int("lgb_lea", 8, 256),
            "feature_fraction": t.suggest_float("lgb_ff", 0.4, 1.0),
            "bagging_fraction": t.suggest_float("lgb_bf", 0.4, 1.0),
            "min_gain_to_split": 0.0, "is_unbalance": True,
            "random_state": 42}}}


    if mdl == "hist_gbm":
        return {"model_type": mdl, "hyper": {"hist_gbm": {
            "learning_rate": t.suggest_float("hgb_lr", 1e-3, 0.3, log=True),
            "max_depth": t.suggest_int("hgb_md", 3, 12),
            "max_iter": t.suggest_int("hgb_it", 200, 1600, step=100)
        }}}

    if mdl == "extra_trees":
        return {"model_type": mdl, "hyper": {"extra_trees": {
            "n_estimators": t.suggest_int("et_n", 400, 4000, step=200),
            "max_depth": t.suggest_int("et_md", 3, 40),
            "min_samples_leaf": t.suggest_int("et_leaf", 1, 32),
            "random_state": 42}}}

    if mdl == "ada_boost":
        return {"model_type": mdl, "hyper": {"ada_boost": {
            "n_estimators": t.suggest_int("ada_n", 200, 2000, step=100),
            "learning_rate": t.suggest_float("ada_lr", 0.01, 2.0, log=True),
            "random_state": 42}}}

    if mdl == "random_forest":
        return {"model_type": mdl, "hyper": {"random_forest": {
            "n_estimators": t.suggest_int("rf_n", 400, 4000, step=200),
            "max_depth": t.suggest_int("rf_md", 3, 40),
            "min_samples_leaf": t.suggest_int("rf_leaf", 1, 32),
            "class_weight": "balanced", "random_state": 42}}}

    if mdl == "linear_svm":
        return {"model_type": mdl, "hyper": {"linear_svm": {
            "C": t.suggest_float("svm_C", 1e-3, 20, log=True)}}}

    if mdl == "naive_bayes":
        return {"model_type": mdl, "hyper": {"naive_bayes": {}}}

    if mdl == "knn":
        return {"model_type": mdl, "hyper": {"knn": {
            "n_neighbors": t.suggest_int("knn_k", 3, 25),
            "weights": t.suggest_categorical("knn_w", ["uniform", "distance"]),
            "metric": "euclidean"}}}

    return {"model_type": "log_reg", "hyper": {"log_reg": {
        "C": t.suggest_float("lr_C", 1e-3, 20, log=True),
        "max_iter": 4000, "solver": "lbfgs", "class_weight": "balanced"}}}

# ─────────── objective ───────────
def objective(trial):
    override = json.dumps(space(trial))
    run = subprocess.run(
        [sys.executable, str(TRAIN), "--override", override],
        cwd=HERE, capture_output=True, text=True, timeout=1800   # 3-minute limit
    )
    if run.returncode != 0:
        print(run.stderr, file=sys.stderr)
        return -1e9
    try:
        return float(run.stdout.strip().splitlines()[-1])
    except Exception:
        return -1e9

# ─────────── CLI ───────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--trials", type=int, default=600)
    p.add_argument("--jobs", type=int, default=os.cpu_count() or 4)
    p.add_argument("--fresh", action="store_true")
    args = p.parse_args()

    if args.fresh and DB.exists():
        DB.unlink()

    study = optuna.create_study(
        direction="maximize",
        study_name="trade_filter",
        storage=f"sqlite:///{DB}",
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=15)
    )
    study.optimize(objective,
                   n_trials=args.trials,
                   n_jobs=args.jobs,
                   timeout=timedelta(hours=6).total_seconds())

    print("best val:", study.best_value)
    print(json.dumps(study.best_params, indent=2))

if __name__ == "__main__":
    main()
