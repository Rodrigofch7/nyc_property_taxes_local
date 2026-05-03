"""
hyperparameter_tuning.py
========================
Reusable hyperparameter search utilities for NYC property tax models.

Key improvement over sklearn RandomizedSearchCV:
  - Manual tuning loop with tqdm progress bar
  - Saves best params after EVERY iteration → safe to Ctrl+C and resume
  - On resume: skips already-evaluated combinations, loads best so far
  - Checkpoint file: models/tuning_checkpoint_<model_key>.json
"""

import json
import os
import time
import gc
import itertools
import random
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    print("  tip: run 'uv add tqdm' for progress bars")

# ── Paths ─────────────────────────────────────────────────────────────────────
DEFAULT_PARAMS_PATH = "/home/rodrigofrancachaves/project-nyc_property_taxes/models/best_params.json"
CHECKPOINT_DIR      = "/home/rodrigofrancachaves/project-nyc_property_taxes/models"


# ── Checkpoint helpers ────────────────────────────────────────────────────────
def _checkpoint_path(model_key: str) -> str:
    return os.path.join(CHECKPOINT_DIR, f"tuning_checkpoint_{model_key}.json")


def _load_checkpoint(model_key: str) -> dict:
    """Load tuning checkpoint: {best_score, best_params, evaluated: [...]}"""
    path = _checkpoint_path(model_key)
    try:
        with open(path) as f:
            ckpt = json.load(f)
        n_done = len(ckpt.get("evaluated", []))
        print(f"  Resuming from checkpoint — {n_done} combinations already evaluated")
        print(f"  Best so far: F1={ckpt['best_score']:.4f} | {ckpt['best_params']}")
        return ckpt
    except FileNotFoundError:
        return {"best_score": -1.0, "best_params": {}, "evaluated": []}


def _save_checkpoint(model_key: str, ckpt: dict) -> None:
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    with open(_checkpoint_path(model_key), "w") as f:
        json.dump(ckpt, f, indent=2)


def _clear_checkpoint(model_key: str) -> None:
    path = _checkpoint_path(model_key)
    if os.path.exists(path):
        os.remove(path)
        print(f"  Cleared checkpoint for '{model_key}'")


# ── Saved best params helpers ─────────────────────────────────────────────────
def save_best_params(model_key: str, params: dict, path: str = DEFAULT_PARAMS_PATH) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        with open(path) as f:
            all_params = json.load(f)
    except FileNotFoundError:
        all_params = {}
    all_params[model_key] = params
    with open(path, "w") as f:
        json.dump(all_params, f, indent=2)
    print(f"  Best params saved → {path} (key: '{model_key}')")


def load_best_params(model_key: str, path: str = DEFAULT_PARAMS_PATH) -> dict | None:
    try:
        with open(path) as f:
            params = json.load(f).get(model_key)
        if params:
            print(f"  Found saved params for '{model_key}': {params}")
        return params
    except FileNotFoundError:
        return None


def clear_saved_params(model_key: str, path: str = DEFAULT_PARAMS_PATH) -> None:
    try:
        with open(path) as f:
            all_params = json.load(f)
        if model_key in all_params:
            del all_params[model_key]
            with open(path, "w") as f:
                json.dump(all_params, f, indent=2)
            print(f"  Cleared saved params for '{model_key}'")
    except FileNotFoundError:
        pass


# ── Parameter grids ───────────────────────────────────────────────────────────
PARAM_GRIDS: dict = {
    "sgd_l2": {
        "alpha": [1e-4, 5e-4, 1e-3, 5e-3, 0.01, 0.05, 0.1, 1.0, 10.0],
    },
    "sgd_l1": {
        "alpha": [1e-4, 5e-4, 1e-3, 5e-3, 0.01, 0.05, 0.1, 1.0, 10.0],
    },
    "sgd_elasticnet": {
        "alpha":    [1e-4, 1e-3, 0.01, 0.1, 1.0],
        "l1_ratio": [0.1, 0.25, 0.5, 0.75, 0.9],
    },
    # Reduced grid — each combo takes ~5-10 min on 300k rows
    # 20 iterations × 3-fold CV = 60 fits total
    "lgbm": {
        "n_estimators":      [300, 500, 800],
        "learning_rate":     [0.05, 0.1, 0.2],
        "num_leaves":        [63, 127, 255],
        "min_child_samples": [20, 50],
        "subsample":         [0.8, 1.0],
        "colsample_bytree":  [0.8, 1.0],
        "reg_alpha":         [0.0, 0.5, 1.0],
        "reg_lambda":        [0.0, 0.5, 1.0],
    },
}


# ── Core: manual tuning loop with checkpointing ───────────────────────────────
def tune_with_checkpoints(
    estimator,
    param_grid: dict,
    X_train,
    y_train,
    *,
    model_key: str,
    n_iter: int = 20,
    cv: int = 3,
    scoring: str = "f1_macro",
    random_state: int = 42,
    force_retune: bool = False,
):
    """
    Manual hyperparameter search with:
      - tqdm progress bar showing iteration, score, elapsed time
      - checkpoint saved after EVERY iteration
      - safe to Ctrl+C and resume — already-evaluated combos are skipped
      - best params saved to best_params.json when done

    Returns the best params dict.
    """
    if force_retune:
        _clear_checkpoint(model_key)
        clear_saved_params(model_key)

    # Load checkpoint (picks up where we left off)
    ckpt = _load_checkpoint(model_key)

    # Build shuffled candidate list
    keys       = list(param_grid.keys())
    values     = list(param_grid.values())
    all_combos = [dict(zip(keys, v)) for v in itertools.product(*values)]
    rng        = random.Random(random_state)
    rng.shuffle(all_combos)
    candidates = all_combos[:n_iter]

    # Skip already evaluated combinations
    evaluated_strs = {json.dumps(e["params"], sort_keys=True) for e in ckpt["evaluated"]}
    remaining      = [c for c in candidates if json.dumps(c, sort_keys=True) not in evaluated_strs]

    if not remaining:
        print(f"  All {n_iter} combinations already evaluated — loading best params")
        return ckpt["best_params"]

    n_done = len(ckpt["evaluated"])
    print(f"\n  {n_done} done, {len(remaining)} remaining out of {n_iter} total")
    print(f"  Best so far: F1={ckpt['best_score']:.4f}")
    print(f"  Checkpoint: {_checkpoint_path(model_key)}")
    print(f"  (Safe to Ctrl+C — progress is saved after each iteration)\n")

    cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    iterator    = tqdm(remaining, desc=f"Tuning {model_key}", unit="combo") if HAS_TQDM else remaining

    for params in iterator:
        t0 = time.time()
        try:
            estimator.set_params(**params)
            scores = cross_val_score(
                estimator, X_train, y_train,
                cv=cv_splitter, scoring=scoring, n_jobs=1
            )
            score = float(scores.mean())
        except Exception as e:
            print(f"\n  WARNING: params {params} failed — {e}")
            score = -1.0

        elapsed = time.time() - t0
        is_best = score > ckpt["best_score"]

        # Update checkpoint
        ckpt["evaluated"].append({"params": params, "score": score})
        if is_best:
            ckpt["best_score"]  = score
            ckpt["best_params"] = params

        # Save after every single iteration
        _save_checkpoint(model_key, ckpt)

        n_total_done = len(ckpt["evaluated"])
        msg = f"  [{n_total_done}/{n_iter}] F1={score:.4f} | best={ckpt['best_score']:.4f} | {elapsed:.0f}s"
        if is_best:
            msg += " ← NEW BEST"

        if HAS_TQDM:
            tqdm.write(msg)
            iterator.set_postfix({
                "best": f"{ckpt['best_score']:.4f}",
                "last": f"{score:.4f}",
            })
        else:
            print(msg)

        gc.collect()

    best = ckpt["best_params"]
    save_best_params(model_key, best)
    print(f"\n  Tuning complete — best F1: {ckpt['best_score']:.4f}")
    print(f"  Best params: {best}")
    return best


# ── Convenience wrappers ──────────────────────────────────────────────────────
def tune_lgbm(
    estimator,
    X_train,
    y_train,
    param_grid: dict | None = None,
    *,
    model_key: str = "lgbm",
    params_path: str = DEFAULT_PARAMS_PATH,
    n_iter: int = 20,
    cv: int = 3,
    scoring: str = "f1_macro",
    random_state: int = 42,
    force_retune: bool | str = False,  # False | True | "safe"
):
    """
    force_retune options:
      False   → load cached params if available, skip CV
      True    → wipe cache, run CV, save whatever is best (even if worse)
      "safe"  → run CV, only overwrite cache if new result is better
    """
    # Load existing best for comparison later
    existing = load_best_params(model_key, path=params_path)

    if force_retune is False and existing:
        print(f"  Skipping search — loading saved params for '{model_key}'")
        estimator.set_params(**existing)
        estimator.fit(X_train, y_train)
        return estimator

    if force_retune is True:
        # Wipe cache and start fresh
        _clear_checkpoint(model_key)
        clear_saved_params(model_key)
        existing = None

    # force_retune is True or "safe" — run the search
    grid = param_grid or PARAM_GRIDS["lgbm"]
    best_params = tune_with_checkpoints(
        estimator, grid, X_train, y_train,
        model_key=model_key,
        n_iter=n_iter, cv=cv, scoring=scoring,
        random_state=random_state,
        force_retune=(force_retune is True),  # only wipe checkpoint if True
    )

    # Get the new best score from checkpoint
    ckpt = _load_checkpoint(model_key)
    new_score = ckpt["best_score"]

    if force_retune == "safe" and existing:
        # Compare new vs old — need to score old params too
        print(f"\n  Comparing new vs existing params...")
        from sklearn.model_selection import cross_val_score, StratifiedKFold
        cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
        estimator.set_params(**existing)
        old_scores = cross_val_score(
            estimator, X_train, y_train,
            cv=cv_splitter, scoring=scoring, n_jobs=1
        )
        old_score = float(old_scores.mean())

        print(f"  Old params CV F1: {old_score:.4f}")
        print(f"  New params CV F1: {new_score:.4f}")

        if new_score > old_score:
            print(f"  ✓ New params are better — updating cache")
            save_best_params(model_key, best_params, path=params_path)
            final_params = best_params
        else:
            print(f"  ✗ Old params are better — keeping cache unchanged")
            final_params = existing
    else:
        final_params = best_params

    print(f"\n  Fitting final model with best params...")
    estimator.set_params(**final_params)
    estimator.fit(X_train, y_train)
    return estimator


def tune_sgd(
    estimator,
    X_train,
    y_train,
    penalty: str = "l2",
    *,
    model_key: str | None = None,
    params_path: str = DEFAULT_PARAMS_PATH,
    cv: int = 5,
    scoring: str = "f1_macro",
    force_retune: bool = False,
):
    key = model_key or f"sgd_{penalty}"
    if not force_retune:
        saved = load_best_params(key, path=params_path)
        if saved:
            print(f"  Skipping search — loading saved params for '{key}'")
            estimator.set_params(**saved)
            estimator.fit(X_train, y_train)
            return estimator

    grid_key    = f"sgd_{penalty}" if f"sgd_{penalty}" in PARAM_GRIDS else "sgd_l2"
    param_grid  = PARAM_GRIDS[grid_key]
    best_params = tune_with_checkpoints(
        estimator, param_grid, X_train, y_train,
        model_key=key, cv=cv, scoring=scoring,
        n_iter=len(list(itertools.product(*param_grid.values()))),
        force_retune=force_retune,
    )
    estimator.set_params(**best_params)
    estimator.fit(X_train, y_train)
    return estimator


def tune_generic(
    estimator,
    X_train,
    y_train,
    model_key: str,
    *,
    n_iter: int = 20,
    cv: int = 3,
    scoring: str = "f1_macro",
    random_state: int = 42,
    force_retune: bool = False,
):
    if not force_retune:
        saved = load_best_params(model_key)
        if saved:
            print(f"  Skipping search — loading saved params for '{model_key}'")
            estimator.set_params(**saved)
            estimator.fit(X_train, y_train)
            return estimator

    if model_key not in PARAM_GRIDS:
        raise KeyError(f"'{model_key}' not in PARAM_GRIDS. Available: {list(PARAM_GRIDS.keys())}")

    best_params = tune_with_checkpoints(
        estimator, PARAM_GRIDS[model_key], X_train, y_train,
        model_key=model_key, n_iter=n_iter, cv=cv, scoring=scoring,
        random_state=random_state, force_retune=force_retune,
    )
    estimator.set_params(**best_params)
    estimator.fit(X_train, y_train)
    return estimator