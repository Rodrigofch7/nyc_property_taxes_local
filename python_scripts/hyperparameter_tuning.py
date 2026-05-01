"""
hyperparameter_tuning.py
========================
Reusable hyperparameter search utilities for NYC property tax models.
Supports GridSearchCV and RandomizedSearchCV with sensible defaults
for SGDClassifier, LGBMClassifier, and any sklearn-compatible estimator.

Usage:
    from hyperparameter_tuning import tune, PARAM_GRIDS
"""

import json
import os
import time
import gc
import numpy as np
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold,
)

# ── Saved params path ─────────────────────────────────────────────────────────
DEFAULT_PARAMS_PATH = "/home/rodrigofrancachaves/project-nyc_property_taxes/models/best_params.json"


def save_best_params(model_key: str, params: dict, path: str = DEFAULT_PARAMS_PATH) -> None:
    """Persist best params for model_key into the shared JSON file."""
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
    """
    Load saved params for model_key from the shared JSON file.
    Returns None if the file or key doesn't exist yet.
    """
    try:
        with open(path) as f:
            params = json.load(f).get(model_key)
        if params:
            print(f"  Found saved params for '{model_key}': {params}")
        return params
    except FileNotFoundError:
        return None

# ── Default parameter grids ───────────────────────────────────────────────────
PARAM_GRIDS: dict = {
    # Linear models
    "sgd_l2": {
        "alpha": [0.0001, 0.001, 0.01, 0.1],
    },
    "sgd_l1": {
        "alpha": [0.0001, 0.001, 0.01, 0.1],
    },
    "sgd_elasticnet": {
        "alpha":    [0.0001, 0.001, 0.01, 0.1],
        "l1_ratio": [0.15, 0.5, 0.85],
    },
    # Non-linear / tree models
    "lgbm": {
        "n_estimators":  [300, 500],
        "max_depth":     [5, 9, -1],
        "learning_rate": [0.05, 0.1],
        "num_leaves":    [31, 127],
    },
    "random_forest": {
        "n_estimators": [100, 300],
        "max_depth":    [10, 20, None],
        "min_samples_split": [2, 5],
    },
    "xgb": {
        "n_estimators":  [300, 500],
        "max_depth":     [4, 6, 8],
        "learning_rate": [0.05, 0.1],
        "subsample":     [0.8, 1.0],
    },
}


# ── Core tuning function ──────────────────────────────────────────────────────
def tune(
    estimator,
    param_grid: dict,
    X_train,
    y_train,
    *,
    method: str = "random",
    n_iter: int = 10,
    cv: int = 3,
    scoring: str = "f1_macro",
    n_jobs: int = 1,
    random_state: int = 42,
    verbose: int = 0,
    refit: bool = True,
):
    """
    Run hyperparameter search and return the fitted search object.

    Parameters
    ----------
    estimator   : sklearn-compatible estimator (unfitted)
    param_grid  : dict of parameter name → list of values
    X_train     : training features (array or DataFrame)
    y_train     : training labels
    method      : 'random' (RandomizedSearchCV) or 'grid' (GridSearchCV)
    n_iter      : number of random combinations (only used when method='random')
    cv          : number of cross-validation folds
    scoring     : sklearn scoring string (default 'f1_macro')
    n_jobs      : parallelism (-1 = all cores); keep at 1 for LightGBM
    random_state: for reproducibility
    verbose     : verbosity level passed to the search object
    refit       : whether to refit the best estimator on full X_train

    Returns
    -------
    search : fitted GridSearchCV or RandomizedSearchCV object
             access best model via search.best_estimator_
    """
    cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)

    common_kwargs = dict(
        estimator=estimator,
        cv=cv_splitter,
        scoring=scoring,
        n_jobs=n_jobs,
        verbose=verbose,
        refit=refit,
    )

    if method == "random":
        search = RandomizedSearchCV(
            param_distributions=param_grid,
            n_iter=n_iter,
            random_state=random_state,
            **common_kwargs,
        )
    elif method == "grid":
        search = GridSearchCV(
            param_grid=param_grid,
            **common_kwargs,
        )
    else:
        raise ValueError(f"method must be 'random' or 'grid', got '{method}'")

    t0 = time.time()
    search.fit(X_train, y_train)
    elapsed = time.time() - t0

    print(f"  Best params  : {search.best_params_}")
    print(f"  CV {scoring}: {search.best_score_:.4f}")
    print(f"  Search time  : {elapsed:.0f}s")

    gc.collect()
    return search


# ── Convenience wrappers ──────────────────────────────────────────────────────
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
    verbose: int = 1,
):
    """
    GridSearch over alpha (and optionally l1_ratio) for an SGDClassifier.
    If saved params exist for model_key, skips search and sets those params
    directly on the estimator instead.
    """
    key = model_key or f"sgd_{penalty}"
    saved = load_best_params(key, path=params_path)
    if saved:
        print(f"  Skipping CV search — loading saved params for '{key}'")
        estimator.set_params(**saved)
        estimator.fit(X_train, y_train)
        return estimator

    grid_key   = f"sgd_{penalty}" if f"sgd_{penalty}" in PARAM_GRIDS else "sgd_l2"
    param_grid = PARAM_GRIDS[grid_key]
    print(f"\nTuning SGD ({penalty.upper()}) — grid search over: {param_grid}")
    search = tune(
        estimator, param_grid, X_train, y_train,
        method="grid", cv=cv, scoring=scoring, n_jobs=1, verbose=verbose,
    )
    save_best_params(key, search.best_params_, path=params_path)
    return search


def tune_lgbm(
    estimator,
    X_train,
    y_train,
    param_grid: dict | None = None,
    *,
    model_key: str = "lgbm",
    params_path: str = DEFAULT_PARAMS_PATH,
    n_iter: int = 10,
    cv: int = 3,
    scoring: str = "f1_macro",
    random_state: int = 42,
    verbose: int = 0,
):
    """
    RandomizedSearch for LGBMClassifier.
    If saved params exist for model_key, skips search and fits directly.
    """
    saved = load_best_params(model_key, path=params_path)
    if saved:
        print(f"  Skipping CV search — loading saved params for '{model_key}'")
        estimator.set_params(**saved)
        estimator.fit(X_train, y_train)
        return estimator

    grid = param_grid or PARAM_GRIDS["lgbm"]
    print(f"\nTuning LightGBM — random search ({n_iter} iterations) over: {list(grid.keys())}")
    search = tune(
        estimator, grid, X_train, y_train,
        method="random", n_iter=n_iter, cv=cv, scoring=scoring,
        n_jobs=1, random_state=random_state, verbose=verbose,
    )
    save_best_params(model_key, search.best_params_, path=params_path)
    return search


def tune_generic(
    estimator,
    X_train,
    y_train,
    model_key: str,
    *,
    method: str = "random",
    params_path: str = DEFAULT_PARAMS_PATH,
    n_iter: int = 10,
    cv: int = 3,
    scoring: str = "f1_macro",
    random_state: int = 42,
    verbose: int = 0,
):
    """
    Tune any estimator using a named grid from PARAM_GRIDS.
    If saved params exist for model_key, skips search and fits directly.

    Example
    -------
        search = tune_generic(XGBClassifier(), X_train, y_sub, "xgb")
    """
    saved = load_best_params(model_key, path=params_path)
    if saved:
        print(f"  Skipping CV search — loading saved params for '{model_key}'")
        estimator.set_params(**saved)
        estimator.fit(X_train, y_train)
        return estimator

    if model_key not in PARAM_GRIDS:
        raise KeyError(
            f"'{model_key}' not in PARAM_GRIDS. "
            f"Available keys: {list(PARAM_GRIDS.keys())}"
        )
    grid = PARAM_GRIDS[model_key]
    print(f"\nTuning {model_key} — {method} search over: {list(grid.keys())}")
    search = tune(
        estimator, grid, X_train, y_train,
        method=method, n_iter=n_iter, cv=cv, scoring=scoring,
        n_jobs=1, random_state=random_state, verbose=verbose,
    )
    save_best_params(model_key, search.best_params_, path=params_path)
    return search
