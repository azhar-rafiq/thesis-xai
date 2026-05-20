"""
Training Run Logger
-------------------
Call `save_models(...)` to export pkl + keras to model-exported/,
then `log_run(...)` to append a result row to training_log.csv.
"""

import csv
import datetime
import joblib
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score

MODEL_DIR = Path("model-exported")
LOG_FILE  = Path("training_log.csv")

FIELDNAMES = [
    "timestamp", "model_pkl", "model_h5",
    "n_train_patients", "n_test_patients", "n_train_slices", "n_test_slices",
    "best_cv_auc",
    "test_auc_epidural", "test_auc_intraparenchymal", "test_auc_intraventricular",
    "test_auc_subarachnoid", "test_auc_subdural", "test_auc_any",
    "test_auc_weighted",
    "filters1", "filters2", "filters3", "lr", "dropout", "epochs", "batch_size",
    "duration_min",
]


def save_models(grid, model_name=None):
    """
    Save best estimator as <model_name>.pkl and <model_name>.keras
    inside model-exported/. Returns (pkl_path, keras_path).

    model_name defaults to rsna_model_<cv_auc>_auc, e.g. rsna_model_0.8181_auc.
    """
    MODEL_DIR.mkdir(exist_ok=True)

    if model_name is None:
        model_name = f"rsna_model_{grid.best_score_:.4f}_auc"

    pkl_path   = MODEL_DIR / f"{model_name}.pkl"
    keras_path = MODEL_DIR / f"{model_name}.keras"

    joblib.dump(grid.best_estimator_, pkl_path, compress=1)
    grid.best_estimator_.named_steps["clf"].model_.save(str(keras_path))

    print(f"✅ Saved  {pkl_path}")
    print(f"✅ Saved  {keras_path}")
    return pkl_path, keras_path


def log_run(
    grid,
    y_test,
    y_pred_proba,
    x_train,
    x_test,
    label_cols,
    N_TRAIN_PATIENTS,
    N_TEST_PATIENTS,
    total_timedelta,
    model_pkl,
    model_h5,
):
    per_class = _per_class_aucs(y_test, y_pred_proba, label_cols)
    try:
        weighted_auc = round(roc_auc_score(y_test, y_pred_proba, average="weighted"), 4)
    except ValueError:
        weighted_auc = float("nan")

    bp = grid.best_params_

    row = {
        "timestamp":                 datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_pkl":                 str(model_pkl),
        "model_h5":                  str(model_h5),
        "n_train_patients":          N_TRAIN_PATIENTS,
        "n_test_patients":           N_TEST_PATIENTS,
        "n_train_slices":            len(x_train),
        "n_test_slices":             len(x_test),
        "best_cv_auc":               round(grid.best_score_, 4),
        "test_auc_epidural":         per_class.get("epidural",         float("nan")),
        "test_auc_intraparenchymal": per_class.get("intraparenchymal", float("nan")),
        "test_auc_intraventricular": per_class.get("intraventricular", float("nan")),
        "test_auc_subarachnoid":     per_class.get("subarachnoid",     float("nan")),
        "test_auc_subdural":         per_class.get("subdural",         float("nan")),
        "test_auc_any":              per_class.get("any",              float("nan")),
        "test_auc_weighted":         weighted_auc,
        "filters1":                  bp.get("clf__model__filters1"),
        "filters2":                  bp.get("clf__model__filters2"),
        "filters3":                  bp.get("clf__model__filters3"),
        "lr":                        bp.get("clf__model__lr"),
        "dropout":                   bp.get("clf__model__dropoutrate"),
        "epochs":                    bp.get("clf__epochs"),
        "batch_size":                bp.get("clf__batch_size"),
        "duration_min":              round(total_timedelta.total_seconds() / 60, 2),
    }

    write_header = not LOG_FILE.exists()
    with LOG_FILE.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    print(f"✅ Run logged → {LOG_FILE}  (total runs: {_count_runs()})")
    return pd.read_csv(LOG_FILE)


def _per_class_aucs(y_test, y_pred_proba, label_cols):
    aucs = {}
    for i, col in enumerate(label_cols):
        try:
            aucs[col] = round(roc_auc_score(y_test[:, i], y_pred_proba[:, i]), 4)
        except ValueError:
            aucs[col] = float("nan")
    return aucs


def _count_runs():
    if not LOG_FILE.exists():
        return 0
    with LOG_FILE.open() as f:
        return sum(1 for _ in f) - 1  # minus header
