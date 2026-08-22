"""
RSNA Intracranial Hemorrhage -- CNN Optuna Hyperparameter Search
replaces the GridSearchCV in 1_train_cnn.py with an Optuna TPE study,
then retrains the best config on the full training set.
Run via sbatch: sbatch 1b_train_cnn_optuna.sh
"""

import gc
import os
import sys
import json
import random
import datetime
import types
import numpy as np
import scipy.ndimage as ndi
import tensorflow as tf
from collections import defaultdict
from tensorflow.keras import models, layers, optimizers, callbacks
from sklearn.metrics import roc_auc_score
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from pathlib import Path
from dotenv import load_dotenv

try:
    import optuna
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'optuna', '--quiet'])
    import optuna

sys.path.insert(0, os.path.expanduser('~/thesis-xai'))
from training_logger import save_models, log_run


class KerasPruningCallback(callbacks.Callback):
    # avoids the optuna-integration[tfkeras] dependency, which isn't installed
    def __init__(self, trial, monitor):
        super().__init__()
        self.trial = trial
        self.monitor = monitor

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        value = logs.get(self.monitor)
        if value is None:
            return
        self.trial.report(value, step=epoch)
        if self.trial.should_prune():
            raise optuna.TrialPruned(f"Trial pruned at epoch {epoch}, {self.monitor}={value}")

load_dotenv()
SEED = 20260605
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ---- optuna study config ----
N_TRIALS    = 50   # number of Optuna trials
N_CV_FOLDS  = 3    # inner CV folds per trial (3 is faster than 5 for large search)
SUB_FRAC    = 0.05 # fraction of train used for search (5% ~= 2x the old 3%)

# additive augmentation (same as 1_train_cnn.py)
AUG_FLIP_PROB     = 0.5
AUG_ROT_PROB      = 0.5
AUG_ZOOM_PROB     = 0.5
AUG_TRANS_PROB    = 0.5
AUG_ROT_DEG       = 5.0
AUG_ZOOM_PCT      = 0.08
AUG_TRANS_PCT     = 0.08
AUG_EXPECTED_MULT = 1.0 + AUG_FLIP_PROB + AUG_ROT_PROB + AUG_ZOOM_PROB + AUG_TRANS_PROB

CACHE_DIR  = Path(os.getenv('KAGGLE_CACHE')) / 'preprocessed'
IMG_SIZE   = 256
BATCH_SIZE = 64
label_cols = ['epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural']

MODEL_EXPORT_DIR = '/rds/projects/k/karwatha-karwath-hds-pg-research/axr1222/models'
os.makedirs(MODEL_EXPORT_DIR, exist_ok=True)


# ---- load preprocessed cache ----
sidecars = sorted(CACHE_DIR.glob(f'*_{IMG_SIZE}_meta.json'))
if not sidecars:
    raise FileNotFoundError(f"No preprocessed cache in {CACHE_DIR}. Run 0_preprocess_rsna.py first.")
meta_path = sidecars[-1]
meta      = json.loads(meta_path.read_text())
base      = str(meta_path).replace('_meta.json', '')

print(f"Loading cache: {meta_path.name}", flush=True)
print(f"  train: {meta['n_train_slices']} slices, {meta['N_TRAIN_PATIENTS']} patients", flush=True)
print(f"  val:   {meta['n_val_slices']} slices,  {meta['N_VAL_PATIENTS']} patients", flush=True)
print(f"  test:  {meta['n_test_slices']} slices,  {meta['N_TEST_PATIENTS']} patients", flush=True)

x_train = np.load(f'{base}_x_train.npy', mmap_mode='r')
y_train = np.load(f'{base}_y_train.npy', mmap_mode='r')
x_val   = np.load(f'{base}_x_val.npy',   mmap_mode='r')
y_val   = np.load(f'{base}_y_val.npy',   mmap_mode='r')
x_test  = np.load(f'{base}_x_test.npy',  mmap_mode='r')
y_test  = np.load(f'{base}_y_test.npy',  mmap_mode='r')

print(f"\nx_train: {x_train.shape}, y_train: {y_train.shape}", flush=True)
print(f"x_val:   {x_val.shape},   y_val:   {y_val.shape}", flush=True)
print(f"x_test:  {x_test.shape},  y_test:  {y_test.shape}", flush=True)


# ---- GPU setup ----
gpus = tf.config.list_physical_devices('GPU')
print(f"\nGPUs available: {len(gpus)}", flush=True)
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
    details = tf.config.experimental.get_device_details(gpu)
    print(f"  {gpu.name} - {details.get('device_name', 'unknown')} "
          f"(compute {details.get('compute_capability', ('?','?'))[0]}."
          f"{details.get('compute_capability', ('?','?'))[1]})", flush=True)
if not gpus:
    print("WARNING: No GPU detected! Training will be slow.", flush=True)


# ---- augmentation helpers (identical to 1_train_cnn.py) ----
def _zoom_and_crop(img, factor):
    h, w = img.shape[:2]
    zoomed = ndi.zoom(img, [factor, factor, 1.0], mode='constant', cval=0.0, order=1)
    zh, zw = zoomed.shape[:2]
    if factor >= 1.0:
        y0 = (zh - h) // 2
        x0 = (zw - w) // 2
        return zoomed[y0:y0 + h, x0:x0 + w, :]
    else:
        result = np.zeros_like(img)
        py = (h - zh) // 2
        px = (w - zw) // 2
        result[py:py + zh, px:px + zw, :] = zoomed
        return result


def _augment_variants(xi):
    variants = []
    if np.random.random() < AUG_FLIP_PROB:
        variants.append(np.flip(xi, axis=1).copy())
    if np.random.random() < AUG_ROT_PROB:
        angle = np.random.uniform(-AUG_ROT_DEG, AUG_ROT_DEG)
        variants.append(ndi.rotate(xi, angle, axes=(0, 1), reshape=False,
                                   mode='constant', cval=0.0, order=1).astype('float32'))
    if np.random.random() < AUG_ZOOM_PROB:
        factor = 1.0 + np.random.uniform(-AUG_ZOOM_PCT, AUG_ZOOM_PCT)
        variants.append(_zoom_and_crop(xi, factor).astype('float32'))
    if np.random.random() < AUG_TRANS_PROB:
        dy = np.random.uniform(-AUG_TRANS_PCT, AUG_TRANS_PCT) * xi.shape[0]
        dx = np.random.uniform(-AUG_TRANS_PCT, AUG_TRANS_PCT) * xi.shape[1]
        variants.append(ndi.shift(xi, [dy, dx, 0], mode='constant', cval=0.0,
                                  order=1).astype('float32'))
    return variants


def make_dataset(X, y, batch_size, shuffle, augment=False):
    n      = len(X)
    y_shape = (y.shape[1],) if y.ndim > 1 else ()
    sig = (
        tf.TensorSpec(shape=X.shape[1:], dtype=tf.float32),
        tf.TensorSpec(shape=y_shape,     dtype=tf.float32),
    )

    def gen():
        while True:
            for i in range(n):
                xi = X[i].astype('float32')
                yi = y[i].astype('float32')
                yield xi, yi
                if augment:
                    for aug_xi in _augment_variants(xi):
                        yield aug_xi, yi

    ds = tf.data.Dataset.from_generator(gen, output_signature=sig)
    if shuffle:
        ds = ds.shuffle(buffer_size=10000)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def build_model(filters1, filters2, filters3, lr, dropoutrate, focal_gamma):
    tf.keras.backend.clear_session()
    cnn = models.Sequential([
        layers.Input((IMG_SIZE, IMG_SIZE, 3)),

        layers.Conv2D(filters1, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(dropoutrate),

        layers.Conv2D(filters2, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(dropoutrate),

        layers.Conv2D(filters3, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(dropoutrate),

        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(dropoutrate),
        layers.Dense(5, activation='sigmoid'),
    ])
    cnn.compile(
        optimizer=optimizers.Adam(learning_rate=lr, clipnorm=1.0),
        loss=tf.keras.losses.BinaryFocalCrossentropy(gamma=focal_gamma),
        metrics=['recall', tf.keras.metrics.AUC(name='auc', multi_label=True)],
    )
    return cnn


# ---- subsample for search (stratified) ----
msss = MultilabelStratifiedShuffleSplit(
    n_splits=1, test_size=1.0 - SUB_FRAC, random_state=SEED)
sub_idx, _ = next(msss.split(x_train, y_train))
x_sub = x_train[sub_idx]
y_sub = y_train[sub_idx]
print(f"\nSearch subsample: {len(x_sub):,} slices ({len(x_sub)/len(x_train)*100:.1f}%)", flush=True)

msss_val = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.9, random_state=SEED)
val_idx, _ = next(msss_val.split(x_val, y_val))
x_val_sub = x_val[val_idx]
y_val_sub = y_val[val_idx]
print(f"Search val subset: {len(x_val_sub):,} slices", flush=True)


# ---- Optuna objective ----
def objective(trial):
    filters1    = trial.suggest_categorical('filters1',    [32])
    filters2    = trial.suggest_categorical('filters2',    [64, 128])
    filters3    = trial.suggest_categorical('filters3',    [128, 256])
    lr          = trial.suggest_float('lr',          1e-4, 1e-2, log=True)
    dropoutrate = trial.suggest_float('dropoutrate', 0.1,  0.5,  step=0.05)
    focal_gamma = trial.suggest_float('focal_gamma', 0.5,  3.0,  step=0.5)
    epochs      = trial.suggest_categorical('epochs', [20])

    # simple train/val split (no inner CV) to keep per-trial cost low
    model = build_model(filters1, filters2, filters3, lr, dropoutrate, focal_gamma)

    n_sub       = len(x_sub)
    steps       = int(np.ceil(n_sub * AUG_EXPECTED_MULT / BATCH_SIZE))
    val_steps   = int(np.ceil(len(x_val_sub) / BATCH_SIZE))
    train_ds    = make_dataset(x_sub,     y_sub,     BATCH_SIZE, shuffle=True,  augment=True)
    val_ds      = make_dataset(x_val_sub, y_val_sub, BATCH_SIZE, shuffle=False)

    cb = [
        callbacks.EarlyStopping(monitor='val_auc', patience=5, mode='max',
                                restore_best_weights=True),
        callbacks.ReduceLROnPlateau(monitor='val_auc', factor=0.5, patience=3,
                                    mode='max', min_lr=1e-6),
        KerasPruningCallback(trial, 'val_auc'),
    ]

    hist = model.fit(
        train_ds,
        steps_per_epoch=steps,
        validation_data=val_ds,
        validation_steps=val_steps,
        epochs=epochs,
        callbacks=cb,
        verbose=0,
    )

    best_val_auc = max(hist.history.get('val_auc', [0.0]))
    del model, train_ds, val_ds
    gc.collect()
    tf.keras.backend.clear_session()
    return best_val_auc


# ---- run study ----
print(f"\n{'='*60}")
print(f"Optuna search started at {datetime.datetime.now()}")
print(f"  n_trials={N_TRIALS}, sub_frac={SUB_FRAC}, batch_size={BATCH_SIZE}")
print(f"{'='*60}\n", flush=True)

optuna.logging.set_verbosity(optuna.logging.INFO)
sampler = optuna.samplers.TPESampler(seed=SEED)
pruner  = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5)
storage_path = os.path.join(MODEL_EXPORT_DIR, 'cnn_optuna.db')
study = optuna.create_study(
    direction='maximize', sampler=sampler, pruner=pruner,
    study_name='cnn_rsna_ich',
    storage=f'sqlite:///{storage_path}',
    load_if_exists=True,
)
print(f"Study storage: {storage_path} ({len(study.trials)} trials already recorded)", flush=True)

trials_csv_path = os.path.join(MODEL_EXPORT_DIR, 'cnn_optuna_trials.csv')

def dump_trial_csv(study, trial):
    import csv
    write_header = not os.path.exists(trials_csv_path)
    with open(trials_csv_path, 'a', newline='') as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(['trial_number', 'value', 'state', 'datetime', *trial.params.keys()])
        writer.writerow([trial.number, trial.value, trial.state.name,
                          datetime.datetime.now().isoformat(), *trial.params.values()])

remaining_trials = max(0, N_TRIALS - len(study.trials))
print(f"Trials remaining this run: {remaining_trials} (target {N_TRIALS} total)", flush=True)

search_start = datetime.datetime.now()
if remaining_trials > 0:
    study.optimize(objective, n_trials=remaining_trials, show_progress_bar=False,
                   callbacks=[dump_trial_csv])
search_time  = datetime.datetime.now() - search_start

print(f"\n{'='*60}")
print(f"Optuna search finished at {datetime.datetime.now()}")
h, rem = divmod(search_time.total_seconds(), 3600)
m, s   = divmod(rem, 60)
print(f"Total search time: {int(h)}h {int(m)}m {int(s)}s")
print(f"Completed trials:  {len(study.trials)}")
print(f"Best trial value:  {study.best_value:.4f}")
print(f"Best params:       {study.best_params}")
print(f"{'='*60}\n", flush=True)

bp = study.best_params
del x_sub, y_sub, x_val_sub, y_val_sub
gc.collect()


# ---- retrain best config on full training set ----
best_epochs = bp['epochs']
best_model  = build_model(
    filters1    = bp['filters1'],
    filters2    = bp['filters2'],
    filters3    = bp['filters3'],
    lr          = bp['lr'],
    dropoutrate = bp['dropoutrate'],
    focal_gamma = bp['focal_gamma'],
)

n_train     = len(x_train)
n_val       = len(x_val)
train_steps = int(np.ceil(n_train * AUG_EXPECTED_MULT / BATCH_SIZE))
val_steps   = int(np.ceil(n_val   / BATCH_SIZE))

train_ds = make_dataset(x_train, y_train, BATCH_SIZE, shuffle=True,  augment=True)
val_ds   = make_dataset(x_val,   y_val,   BATCH_SIZE, shuffle=False)

ckpt_path = os.path.join(
    MODEL_EXPORT_DIR,
    f'cnn_optuna_ckpt_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.keras'
)

print(f"Retraining best CNN on full x_train ({n_train:,} slices)...", flush=True)
retrain_started = datetime.datetime.now()
best_model.fit(
    train_ds,
    steps_per_epoch=train_steps,
    validation_data=val_ds,
    validation_steps=val_steps,
    epochs=best_epochs,
    callbacks=[
        callbacks.EarlyStopping(monitor='val_auc', patience=5, mode='max',
                                restore_best_weights=True),
        callbacks.ReduceLROnPlateau(monitor='val_auc', factor=0.5, patience=3,
                                    mode='max', min_lr=1e-6),
        callbacks.ModelCheckpoint(filepath=ckpt_path, monitor='val_auc', mode='max',
                                  save_best_only=True, verbose=1),
    ],
)
total = datetime.datetime.now() - retrain_started
print(f"Full retrain complete.", flush=True)


# ---- test set evaluation ----
y_pred_proba = np.vstack([
    best_model.predict_on_batch(x_test[i : i + BATCH_SIZE].copy())
    for i in range(0, len(x_test), BATCH_SIZE)
])

print("\nTest Set AUC per class:")
for i, col in enumerate(label_cols):
    try:
        auc = roc_auc_score(y_test[:, i], y_pred_proba[:, i])
        print(f"  {col:25s} AUC: {auc:.4f}")
    except ValueError:
        print(f"  {col:25s} AUC: N/A (single class)")
print(f"  {'weighted average':25s} AUC: {roc_auc_score(y_test, y_pred_proba, average='weighted'):.4f}")


# ---- save and log ----
os.chdir(os.path.expanduser('~/thesis-xai'))

dummy_grid = types.SimpleNamespace(
    best_params_={
        'clf__batch_size':          BATCH_SIZE,
        'clf__epochs':              best_epochs,
        'clf__model__filters1':     bp['filters1'],
        'clf__model__filters2':     bp['filters2'],
        'clf__model__filters3':     bp['filters3'],
        'clf__model__lr':           bp['lr'],
        'clf__model__dropoutrate':  bp['dropoutrate'],
    },
    best_score_=study.best_value,
)

pkl_path, keras_path = save_models(dummy_grid, keras_model=best_model, method='cnn_optuna')

log_run(
    grid=dummy_grid,
    y_test=y_test,
    y_pred_proba=y_pred_proba,
    method='cnn_optuna',
    loss='focal',
    x_train=x_train,
    x_test=x_test,
    label_cols=label_cols,
    N_TRAIN_PATIENTS=meta['N_TRAIN_PATIENTS'],
    N_TEST_PATIENTS=meta['N_TEST_PATIENTS'],
    total_timedelta=total,
    model_pkl=pkl_path,
    model_h5=keras_path,
)

print("\nDone!")
