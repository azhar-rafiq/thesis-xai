"""
RSNA Intracranial Hemorrhage -- Hybrid Ensemble (CNN + ViT)
averages softmax outputs of the best CNN and ViT checkpoints. no re-training needed.
Run via sbatch: sbatch 3_train_hybrid.sh
"""

import gc
import os
import sys
import json
import datetime
import types
import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score
from pathlib import Path
from dotenv import load_dotenv

sys.path.insert(0, os.path.expanduser('~/thesis-xai'))
from training_logger import save_models, log_run

load_dotenv()
SEED = 20260605
tf.random.set_seed(SEED)
np.random.seed(SEED)

CACHE_DIR        = Path(os.getenv('KAGGLE_CACHE')) / 'preprocessed'
MODEL_EXPORT_DIR = '/rds/projects/k/karwatha-karwath-hds-pg-research/axr1222/models'
IMG_SIZE         = 256
BATCH_SIZE       = 64
label_cols       = ['epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural']


# ---- load preprocessed data ----
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
    print("WARNING: No GPU detected! Inference will be slow.", flush=True)


# ---- load CNN checkpoint ----
CNN_CKPT = os.path.join(MODEL_EXPORT_DIR, 'rsna_cnn_20260605_05000_auc.keras')
if not os.path.exists(CNN_CKPT):
    raise FileNotFoundError(f"CNN checkpoint not found: {CNN_CKPT}")
print(f"\nLoading CNN: {os.path.basename(CNN_CKPT)}", flush=True)
cnn_model = tf.keras.models.load_model(CNN_CKPT)
print(f"  layers: {len(cnn_model.layers)}", flush=True)


# ---- load ViT checkpoint ----
class PositionEmbedding(tf.keras.layers.Layer):
    def __init__(self, num_patches, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.num_patches = num_patches
        self.embed       = tf.keras.layers.Embedding(input_dim=num_patches, output_dim=embed_dim)

    def call(self, x):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        return x + self.embed(positions)

    def get_config(self):
        config = super().get_config()
        config.update({'num_patches': self.num_patches,
                       'embed_dim':   self.embed.output_dim})
        return config

VIT_CKPT = os.path.join(MODEL_EXPORT_DIR, 'rsna_vit_20260605_08034_auc.keras')
if not os.path.exists(VIT_CKPT):
    raise FileNotFoundError(f"ViT checkpoint not found: {VIT_CKPT}")
print(f"Loading ViT:  {os.path.basename(VIT_CKPT)}", flush=True)
vit_model = tf.keras.models.load_model(
    VIT_CKPT, custom_objects={'PositionEmbedding': PositionEmbedding}
)
print(f"  layers: {len(vit_model.layers)}", flush=True)


# ---- build ensemble ----
# freeze both sub-models; no weights are updated
cnn_model.trainable = False
vit_model.trainable = False

inputs       = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name='input')
ensemble_out = tf.keras.layers.Average()([cnn_model(inputs), vit_model(inputs)])
ensemble     = tf.keras.Model(inputs, ensemble_out, name='ensemble_cnn_vit')
print(f"\nEnsemble built. total params: {ensemble.count_params():,}", flush=True)


# ---- batch inference helper ----
def predict_batched(model, X, batch_size=BATCH_SIZE):
    return np.vstack([
        model.predict_on_batch(X[i : i + batch_size].copy())
        for i in range(0, len(X), batch_size)
    ])


# ---- val set: compare CNN, ViT, ensemble ----
print(f"\nVal set inference...", flush=True)
started = datetime.datetime.now()

y_val_cnn = predict_batched(cnn_model, x_val)
y_val_vit = predict_batched(vit_model, x_val)
y_val_ens = (y_val_cnn + y_val_vit) / 2.0

print(f"  done in {(datetime.datetime.now() - started).seconds}s", flush=True)
print(f"\nVal AUC (weighted):")
for name, preds in [('CNN',      y_val_cnn),
                    ('ViT',      y_val_vit),
                    ('Ensemble', y_val_ens)]:
    auc = roc_auc_score(y_val, preds, average='weighted')
    print(f"  {name:10s}  {auc:.4f}", flush=True)

val_auc = roc_auc_score(y_val, y_val_ens, average='weighted')
del y_val_cnn, y_val_vit, y_val_ens
gc.collect()


# ---- test set ----
print(f"\nTest set inference...", flush=True)
started = datetime.datetime.now()

y_test_cnn   = predict_batched(cnn_model, x_test)
y_test_vit   = predict_batched(vit_model, x_test)
y_pred_proba = (y_test_cnn + y_test_vit) / 2.0

print(f"  done in {(datetime.datetime.now() - started).seconds}s", flush=True)

print(f"\nTest AUC per class:")
print(f"  {'subtype':25s}  {'CNN':>7}  {'ViT':>7}  {'Ensemble':>8}")
for i, col in enumerate(label_cols):
    try:
        auc_cnn = roc_auc_score(y_test[:, i], y_test_cnn[:, i])
        auc_vit = roc_auc_score(y_test[:, i], y_test_vit[:, i])
        auc_ens = roc_auc_score(y_test[:, i], y_pred_proba[:, i])
        print(f"  {col:25s}  {auc_cnn:.4f}   {auc_vit:.4f}   {auc_ens:.4f}")
    except ValueError:
        print(f"  {col:25s}  N/A")

w_cnn = roc_auc_score(y_test, y_test_cnn, average='weighted')
w_vit = roc_auc_score(y_test, y_test_vit, average='weighted')
w_ens = roc_auc_score(y_test, y_pred_proba, average='weighted')
print(f"  {'weighted average':25s}  {w_cnn:.4f}   {w_vit:.4f}   {w_ens:.4f}")

del y_test_cnn, y_test_vit
gc.collect()


# ---- save and log ----
os.chdir(os.path.expanduser('~/thesis-xai'))

dummy_grid = types.SimpleNamespace(
    best_params_={
        'clf__batch_size': BATCH_SIZE,
        'clf__epochs':     0,
        'cnn_checkpoint':  os.path.basename(CNN_CKPT),
        'vit_checkpoint':  os.path.basename(VIT_CKPT),
    },
    best_score_=val_auc,
)

pkl_path, keras_path = save_models(dummy_grid, keras_model=ensemble, method='hybrid')

log_run(
    grid=dummy_grid,
    y_test=y_test,
    y_pred_proba=y_pred_proba,
    method='hybrid',
    loss='focal',
    x_train=x_train,
    x_test=x_test,
    label_cols=label_cols,
    N_TRAIN_PATIENTS=meta['N_TRAIN_PATIENTS'],
    N_TEST_PATIENTS=meta['N_TEST_PATIENTS'],
    total_timedelta=datetime.timedelta(0),
    model_pkl=pkl_path,
    model_h5=keras_path,
)

print("\nDone!")
