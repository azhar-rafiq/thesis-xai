"""
RSNA intracranial hemorrhage: ViT grid search training script.
run via sbatch: sbatch 2_train_vit.sh
"""

import gc
import os
import sys
import json
import random
import datetime
import types
import numpy as np
import tensorflow as tf
from collections import defaultdict
from tensorflow.keras import models, layers, optimizers, callbacks
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score
from scikeras.wrappers import KerasClassifier
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold, MultilabelStratifiedShuffleSplit
from pathlib import Path
from dotenv import load_dotenv

sys.path.insert(0, os.path.expanduser('~/thesis-xai'))
from training_logger import save_models, log_run

load_dotenv()
SEED       = 20260605
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

CACHE_DIR  = Path(os.getenv('KAGGLE_CACHE')) / 'preprocessed'
IMG_SIZE   = 256
label_cols = ['epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural']

# locate most recent preprocessed cache via its sidecar
sidecars = sorted(CACHE_DIR.glob(f'*_{IMG_SIZE}_meta.json'))
if not sidecars:
    raise FileNotFoundError(f"No preprocessed cache found in {CACHE_DIR}. Run 0_preprocess_rsna.py first.")
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

# enable GPU memory growth and log device details
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


class DataStreamKerasClassifier(KerasClassifier):
    # generator-based streaming so TF never allocates the full array as a constant; shuffling operates on individual samples

    @staticmethod
    def _make_dataset(X, y, sample_weight, batch_size, shuffle, augment=False):
        n = len(X)
        y_shape = (y.shape[1],) if y.ndim > 1 else ()
        sig = (
            tf.TensorSpec(shape=X.shape[1:], dtype=tf.float32),
            tf.TensorSpec(shape=y_shape,     dtype=tf.float32),
        )
        if sample_weight is not None:
            sig += (tf.TensorSpec(shape=(), dtype=tf.float32),)

        def gen():
            # sequential NFS reads avoid random mmap seeks; shuffle buffer randomises at sample level
            while True:
                for i in range(n):
                    xi = X[i].astype('float32')
                    yi = y[i].astype('float32')
                    if sample_weight is not None:
                        yield xi, yi, sample_weight[i].astype('float32')
                    else:
                        yield xi, yi

        ds = tf.data.Dataset.from_generator(gen, output_signature=sig)
        if shuffle:
            ds = ds.shuffle(buffer_size=10000)
        ds = ds.batch(batch_size)

        if augment:
            # matches CNN augmentation: ±5 deg rotation and ±8% zoom/translation for scanner variance
            augmenter = tf.keras.Sequential([
                layers.RandomFlip("horizontal"),
                layers.RandomRotation(factor=5/360, fill_mode='constant'),
                layers.RandomZoom(height_factor=0.08, width_factor=0.08, fill_mode='constant'),
                layers.RandomTranslation(height_factor=0.08, width_factor=0.08, fill_mode='constant'),
            ])
            if sample_weight is not None:
                ds = ds.map(lambda x, y, w: (augmenter(x, training=True), y, w),
                            num_parallel_calls=tf.data.AUTOTUNE)
            else:
                ds = ds.map(lambda x, y: (augmenter(x, training=True), y),
                            num_parallel_calls=tf.data.AUTOTUNE)

        return ds.prefetch(tf.data.AUTOTUNE)

    def _fit_keras_model(self, X, y, sample_weight, warm_start, epochs, initial_epoch, **kwargs):
        batch_size      = self.get_params().get('batch_size', 32)
        steps_per_epoch = int(np.ceil(len(X) / batch_size))

        train_ds = self._make_dataset(X, y, sample_weight, batch_size, shuffle=True, augment=True)

        val_data  = kwargs.pop('validation_data', None)
        val_steps = None
        if val_data is not None:
            vx, vy    = val_data[0], val_data[1]
            val_steps = int(np.ceil(len(vx) / batch_size))
            kwargs['validation_data'] = self._make_dataset(vx, vy, None, batch_size, shuffle=False)

        kwargs.pop('batch_size', None)

        hist = self.model_.fit(
            train_ds,
            steps_per_epoch=steps_per_epoch,
            validation_steps=val_steps,
            epochs=initial_epoch + epochs,
            initial_epoch=initial_epoch,
            callbacks=self._fit_callbacks,
            **kwargs,
        )

        if not warm_start or not hasattr(self, 'history_') or initial_epoch == 0:
            self.history_ = defaultdict(list)
        for key, val in hist.history.items():
            self.history_[key] += val


class PositionEmbedding(layers.Layer):
    def __init__(self, num_patches, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.num_patches = num_patches
        self.embed       = layers.Embedding(input_dim=num_patches, output_dim=embed_dim)
        # docs Embedding: https://www.tensorflow.org/api_docs/python/tf/keras/layers/Embedding

    def call(self, x):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        return x + self.embed(positions)

    def get_config(self):
        config = super().get_config()
        config.update({'num_patches': self.num_patches,
                       'embed_dim':   self.embed.output_dim})
        return config


def getModel(patch_size=16, embed_dim=64, num_heads=4,
             num_blocks=2, mlp_dim=128, lr=0.0001, dropoutrate=0.1):
    """mini ViT: patch embedding -> learnable pos embed -> N transformer blocks -> GAP -> Dense(5, sigmoid)."""
    # release the previous fold's model and tensors before building a new one
    tf.keras.backend.clear_session()

    num_patches = (IMG_SIZE // patch_size) ** 2

    inputs = layers.Input((IMG_SIZE, IMG_SIZE, 3))

    # Conv2D doubles as patch extractor: stride=patch_size gives non-overlapping patches projected to embed_dim
    x = layers.Conv2D(embed_dim, kernel_size=patch_size, strides=patch_size, padding='valid')(inputs)
    x = layers.Reshape((num_patches, embed_dim))(x)

    x = PositionEmbedding(num_patches, embed_dim)(x)

    # docs MultiHeadAttention: https://www.tensorflow.org/api_docs/python/tf/keras/layers/MultiHeadAttention
    # docs LayerNormalization: https://www.tensorflow.org/api_docs/python/tf/keras/layers/LayerNormalization
    for _ in range(num_blocks):
        attn = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim // num_heads)(x, x)
        attn = layers.Dropout(dropoutrate)(attn)
        x    = layers.LayerNormalization(epsilon=1e-6)(x + attn)

        ffn = layers.Dense(mlp_dim, activation='gelu')(x)
        ffn = layers.Dropout(dropoutrate)(ffn)
        ffn = layers.Dense(embed_dim)(ffn)
        ffn = layers.Dropout(dropoutrate)(ffn)
        x   = layers.LayerNormalization(epsilon=1e-6)(x + ffn)

    # docs GlobalAveragePooling1D: https://www.tensorflow.org/api_docs/python/tf/keras/layers/GlobalAveragePooling1D
    x       = layers.GlobalAveragePooling1D()(x)
    x       = layers.Dense(128, activation='gelu')(x)
    x       = layers.Dropout(dropoutrate)(x)
    outputs = layers.Dense(5, activation='sigmoid')(x)

    vit = models.Model(inputs, outputs)
    vit.compile(
        optimizer=optimizers.Adam(learning_rate=lr),
        loss=tf.keras.losses.BinaryFocalCrossentropy(gamma=2.0),
        metrics=['recall', tf.keras.metrics.AUC(name='auc', multi_label=True)]
    )
    return vit


# set True to skip grid search and reuse hardcoded best params from a prior run
SKIP_GRID_SEARCH = False

if not SKIP_GRID_SEARCH:
    # 3% subsample keeps per-fold memory low so accumulated folds stay within node budget
    msss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.97, random_state=SEED)
    sub_idx, _ = next(msss.split(x_train, y_train))
    x_sub = x_train[sub_idx]
    y_sub = y_train[sub_idx]
    print(f"Subsample: {len(x_sub):,} slices ({len(x_sub)/len(x_train)*100:.1f}%)", flush=True)

    msss_val = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.95, random_state=SEED)
    val_cv_idx, _ = next(msss_val.split(x_val, y_val))
    x_val_cv = x_val[val_cv_idx]
    y_val_cv = y_val[val_cv_idx]
    print(f"Val CV subset: {len(x_val_cv):,} slices ({len(x_val_cv)/len(x_val)*100:.1f}%)", flush=True)

    clf = DataStreamKerasClassifier(
        model=getModel,
        callbacks=[
            callbacks.EarlyStopping(monitor='val_auc', patience=5, mode='max', restore_best_weights=True),
            callbacks.ReduceLROnPlateau(monitor='val_auc', factor=0.5, patience=3, mode='max', min_lr=1e-6),
        ]
    )
    pipe = Pipeline([('clf', clf)])

    def multi_label_auc_scorer(estimator, X, y):
        # callable scorer bypasses sklearn's is_classifier() check that fails on DataStreamKerasClassifier in sklearn 1.8
        y_pred = estimator.predict_proba(X)
        return roc_auc_score(y, y_pred, average='weighted')

    grid = GridSearchCV(
        estimator=pipe,
        param_grid=[{
            'clf__model__patch_size':  [16],
            'clf__model__embed_dim':   [64, 128],
            'clf__model__num_heads':   [4],
            'clf__model__num_blocks':  [4],
            'clf__model__mlp_dim':     [128, 256],
            'clf__model__lr':          [0.0001],
            'clf__model__dropoutrate': [0.1],
            'clf__epochs':             [30],
            'clf__batch_size':         [64],
        }],
        scoring=multi_label_auc_scorer,
        cv=MultilabelStratifiedKFold(n_splits=5, shuffle=True, random_state=SEED),
        n_jobs=1,
        error_score='raise',
        verbose=2,
    )

    print(f"\n{'='*60}")
    print(f"Grid Search started at {datetime.datetime.now()}")
    print(f"{'='*60}\n", flush=True)

    gs_start = datetime.datetime.now()
    grid.fit(x_sub, y_sub.astype(int),
             clf__validation_data=(x_val_cv, y_val_cv))
    gs_end  = datetime.datetime.now()
    gs_time = gs_end - gs_start
    h, rem  = divmod(gs_time.total_seconds(), 3600)
    m, s    = divmod(rem, 60)

    print(f"\n{'='*60}")
    print(f"Grid Search finished at {gs_end}")
    print(f"Total time: {int(h)}h {int(m)}m {int(s)}s")
    print(f"Best CV score: {grid.best_score_:.4f}")
    print(f"Best params:   {grid.best_params_}")
    print(f"{'='*60}\n", flush=True)

    del x_sub, y_sub, x_val_cv, y_val_cv
    gc.collect()

else:
    # update these after the first successful grid search run
    _best = {
        'clf__batch_size':           64,
        'clf__epochs':               30,
        'clf__model__patch_size':    16,
        'clf__model__embed_dim':     128,
        'clf__model__num_heads':     4,
        'clf__model__num_blocks':    4,
        'clf__model__mlp_dim':       256,
        'clf__model__lr':            0.0001,
        'clf__model__dropoutrate':   0.1,
    }
    grid = types.SimpleNamespace(best_params_=_best, best_score_=float('nan'))
    print(f"Skipping grid search, using hardcoded params: {_best}", flush=True)

# build best model directly, bypassing scikeras to avoid allocating train and val arrays as TF constants
_bp             = grid.best_params_
best_batch_size = _bp['clf__batch_size']
best_epochs     = _bp['clf__epochs']
model_kwargs    = {k.replace('clf__model__', ''): v
                   for k, v in _bp.items() if k.startswith('clf__model__')}

best_model = getModel(**model_kwargs)

n_train     = len(x_train)
n_val       = len(x_val)
train_steps = int(np.ceil(n_train / best_batch_size))
val_steps   = int(np.ceil(n_val   / best_batch_size))

train_ds = DataStreamKerasClassifier._make_dataset(
    x_train, y_train, None, best_batch_size, shuffle=True, augment=True)
val_ds   = DataStreamKerasClassifier._make_dataset(
    x_val, y_val, None, best_batch_size, shuffle=False)

# saves best weights each epoch so the checkpoint survives if SLURM walltime hits before fit() returns
MODEL_EXPORT_DIR = '/rds/projects/k/karwatha-karwath-hds-pg-research/axr1222/models'
os.makedirs(MODEL_EXPORT_DIR, exist_ok=True)
checkpoint_path = os.path.join(MODEL_EXPORT_DIR,
    f'vit_checkpoint_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.keras')

total_steps  = train_steps * best_epochs
warmup_steps = max(1, int(0.1 * total_steps))
lr           = _bp['clf__model__lr']
lr_schedule  = tf.keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=0.0,
    decay_steps=total_steps,
    warmup_steps=warmup_steps,
    warmup_target=lr,
    alpha=1e-6 / lr,
)
best_model.compile(
    optimizer=tf.keras.optimizers.AdamW(learning_rate=lr_schedule, weight_decay=1e-4),
    loss=tf.keras.losses.BinaryFocalCrossentropy(gamma=2.0),
    metrics=['recall', tf.keras.metrics.AUC(name='auc', multi_label=True)]
)
print(f"Retrain: AdamW + cosine warmup ({warmup_steps} warmup / {total_steps} total steps)", flush=True)

retrain_started = datetime.datetime.now()
print(f"Retraining best ViT on full x_train...", flush=True)
best_model.fit(
    train_ds,
    steps_per_epoch=train_steps,
    validation_data=val_ds,
    validation_steps=val_steps,
    epochs=best_epochs,
    callbacks=[
        callbacks.EarlyStopping(monitor='val_auc', patience=5, mode='max', restore_best_weights=True),
        callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor='val_auc',
            mode='max',
            save_best_only=True,
            verbose=1,
        ),
    ],
)
total = datetime.datetime.now() - retrain_started
print(f"Full retrain complete.", flush=True)

# predict in batches to avoid converting x_test to a TF constant
y_pred_proba = np.vstack([
    best_model.predict_on_batch(x_test[i : i + best_batch_size].copy())
    for i in range(0, len(x_test), best_batch_size)
])

print("\nTest Set AUC per class:")
for i, col in enumerate(label_cols):
    try:
        auc = roc_auc_score(y_test[:, i], y_pred_proba[:, i])
        print(f"  {col:25s} AUC: {auc:.4f}")
    except ValueError:
        print(f"  {col:25s} AUC: N/A (single class)")
print(f"  {'weighted average':25s} AUC: {roc_auc_score(y_test, y_pred_proba, average='weighted'):.4f}")

os.chdir(os.path.expanduser('~/thesis-xai'))

pkl_path, keras_path = save_models(grid, keras_model=best_model, method='vit')

log_run(
    grid=grid,
    y_test=y_test,
    y_pred_proba=y_pred_proba,
    x_train=x_train,
    x_test=x_test,
    label_cols=label_cols,
    N_TRAIN_PATIENTS=meta['N_TRAIN_PATIENTS'],
    N_TEST_PATIENTS=meta['N_TEST_PATIENTS'],
    total_timedelta=total,
    model_pkl=pkl_path,
    model_h5=keras_path,
    method='vit',
    loss='focal'
)

print("\nDone!")
