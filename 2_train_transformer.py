"""
RSNA Intracranial Hemorrhage — Vision Transformer Training Script
Run via sbatch: sbatch 2_train_transformer.sh
"""

import os
import sys
import datetime
import numpy as np
import tensorflow as tf
from tensorflow.keras import models, layers, optimizers, callbacks
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score, make_scorer
from sklearn.utils.class_weight import compute_sample_weight
from scikeras.wrappers import KerasClassifier
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from pathlib import Path

sys.path.insert(0, os.path.expanduser('~/thesis-xai'))
from training_logger import save_models, log_run

# Config
IMG_SIZE         = 256
N_TRAIN_PATIENTS = 3000
N_TEST_PATIENTS  = 750

from dotenv import load_dotenv
load_dotenv()
os.environ['KAGGLEHUB_CACHE'] = os.getenv('KAGGLE_CACHE')
CACHE_DIR  = Path(os.environ['KAGGLEHUB_CACHE']) / 'preprocessed'
label_cols = ['epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural']

# Load cached data
cache_file = CACHE_DIR / f'rsna_train{N_TRAIN_PATIENTS}_test{N_TEST_PATIENTS}_{IMG_SIZE}.npz'
if not cache_file.exists():
    raise FileNotFoundError(f"Cache not found: {cache_file}. Run 0_preprocess_rsna.py first.")

print(f"Loading from cache: {cache_file}")
data = np.load(cache_file)
x_train, y_train = data['x_train'], data['y_train']
x_test,  y_test  = data['x_test'],  data['y_test']

y_train = y_train[:, :5]
y_test  = y_test[:, :5]

print(f"x_train: {x_train.shape}, y_train: {y_train.shape}")
print(f"x_test:  {x_test.shape},  y_test:  {y_test.shape}")

# GPU memory growth
gpus = tf.config.list_physical_devices('GPU')
print(f"GPUs available: {gpus}")
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print("Memory growth enabled.")
else:
    print("WARNING: No GPU detected! Training will be slow.")


# Positional embedding layer
# Learns a vector for each patch position and adds it to the patch tokens.
# This tells the transformer WHERE each patch came from in the image.
class PositionEmbedding(layers.Layer):
    def __init__(self, num_patches, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.num_patches = num_patches
        self.embed       = layers.Embedding(input_dim=num_patches, output_dim=embed_dim)

    def call(self, x):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        return x + self.embed(positions)

    def get_config(self):
        config = super().get_config()
        config.update({'num_patches': self.num_patches,
                       'embed_dim':   self.embed.output_dim})
        return config


# Model
def getModel(patch_size=16, embed_dim=64, num_heads=4,
             num_blocks=2, mlp_dim=128, lr=0.0001, dropoutrate=0.1):
    """
    Mini Vision Transformer (ViT) for multi-label ICH classification.

    Steps:
      1. Split image into non-overlapping patches via Conv2D(stride=patch_size)
      2. Add learnable positional embeddings
      3. Run through N transformer blocks (attention + FFN)
      4. Global average pool → Dense(5, sigmoid)
    """
    num_patches = (IMG_SIZE // patch_size) ** 2   # e.g. patch_size=16 -> 256 patches

    inputs = layers.Input((IMG_SIZE, IMG_SIZE, 1))

    # Step 1 — Patch embedding
    # Conv2D with stride=patch_size extracts non-overlapping patches and projects
    # each one to embed_dim dimensions in a single step (standard ViT trick)
    x = layers.Conv2D(embed_dim, kernel_size=patch_size, strides=patch_size, padding='valid')(inputs)
    x = layers.Reshape((num_patches, embed_dim))(x)   # (batch, num_patches, embed_dim)

    # Step 2 — Positional embedding
    x = PositionEmbedding(num_patches, embed_dim)(x)

    # Step 3 — Transformer blocks
    for _ in range(num_blocks):

        # Multi-head self-attention (each patch attends to every other patch)
        attn = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embed_dim // num_heads
        )(x, x)
        attn = layers.Dropout(dropoutrate)(attn)
        x    = layers.LayerNormalization(epsilon=1e-6)(x + attn)   # residual + norm

        # Feed-forward network (two Dense layers, GELU activation)
        ffn  = layers.Dense(mlp_dim, activation='gelu')(x)
        ffn  = layers.Dropout(dropoutrate)(ffn)
        ffn  = layers.Dense(embed_dim)(ffn)
        ffn  = layers.Dropout(dropoutrate)(ffn)
        x    = layers.LayerNormalization(epsilon=1e-6)(x + ffn)    # residual + norm

    # Step 4 — Classification head
    x       = layers.GlobalAveragePooling1D()(x)       # average across all patch tokens
    x       = layers.Dense(128, activation='gelu')(x)
    x       = layers.Dropout(dropoutrate)(x)
    outputs = layers.Dense(5, activation='sigmoid')(x) # one output per ICH subtype

    vit = models.Model(inputs, outputs)
    vit.compile(
        optimizer=optimizers.Adam(learning_rate=lr),
        loss='binary_crossentropy',
        metrics=['recall', tf.keras.metrics.AUC(name='auc', multi_label=True)]
    )
    return vit


# Pipeline
clf = KerasClassifier(
    model=getModel,
    validation_split=0.15,
    callbacks=[callbacks.EarlyStopping(
        monitor='val_auc',
        patience=5,
        mode='max',
        restore_best_weights=True
    )]
)

pipe = Pipeline([('clf', clf)])

# Sample weights
any_hemorrhage = (y_train.max(axis=1) > 0).astype(int)
sample_weights = compute_sample_weight('balanced', any_hemorrhage)

# Hyperparameter grid
param_grid = [{
    'clf__model__patch_size':  [16, 32],   # 256 tokens vs 64 tokens
    'clf__model__embed_dim':   [64],
    'clf__model__num_heads':   [4],
    'clf__model__num_blocks':  [2, 4],     # shallow vs deeper transformer
    'clf__model__mlp_dim':     [128],
    'clf__model__lr':          [0.0001],
    'clf__model__dropoutrate': [0.1],
    'clf__epochs':             [30],
    'clf__batch_size':         [64],
}]

# Scorer
def multi_label_auc(y_true, y_pred, **kwargs):
    return roc_auc_score(y_true, y_pred, average='weighted', multi_class='ovr')

roc_auc = make_scorer(multi_label_auc)

# Grid Search
grid = GridSearchCV(
    estimator=pipe,
    param_grid=param_grid,
    scoring=roc_auc,
    cv=MultilabelStratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    n_jobs=1,
    error_score='raise',
    verbose=2
)

print(f"\n{'='*60}")
print(f"Grid Search started at {datetime.datetime.now()}")
print(f"{'='*60}\n")

started  = datetime.datetime.now()
grid.fit(x_train, y_train.astype(int), clf__sample_weight=sample_weights)
finished = datetime.datetime.now()

total = finished - started
h, remainder = divmod(total.total_seconds(), 3600)
m, s         = divmod(remainder, 60)

print(f"\n{'='*60}")
print(f"Grid Search finished at {finished}")
print(f"Total time:    {int(h)}h {int(m)}m {int(s)}s")
print(f"Best CV score: {grid.best_score_:.4f}")
print(f"Best params:   {grid.best_params_}")
print(f"{'='*60}\n")

# Evaluate on test set
y_pred_proba = grid.best_estimator_.predict_proba(x_test)

print("\nTest Set AUC per class:")
for i, col in enumerate(label_cols):
    try:
        auc = roc_auc_score(y_test[:, i], y_pred_proba[:, i])
        print(f"  {col:25s} AUC: {auc:.4f}")
    except ValueError:
        print(f"  {col:25s} AUC: N/A (single class)")

weighted_auc = roc_auc_score(y_test, y_pred_proba, average='weighted')
print(f"\n  {'weighted avg':25s} AUC: {weighted_auc:.4f}")

# Save
os.chdir(os.path.expanduser('~/thesis-xai'))
pkl_path, keras_path = save_models(grid)

log_run(
    grid=grid,
    y_test=y_test,
    y_pred_proba=y_pred_proba,
    x_train=x_train,
    x_test=x_test,
    label_cols=label_cols,
    N_TRAIN_PATIENTS=N_TRAIN_PATIENTS,
    N_TEST_PATIENTS=N_TEST_PATIENTS,
    total_timedelta=total,
    model_pkl=pkl_path,
    model_h5=keras_path,
)

print("\nDone!")