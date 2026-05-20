"""
RSNA Intracranial Hemorrhage — Grid Search Training Script
Run via sbatch: sbatch 1_train_mult_rsna.sh
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

# Add thesis-xai to path so we can import training_logger
sys.path.insert(0, os.path.expanduser('~/thesis-xai'))
from training_logger import save_models, log_run

# Config
IMG_SIZE = 256
N_TRAIN_PATIENTS = 800
N_TEST_PATIENTS = 200

from dotenv import load_dotenv

load_dotenv()
os.environ['KAGGLEHUB_CACHE'] = os.getenv('KAGGLE_CACHE')
CACHE_DIR = Path(os.environ['KAGGLEHUB_CACHE']) / 'preprocessed'
label_cols = ['epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural']

# Load cached data
cache_file = CACHE_DIR / f'rsna_train{N_TRAIN_PATIENTS}_test{N_TEST_PATIENTS}_{IMG_SIZE}.npz'
if not cache_file.exists():
    raise FileNotFoundError(f"Cache not found: {cache_file}. Run the notebook first to preprocess.")

print(f"Loading from cache: {cache_file}")
data = np.load(cache_file)
x_train, y_train = data['x_train'], data['y_train']
x_test, y_test = data['x_test'], data['y_test']

# drop 'any' column (index 5) model outputs 5 classes
y_train = y_train[:, :5]
y_test = y_test[:, :5]

print(f"x_train: {x_train.shape}, y_train: {y_train.shape}")
print(f"x_test: {x_test.shape}, y_test: {y_test.shape}")

# GPU memory growth (prevent OOM)
gpus = tf.config.list_physical_devices('GPU')
print(f"GPUs available: {gpus}")
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print("Memory growth enabled for all GPUs.")
else:
    print("WARNING: No GPU detected! Training will be slow.")

# Model
def getModel(filters1=16, filters2=32, filters3=64,
             lr=0.0001, layer_number=3, dropoutrate=0.3):

    img_dimension = (IMG_SIZE, IMG_SIZE, 1)

    cnn = models.Sequential()
    cnn.add(layers.Input(img_dimension))

    cnn.add(layers.Conv2D(filters=filters1, kernel_size=(3, 3), padding='same'))
    cnn.add(layers.BatchNormalization())
    cnn.add(layers.Activation('relu'))
    cnn.add(layers.MaxPooling2D(pool_size=(2, 2)))
    cnn.add(layers.Dropout(dropoutrate))

    if layer_number >= 2:
        cnn.add(layers.Conv2D(filters=filters2, kernel_size=(3, 3), padding='same'))
        cnn.add(layers.BatchNormalization())
        cnn.add(layers.Activation('relu'))
        cnn.add(layers.MaxPooling2D(pool_size=(2, 2)))
        cnn.add(layers.Dropout(dropoutrate))

    if layer_number >= 3:
        cnn.add(layers.Conv2D(filters=filters3, kernel_size=(3, 3), padding='same'))
        cnn.add(layers.BatchNormalization())
        cnn.add(layers.Activation('relu'))
        cnn.add(layers.MaxPooling2D(pool_size=(2, 2)))
        cnn.add(layers.Dropout(dropoutrate))

    cnn.add(layers.Flatten())
    cnn.add(layers.Dense(128, activation='relu'))
    cnn.add(layers.Dropout(dropoutrate))
    cnn.add(layers.Dense(5, activation='sigmoid'))

    cnn.compile(
        optimizer=optimizers.Adam(learning_rate=lr),
        loss='binary_crossentropy',
        metrics=['recall', tf.keras.metrics.AUC(name='auc', multi_label=True)]
    )
    return cnn

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

pipe = Pipeline([
    ('clf', clf),
])

# Sample weights
any_hemorrhage = (y_train.max(axis=1) > 0).astype(int)
sample_weights = compute_sample_weight('balanced', any_hemorrhage)

# Hyperparameter grid
param_grid = [
    {
        'clf__model__filters1': [16],
        'clf__model__filters2': [32, 64],
        'clf__model__filters3': [128],
        'clf__model__lr': [0.001, 0.0001],
        'clf__model__layer_number': [3],
        'clf__model__dropoutrate': [0.25],
        'clf__epochs': [30],
        'clf__batch_size': [64],
    },
]

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

started = datetime.datetime.now()
grid.fit(x_train, y_train.astype(int), clf__sample_weight=sample_weights)
finished = datetime.datetime.now()

total = finished - started
h, remainder = divmod(total.total_seconds(), 3600)
m, s = divmod(remainder, 60)

print(f"\n{'='*60}")
print(f"Grid Search finished at {finished}")
print(f"Total time: {int(h)}h {int(m)}m {int(s)}s")
print(f"Best CV score: {grid.best_score_:.4f}")
print(f"Best params: {grid.best_params_}")
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

# Save using training_logger
os.chdir(os.path.expanduser('~/thesis-xai'))  # so logger saves to thesis-xai/

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
