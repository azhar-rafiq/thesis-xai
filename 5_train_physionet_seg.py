"""
PhysioNet CT-ICH -- supervised U-Net segmentation
encoder initialised from RSNA CNN checkpoint (step 1 pretrained weights).
trains directly on PhysioNet pixel masks; no slice-level labels needed.
Run via sbatch: sbatch 5_train_physionet_seg.sh
"""

import gc
import os
import sys
import datetime
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, optimizers, callbacks
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit
from skimage.transform import resize
from dotenv import load_dotenv

try:
    import nibabel as nib
except ImportError:
    raise ImportError("nibabel not installed -- run: pip install nibabel")

sys.path.insert(0, os.path.expanduser('~/thesis-xai'))
load_dotenv()

SEED = 20260605
tf.random.set_seed(SEED)
np.random.seed(SEED)

PHYSIONET_DIR    = Path('/rds/projects/k/karwatha-karwath-hds-pg-research/axr1222/data/physionet.org/files/ct-ich/1.3.1')
CT_DIR           = PHYSIONET_DIR / 'ct_scans'
MASK_DIR         = PHYSIONET_DIR / 'masks'
LABELS_CSV       = PHYSIONET_DIR / 'hemorrhage_diagnosis_raw_ct.csv'
MODEL_EXPORT_DIR = Path('/rds/projects/k/karwatha-karwath-hds-pg-research/axr1222/models')

IMG_SIZE   = 256
BATCH_SIZE = 16
EPOCHS     = 50
LR         = 1e-4


# ---- same windowing as 0_preprocess_rsna.py ----
WINDOWS = {
    'brain':    {'wc': 40,  'ww': 80},
    'subdural': {'wc': 80,  'ww': 200},
    'soft':     {'wc': 40,  'ww': 380},
}

def apply_window(hu, wc, ww):
    return np.clip((hu - (wc - ww / 2.0)) / ww, 0.0, 1.0)

def hu_to_3channel(hu_slice):
    img = np.stack([
        apply_window(hu_slice, **WINDOWS['brain']),
        apply_window(hu_slice, **WINDOWS['subdural']),
        apply_window(hu_slice, **WINDOWS['soft']),
    ], axis=-1)
    return resize(img, (IMG_SIZE, IMG_SIZE), anti_aliasing=True).astype(np.float32)


# ---- load all slices for one patient ----
def load_patient(patient_num):
    ct_vol   = nib.load(str(CT_DIR   / f'{patient_num:03d}.nii')).get_fdata().astype(np.float32)
    mask_vol = nib.load(str(MASK_DIR / f'{patient_num:03d}.nii')).get_fdata().astype(np.float32)
    xs, ys = [], []
    for s in range(ct_vol.shape[2]):
        xs.append(hu_to_3channel(ct_vol[:, :, s]))
        mask = resize(mask_vol[:, :, s], (IMG_SIZE, IMG_SIZE), anti_aliasing=False)
        ys.append((mask > 0.5).astype(np.float32)[..., np.newaxis])  # (256, 256, 1)
    return xs, ys


# ---- load labels and list available patients ----
labels_df   = pd.read_csv(LABELS_CSV)
available   = sorted([int(p.stem) for p in CT_DIR.glob('*.nii')])
labels_df   = labels_df[labels_df['PatientNumber'].isin(available)]
patients    = labels_df['PatientNumber'].unique()

ich_count = (labels_df.groupby('PatientNumber')
             [['Intraventricular', 'Intraparenchymal', 'Subarachnoid', 'Epidural', 'Subdural']]
             .max().max(axis=1) > 0).sum()
print(f"PhysioNet: {len(patients)} patients ({ich_count} with ICH), {len(labels_df)} slices", flush=True)


# ---- patient-level 80/20 split ----
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
train_idx, test_idx = next(gss.split(labels_df, groups=labels_df['PatientNumber']))
train_patients = labels_df.iloc[train_idx]['PatientNumber'].unique()
test_patients  = labels_df.iloc[test_idx]['PatientNumber'].unique()
print(f"Split -- train: {len(train_patients)} patients, test: {len(test_patients)} patients", flush=True)


# ---- load splits into RAM (small dataset, ~2 GB total) ----
def load_split(patient_list, split_name):
    xs, ys = [], []
    for p in patient_list:
        try:
            x_slices, y_slices = load_patient(p)
            xs.extend(x_slices)
            ys.extend(y_slices)
        except Exception as e:
            print(f"  [WARN] skipping patient {p}: {e}", flush=True)
    X = np.array(xs, dtype=np.float32)
    Y = np.array(ys, dtype=np.float32)
    pos = (Y.max(axis=(1, 2, 3)) > 0).sum()
    print(f"  {split_name}: {X.shape[0]} slices, {pos} positive", flush=True)
    return X, Y

print("Loading data...", flush=True)
x_train, y_train = load_split(train_patients, 'train')
x_test,  y_test  = load_split(test_patients,  'test')


# ---- GPU setup ----
gpus = tf.config.list_physical_devices('GPU')
print(f"\nGPUs available: {len(gpus)}", flush=True)
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
    details = tf.config.experimental.get_device_details(gpu)
    print(f"  {gpu.name} - {details.get('device_name', 'unknown')}", flush=True)
if not gpus:
    print("WARNING: No GPU detected!", flush=True)


# ---- load CNN encoder from step 1 ----
CNN_CKPT = MODEL_EXPORT_DIR / 'rsna_cnn_20260605_05000_auc.keras'
if not CNN_CKPT.exists():
    raise FileNotFoundError(f"CNN checkpoint not found: {CNN_CKPT}")
print(f"\nLoading CNN encoder: {CNN_CKPT.name}", flush=True)
cnn_base = tf.keras.models.load_model(str(CNN_CKPT))
print(f"  CNN layers: {len(cnn_base.layers)}", flush=True)


# ---- build U-Net segmenter on top of CNN encoder ----
def build_unet_segmenter(cnn_base, freeze_encoder=True):
    """
    same skip-connection U-Net as 4_segmentation_xai.py but outputs
    a single binary mask (256, 256, 1) trained with pixel-level supervision.
    """
    cnn_base.trainable = not freeze_encoder

    # CNN is a Sequential model; trace through layers functionally to get skip tensors
    inp = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name='input')
    x   = inp
    act_outputs  = []
    drop_outputs = []
    for layer in cnn_base.layers:
        x = layer(x)
        if isinstance(layer, tf.keras.layers.Activation):
            act_outputs.append(x)
        elif isinstance(layer, tf.keras.layers.Dropout):
            drop_outputs.append(x)

    if len(act_outputs) < 3 or len(drop_outputs) < 3:
        raise ValueError(f"expected >=3 Activation and Dropout layers in CNN")

    skip1      = act_outputs[0]   # (256, 256, filters1)
    skip2      = act_outputs[1]   # (128, 128, filters2)
    skip3      = act_outputs[2]   # (64,  64,  filters3)
    bottleneck = drop_outputs[2]  # (32,  32,  filters3)

    # decoder: 32x32 -> 64x64 -> 128x128 -> 256x256
    xd = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same', name='dec_up1')(bottleneck)
    xd = layers.Concatenate(name='dec_cat1')([xd, skip3])
    xd = layers.Conv2D(128, (3, 3), padding='same', name='dec_conv1')(xd)
    xd = layers.BatchNormalization(name='dec_bn1')(xd)
    xd = layers.Activation('relu', name='dec_act1')(xd)

    xd = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same', name='dec_up2')(xd)
    xd = layers.Concatenate(name='dec_cat2')([xd, skip2])
    xd = layers.Conv2D(64, (3, 3), padding='same', name='dec_conv2')(xd)
    xd = layers.BatchNormalization(name='dec_bn2')(xd)
    xd = layers.Activation('relu', name='dec_act2')(xd)

    xd = layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same', name='dec_up3')(xd)
    xd = layers.Concatenate(name='dec_cat3')([xd, skip1])
    xd = layers.Conv2D(32, (3, 3), padding='same', name='dec_conv3')(xd)
    xd = layers.BatchNormalization(name='dec_bn3')(xd)
    xd = layers.Activation('relu', name='dec_act3')(xd)

    output = layers.Conv2D(1, (1, 1), activation='sigmoid', name='seg_output')(xd)
    return tf.keras.Model(inputs=inp, outputs=output, name='unet_physionet')


tf.keras.backend.clear_session()
unet = build_unet_segmenter(cnn_base, freeze_encoder=True)
trainable = sum(int(np.prod(v.shape)) for v in unet.trainable_variables)
print(f"Trainable params: {trainable:,}  (encoder frozen)", flush=True)


# ---- loss: BCE + Dice handles sparse positive pixels ----
def dice_coef(y_true, y_pred, smooth=1.0):
    y_true_f = tf.reshape(tf.cast(y_true, tf.float32), [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    inter    = tf.reduce_sum(y_true_f * y_pred_f)
    return (2.0 * inter + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

def bce_dice_loss(y_true, y_pred):
    return tf.keras.losses.binary_crossentropy(y_true, y_pred) + (1.0 - dice_coef(y_true, y_pred))

unet.compile(
    optimizer=optimizers.Adam(learning_rate=LR),
    loss=bce_dice_loss,
    metrics=[dice_coef],
)


# ---- TF dataset with augmentation (train only) ----
@tf.function
def augment(img, mask):
    # concat so the same spatial transform hits both image and mask
    combined = tf.concat([img, mask], axis=-1)                      # (256, 256, 4)
    combined = tf.image.random_flip_left_right(combined, seed=SEED)
    combined = tf.image.random_flip_up_down(combined, seed=SEED)
    return combined[..., :3], combined[..., 3:]

def make_dataset(X, Y, shuffle):
    ds = tf.data.Dataset.from_tensor_slices((X, Y))
    if shuffle:
        ds = ds.shuffle(len(X), seed=SEED).map(augment, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

train_ds = make_dataset(x_train, y_train, shuffle=True)
test_ds  = make_dataset(x_test,  y_test,  shuffle=False)


# ---- train ----
ts        = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
ckpt_path = str(MODEL_EXPORT_DIR / f'physionet_unet_{ts}.keras')

print(f"\nTraining PhysioNet U-Net (encoder frozen)...", flush=True)
train_started = datetime.datetime.now()
unet.fit(
    train_ds,
    validation_data=test_ds,
    epochs=EPOCHS,
    callbacks=[
        callbacks.EarlyStopping(monitor='val_dice_coef', patience=10, mode='max',
                                restore_best_weights=True),
        callbacks.ReduceLROnPlateau(monitor='val_dice_coef', factor=0.5, patience=5,
                                    mode='max', min_lr=1e-6),
        callbacks.ModelCheckpoint(filepath=ckpt_path, monitor='val_dice_coef',
                                   mode='max', save_best_only=True, verbose=1),
    ],
)
total = datetime.datetime.now() - train_started
print(f"Training complete. ({int(total.total_seconds() // 60)} min)", flush=True)


# ---- evaluate ----
print("\nTest set evaluation...", flush=True)
y_pred      = unet.predict(test_ds)
y_pred_bin  = (y_pred > 0.5).astype(np.float32)

def seg_metrics(y_true, y_pred_bin):
    yt = y_true.ravel().astype(np.float32)
    yp = y_pred_bin.ravel()
    inter     = (yt * yp).sum()
    dice      = (2 * inter + 1) / (yt.sum() + yp.sum() + 1)
    iou       = (inter + 1) / (yt.sum() + yp.sum() - inter + 1)
    recall    = inter / (yt.sum() + 1e-8)
    precision = inter / (yp.sum() + 1e-8)
    return dice, iou, recall, precision

dice, iou, rec, prec = seg_metrics(y_test, y_pred_bin)
print(f"  all test slices:       Dice {dice:.4f}  IoU {iou:.4f}  Recall {rec:.4f}  Precision {prec:.4f}")

# positive slices only (more meaningful -- negative slices trivially score high)
pos_mask = y_test.max(axis=(1, 2, 3)) > 0
if pos_mask.sum() > 0:
    d, i, r, p = seg_metrics(y_test[pos_mask], y_pred_bin[pos_mask])
    print(f"  positive slices ({pos_mask.sum():>3}): Dice {d:.4f}  IoU {i:.4f}  Recall {r:.4f}  Precision {p:.4f}")


# ---- save outputs for comparison in step 6 ----
maps_path = str(MODEL_EXPORT_DIR / f'physionet_seg_maps_{ts}.npy')
idx_path  = str(MODEL_EXPORT_DIR / f'physionet_test_patients_{ts}.npy')

np.save(maps_path, y_pred.astype('float16'))
np.save(idx_path,  test_patients)
print(f"\nSaved seg maps:    physionet_seg_maps_{ts}.npy    shape: {y_pred.shape}", flush=True)
print(f"Saved test patients: physionet_test_patients_{ts}.npy", flush=True)

print("\nDone!")
