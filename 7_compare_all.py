"""
Cross-dataset XAI comparison: RSNA + PhysioNet CT-ICH + Seg-CQ500

PhysioNet and CQ500 (pixel masks available):
  compare CNN / ViT / Hybrid GradCAM and supervised U-Net vs ground-truth pixel masks.
  Metrics: Dice, IoU, recall, precision, pointing game.

RSNA (slice-level labels only, no pixel masks):
  qualitative GradCAM panel -- CT + CNN / ViT / Hybrid activation maps
  with true and predicted class labels in panel titles.

Classification arm (all three datasets):
  ROC / AUC for CNN / ViT / Hybrid / U-Net. RSNA is the internal test set,
  PhysioNet and CQ500 are zero-shot external validation for the RSNA-trained
  classifiers scored as slice-level "any ICH".

Inferential statistics (stats_utils.py):
  DeLong for paired AUCs, Friedman + Wilcoxon + McNemar for localisation
  metrics, Mann-Whitney across datasets, all Holm-corrected, plus a patient
  clustered bootstrap for every Dice/IoU confidence interval.

Run via sbatch: sbatch 7_compare_all.sh
"""

import csv
import gc
import json
import os
import sys
import datetime
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
from pathlib import Path
from skimage.transform import resize
from sklearn.metrics import average_precision_score
from dotenv import load_dotenv

import stats_utils as su

try:
    import nibabel as nib
except ImportError:
    raise ImportError("nibabel not installed -- run: pip install nibabel")

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    MATPLOTLIB = True
except ImportError:
    MATPLOTLIB = False

load_dotenv()

SEED = 20260605
tf.random.set_seed(SEED)
np.random.seed(SEED)

PHYSIONET_DIR    = Path('/rds/projects/k/karwatha-karwath-hds-pg-research/axr1222/data/physionet.org/files/ct-ich/1.3.1')
PHYSIONET_CT     = PHYSIONET_DIR / 'ct_scans'
PHYSIONET_MASK   = PHYSIONET_DIR / 'masks'

CQ500_VOLUMES    = Path('/rds/projects/k/karwatha-karwath-hds-pg-research/axr1222/data/seg-cq500/Seg-CQ500/data/volumes')

MODEL_EXPORT_DIR = Path('/rds/projects/k/karwatha-karwath-hds-pg-research/axr1222/models')
OUTPUT_DIR       = Path(os.path.expanduser('~/thesis-xai/result'))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IMG_SIZE    = 256
BATCH_SIZE  = 16
THRESHOLD   = 0.5
N_CLASSES   = 5
N_QUAL_IMGS = 2
N_QUAL_RSNA = 3

# figure width
FIG_W       = 6.7
FIG_W_WIDE  = 9.6
FIG_DPI     = 300
TITLE_FS    = 7
SUPTITLE_FS = 9


def grid_figsize(n_cols, n_rows, width_in=FIG_W_WIDE,
                 title_in=0.42, sup_in=0.35):
    """square panels that fill width_in exactly, plus room for titles."""
    panel = width_in / n_cols
    return (width_in, n_rows * (panel + title_in) + sup_in)

# pure forward passes can use a larger batch than the gradient-tape GradCAM loop
INFER_BATCH = 64

# cap the RSNA classification pass for a quick smoke test; None uses every test slice
N_RSNA_AUC  = None

# bootstrap resamples for the patient-clustered Dice/IoU confidence intervals
N_BOOTSTRAP = 2000

# chance baseline: a uniform random saliency map, seeded off SEED so it is
# reproducible and so it cannot collide with any other rng stream in this script
CHANCE_SEED  = SEED + 1
CHANCE_DRAWS = 1000

RSNA_LABEL_COLS = ['epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural']
# the first three characters collide (intra..., sub...), so use the clinical codes
RSNA_LABEL_SHORT = {'epidural':         'EDH',
                    'intraparenchymal': 'IPH',
                    'intraventricular': 'IVH',
                    'subarachnoid':     'SAH',
                    'subdural':         'SDH'}

# ---- load RSNA test sample (qualitative only -- no pixel masks) ----
CACHE_DIR = Path(os.getenv('KAGGLE_CACHE')) / 'preprocessed'
_sidecars = sorted(CACHE_DIR.glob(f'*_{IMG_SIZE}_meta.json'))
if not _sidecars:
    raise FileNotFoundError(f"No RSNA preprocessed cache found in {CACHE_DIR}. Run 0_preprocess_rsna.py first.")
_meta_path = _sidecars[-1]
_base      = str(_meta_path).replace('_meta.json', '')
print(f"RSNA cache: {_meta_path.name}", flush=True)

# the memmaps stay open: the classification arm streams the whole test set through
# each model in batches, so it is never materialised in RAM (101k slices is ~80 GB)
rsna_x_mm = np.load(f'{_base}_x_test.npy', mmap_mode='r')
rsna_y_mm = np.load(f'{_base}_y_test.npy', mmap_mode='r')

# pick N_QUAL_RSNA ICH-positive slices spread across the test set
_pos_mask   = rsna_y_mm.sum(axis=1) > 0
_pos_indices = np.where(_pos_mask)[0]
np.random.seed(SEED)
_chosen = np.random.choice(_pos_indices,
                            size=min(N_QUAL_RSNA, len(_pos_indices)),
                            replace=False)
_chosen.sort()
rsna_sample_x = rsna_x_mm[_chosen].copy().astype(np.float32)  # (N_QUAL_RSNA, H, W, 3)
rsna_sample_y = rsna_y_mm[_chosen].copy().astype(np.float32)  # (N_QUAL_RSNA, 5)
print(f"  RSNA sample: {len(_chosen)} ICH-positive slices (indices {_chosen.tolist()})", flush=True)

# slice range used for the classification AUC arm
N_RSNA_EVAL = len(rsna_x_mm) if N_RSNA_AUC is None else min(int(N_RSNA_AUC), len(rsna_x_mm))
rsna_y_eval = np.asarray(rsna_y_mm[:N_RSNA_EVAL], dtype=np.float32)
print(f"  RSNA classification arm: {N_RSNA_EVAL:,} slices "
      f"({'full test set' if N_RSNA_AUC is None else 'capped by N_RSNA_AUC'})", flush=True)

rsna_cam_maps   = {}   # method -> (N_QUAL_RSNA, H, W, 1) float32
rsna_preds      = {}   # method -> (N_QUAL_RSNA, 5) float32

# classification probabilities collected for the AUC / ROC / statistics arm
rsna_probs  = {}   # model -> (N_RSNA_EVAL, 5) float32
phys_probs  = {}   # model -> (n_phys_slices, 5) float32
cq500_probs = {}   # model -> (n_cq500_slices, 5) float32

CNN_CKPT    = str(MODEL_EXPORT_DIR / 'rsna_cnn_20260722-BD_nan_auc.keras')
VIT_CKPT    = str(MODEL_EXPORT_DIR / 'rsna_vit_20260713-ET_08916_auc.keras')
# no HYBRID_CKPT: the hybrid is the mean of the CNN and ViT probabilities and CAMs
# (see 3_train_hybrid.py), reconstructed below rather than reloaded
UNET_CLS_CKPT = str(MODEL_EXPORT_DIR / 'rsna_unet_20260802-NB_nan_auc.keras')


# ---- CT windowing ----
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


# ---- data loaders ----
def load_physionet_patient(patient_num):
    ct_vol   = nib.load(str(PHYSIONET_CT   / f'{patient_num:03d}.nii')).get_fdata().astype(np.float32)
    mask_vol = nib.load(str(PHYSIONET_MASK / f'{patient_num:03d}.nii')).get_fdata().astype(np.float32)
    xs, ys = [], []
    for s in range(ct_vol.shape[2]):
        xs.append(hu_to_3channel(ct_vol[:, :, s]))
        mask = resize(mask_vol[:, :, s], (IMG_SIZE, IMG_SIZE), anti_aliasing=False)
        ys.append((mask > 0.5).astype(np.float32)[..., np.newaxis])
    return np.array(xs, dtype=np.float32), np.array(ys, dtype=np.float32)

def load_cq500_patient(patient_name):
    patient_dir = CQ500_VOLUMES / patient_name
    ct_vol   = nib.load(str(patient_dir / 'CT.nii')).get_fdata().astype(np.float32)
    mask_vol = nib.load(str(patient_dir / 'ICH_mask.nii.gz')).get_fdata().astype(np.float32)
    xs, ys = [], []
    for s in range(ct_vol.shape[2]):
        xs.append(hu_to_3channel(ct_vol[:, :, s]))
        mask = resize(mask_vol[:, :, s], (IMG_SIZE, IMG_SIZE), anti_aliasing=False)
        ys.append((mask > 0.5).astype(np.float32)[..., np.newaxis])
    return np.array(xs, dtype=np.float32), np.array(ys, dtype=np.float32)


# ---- metrics ----
def pointing_game(y_true_hw1, cam_hw1):
    peak = np.unravel_index(np.argmax(cam_hw1[..., 0]), cam_hw1[..., 0].shape)
    return int(y_true_hw1[peak[0], peak[1], 0] > 0.5)

def seg_metrics(y_true, y_pred_bin):
    yt = y_true.ravel().astype(np.float32)
    yp = y_pred_bin.ravel().astype(np.float32)
    inter     = (yt * yp).sum()
    dice      = (2 * inter + 1) / (yt.sum() + yp.sum() + 1)
    iou       = (inter + 1) / (yt.sum() + yp.sum() - inter + 1)
    recall    = inter / (yt.sum() + 1e-8)
    precision = inter / (yp.sum() + 1e-8)
    return float(dice), float(iou), float(recall), float(precision)

def per_slice_metrics(maps_nchw1, y_test, pos_idx):
    """per-slice metric arrays, kept so the paired significance tests can use them."""
    maps_bin = (maps_nchw1.astype(np.float32) > THRESHOLD).astype(np.float32)
    dices, ious, recs, precs, pg_hits, aps = [], [], [], [], [], []
    for idx in pos_idx:
        d, iou, rec, prec = seg_metrics(y_test[idx], maps_bin[idx])
        dices.append(d); ious.append(iou); recs.append(rec); precs.append(prec)
        pg_hits.append(pointing_game(y_test[idx], maps_nchw1[idx]))
        # threshold-free ranking of the soft map against the binary mask. every idx
        # here is an ICH-positive slice, so the mask always has a positive pixel and
        # average_precision_score is always defined.
        aps.append(float(average_precision_score(
            (y_test[idx, :, :, 0] > 0.5).ravel().astype(np.int8),
            maps_nchw1[idx, :, :, 0].astype(np.float32).ravel())))
    return {
        'dice':      np.asarray(dices,   dtype=np.float32),
        'iou':       np.asarray(ious,    dtype=np.float32),
        'recall':    np.asarray(recs,    dtype=np.float32),
        'precision': np.asarray(precs,   dtype=np.float32),
        'pg':        np.asarray(pg_hits, dtype=np.float32),
        'ap':        np.asarray(aps,     dtype=np.float32),
    }


def chance_pointing_game(y_test, pos_idx, n_draws=CHANCE_DRAWS, seed=CHANCE_SEED):
    """
    reference value the observed pointing game must be read against.

    a uniformly random peak lands inside the lesion with probability equal to that
    slice's lesion area fraction, so the analytic expectation is the mean area
    fraction. the monte carlo draws are there to confirm that value, not replace it.
    """
    rng   = np.random.default_rng(seed)
    fracs = np.asarray([float((y_test[i, :, :, 0] > 0.5).sum()) / (IMG_SIZE * IMG_SIZE)
                        for i in pos_idx], dtype=np.float64)
    hits  = 0
    for i in pos_idx:
        m  = y_test[i, :, :, 0] > 0.5
        rr = rng.integers(0, IMG_SIZE, size=n_draws)
        cc = rng.integers(0, IMG_SIZE, size=n_draws)
        hits += int(m[rr, cc].sum())
    return float(fracs.mean()), hits / float(len(pos_idx) * n_draws)


def eval_method(maps_nchw1, y_test, pos_idx, per_slice=None):
    ps = per_slice if per_slice is not None else per_slice_metrics(maps_nchw1, y_test, pos_idx)
    return {
        'dice_mean':      float(ps['dice'].mean()),
        'dice_std':       float(ps['dice'].std()),
        'iou_mean':       float(ps['iou'].mean()),
        'iou_std':        float(ps['iou'].std()),
        'recall_mean':    float(ps['recall'].mean()),
        'recall_std':     float(ps['recall'].std()),
        'precision_mean': float(ps['precision'].mean()),
        'precision_std':  float(ps['precision'].std()),
        'pointing_game':  float(ps['pg'].mean()),
        'ap_mean':        float(ps['ap'].mean()),
        'ap_std':         float(ps['ap'].std()),
        'n_slices':       len(pos_idx),
    }


# ---- GradCAM ----
def compute_gradcam(gradcam_model, x_batch, out_size=IMG_SIZE, n_classes=N_CLASSES):
    x_tensor = tf.constant(x_batch.astype('float32'))
    cls_maps = []
    for cls in range(n_classes):
        with tf.GradientTape() as tape:
            tape.watch(x_tensor)
            conv_out, preds = gradcam_model(x_tensor, training=False)
            score           = tf.reduce_sum(preds[:, cls])
        grads   = tape.gradient(score, conv_out)
        weights = tf.reduce_mean(grads, axis=[1, 2])
        cam     = tf.reduce_sum(conv_out * weights[:, None, None, :], axis=-1)
        cam     = tf.nn.relu(cam)
        cam     = cam / (tf.reduce_max(cam, axis=[1, 2], keepdims=True) + 1e-8)
        cam     = tf.image.resize(cam[..., tf.newaxis], [out_size, out_size])[..., 0]
        cls_maps.append(cam.numpy().astype('float16'))
    return np.stack(cls_maps, axis=-1)   # (B, H, W, N_CLASSES)

def run_gradcam(gradcam_model, X):
    results = []
    for i in range(0, len(X), BATCH_SIZE):
        results.append(compute_gradcam(gradcam_model, X[i:i + BATCH_SIZE].copy()))
    return np.vstack(results)

def aggregate_gradcam(cam_maps):
    return cam_maps.max(axis=-1, keepdims=True)

def get_preds(gradcam_model, X):
    """get class probabilities from a GradCAM model (2nd output) for a small batch."""
    _, preds = gradcam_model(X.astype('float32'), training=False)
    return preds.numpy()


def predict_batched(model, X, batch_size=INFER_BATCH, desc='', log_every=200):
    """
    stream X through model in batches. X may be a memmap, so slices are copied
    one batch at a time and the full array is never resident.
    """
    started  = datetime.datetime.now()
    n_batch  = int(np.ceil(len(X) / batch_size))
    out      = []
    for bi, i in enumerate(range(0, len(X), batch_size)):
        xb = np.asarray(X[i:i + batch_size], dtype=np.float32)
        out.append(np.asarray(model.predict_on_batch(xb), dtype=np.float32))
        if desc and (bi % log_every == 0 or bi == n_batch - 1):
            done    = min(i + batch_size, len(X))
            elapsed = (datetime.datetime.now() - started).total_seconds()
            rate    = done / elapsed if elapsed > 0 else 0.0
            eta     = (len(X) - done) / rate if rate > 0 else 0.0
            print(f"    {desc}: {done:,}/{len(X):,} slices  "
                  f"{rate:.0f} slice/s  ETA {eta / 60:.1f} min", flush=True)
    return np.vstack(out).astype(np.float32)


def any_ich_score(prob_n5):
    """
    collapse the five RSNA subtype probabilities into one slice-level ICH score.
    max matches how the GradCAM maps are aggregated across classes; noisy-OR is
    returned alongside as a sensitivity check.
    """
    p = np.asarray(prob_n5, dtype=np.float64)
    return p.max(axis=1).astype(np.float32), (1.0 - np.prod(1.0 - p, axis=1)).astype(np.float32)


def unet_slice_score(seg_maps_nchw1):
    """slice-level ICH score for a segmentation U-Net: its most confident pixel."""
    m = np.asarray(seg_maps_nchw1, dtype=np.float32)
    return m.reshape(len(m), -1).max(axis=1)


# PositionEmbedding needed to deserialise ViT and Hybrid checkpoints
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


def error_rgb(y_true_hw1, y_pred_bin_hw1):
    yt  = y_true_hw1[..., 0]     > 0.5
    yp  = y_pred_bin_hw1[..., 0] > 0.5
    rgb = np.zeros((*yt.shape, 3), dtype=np.uint8)
    rgb[~yt & ~yp] = [30,  30,  30]
    rgb[ yt &  yp] = [0,  180,   0]
    rgb[~yt &  yp] = [200,  0,   0]
    rgb[ yt & ~yp] = [0,   80, 200]
    return rgb


# ===========================================================================
# 1. discover step 5 (physionet) and step 6 (cq500) outputs
# ===========================================================================
print("Discovering step outputs...", flush=True)

phys_seg_files     = sorted(MODEL_EXPORT_DIR.glob('physionet_seg_maps_*.npy'))
phys_patient_files = sorted(MODEL_EXPORT_DIR.glob('physionet_test_patients_*.npy'))
if not phys_seg_files or not phys_patient_files:
    raise FileNotFoundError(
        f"PhysioNet step 5 outputs not found in {MODEL_EXPORT_DIR}. "
        "Run 5_train_physionet_seg.py first.")
phys_seg_path     = phys_seg_files[-1]
phys_patient_path = phys_patient_files[-1]
print(f"  PhysioNet U-Net preds: {phys_seg_path.name}", flush=True)
print(f"  PhysioNet test patients: {phys_patient_path.name}", flush=True)

cq500_seg_files     = sorted(MODEL_EXPORT_DIR.glob('cq500_seg_maps_*.npy'))
cq500_patient_files = sorted(MODEL_EXPORT_DIR.glob('cq500_test_patients_*.npy'))
if not cq500_seg_files or not cq500_patient_files:
    raise FileNotFoundError(
        f"Seg-CQ500 step 6 outputs not found in {MODEL_EXPORT_DIR}. "
        "Run 6_train_cq500.py first.")
cq500_seg_path     = cq500_seg_files[-1]
cq500_patient_path = cq500_patient_files[-1]
print(f"  CQ500 U-Net preds:    {cq500_seg_path.name}", flush=True)
print(f"  CQ500 test patients:  {cq500_patient_path.name}", flush=True)

phys_unet_preds  = np.load(str(phys_seg_path)).astype(np.float32)
phys_test_patients = np.load(str(phys_patient_path))

cq500_unet_preds   = np.load(str(cq500_seg_path)).astype(np.float32)
cq500_test_patients = np.load(str(cq500_patient_path))


# ===========================================================================
# 2. load test slices for both datasets
# ===========================================================================
print("\nLoading PhysioNet test slices...", flush=True)
xs_p, ys_p, info_p = [], [], []
for p in phys_test_patients:
    try:
        x_p, y_p = load_physionet_patient(int(p))
        for s in range(len(x_p)):
            info_p.append({'patient': int(p), 'slice': s, 'has_ich': int(y_p[s].max() > 0)})
        xs_p.append(x_p); ys_p.append(y_p)
    except Exception as e:
        print(f"  [WARN] skipping physionet patient {p}: {e}", flush=True)
x_phys  = np.concatenate(xs_p, axis=0)
y_phys  = np.concatenate(ys_p, axis=0)
phys_df = pd.DataFrame(info_p)
print(f"  {len(x_phys)} slices, {phys_df['has_ich'].sum()} ICH-positive", flush=True)

if len(x_phys) != len(phys_unet_preds):
    raise ValueError(
        f"PhysioNet slice count mismatch: loaded {len(x_phys)} vs U-Net preds {len(phys_unet_preds)}. "
        "A patient may have failed to load -- check step 5 log for [WARN] lines.")

print("\nLoading Seg-CQ500 test slices...", flush=True)
xs_c, ys_c, info_c = [], [], []
for p in cq500_test_patients:
    try:
        x_p, y_p = load_cq500_patient(str(p))
        for s in range(len(x_p)):
            info_c.append({'patient': str(p), 'slice': s, 'has_ich': int(y_p[s].max() > 0)})
        xs_c.append(x_p); ys_c.append(y_p)
    except Exception as e:
        print(f"  [WARN] skipping cq500 patient {p}: {e}", flush=True)
x_cq500  = np.concatenate(xs_c, axis=0)
y_cq500  = np.concatenate(ys_c, axis=0)
cq500_df = pd.DataFrame(info_c)
print(f"  {len(x_cq500)} slices, {cq500_df['has_ich'].sum()} ICH-positive", flush=True)

if len(x_cq500) != len(cq500_unet_preds):
    raise ValueError(
        f"CQ500 slice count mismatch: loaded {len(x_cq500)} vs U-Net preds {len(cq500_unet_preds)}. "
        "A patient may have failed to load -- check step 6 log for [WARN] lines.")

phys_pos_idx  = phys_df[phys_df['has_ich']  == 1].index.values
cq500_pos_idx = cq500_df[cq500_df['has_ich'] == 1].index.values
print(f"\n  PhysioNet positive slices: {len(phys_pos_idx)}", flush=True)
print(f"  CQ500 positive slices:     {len(cq500_pos_idx)}", flush=True)


# ===========================================================================
# 3. GPU setup
# ===========================================================================
gpus = tf.config.list_physical_devices('GPU')
print(f"\nGPUs available: {len(gpus)}", flush=True)
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
    details = tf.config.experimental.get_device_details(gpu)
    print(f"  {gpu.name} - {details.get('device_name', 'unknown')}", flush=True)
if not gpus:
    print("WARNING: No GPU detected!", flush=True)


# ===========================================================================
# 4. generate GradCAMs -- load each model once, run on both datasets
# ===========================================================================
# cam_maps['PhysioNet'][method] and cam_maps['CQ500'][method] -> (N, H, W, 1) float32
cam_maps = {'PhysioNet': {}, 'CQ500': {}}

# CNN
if not os.path.exists(CNN_CKPT):
    print(f"\nWARNING: CNN checkpoint not found ({CNN_CKPT}) -- skipping CNN GradCAM.", flush=True)
else:
    print(f"\nGenerating CNN GradCAM (both datasets)...", flush=True)
    cnn_base = tf.keras.models.load_model(CNN_CKPT)
    inp_g = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name='cnn_gradcam_input')
    xg = inp_g
    last_conv_out = None
    for lyr in cnn_base.layers:
        xg = lyr(xg)
        if isinstance(lyr, tf.keras.layers.Conv2D):
            last_conv_out = xg
    cnn_gradcam_m = tf.keras.Model(inputs=inp_g, outputs=[last_conv_out, xg])

    raw = run_gradcam(cnn_gradcam_m, x_phys)
    cam_maps['PhysioNet']['CNN'] = aggregate_gradcam(raw).astype(np.float32)
    print(f"  PhysioNet CNN GradCAM: {cam_maps['PhysioNet']['CNN'].shape}", flush=True)

    raw = run_gradcam(cnn_gradcam_m, x_cq500)
    cam_maps['CQ500']['CNN'] = aggregate_gradcam(raw).astype(np.float32)
    print(f"  CQ500     CNN GradCAM: {cam_maps['CQ500']['CNN'].shape}", flush=True)

    raw = run_gradcam(cnn_gradcam_m, rsna_sample_x)
    rsna_cam_maps['CNN'] = aggregate_gradcam(raw).astype(np.float32)
    rsna_preds['CNN']    = get_preds(cnn_gradcam_m, rsna_sample_x)
    print(f"  RSNA      CNN GradCAM: {rsna_cam_maps['CNN'].shape}", flush=True)

    # classification arm: the model is already resident, so score all three datasets now
    print("  CNN classification pass...", flush=True)
    phys_probs['CNN']  = predict_batched(cnn_base, x_phys,  desc='CNN PhysioNet')
    cq500_probs['CNN'] = predict_batched(cnn_base, x_cq500, desc='CNN CQ500')
    rsna_probs['CNN']  = predict_batched(cnn_base, rsna_x_mm[:N_RSNA_EVAL], desc='CNN RSNA')

    del cnn_base, cnn_gradcam_m, raw; gc.collect()

# ViT
if not os.path.exists(VIT_CKPT):
    print(f"\nWARNING: ViT checkpoint not found ({VIT_CKPT}) -- skipping ViT GradCAM.", flush=True)
else:
    print(f"\nGenerating ViT GradCAM (both datasets)...", flush=True)
    vit_base        = tf.keras.models.load_model(
        VIT_CKPT, custom_objects={'PositionEmbedding': PositionEmbedding})
    vit_conv_layers = [l for l in vit_base.layers if isinstance(l, tf.keras.layers.Conv2D)]
    vit_gradcam_m   = tf.keras.Model(
        inputs=vit_base.input,
        outputs=[vit_conv_layers[0].output, vit_base.output])

    raw = run_gradcam(vit_gradcam_m, x_phys)
    cam_maps['PhysioNet']['ViT'] = aggregate_gradcam(raw).astype(np.float32)
    print(f"  PhysioNet ViT GradCAM: {cam_maps['PhysioNet']['ViT'].shape}", flush=True)

    raw = run_gradcam(vit_gradcam_m, x_cq500)
    cam_maps['CQ500']['ViT'] = aggregate_gradcam(raw).astype(np.float32)
    print(f"  CQ500     ViT GradCAM: {cam_maps['CQ500']['ViT'].shape}", flush=True)

    raw = run_gradcam(vit_gradcam_m, rsna_sample_x)
    rsna_cam_maps['ViT'] = aggregate_gradcam(raw).astype(np.float32)
    rsna_preds['ViT']    = get_preds(vit_gradcam_m, rsna_sample_x)
    print(f"  RSNA      ViT GradCAM: {rsna_cam_maps['ViT'].shape}", flush=True)

    print("  ViT classification pass...", flush=True)
    phys_probs['ViT']  = predict_batched(vit_base, x_phys,  desc='ViT PhysioNet')
    cq500_probs['ViT'] = predict_batched(vit_base, x_cq500, desc='ViT CQ500')
    rsna_probs['ViT']  = predict_batched(vit_base, rsna_x_mm[:N_RSNA_EVAL], desc='ViT RSNA')

    del vit_base, vit_gradcam_m, raw; gc.collect()

# Hybrid (average of CNN + ViT CAMs)
print(f"\nGenerating Hybrid GradCAM...", flush=True)
for ds in ('PhysioNet', 'CQ500'):
    if 'CNN' in cam_maps[ds] and 'ViT' in cam_maps[ds]:
        cam_maps[ds]['Hybrid'] = (
            (cam_maps[ds]['CNN'] + cam_maps[ds]['ViT']) / 2.0).astype(np.float32)
        print(f"  {ds} Hybrid GradCAM: {cam_maps[ds]['Hybrid'].shape}", flush=True)
    else:
        print(f"  WARNING: CNN or ViT missing for {ds} -- skipping Hybrid.", flush=True)

if 'CNN' in rsna_cam_maps and 'ViT' in rsna_cam_maps:
    rsna_cam_maps['Hybrid'] = (
        (rsna_cam_maps['CNN'] + rsna_cam_maps['ViT']) / 2.0).astype(np.float32)
    rsna_preds['Hybrid']    = (rsna_preds['CNN'] + rsna_preds['ViT']) / 2.0
    print(f"  RSNA  Hybrid GradCAM: {rsna_cam_maps['Hybrid'].shape}", flush=True)

# hybrid classification probabilities, matching how 3_train_hybrid.py defines the ensemble
for store in (rsna_probs, phys_probs, cq500_probs):
    if 'CNN' in store and 'ViT' in store:
        store['Hybrid'] = ((store['CNN'] + store['ViT']) / 2.0).astype(np.float32)
if 'Hybrid' in rsna_probs:
    print(f"  Hybrid classification probabilities built (mean of CNN and ViT)", flush=True)

# U-Net predictions
cam_maps['PhysioNet']['U-Net'] = phys_unet_preds
cam_maps['CQ500']['U-Net']     = cq500_unet_preds

# ---- Chance baseline ----
# a uniform random saliency map at the same threshold, entered as just another
# method key so it flows through every table and figure with no special-casing.
# its argmax is a uniformly random pixel, so its pointing game is Bernoulli(lesion
# area fraction), and its average precision is the slice prevalence.
print(f"\nBuilding the Chance baseline (uniform random maps, seed={CHANCE_SEED})...", flush=True)
_chance_rng = np.random.default_rng(CHANCE_SEED)
for _ds, _ref in (('PhysioNet', phys_unet_preds), ('CQ500', cq500_unet_preds)):
    cam_maps[_ds]['Chance'] = _chance_rng.random(_ref.shape, dtype=np.float32)
    print(f"  {_ds} Chance map: {cam_maps[_ds]['Chance'].shape}", flush=True)


# ---- U-Net classifier on RSNA (the segmentation U-Nets cover PhysioNet and CQ500) ----
if not os.path.exists(UNET_CLS_CKPT):
    print(f"\nWARNING: RSNA U-Net classifier not found ({UNET_CLS_CKPT}) "
          "-- skipping its classification row.", flush=True)
else:
    print(f"\nRunning RSNA U-Net classifier...", flush=True)
    try:
        unet_cls = tf.keras.models.load_model(
            UNET_CLS_CKPT, custom_objects={'PositionEmbedding': PositionEmbedding})
        rsna_probs['U-Net'] = predict_batched(unet_cls, rsna_x_mm[:N_RSNA_EVAL],
                                              desc='U-Net RSNA')
        del unet_cls; gc.collect()
    except Exception as e:
        print(f"  [WARN] could not load or run the RSNA U-Net classifier: {e}", flush=True)


# ===========================================================================
# 5. evaluate metrics
# ===========================================================================
print(f"\nEvaluating (threshold={THRESHOLD})...", flush=True)

records   = []
per_slice = {'PhysioNet': {}, 'CQ500': {}}   # dataset -> method -> metric -> (n_pos,) array
for ds_name, y_test, pos_idx in [
        ('PhysioNet', y_phys,  phys_pos_idx),
        ('CQ500',     y_cq500, cq500_pos_idx)]:
    if not cam_maps[ds_name]:
        print(f"  no maps for {ds_name}, skipping.", flush=True)
        continue
    for method, maps in cam_maps[ds_name].items():
        ps  = per_slice_metrics(maps, y_test, pos_idx)
        row = eval_method(maps, y_test, pos_idx, per_slice=ps)
        per_slice[ds_name][method] = ps
        row['dataset'] = ds_name
        row['method']  = method
        records.append(row)
        print(f"  {ds_name:<12} {method:<8} Dice {row['dice_mean']:.4f} +/- {row['dice_std']:.4f}",
              flush=True)

results_df = pd.DataFrame(records).set_index(['dataset', 'method'])


# ===========================================================================
# 6. print summary table
# ===========================================================================
print("\n" + "=" * 100, flush=True)
print(f"RESULTS  (ICH-positive slices, threshold={THRESHOLD})", flush=True)
print(f"  PhysioNet n={len(phys_pos_idx)},  CQ500 n={len(cq500_pos_idx)}", flush=True)
print("=" * 100, flush=True)
print(f"{'Dataset':<12} {'Method':<8} {'Dice':>17} {'IoU':>17} {'Recall':>17} {'Precision':>17} {'PG':>6}",
      flush=True)
print("-" * 100, flush=True)
for (ds_name, method), row in results_df.iterrows():
    print(
        f"{ds_name:<12} {method:<8} "
        f"{row['dice_mean']:.4f} +/- {row['dice_std']:.4f}  "
        f"{row['iou_mean']:.4f} +/- {row['iou_std']:.4f}  "
        f"{row['recall_mean']:.4f} +/- {row['recall_std']:.4f}  "
        f"{row['precision_mean']:.4f} +/- {row['precision_std']:.4f}  "
        f"PG {row['pointing_game']:.4f}",
        flush=True)
print("=" * 100, flush=True)


# ===========================================================================
# 7. save results + figures
# ===========================================================================
ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

csv_path = OUTPUT_DIR / f'all_comparison_results_{ts}.csv'
results_df.to_csv(str(csv_path))
print(f"\nSaved results: {csv_path.name}", flush=True)

if not MATPLOTLIB:
    print("matplotlib not available -- skipping figures.", flush=True)
else:
    # ---- per-dataset bar charts ----
    metric_specs = [('dice', 'Dice'), ('iou', 'IoU'),
                    ('recall', 'Recall'), ('precision', 'Precision'),
                    ('pointing_game', 'Pointing Game'), ('ap', 'Avg Precision')]

    for ds_name, x_test, y_test, pos_idx, slice_df in [
            ('PhysioNet', x_phys,  y_phys,  phys_pos_idx,  phys_df),
            ('CQ500',     x_cq500, y_cq500, cq500_pos_idx, cq500_df)]:

        if ds_name not in [r[0] for r in results_df.index]:
            continue
        ds_results = results_df.loc[ds_name]
        methods    = ds_results.index.tolist()
        n_m        = len(methods)
        bin_maps   = {m: (cam_maps[ds_name][m].astype(np.float32) > THRESHOLD).astype(np.float32)
                      for m in methods}

        # bar chart
        fig, axes = plt.subplots(2, 3,
                                 figsize=(FIG_W_WIDE, FIG_W_WIDE * 0.52))
        
        clusters = slice_df.loc[pos_idx, 'patient'].astype(str).values
        for ax, (metric, label) in zip(axes.ravel(), metric_specs):
            key   = 'pg' if metric == 'pointing_game' else metric
            means, lo_arm, hi_arm = [], [], []
            for m in methods:
                v  = np.asarray(per_slice[ds_name][m][key], dtype=float)
                mu = float(v.mean())
                lo, hi = su.cluster_bootstrap_ci(v, clusters,
                                                 n_boot=N_BOOTSTRAP, seed=SEED)
                means.append(mu)
                lo_arm.append(max(mu - lo, 0.0))
                hi_arm.append(max(hi - mu, 0.0))
            means  = np.asarray(means)
            lo_arm = np.asarray(lo_arm)
            hi_arm = np.asarray(hi_arm)
            bars   = ax.barh(methods, means, xerr=np.vstack([lo_arm, hi_arm]),
                             capsize=2.5, color='steelblue', height=0.5,
                             error_kw={'elinewidth': 0.9})
            # the value sits past the end of the whisker, and the axis is widened
            # to leave room for it, otherwise the text lands on the error bar
            reach = max(float(np.nanmax(means + hi_arm)), 0.05)
            ax.set_xlim(0, reach * 1.30)
            pad = reach * 0.035
            ax.set_xlabel(label)
            ax.set_title(f'{label}  (mean, 95% CI)')
            ax.invert_yaxis()
            for bar, v, e in zip(bars, means, hi_arm):
                ax.text(v + e + pad, bar.get_y() + bar.get_height() / 2,
                        f'{v:.3f}', va='center', fontsize=TITLE_FS)
        plt.suptitle(f'{ds_name}  (n={len(pos_idx)} ICH-positive slices, threshold={THRESHOLD})',
                     fontsize=SUPTITLE_FS)
        plt.tight_layout()
        bar_path = OUTPUT_DIR / f'all_comparison_bar_{ds_name.lower()}_{ts}.png'
        plt.savefig(str(bar_path), dpi=FIG_DPI, bbox_inches='tight')
        plt.close()
        print(f"Saved bar chart: {bar_path.name}", flush=True)

        # qualitative: one representative slice per positive patient (largest GT lesion)
        pos_by_patient = slice_df[slice_df['has_ich'] == 1].groupby('patient').groups
        patient_keys   = list(pos_by_patient.keys())
        np.random.shuffle(patient_keys)
        sample_idx = []
        for pid in patient_keys[:N_QUAL_IMGS]:
            idxs = pos_by_patient[pid].tolist()
            best = max(idxs, key=lambda i: float(y_test[i, :, :, 0].sum()))
            sample_idx.append(best)

        # one figure per sample: 6 panels in a 3x2 grid, portrait width, so each
        # brain is large enough to read on the page
        for sl in sample_idx:
            p_id   = slice_df.loc[sl, 'patient']
            s_id   = slice_df.loc[sl, 'slice']
            gt_bin = y_test[sl, :, :, 0] > 0.5
            tag    = f'p{p_id}_s{s_id}'

            # soft maps: CT with the mask outlined, then one panel per method
            fig, axes = plt.subplots(2, 3, figsize=grid_figsize(3, 2, width_in=FIG_W),
                                     squeeze=False)
            flat = axes.ravel()
            flat[0].imshow(x_test[sl, :, :, 0], cmap='gray')
            flat[0].contour(gt_bin, levels=[0.5], colors='cyan', linewidths=0.9)
            flat[0].set_title('CT  (cyan = radiologist mask)', fontsize=TITLE_FS)
            for ax_i, method in enumerate(methods, start=1):
                d, iou, *_ = seg_metrics(y_test[sl], bin_maps[method][sl])
                flat[ax_i].imshow(cam_maps[ds_name][method][sl, :, :, 0].astype(np.float32),
                                  cmap='hot', vmin=0, vmax=1)
                flat[ax_i].set_title(f'{method}\nDice {d:.3f}  IoU {iou:.3f}',
                                     fontsize=TITLE_FS)
            for ax in flat:
                ax.axis('off')
            plt.suptitle(f'{ds_name} {tag} -- unthresholded saliency maps',
                         fontsize=SUPTITLE_FS, y=1.00)
            plt.tight_layout()
            soft_path = OUTPUT_DIR / f'all_comparison_soft_{ds_name.lower()}_{tag}_{ts}.png'
            plt.savefig(str(soft_path), dpi=FIG_DPI, bbox_inches='tight')
            plt.close()
            print(f"Saved soft-map figure: {soft_path.name}", flush=True)

            # error maps: same grid, same first panel
            fig, axes = plt.subplots(2, 3, figsize=grid_figsize(3, 2, width_in=FIG_W,
                                                                title_in=0.5),
                                     squeeze=False)
            flat = axes.ravel()
            flat[0].imshow(x_test[sl, :, :, 0], cmap='gray')
            flat[0].contour(gt_bin, levels=[0.5], colors='cyan', linewidths=0.9)
            flat[0].set_title('CT  (cyan = radiologist mask)', fontsize=TITLE_FS)
            for ax_i, method in enumerate(methods, start=1):
                d, iou, rec, prec = seg_metrics(y_test[sl], bin_maps[method][sl])
                flat[ax_i].imshow(error_rgb(y_test[sl], bin_maps[method][sl]))
                flat[ax_i].set_title(f'{method}\nDice {d:.3f}  Rec {rec:.3f}',
                                     fontsize=TITLE_FS)
            for ax in flat:
                ax.axis('off')
            legend_patches = [
                mpatches.Patch(color=(0/255, 180/255, 0/255),  label='TP'),
                mpatches.Patch(color=(200/255, 0/255, 0/255),  label='FP'),
                mpatches.Patch(color=(0/255, 80/255, 200/255), label='FN'),
                mpatches.Patch(color=(30/255, 30/255, 30/255), label='TN'),
            ]
            fig.legend(handles=legend_patches, loc='lower center', ncol=4,
                       fontsize=TITLE_FS, frameon=True, bbox_to_anchor=(0.5, -0.03))
            plt.suptitle(f'{ds_name} {tag} -- error maps', fontsize=SUPTITLE_FS, y=1.00)
            plt.tight_layout()
            err_path = OUTPUT_DIR / f'all_comparison_error_{ds_name.lower()}_{tag}_{ts}.png'
            plt.savefig(str(err_path), dpi=FIG_DPI, bbox_inches='tight')
            plt.close()
            print(f"Saved error-map figure: {err_path.name}", flush=True)

    # ---- combined Dice bar chart (both datasets side-by-side) ----
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_W * 0.55))
    all_methods = list(dict.fromkeys(
        m for (_, m) in results_df.index))   # preserve insertion order, dedupe
    x      = np.arange(len(all_methods))
    width  = 0.35
    colors = {'PhysioNet': 'steelblue', 'CQ500': 'darkorange'}

    top = 0.0
    for i, (ds_name, color) in enumerate(colors.items()):
        if ds_name not in results_df.index.get_level_values('dataset'):
            continue
        clusters = (phys_df if ds_name == 'PhysioNet' else cq500_df).loc[
            phys_pos_idx if ds_name == 'PhysioNet' else cq500_pos_idx,
            'patient'].astype(str).values
        means, lo_arm, hi_arm = [], [], []
        for m in all_methods:
            v = per_slice[ds_name].get(m, {}).get('dice')
            if v is None:
                means.append(0.0); lo_arm.append(0.0); hi_arm.append(0.0); continue
            v  = np.asarray(v, dtype=float)
            mu = float(v.mean())
            lo, hi = su.cluster_bootstrap_ci(v, clusters, n_boot=N_BOOTSTRAP, seed=SEED)
            means.append(mu)
            lo_arm.append(max(mu - lo, 0.0))
            hi_arm.append(max(hi - mu, 0.0))
        ax.bar(x + i * width, means, width,
               yerr=np.vstack([lo_arm, hi_arm]), label=ds_name,
               color=color, capsize=3, alpha=0.85, error_kw={'elinewidth': 0.9})
        top = max(top, max(np.asarray(means) + np.asarray(hi_arm)))

    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(all_methods)
    ax.set_ylim(0, max(top, 0.05) * 1.12)
    ax.set_ylabel('Dice (mean, 95% CI)')
    ax.set_title(f'Dice -- PhysioNet vs Seg-CQ500  (threshold={THRESHOLD})')
    ax.legend()
    plt.tight_layout()
    combined_path = OUTPUT_DIR / f'all_comparison_dice_combined_{ts}.png'
    plt.savefig(str(combined_path), dpi=FIG_DPI, bbox_inches='tight')
    plt.close()
    print(f"Saved combined Dice chart: {combined_path.name}", flush=True)

# ===========================================================================
# 8. RSNA qualitative GradCAM panel (no pixel-mask GT available)
# ===========================================================================
if MATPLOTLIB and rsna_cam_maps:
    methods_rsna = list(rsna_cam_maps.keys())
    n_cols = 1 + len(methods_rsna)   # CT + one column per method

    fig, axes = plt.subplots(len(rsna_sample_x), n_cols,
                             figsize=grid_figsize(n_cols, len(rsna_sample_x),
                                                  title_in=0.62),
                             squeeze=False)

    for row_i in range(len(rsna_sample_x)):
        true_labels = [RSNA_LABEL_COLS[j] for j in range(len(RSNA_LABEL_COLS))
                       if rsna_sample_y[row_i, j] > 0.5]
        true_str = (', '.join(RSNA_LABEL_SHORT[t] for t in true_labels)
                    if true_labels else 'none')

        axes[row_i, 0].imshow(rsna_sample_x[row_i, :, :, 0], cmap='gray')
        axes[row_i, 0].set_title(f'RSNA slice {_chosen[row_i]}\nTrue: {true_str}',
                                 fontsize=TITLE_FS)
        axes[row_i, 0].axis('off')

        for col_i, method in enumerate(methods_rsna):
            cam = rsna_cam_maps[method][row_i, :, :, 0].astype(np.float32)
            axes[row_i, 1 + col_i].imshow(cam, cmap='hot', vmin=0, vmax=1)
            if method in rsna_preds:
                parts = [f'{RSNA_LABEL_SHORT[c]} {rsna_preds[method][row_i, j]:.2f}'
                         for j, c in enumerate(RSNA_LABEL_COLS)]
                # three per line, otherwise the string runs into the next panel
                pred_str = '\n'.join('  '.join(parts[i:i + 3])
                                     for i in range(0, len(parts), 3))
                axes[row_i, 1 + col_i].set_title(f'{method}\n{pred_str}', fontsize=TITLE_FS)
            else:
                axes[row_i, 1 + col_i].set_title(method, fontsize=TITLE_FS)
            axes[row_i, 1 + col_i].axis('off')

    plt.suptitle('RSNA test set -- GradCAM activations (no pixel-mask GT)\n'
                 'EDH epidural, IPH intraparenchymal, IVH intraventricular, '
                 'SAH subarachnoid, SDH subdural',
                 fontsize=SUPTITLE_FS, y=1.00)
    plt.tight_layout()
    rsna_fig_path = OUTPUT_DIR / f'all_comparison_rsna_gradcam_{ts}.png'
    plt.savefig(str(rsna_fig_path), dpi=FIG_DPI, bbox_inches='tight')
    plt.close()
    print(f"Saved RSNA qualitative figure: {rsna_fig_path.name}", flush=True)


# ===========================================================================
# 9. classification AUC + inferential statistics
# ===========================================================================
print("\n" + "=" * 100, flush=True)
print("CLASSIFICATION + INFERENTIAL STATISTICS", flush=True)
print("=" * 100, flush=True)

bundle = {
    'meta': {
        'generated':     ts,
        'seed':          SEED,
        'img_size':      IMG_SIZE,
        'cam_threshold': THRESHOLD,
        'rsna_cache':    _meta_path.name,
        'rsna_slices':   int(N_RSNA_EVAL),
        'cnn_ckpt':      os.path.basename(CNN_CKPT),
        'vit_ckpt':      os.path.basename(VIT_CKPT),
        'unet_cls_ckpt': os.path.basename(UNET_CLS_CKPT),
        'physionet_unet': phys_seg_path.name,
        'cq500_unet':     cq500_seg_path.name,
        'hybrid_definition': 'mean of CNN and ViT probabilities / CAMs',
        'any_ich_score':     'max over the five subtype probabilities',
        'chance_seed':       CHANCE_SEED,
        'chance_definition': 'uniform random saliency map at the same threshold; its '
                             'pointing game expectation is the mean lesion area fraction',
    },
    'rsna': None,
    'external': {},
}

if rsna_probs:
    bundle['rsna'] = {
        'y':          rsna_y_eval,
        'label_cols': RSNA_LABEL_COLS,
        'probs':      {m: rsna_probs[m] for m in su.order_methods(list(rsna_probs))},
    }
    print(f"  RSNA classification models: {list(bundle['rsna']['probs'])}", flush=True)
else:
    print("  no RSNA classification probabilities -- classification arm limited to "
          "the external datasets.", flush=True)

# noisy-OR is a sensitivity check only, max stays the headline score
noisy_or_rows = []

for ds_name, df_slices, probs, unet_maps in [
        ('PhysioNet', phys_df,  phys_probs,  phys_unet_preds),
        ('CQ500',     cq500_df, cq500_probs, cq500_unet_preds)]:

    if not per_slice[ds_name]:
        continue

    y_any    = df_slices['has_ich'].values.astype(np.int8)
    patients = df_slices['patient'].astype(str).values
    pos_idx  = phys_pos_idx if ds_name == 'PhysioNet' else cq500_pos_idx

    scores = {}
    for model in su.order_methods(list(probs)):
        s_max, s_nor  = any_ich_score(probs[model])
        scores[model] = s_max
        if y_any.sum() and y_any.sum() < len(y_any):
            auc_max = su.delong_auc_ci(y_any, s_max)[0]
            auc_nor = su.delong_auc_ci(y_any, s_nor)[0]
            noisy_or_rows.append({'dataset': ds_name, 'model': model,
                                  'auc_max': auc_max, 'auc_noisy_or': auc_nor})
    scores['U-Net'] = unet_slice_score(unet_maps)

    bundle['external'][ds_name] = {
        'y_any':       y_any,
        'patient':     patients,
        'pos_patient': df_slices.loc[pos_idx, 'patient'].astype(str).values,
        'scores':      scores,
        'seg':         per_slice[ds_name],
    }
    print(f"  {ds_name}: {len(y_any)} slices, {int(y_any.sum())} ICH-positive, "
          f"classification models {list(scores)}", flush=True)

# the bundle does not store lesion areas, so this is carried rather than recomputed
for ds_name, y_test, pos_idx in [('PhysioNet', y_phys,  phys_pos_idx),
                                 ('CQ500',     y_cq500, cq500_pos_idx)]:
    if ds_name not in bundle['external']:
        continue
    analytic, empirical = chance_pointing_game(y_test, pos_idx)
    bundle['external'][ds_name]['pg_chance'] = {'analytic': analytic,
                                                'empirical': empirical}

bundle_path = OUTPUT_DIR / f'all_raw_{ts}.npz'
su.save_bundle(bundle_path, bundle)
print(f"\nSaved raw bundle: {bundle_path.name}", flush=True)
print(f"  re-run the statistics on CPU from thesis.ipynb using {bundle_path.name}", flush=True)

if noisy_or_rows:
    nor_path = OUTPUT_DIR / f'all_sensitivity_noisy_or_{ts}.csv'
    pd.DataFrame(noisy_or_rows).to_csv(str(nor_path), index=False)
    print(f"Saved noisy-OR sensitivity check: {nor_path.name}", flush=True)

tables, written = su.run_all_analysis(bundle, OUTPUT_DIR, ts,
                                      prefix='all', n_boot=N_BOOTSTRAP, seed=SEED)

# ---- console summary of the headline numbers ----
auc_df = tables['auc']
if not auc_df.empty:
    print("\n" + "=" * 100, flush=True)
    print("CLASSIFICATION AUC SUMMARY", flush=True)
    print("=" * 100, flush=True)
    print(f"{'Dataset':<12} {'Target':<22} {'Model':<8} {'AUC':>8} {'95% CI':>22} {'AvgPrec':>9}",
          flush=True)
    print("-" * 100, flush=True)
    headline = auc_df[auc_df['target'].isin(['any ICH', 'weighted average'])]
    for _, r in headline.iterrows():
        ci = f"[{r['ci_lo']:.4f}, {r['ci_hi']:.4f}]" if np.isfinite(r['ci_lo']) else 'n/a'
        print(f"{r['dataset']:<12} {r['target']:<22} {r['model']:<8} "
              f"{r['auc']:>8.4f} {ci:>22} {r['ap']:>9.4f}", flush=True)
    print("=" * 100, flush=True)

print("\nDone!")
