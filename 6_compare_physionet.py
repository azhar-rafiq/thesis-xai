"""
PhysioNet CT-ICH -- XAI comparison: GradCAM vs supervised U-Net vs ground truth
for each ICH-positive PhysioNet test slice, compare:
  - CNN GradCAM   (RSNA-trained, max over 5 ICH class channels)
  - ViT GradCAM   (RSNA-trained, max over 5 ICH class channels)
  - hybrid GradCAM (RSNA-trained, max over 5 ICH class channels)
  - supervised U-Net mask (PhysioNet-trained, from step 5)
vs. PhysioNet ground-truth binary pixel mask.
metrics: Dice, IoU, pixel recall, pixel precision (mean +/- std over positive slices).
run via sbatch: sbatch 6_compare_physionet.sh
"""

import gc
import os
import sys
import datetime
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
from pathlib import Path
from skimage.transform import resize
from dotenv import load_dotenv

try:
    import nibabel as nib
except ImportError:
    raise ImportError("nibabel not installed -- run: pip install nibabel")

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    MATPLOTLIB = True
except ImportError:
    MATPLOTLIB = False

load_dotenv()

SEED = 20260605
tf.random.set_seed(SEED)
np.random.seed(SEED)

PHYSIONET_DIR    = Path('/rds/projects/k/karwatha-karwath-hds-pg-research/axr1222/data/physionet.org/files/ct-ich/1.3.1')
CT_DIR           = PHYSIONET_DIR / 'ct_scans'
MASK_DIR         = PHYSIONET_DIR / 'masks'
MODEL_EXPORT_DIR = Path('/rds/projects/k/karwatha-karwath-hds-pg-research/axr1222/models')
OUTPUT_DIR       = Path(os.path.expanduser('~/thesis-xai'))

IMG_SIZE     = 256
BATCH_SIZE   = 16
THRESHOLD    = 0.5   # binarisation threshold for all methods
N_CLASSES    = 5
N_QUAL_IMGS  = 5     # sample slices saved for visual inspection

# same model paths as steps 4 and 5
CNN_CKPT    = str(MODEL_EXPORT_DIR / 'rsna_cnn_20260605_05000_auc.keras')
VIT_CKPT    = str(MODEL_EXPORT_DIR / 'rsna_vit_20260608-ZU_08740_auc.keras')
HYBRID_CKPT = str(MODEL_EXPORT_DIR / 'rsna_hybrid_20260608-AR_09512_auc.keras')


# ---- CT windowing (same as all other steps) ----
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


def load_patient(patient_num):
    """load one PhysioNet patient; returns (images, masks) arrays over slices."""
    ct_vol   = nib.load(str(CT_DIR   / f'{patient_num:03d}.nii')).get_fdata().astype(np.float32)
    mask_vol = nib.load(str(MASK_DIR / f'{patient_num:03d}.nii')).get_fdata().astype(np.float32)
    xs, ys = [], []
    for s in range(ct_vol.shape[2]):
        xs.append(hu_to_3channel(ct_vol[:, :, s]))
        mask_s = resize(mask_vol[:, :, s], (IMG_SIZE, IMG_SIZE), anti_aliasing=False)
        ys.append((mask_s > 0.5).astype(np.float32)[..., np.newaxis])
    return np.array(xs, dtype=np.float32), np.array(ys, dtype=np.float32)


# ---- per-slice segmentation metrics ----
def pointing_game(y_true_hw1, cam_hw1):
    """
    1 if the peak activation pixel falls inside the GT mask, else 0.
    Selvaraju et al. (2020), Zotero 5FCJIKMB, doi:10.1007/s11263-019-01228-7
    """
    peak = np.unravel_index(np.argmax(cam_hw1[..., 0]), cam_hw1[..., 0].shape)
    return int(y_true_hw1[peak[0], peak[1], 0] > 0.5)


def seg_metrics(y_true, y_pred_bin):
    """compute Dice, IoU, recall, precision for a single slice (ravelled)."""
    yt = y_true.ravel().astype(np.float32)
    yp = y_pred_bin.ravel().astype(np.float32)
    inter     = (yt * yp).sum()
    dice      = (2 * inter + 1) / (yt.sum() + yp.sum() + 1)
    iou       = (inter + 1) / (yt.sum() + yp.sum() - inter + 1)
    recall    = inter / (yt.sum() + 1e-8)
    precision = inter / (yp.sum() + 1e-8)
    return float(dice), float(iou), float(recall), float(precision)


def eval_method(maps_nchw1, y_test, pos_idx):
    """
    evaluate a map array (N, H, W, 1) against y_test on pos_idx slices.
    pointing_game uses the raw soft map (threshold-free); all others use binarised.
    """
    maps_bin = (maps_nchw1.astype(np.float32) > THRESHOLD).astype(np.float32)
    dices, ious, recs, precs, pg_hits = [], [], [], [], []
    for idx in pos_idx:
        d, iou, rec, prec = seg_metrics(y_test[idx], maps_bin[idx])
        dices.append(d); ious.append(iou); recs.append(rec); precs.append(prec)
        pg_hits.append(pointing_game(y_test[idx], maps_nchw1[idx]))
    return {
        'dice_mean':      float(np.mean(dices)),
        'dice_std':       float(np.std(dices)),
        'iou_mean':       float(np.mean(ious)),
        'iou_std':        float(np.std(ious)),
        'recall_mean':    float(np.mean(recs)),
        'recall_std':     float(np.std(recs)),
        'precision_mean': float(np.mean(precs)),
        'precision_std':  float(np.std(precs)),
        'pointing_game':  float(np.mean(pg_hits)),   # proportion of hits (threshold-free)
        'n_slices':       len(pos_idx),
    }


# ---- GradCAM (same implementation as step 4) ----
def compute_gradcam(gradcam_model, x_batch, out_size=IMG_SIZE, n_classes=N_CLASSES):
    """
    per-class GradCAM for a batch.
    returns float16 of shape (B, out_size, out_size, n_classes).
    """
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
    return np.stack(cls_maps, axis=-1)  # (B, H, W, N_CLASSES)


def run_gradcam(gradcam_model, X):
    """run GradCAM over all slices in batches; returns (N, H, W, N_CLASSES) float16."""
    results = []
    for i in range(0, len(X), BATCH_SIZE):
        results.append(compute_gradcam(gradcam_model, X[i:i + BATCH_SIZE].copy()))
    return np.vstack(results)


def aggregate_gradcam(cam_maps):
    """(N, H, W, N_CLASSES) -> (N, H, W, 1): max over ICH class channels."""
    return cam_maps.max(axis=-1, keepdims=True)


def error_rgb(y_true_hw1, y_pred_bin_hw1):
    """(H, W, 1) binary arrays -> (H, W, 3) uint8: TP=green FP=red FN=blue TN=dark."""
    yt  = y_true_hw1[..., 0]    > 0.5
    yp  = y_pred_bin_hw1[..., 0] > 0.5
    rgb = np.zeros((*yt.shape, 3), dtype=np.uint8)
    rgb[~yt & ~yp] = [30,  30,  30]   # TN -- dark gray
    rgb[ yt &  yp] = [0,  180,   0]   # TP -- green
    rgb[~yt &  yp] = [200,  0,   0]   # FP -- red
    rgb[ yt & ~yp] = [0,   80, 200]   # FN -- blue
    return rgb


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


# 1. discover step 5 outputs (most recent by filename sort)
print("Discovering step 5 outputs...", flush=True)
seg_files     = sorted(MODEL_EXPORT_DIR.glob('physionet_seg_maps_*.npy'))
patient_files = sorted(MODEL_EXPORT_DIR.glob('physionet_test_patients_*.npy'))

if not seg_files or not patient_files:
    raise FileNotFoundError(
        f"Step 5 outputs not found in {MODEL_EXPORT_DIR}. "
        "Run 5_train_physionet_seg.py first."
    )

seg_path     = seg_files[-1]
patient_path = patient_files[-1]
print(f"  U-Net preds:   {seg_path.name}", flush=True)
print(f"  test patients: {patient_path.name}", flush=True)

test_patients = np.load(str(patient_path))
unet_preds    = np.load(str(seg_path)).astype(np.float32)   # (N_slices, H, W, 1)
print(f"  test patients: {test_patients.tolist()}", flush=True)
print(f"  unet_preds shape: {unet_preds.shape}", flush=True)


# 2. load PhysioNet test slices (same patient order as step 5)
print("\nLoading PhysioNet test slices...", flush=True)
xs_list, ys_list, slice_info = [], [], []
loaded_patients = []

for p in test_patients:
    try:
        x_p, y_p = load_patient(int(p))
        for s in range(len(x_p)):
            slice_info.append({'patient': int(p), 'slice': s,
                               'has_ich': int(y_p[s].max() > 0)})
        xs_list.append(x_p)
        ys_list.append(y_p)
        loaded_patients.append(int(p))
    except Exception as e:
        print(f"  [WARN] skipping patient {p}: {e}", flush=True)

x_test   = np.concatenate(xs_list, axis=0)    # (N, H, W, 3)
y_test   = np.concatenate(ys_list, axis=0)    # (N, H, W, 1)
slice_df = pd.DataFrame(slice_info)
print(f"  {len(x_test)} slices, {slice_df['has_ich'].sum()} ICH-positive", flush=True)

if len(x_test) != len(unet_preds):
    raise ValueError(
        f"Slice count mismatch: PhysioNet {len(x_test)} vs U-Net preds {len(unet_preds)}. "
        "Possible cause: a patient failed to load in step 5 but is listed in test_patients. "
        "Check the step 5 log for [WARN] lines."
    )

pos_idx = slice_df[slice_df['has_ich'] == 1].index.values
print(f"  positive slices for evaluation: {len(pos_idx)}", flush=True)


# 3. GPU setup
gpus = tf.config.list_physical_devices('GPU')
print(f"\nGPUs available: {len(gpus)}", flush=True)
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
    details = tf.config.experimental.get_device_details(gpu)
    print(f"  {gpu.name} - {details.get('device_name', 'unknown')}", flush=True)
if not gpus:
    print("WARNING: No GPU detected!", flush=True)


# 4. generate GradCAM on PhysioNet test slices
#    (models are RSNA-trained; GradCAM is generated on PhysioNet data)
all_cam_maps = {}  # method -> (N, H, W, 1) float32, in [0,1]

# CNN
if not os.path.exists(CNN_CKPT):
    print(f"\nWARNING: CNN checkpoint not found ({CNN_CKPT}) -- skipping CNN GradCAM.", flush=True)
else:
    print(f"\nGenerating CNN GradCAM...", flush=True)
    cnn_base = tf.keras.models.load_model(CNN_CKPT)
    inp_g = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name='cnn_gradcam_input')
    xg = inp_g
    last_conv_out = None
    for layer in cnn_base.layers:
        xg = layer(xg)
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv_out = xg
    cnn_gradcam_m = tf.keras.Model(inputs=inp_g, outputs=[last_conv_out, xg])
    raw = run_gradcam(cnn_gradcam_m, x_test)
    all_cam_maps['CNN'] = aggregate_gradcam(raw).astype(np.float32)
    print(f"  CNN GradCAM shape: {all_cam_maps['CNN'].shape}", flush=True)
    del cnn_base, cnn_gradcam_m, raw; gc.collect()

# ViT -- patch embedding is the only Conv2D; GradCAM upsamples from 16x16 to 256x256
if not os.path.exists(VIT_CKPT):
    print(f"\nWARNING: ViT checkpoint not found ({VIT_CKPT}) -- skipping ViT GradCAM.", flush=True)
else:
    print(f"\nGenerating ViT GradCAM...", flush=True)
    vit_base        = tf.keras.models.load_model(
        VIT_CKPT, custom_objects={'PositionEmbedding': PositionEmbedding}
    )
    vit_conv_layers = [l for l in vit_base.layers if isinstance(l, tf.keras.layers.Conv2D)]
    vit_gradcam_m   = tf.keras.Model(
        inputs=vit_base.input,
        outputs=[vit_conv_layers[0].output, vit_base.output],
    )
    raw = run_gradcam(vit_gradcam_m, x_test)
    all_cam_maps['ViT'] = aggregate_gradcam(raw).astype(np.float32)
    print(f"  ViT GradCAM shape: {all_cam_maps['ViT'].shape}", flush=True)
    del vit_base, vit_gradcam_m, raw; gc.collect()

# hybrid -- average of CNN and ViT GradCAMs (mathematically equivalent to hybrid model GradCAM
# since the hybrid is just (CNN + ViT) / 2 and normalisation cancels the 1/2 factor)
print(f"\nGenerating Hybrid GradCAM...", flush=True)
if 'CNN' in all_cam_maps and 'ViT' in all_cam_maps:
    all_cam_maps['Hybrid'] = ((all_cam_maps['CNN'] + all_cam_maps['ViT']) / 2.0).astype(np.float32)
    print(f"  Hybrid GradCAM shape: {all_cam_maps['Hybrid'].shape}", flush=True)
else:
    print("  WARNING: CNN or ViT GradCAM missing -- skipping Hybrid GradCAM.", flush=True)

# add U-Net predictions (already (N, H, W, 1) from step 5)
all_cam_maps['U-Net'] = unet_preds

if not all_cam_maps:
    raise RuntimeError("No maps to evaluate -- check that model checkpoints exist.")


# 5. compute metrics on ICH-positive slices
print(f"\nEvaluating on {len(pos_idx)} ICH-positive slices (threshold={THRESHOLD})...", flush=True)

records = []
for method, maps in all_cam_maps.items():
    row = eval_method(maps, y_test, pos_idx)
    row['method'] = method
    records.append(row)
    print(f"  {method}: Dice {row['dice_mean']:.4f} +/- {row['dice_std']:.4f}", flush=True)

results_df = pd.DataFrame(records).set_index('method')


# 6. print summary table
print("\n" + "=" * 80, flush=True)
print(f"RESULTS  --  ICH-positive slices  (n={len(pos_idx)}, threshold={THRESHOLD})", flush=True)
print("=" * 80, flush=True)
print(f"{'Method':<10} {'Dice':>17} {'IoU':>17} {'Recall':>17} {'Precision':>17} {'PG':>6}", flush=True)
print("-" * 88, flush=True)
for method, row in results_df.iterrows():
    print(
        f"{method:<10} "
        f"{row['dice_mean']:.4f} +/- {row['dice_std']:.4f}  "
        f"{row['iou_mean']:.4f} +/- {row['iou_std']:.4f}  "
        f"{row['recall_mean']:.4f} +/- {row['recall_std']:.4f}  "
        f"{row['precision_mean']:.4f} +/- {row['precision_std']:.4f}  "
        f"PG {row['pointing_game']:.4f}",
        flush=True,
    )
print("=" * 88, flush=True)


# 7. save results CSV + figures
ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

csv_path = OUTPUT_DIR / f'comparison_results_{ts}.csv'
results_df.to_csv(str(csv_path))
print(f"\nSaved results: {csv_path.name}", flush=True)

if MATPLOTLIB:
    import matplotlib.patches as mpatches

    methods  = results_df.index.tolist()
    n_m      = len(methods)
    # pre-binarise once for reuse in figures
    bin_maps = {m: (all_cam_maps[m].astype(np.float32) > THRESHOLD).astype(np.float32)
                for m in methods}

    # ---- bar chart: all 4 metrics ----
    metric_specs = [('dice', 'Dice'), ('iou', 'IoU'),
                    ('recall', 'Recall'), ('precision', 'Precision'),
                    ('pointing_game', 'Pointing Game')]
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    for ax, (metric, label) in zip(axes.ravel(), metric_specs):
        col      = metric if metric == 'pointing_game' else f'{metric}_mean'
        std_col  = f'{metric}_std' if f'{metric}_std' in results_df.columns else None
        means    = results_df[col].values
        stds     = results_df[std_col].values if std_col else None
        title    = label if metric == 'pointing_game' else f'{label}  (mean +/- std)'
        bars     = ax.barh(methods, means, xerr=stds, capsize=4, color='steelblue', height=0.5)
        ax.set_xlim(0, 1.05)
        ax.set_xlabel(label)
        ax.set_title(title)
        ax.invert_yaxis()
        for bar, v in zip(bars, means):
            ax.text(v + 0.01, bar.get_y() + bar.get_height() / 2,
                    f'{v:.3f}', va='center', fontsize=9)
    # hide the unused 6th subplot (2x3 grid, 5 metrics)
    axes.ravel()[-1].set_visible(False)
    plt.suptitle(f'Method comparison  (n={len(pos_idx)} ICH-positive slices, '
                 f'threshold={THRESHOLD})', fontsize=11)
    plt.tight_layout()
    bar_path = OUTPUT_DIR / f'comparison_bar_{ts}.png'
    plt.savefig(str(bar_path), dpi=120, bbox_inches='tight')
    plt.close()
    print(f"Saved bar chart: {bar_path.name}", flush=True)

    # one representative slice per positive patient (largest GT lesion area),
    # up to N_QUAL_IMGS patients -- avoids all samples coming from the same patient
    pos_by_patient = (slice_df[slice_df['has_ich'] == 1]
                      .groupby('patient').groups)  # {patient_id: Index of slice rows}
    patient_keys = list(pos_by_patient.keys())
    np.random.shuffle(patient_keys)
    sample_idx = []
    for pid in patient_keys[:N_QUAL_IMGS]:
        idxs = pos_by_patient[pid].tolist()
        best = max(idxs, key=lambda i: float(y_test[i, :, :, 0].sum()))
        sample_idx.append(best)

    if len(sample_idx) > 0:

        # ---- figure 1: soft maps with per-slice Dice / IoU ----
        n_cols = 2 + n_m
        fig, axes = plt.subplots(N_QUAL_IMGS, n_cols,
                                  figsize=(3.5 * n_cols, 3.5 * N_QUAL_IMGS),
                                  squeeze=False)
        for row_i, sl in enumerate(sample_idx):
            p_id = slice_df.loc[sl, 'patient']
            s_id = slice_df.loc[sl, 'slice']

            axes[row_i, 0].imshow(x_test[sl, :, :, 0], cmap='gray')
            axes[row_i, 0].set_title(f'CT  p{p_id} s{s_id}', fontsize=8)
            axes[row_i, 0].axis('off')

            axes[row_i, 1].imshow(y_test[sl, :, :, 0], cmap='hot', vmin=0, vmax=1)
            axes[row_i, 1].set_title('GT mask', fontsize=8)
            axes[row_i, 1].axis('off')

            for col_i, method in enumerate(methods):
                d, iou, *_ = seg_metrics(y_test[sl], bin_maps[method][sl])
                axes[row_i, 2 + col_i].imshow(
                    all_cam_maps[method][sl, :, :, 0].astype(np.float32),
                    cmap='hot', vmin=0, vmax=1)
                axes[row_i, 2 + col_i].set_title(
                    f'{method}\nDice {d:.3f}  IoU {iou:.3f}', fontsize=8)
                axes[row_i, 2 + col_i].axis('off')

        plt.suptitle('Soft maps  (hot colormap, 0-1)', fontsize=11, y=1.01)
        plt.tight_layout()
        soft_path = OUTPUT_DIR / f'comparison_soft_{ts}.png'
        plt.savefig(str(soft_path), dpi=120, bbox_inches='tight')
        plt.close()
        print(f"Saved soft-map figure: {soft_path.name}", flush=True)

        # ---- figure 2: TP / FP / FN error maps ----
        # col 0: CT with GT boundary contour; cols 1+: error map per method
        n_cols = 1 + n_m
        fig, axes = plt.subplots(N_QUAL_IMGS, n_cols,
                                  figsize=(3.5 * n_cols, 3.5 * N_QUAL_IMGS),
                                  squeeze=False)
        for row_i, sl in enumerate(sample_idx):
            p_id = slice_df.loc[sl, 'patient']
            s_id = slice_df.loc[sl, 'slice']
            gt_bin = y_test[sl, :, :, 0] > 0.5

            axes[row_i, 0].imshow(x_test[sl, :, :, 0], cmap='gray')
            axes[row_i, 0].contour(gt_bin, levels=[0.5], colors='cyan', linewidths=1.0)
            axes[row_i, 0].set_title(f'CT  p{p_id} s{s_id}\n(cyan = GT boundary)', fontsize=8)
            axes[row_i, 0].axis('off')

            for col_i, method in enumerate(methods):
                d, iou, rec, prec = seg_metrics(y_test[sl], bin_maps[method][sl])
                err = error_rgb(y_test[sl], bin_maps[method][sl])
                axes[row_i, 1 + col_i].imshow(err)
                axes[row_i, 1 + col_i].set_title(
                    f'{method}\nDice {d:.3f}  IoU {iou:.3f}\n'
                    f'Rec {rec:.3f}  Prec {prec:.3f}', fontsize=8)
                axes[row_i, 1 + col_i].axis('off')

        legend_patches = [
            mpatches.Patch(color=(0/255, 180/255, 0/255),  label='TP'),
            mpatches.Patch(color=(200/255, 0/255, 0/255),  label='FP'),
            mpatches.Patch(color=(0/255, 80/255, 200/255), label='FN'),
            mpatches.Patch(color=(30/255, 30/255, 30/255), label='TN'),
        ]
        fig.legend(handles=legend_patches, loc='lower center', ncol=4,
                   fontsize=10, frameon=True, bbox_to_anchor=(0.5, -0.02))
        plt.suptitle('Error maps  (TP=green  FP=red  FN=blue  TN=dark)', fontsize=11, y=1.01)
        plt.tight_layout()
        err_path = OUTPUT_DIR / f'comparison_error_{ts}.png'
        plt.savefig(str(err_path), dpi=120, bbox_inches='tight')
        plt.close()
        print(f"Saved error-map figure: {err_path.name}", flush=True)

        # ---- figure 3: consensus / agreement heatmap ----
        # shows how many of the n_m methods predict positive at each pixel.
        # GT boundary overlaid in cyan so misalignment is immediately visible.
        fig, axes = plt.subplots(N_QUAL_IMGS, 3,
                                  figsize=(10, 3.5 * N_QUAL_IMGS),
                                  squeeze=False)
        for row_i, sl in enumerate(sample_idx):
            p_id = slice_df.loc[sl, 'patient']
            s_id = slice_df.loc[sl, 'slice']
            gt_bin = y_test[sl, :, :, 0] > 0.5

            # sum binary predictions across all methods (int 0..n_m per pixel)
            consensus = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)
            for method in methods:
                consensus += bin_maps[method][sl, :, :, 0]

            axes[row_i, 0].imshow(x_test[sl, :, :, 0], cmap='gray')
            axes[row_i, 0].set_title(f'CT  p{p_id} s{s_id}', fontsize=8)
            axes[row_i, 0].axis('off')

            axes[row_i, 1].imshow(y_test[sl, :, :, 0], cmap='hot', vmin=0, vmax=1)
            axes[row_i, 1].set_title('GT mask', fontsize=8)
            axes[row_i, 1].axis('off')

            im = axes[row_i, 2].imshow(consensus, cmap='YlOrRd', vmin=0, vmax=n_m)
            axes[row_i, 2].contour(gt_bin, levels=[0.5], colors='cyan', linewidths=1.2)
            plt.colorbar(im, ax=axes[row_i, 2], fraction=0.04, pad=0.02,
                         label=f'# methods (0-{n_m})')
            axes[row_i, 2].set_title('Consensus  (cyan = GT boundary)', fontsize=8)
            axes[row_i, 2].axis('off')

        plt.suptitle('Method consensus  (how many methods agree per pixel)', fontsize=11, y=1.01)
        plt.tight_layout()
        con_path = OUTPUT_DIR / f'comparison_consensus_{ts}.png'
        plt.savefig(str(con_path), dpi=120, bbox_inches='tight')
        plt.close()
        print(f"Saved consensus figure: {con_path.name}", flush=True)


print("\nDone!")
