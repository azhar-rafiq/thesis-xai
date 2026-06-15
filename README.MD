# thesis-xai -- ICH detection with XAI

Azhar | MSc Health Data Science, UoB

---

## what am I doing

| RQ | short version | approach | status |
|----|--------------|----------|--------|
| RQ1 | classify ICH subtypes from CT slices | CNN, ViT, Hybrid ensemble | done -- CNN 0.9243, ViT 0.9473, Hybrid 0.9513 |
| RQ2 | segment haemorrhage at pixel level | supervised U-Net on PhysioNet pixel masks | done -- Dice 0.21 (baseline, frozen encoder) |
| RQ3 | do heatmaps actually point at the bleed? | GradCAM from RQ1 models, evaluated on PhysioNet | done |
| RQ4 | how does XAI compare to supervised segmentation? | IoU / Dice: GradCAM vs U-Net vs PhysioNet masks | done -- `6_compare_physionet.py` written and run |

---

## data

| dataset | format | size | labels | used for |
|---------|--------|------|--------|----------|
| RSNA 2019 | DICOM | 752k slices, 18,938 patients (raw) | 5 subtypes (slice-level only) | RQ1 training |
| PhysioNet CT-ICH v1.3.1 | NIfTI | 76 patients, ~2,800 slices | pixel masks (binary) | RQ2 training + RQ3/RQ4 eval |

PhysioNet path: `/rds/projects/k/karwatha-karwath-hds-pg-research/axr1222/data/physionet.org/files/ct-ich/1.3.1/`
PhysioNet masks are binary (0/1, no per-subtype distinction in mask). Per-subtype labels available in `hemorrhage_diagnosis_raw_ct.csv`.

**Preprocessed RSNA cache** (after cleaning multi-study conflicts and corrupt DICOMs):

| split | patients | slices | % of total |
|-------|----------|--------|------------|
| train | 12,784 | 475,870 | 70% |
| val | 2,731 | 103,101 | 15% |
| test | 2,738 | 101,823 | 15% |
| **total** | **18,253** | **680,794** | |

Split is patient-level (GroupShuffleSplit, SEED=20260605) -- no data leakage.

**Preprocessing pipeline** (Burduja et al. 2020 windowing):
1. Raw DICOM pixels -> Hounsfield Units (HU) via RescaleSlope + RescaleIntercept
2. Apply 3 CT windows stacked as RGB channels:
   - Brain window: WC=40, WW=80
   - Subdural window: WC=80, WW=200
   - Soft-tissue window: WC=40, WW=380
3. Normalise each channel to [0, 1]
4. Resize to 256x256
5. Save as float32 .npy (x_train: 349 GB, x_val: 76 GB, x_test: 75 GB)

Cache location: `/rds/projects/k/karwatha-karwath-hds-pg-research/axr1222/data/preprocessed/`

Same 3-window preprocessing applied to PhysioNet NIfTIs (loaded via nibabel).

---

## pipeline

```
0_preprocess_rsna.py       DICOM -> HU -> 3-window stack -> memmap (.npy)
1_train_cnn.py             CNN grid search + full retrain on RSNA              (RQ1)
2_train_vit.py             ViT grid search + full retrain on RSNA              (RQ1)
3_train_hybrid.py          ensemble: average CNN + ViT outputs (no retraining) (RQ1)
4_segmentation_xai.py      GradCAM maps from all 3 models + RSNA weakly-sup U-Net  (RQ3)
5_train_physionet_seg.py   supervised U-Net on PhysioNet pixel masks             (RQ2)
6_compare_physionet.py     GradCAM vs U-Net vs GT masks: Dice/IoU/recall/PG      (RQ4)
```

**Model naming:** `rsna_{method}_{date}-{XX}_{cv_auc}_auc.keras` where `{XX}` is a random 2-letter tag to prevent collisions. e.g. `rsna_vit_20260608-ZU_08740_auc.keras`

---

## model design

### CNN (`1_train_cnn.py`)
- 3-layer Conv2D (filters: 16, 32/64 grid, 128) + BatchNorm + MaxPool + Dropout
- Augmentation baked into model: RandomFlip(h), RandomRotation(±5°), RandomZoom(±8%), RandomTranslation(±8%), `fill_mode='constant'` (CT-appropriate)
- Loss: **BCE** -- focal loss (gamma=1.0 and 2.0) was tested; both gave lower AUC than BCE
- Optimizer: Adam + ReduceLROnPlateau (grid search and retrain)
- Grid: filters2=[32,64], lr=[0.001, 0.0001]
- Best model: `rsna_cnn_20260605_05000_auc.keras` (AUC 0.9243)

### ViT (`2_train_vit.py`)
- Patch embedding (Conv2D stride=patch_size) -> PositionEmbedding -> N transformer blocks -> GAP -> Dense(5, sigmoid)
- Augmentation in dataset pipeline (training only): RandomFlip(h), RandomRotation(±5°), RandomZoom(±8%), RandomTranslation(±8%)
- Loss: **Focal loss** (`gamma=2.0`) -- AUC 0.8985 -> 0.9470 vs BCE
- Optimizer (grid search): Adam + ReduceLROnPlateau
- Optimizer (retrain): **AdamW** + **cosine decay with 10% linear warmup** -- standard for ViTs
- Grid: embed_dim=[64,128], mlp_dim=[128,256], patch_size=16, num_blocks=4
- XLA note: `TF_XLA_FLAGS="--tf_xla_auto_jit=0"` required in sbatch -- XLA JIT deadlocks on the validation pass for transformer models with large val sets
- Best model: `rsna_vit_20260608-ZU_08740_auc.keras` (AUC 0.9473)

### Hybrid (`3_train_hybrid.py`)
- Simple average of CNN and ViT softmax outputs -- no retraining
- CNN component uses BCE loss; ViT component uses focal loss
- Models hardcoded by filename to prevent picking up failed/nan runs
- Best model: `rsna_hybrid_20260608-AR_09512_auc.keras` (AUC 0.9513)

---

## segmentation map files (step 4 + step 5 outputs)

All files saved to `/rds/projects/k/karwatha-karwath-hds-pg-research/axr1222/models/`.
Timestamp in filename = when the Python script started (not when the SLURM job was submitted).

**GradCAM + RSNA U-Net maps:**

| file | generated from |
|------|---------------|
| `cnn_gradcam_maps_{ts}.npy` | `rsna_cnn_20260605_05000_auc.keras` |
| `vit_gradcam_maps_{ts}.npy` | `rsna_vit_20260608-ZU_08740_auc.keras` |
| `hybrid_gradcam_maps_{ts}.npy` | avg of CNN + ViT GradCAM |
| `unet_seg_maps_{ts}.npy` | frozen CNN encoder, RSNA labels |
| `seg_maps_test_idx_{ts}.npy` | indices of 2000 RSNA test slices |

**PhysioNet U-Net maps** (timestamp `20260607_082612`, written Jun 7 2026):

| file | generated from |
|------|---------------|
| `physionet_seg_maps_20260607_082612.npy` | frozen encoder from `rsna_cnn_20260605_05000_auc.keras`, decoder trained on PhysioNet pixel masks |
| `physionet_test_patients_20260607_082612.npy` | patient IDs for the 15 PhysioNet test patients |

## step 5 design notes

- encoder initialised from RSNA CNN checkpoint (`rsna_cnn_20260605_05000_auc.keras`) -- transfer learning, helps with small PhysioNet dataset (76 patients)
- patient-level 80/20 train/test split (76 total -> ~61 train / 15 test)
- loss: BCE + Dice (handles sparse positive pixels better than BCE alone)
- augmentation: horizontal + vertical flip
- encoder frozen during training (only decoder learns)
- test metrics: Dice 0.21, IoU ~0.18 (expected given 61 training patients and frozen encoder -- baseline for RQ4 comparison)
- outputs: `physionet_seg_maps_*.npy` (float16) + test patient list for step 6

---

## step 6 design

For each ICH-positive PhysioNet test slice:
- re-generate GradCAM from CNN, ViT, Hybrid on PhysioNet slices directly (step 4 maps are on RSNA test data -- not reused here)
- load U-Net prediction from step 5 (PhysioNet pixel-supervised)
- compare all 4 methods against PhysioNet ground-truth binary mask

Metrics per method: **Dice**, **IoU**, **pixel recall**, **pixel precision**, **Pointing Game** (Selvaraju et al. 2020, doi:10.1007/s11263-019-01228-7 -- whether peak activation pixel falls inside GT mask; threshold-free)

Qualitative samples: one representative slice per positive patient (largest GT lesion area), up to 5 patients.

Outputs: `comparison_results_{ts}.csv` + bar chart, soft maps, TP/FP/FN error maps, consensus heatmap

---

## status

**all RQs done**
- HPC setup, GitHub, venv
- DICOM index (873,945 files -> `rsna_dicom_index_2.csv`)
- EDA on both datasets
- Full RSNA preprocessing (18,253 patients, SEED=20260605, 3-channel CT windowing)
- CNN training -- BCE, seed-locked, full dataset (job 44259522, AUC 0.9243)
- ViT training -- focal loss (gamma=2.0) + AdamW + cosine warmup (retrained job 44366721, AUC 0.9473)
- Hybrid ensemble (AUC 0.9513) -- CNN (BCE) + ViT (focal), retrained job 44424197
- RSNA weakly-supervised U-Net (AUC 0.8783) -- part of step 4
- PhysioNet pixel-supervised U-Net (Dice 0.21) -- step 5
- GradCAM maps generated (step 4)
- `6_compare_physionet.py` written and run

**next**
- rerun `6_compare_physionet.py`
- try SHAP (`GradientExplainer`) as additional XAI method
- check Seg-CQ500 (51 scans from CQ500 with pixel masks -- RQ3/RQ4 external validation)
- try LIME
- 

---

## results

Runs 0-4 used inconsistent seeds or failed -- splits not reproducible, results are preliminary only.

| # | date | method | loss | patients | test AUC (weighted) | cv AUC | duration | notes |
|---|------|--------|------|----------|--------------------:|-------:|---------:|-------|
| 0 | 2026-05-19 | CNN | BCE | 800 | 0.7814 | 0.6465 | 40 min | preliminary |
| 1 | 2026-05-20 | CNN | BCE | 800 | 0.7886 | 0.6302 | 90 min | preliminary |
| 2 | 2026-06-04 | CNN | BCE | 12,784 | 0.9265 | -- | 133 min | preliminary, gs skipped |
| 3 | 2026-06-05 | ViT | BCE | 12,784 | 0.9062 | 0.8107 | 82 min | preliminary |
| 4 | 2026-06-05 | CNN | BCE | 12,784 | 0.5000 | -- | 47 min | **failed** -- model did not train |
| 5 | 2026-06-05 | ViT | BCE | 12,784 | 0.8985 | 0.8034 | 147 min | job 44265323, seed-locked |
| 6 | 2026-06-05 | CNN | BCE | 12,784 | 0.9243 | 0.5* | 333 min | job 44259522, seed-locked; *cv_auc unreliable (make_scorer bug) |
| 7 | 2026-06-05 | Hybrid | BCE | 12,784 | 0.9325 | -- | -- | ensemble of runs 5+6 |
| 8 | 2026-06-06 | CNN | focal γ=2.0 | 12,784 | 0.8774 | 0.8386 | 160 min | lower than BCE |
| 9 | 2026-06-06 | CNN | focal γ=1.0 | 12,784 | 0.8950 | 0.8369 | 230 min | lower than BCE |
| 10 | 2026-06-07 | ViT | focal γ=2.0 | 12,784 | 0.9470 | 0.8727 | 467 min | job 44333718, ZU model -- **corrupted** |
| 11 | 2026-06-07 | U-Net (RSNA) | BCE | 12,784 | 0.8783 | -- | 552 min | weakly supervised; part of step 4 |
| 12 | 2026-06-07 | Hybrid | focal (ViT) | 12,784 | 0.9513 | -- | -- | job 44361117 -- **corrupted** |
| 13 | 2026-06-08 | ViT | focal γ=2.0 | 12,784 | **0.9473** | 0.874 | 672 min | job 44366721, retrain of run 10; `rsna_vit_20260608-ZU_08740_auc.keras` |
| 14 | 2026-06-08 | Hybrid | focal (ViT) | 12,784 | **0.9513** | -- | -- | job 44424197, retrain of run 12; `rsna_hybrid_20260608-AR_09512_auc.keras` |

**Per-class AUCs (canonical models: run 13 ViT, run 14 Hybrid):**

| class | CNN (run 6) | ViT (run 13) | Hybrid (run 14) |
|-------|------------|-------------|----------------|
| epidural | 0.8907 | 0.9343 | 0.9296 |
| intraparenchymal | 0.9422 | 0.9603 | 0.9661 |
| intraventricular | 0.9627 | 0.9733 | 0.9775 |
| subarachnoid | 0.9033 | 0.9323 | 0.9406 |
| subdural | 0.9064 | 0.9345 | 0.9345 |
| **weighted avg** | **0.9243** | **0.9473** | **0.9513** |

Models saved to: `/rds/projects/k/karwatha-karwath-hds-pg-research/axr1222/models/`

---

## scripts (`~/thesis-xai/`)

| file | what it does |
|------|-------------|
| `0_preprocess_rsna.py` | DICOM -> 3-channel float32 .npy cache (mmap-safe, O(1) RAM) |
| `0_preprocess_rsna.sh` | sbatch for above (CPU, bbdefault, 64G, 24h) |
| `1_train_cnn.py` | CNN grid search + full retrain |
| `1_train_cnn.sh` | sbatch for above (GPU, A100, 48h, 490G) |
| `2_train_vit.py` | ViT grid search + full retrain |
| `2_train_vit.sh` | sbatch for above (GPU, A100, 48h, 490G); sets `TF_XLA_FLAGS="--tf_xla_auto_jit=0"` |
| `3_train_hybrid.py` | ensemble CNN + ViT (inference only, no training) |
| `3_train_hybrid.sh` | sbatch for above (GPU, A100, 4h, 64G) |
| `4_segmentation_xai.py` | GradCAM maps (CNN/ViT/Hybrid) + weakly-supervised RSNA U-Net |
| `4_segmentation_xai.sh` | sbatch for above (GPU, A100, 48h, 490G) |
| `5_train_physionet_seg.py` | pixel-supervised U-Net on PhysioNet masks (transfer from CNN encoder) |
| `5_train_physionet_seg.sh` | sbatch for above (GPU, A100, 4h, 64G) |
| `6_compare_physionet.py` | re-generate GradCAM on PhysioNet test slices, compare vs U-Net + GT masks; Dice/IoU/recall/precision/Pointing Game; outputs 4 PNGs + CSV to `MODEL_EXPORT_DIR` |
| `training_logger.py` | saves model + appends row to `training_log.csv`; `model_status` column tracks ok/corrupted/deleted/retraining |
| `training_log.csv` | all run results (timestamp, method, loss, AUCs, hyperparams, duration, model_status) |
| `check_log.py` | parse a training log: status, val_auc per epoch, speed, ETA |
| `thesis.ipynb` | main notebook -- EDA, eval, plots |

---

## OOM notes (important for HPC submissions)

The full RSNA dataset is 349 GB (x_train) + 76 GB (x_val) + 75 GB (x_test) as float32.
Several failure modes were discovered and fixed:

| problem | cause | fix |
|---------|-------|-----|
| OOM during grid search | TF held fold references; accumulated ~8 GB per fold -> ~40 GB after 5 folds | `DataStreamKerasClassifier` with generator + `clear_session()` in `getModel` |
| OOM during final retrain | `model.fit(numpy)` calls `tf.constant()` on full 349 GB array | Direct `model.fit(tf.data.Dataset)` with sequential generator, bypassing scikeras |
| 2.5h/epoch on NFS | random mmap seeks across 349 GB for each shuffled batch | Sequential reads + TF shuffle buffer (10,000 samples = 7.5 GB) -> ~7 min/epoch |
| CV AUC always 0.5 | `make_scorer(fn, needs_proba=True)` mangles multilabel `predict_proba` output in sklearn 1.8+ | Replace with raw callable scorer: `scoring=fn(estimator, X, y)` that calls `estimator.predict_proba(X)` directly |
| Node memory cap | `--mem=512G` fails; GPU nodes max at 490G | Use `--mem=490G` |
| XLA JIT deadlock (ViT) | `dot_merger.cc` hangs during first validation pass; XLA compiles too many variants | `export TF_XLA_FLAGS="--tf_xla_auto_jit=0"` in sbatch |
| Sequential model `.input` error | Keras 3 Sequential models loaded via `load_model` have no symbolic `.input`/`.output` | Functional tracing: loop `cnn_base.layers`, build new graph with `tf.keras.Input` |
| Keras 3 layer name collision | `load_model` doesn't update global name counter; new decoder layers get names colliding with CNN | Explicit `name='dec_conv1'`, `name='dec_bn1'`, etc. on all decoder layers |

PhysioNet dataset is ~2 GB total (fits in RAM, no memmap needed).

---

## rules

- always split by `PatientID` -- prevent data leakage
- drop `any` label before training -- it's derived, causes leakage
- HU conversion before anything else -- raw pixels are scanner-dependent
- `set_memory_growth(gpu, True)` always -- otherwise TF grabs all VRAM
- sbatch for training, Jupyter for eval -- kernel dies in Jupyter
- always use `DataStreamKerasClassifier` for large datasets -- avoids TF materialising 349 GB as a constant
- always add `ModelCheckpoint` to long jobs -- SLURM will kill at walltime with no warning
- all experiments use SEED=20260605 -- same seed everywhere for reproducible splits
- augmentation only during training -- `augment=False` for val/test in `_make_dataset`
- hardcode model paths/filename in `3_train_hybrid.py`, `4_segmentation_xai.py`, `6_compare_physionet.py`

---

## things to watch

- `InstanceNumber` all NaN in DICOM index
- epidural AUC is lowest due to extreme rarity (0.27% prevalence)
- no learning curves saved yet -- add `history_` extraction to eval notebook
- PhysioNet CSV column order differs from RSNA: Intraventricular, Intraparenchymal, Subarachnoid, Epidural, Subdural (capitalised)

---

*updated: 2026-06-15*
