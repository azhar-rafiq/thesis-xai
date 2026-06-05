# thesis-xai -- ICH detection with XAI

Azhar | MSc Health Data Science, UoB

---

## what am I doing

| RQ | short version | approach | status |
|----|--------------|----------|--------|
| RQ1 | classify ICH subtypes from CT slices | CNN, ViT, Hybrid ensemble | 🟡 models done, XAI pending |
| RQ2 | segment haemorrhage at pixel level | supervised U-Net on PhysioNet pixel masks | 🔴 not started |
| RQ3 | do heatmaps actually point at the bleed? | GradCAM from RQ1 models, evaluated on PhysioNet | 🔴 not started |
| RQ4 | how does XAI compare to supervised segmentation? | IoU / Dice: GradCAM vs U-Net vs PhysioNet masks | 🔴 not started |

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
4_segmentation_xai.py*      GradCAM maps from all 3 models on RSNA test set     (RQ3)
5_train_physionet_seg.py*   supervised U-Net on PhysioNet pixel masks           (RQ2)
6_compare_physionet.py*     GradCAM vs U-Net vs ground truth: IoU/Dice/recall   (RQ4)
```

**Model naming:** `rsna_{method}_{date}-{XX}_{cv_auc}_auc.keras` where `{XX}` is a random 2-letter tag to prevent collisions. e.g. `rsna_vit_20260605-AB_08034_auc.keras`

---

## model design

### CNN (`1_train_cnn.py`)
- 3-layer Conv2D (filters: 16, 32/64 grid, 128) + BatchNorm + MaxPool + Dropout
- Augmentation baked into model: RandomFlip(h), RandomRotation(±5°), RandomZoom(±8%), RandomTranslation(±8%), `fill_mode='constant'` (CT-appropriate)
- Loss: **Focal loss** (`gamma=2.0`) -- replaced BCE to handle class imbalance without sample weights
- Optimizer: Adam + ReduceLROnPlateau (grid search) / Adam + ReduceLROnPlateau (retrain)
- Grid: filters2=[32,64], lr=[0.001, 0.0001]

### ViT (`2_train_vit.py`)
- Patch embedding (Conv2D stride=patch_size) -> PositionEmbedding -> N transformer blocks -> GAP -> Dense(5, sigmoid)
- Augmentation in dataset pipeline (training only): RandomFlip(h), RandomRotation(±5°), RandomZoom(±8%), RandomTranslation(±8%)
- Loss: **Focal loss** (`gamma=2.0`)
- Optimizer (grid search): Adam + ReduceLROnPlateau
- Optimizer (retrain): **AdamW** + **cosine decay with 10% linear warmup** -- standard for ViTs, prevents patch embedding destruction early in training
- Grid: embed_dim=[64,128], mlp_dim=[128,256], patch_size=16, num_blocks=4

### Hybrid (`3_train_hybrid.py`)
- Simple average of CNN and ViT softmax outputs -- no retraining
- Models hardcoded by filename to prevent picking up failed/nan runs

---

## step 5 design notes

- encoder initialised from RSNA CNN checkpoint (transfer learning -- helps with small PhysioNet dataset)
- patient-level 80/20 train/test split (76 total -> ~61 train / 15 test)
- loss: BCE + Dice (handles sparse positive pixels better than BCE alone)
- augmentation: horizontal + vertical flip
- encoder frozen during training (only decoder learns)
- outputs: `physionet_seg_maps_*.npy` (float16) + test patient list for step 6

---

## step 6 design (planned)

For each test patient slice with ICH:
- generate GradCAM from CNN, ViT, Hybrid (all trained on RSNA)
- get U-Net prediction (trained on PhysioNet)
- compare all 4 against PhysioNet pixel mask

Metrics per method: **Dice**, **IoU**, **pixel recall**, **pixel precision**

---

## status

**done**
- HPC setup, GitHub, venv
- DICOM index (873,945 files -> `rsna_dicom_index_2.csv`)
- EDA on both datasets
- Full RSNA preprocessing (18,253 patients, SEED=20260605, 3-channel CT windowing)
- CNN training -- seed-locked, full dataset (job 44259522, AUC 0.9243)
- ViT training -- seed-locked, full dataset (job 44265323, AUC 0.8985)
- Hybrid ensemble (AUC 0.9325)
- Model improvements: focal loss, CT-appropriate augmentation, AdamW+warmup for ViT retrain

**queued / next**
- ViT retrain with focal loss + augmentation + larger dims (embed_dim=128, mlp_dim=256)
- GradCAM map generation (`4_segmentation_xai.py`)

**not started**
- PhysioNet supervised U-Net (`5_train_physionet_seg.py`)
- Comparison script (`6_compare_physionet.py`)

---

## results

Runs 1-4 used inconsistent seeds -- splits not reproducible, results are preliminary only.
Run 5 (CNN) was a failed run (AUC=0.5, model did not train). Runs 6-8 are seed-locked on full dataset.

| # | date | method | loss | patients | seed | test AUC (weighted) | cv AUC | duration | notes |
|---|------|--------|------|----------|------|--------------------:|-------:|---------:|-------|
| 1 | 2026-05-19 | CNN | BCE | 800 | no | 0.7814 | 0.6465 | 40 min | preliminary |
| 2 | 2026-05-20 | CNN | BCE | 800 | no | 0.7886 | 0.6302 | 90 min | preliminary |
| 3 | 2026-06-04 | CNN | BCE | 12,784 | no | 0.9265 | — | 133 min | preliminary, gs skipped |
| 4 | 2026-06-05 | ViT | BCE | 12,784 | no | 0.9062 | 0.8107 | 82 min | preliminary |
| 5 | 2026-06-05 | CNN | BCE | 12,784 | yes | 0.5000 | — | 47 min | **failed** -- model did not train |
| 6 | 2026-06-05 | ViT | BCE | 12,784 | yes | 0.8985 | 0.8034 | 147 min | job 44265323 |
| 7 | 2026-06-05 | CNN | BCE | 12,784 | yes | 0.9243 | 0.5* | 333 min | job 44259522; *cv_auc unreliable (make_scorer bug) |
| 8 | 2026-06-05 | Hybrid | BCE | 12,784 | yes | **0.9325** | — | — | ensemble of runs 6+7 |

*Note: runs 6-8 used BCE. All subsequent runs use focal loss.*

**Run 7 per-class AUCs (CNN, seed-locked):**

| class | AUC | prevalence |
|-------|-----|-----------|
| epidural | 0.8907 | 0.27% (rarest) |
| intraparenchymal | 0.9422 | 4.3% |
| intraventricular | 0.9627 | 3.2% |
| subarachnoid | 0.9033 | 4.2% |
| subdural | 0.9064 | 5.8% |
| **weighted avg** | **0.9243** | |

**Run 8 per-class AUCs (Hybrid, seed-locked):**

| class | CNN | ViT | Hybrid |
|-------|-----|-----|--------|
| epidural | 0.8907 | 0.8564 | — |
| intraparenchymal | 0.9422 | 0.9293 | — |
| intraventricular | 0.9627 | 0.9495 | — |
| subarachnoid | 0.9033 | 0.8698 | — |
| subdural | 0.9064 | 0.8697 | — |
| **weighted avg** | **0.9243** | **0.8985** | **0.9325** |

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
| `2_train_vit.sh` | sbatch for above (GPU, A100, 48h, 490G) |
| `3_train_hybrid.py` | ensemble CNN + ViT (inference only, no training) |
| `3_train_hybrid.sh` | sbatch for above (GPU, A100, 4h, 64G) |
| `training_logger.py` | saves model + appends row to training_log.csv |
| `training_log.csv` | all run results (timestamp, method, loss, AUCs, hyperparams, duration) |
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
- hardcode model paths/filename in `3_train_hybrid.py`

---

## things to watch

- `InstanceNumber` all NaN in DICOM index
- epidural AUC is lowest due to extreme rarity (0.27% prevalence)
- no learning curves saved yet -- add `history_` extraction to eval notebook
- PhysioNet CSV column order differs from RSNA: Intraventricular, Intraparenchymal, Subarachnoid, Epidural, Subdural (capitalised)
- focal loss (`gamma=2.0`) replaces BCE + sample_weight

---

## next steps

1. `sbatch 2_train_vit.sh` -- ViT with focal loss + augmentation + embed_dim=128, mlp_dim=256
2. `sbatch 3_train_hybrid.sh` -- new hybrid using updated CNN + ViT models
5. Write + run `4_segmentation_xai.sh`,`5_train_physionet_seg.sh`,`6_compare_physionet.py`

---

*updated: 2026-06-05*
