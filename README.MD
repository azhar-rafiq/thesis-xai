# thesis-xai -- ICH detection with XAI

Azhar | MSc Health Data Science, UoB

---

## what am I doing

| RQ | short version | approach | status |
|----|--------------|----------|--------|
| RQ1 | classify ICH subtypes from CT slices | CNN -> Transformer | 🟢 CNN done, ViT next |
| RQ2 | segment haemorrhage at pixel level | U-Net | 🔴 not started |
| RQ3 | do heatmaps actually point at the bleed? | Grad-CAM, SHAP | 🔴 not started |
| RQ4 | how do you measure if XAI is any good? | IoU vs PhysioNet masks | 🔴 not started |

---

## data

| dataset | format | size | labels | used for |
|---------|--------|------|--------|----------|
| RSNA 2019 | DICOM | 752k slices, 18,938 patients (raw) | 5 subtypes (slice-level) | classification |
| PhysioNet CT-ICH | NIfTI | 75 patients, 2,814 slices | pixel masks | segmentation + XAI eval |

PhysioNet untouched so far -- all current work is RSNA.

**Preprocessed RSNA cache** (after cleaning multi-study conflicts and corrupt DICOMs):

| split | patients | slices | % of total |
|-------|----------|--------|------------|
| train | 12,784 | 475,870 | 70% |
| val | 2,731 | 103,101 | 15% |
| test | 2,738 | 101,823 | 15% |
| **total** | **18,253** | **680,794** | |

Split is patient-level (GroupShuffleSplit) -- no data leakage.

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

---

## status

**done**
- HPC setup, GitHub, venv
- DICOM index (873,945 files -> `rsna_dicom_index_2.csv`)
- EDA on both datasets
- Full RSNA preprocessing (18,253 patients, 3-channel CT windowing)
- CNN baseline (800 patients) -- test AUC 0.7814
- **CNN full dataset (12,784 patients) -- test AUC 0.9265** (weighted avg)

**next**
- ViT grid search: `sbatch 2_train_transformer.sh`
- Compare CNN vs ViT AUC

**not started**
- PhysioNet NIfTI pipeline
- U-Net (RQ2)
- Grad-CAM / SHAP (RQ3)
- XAI evaluation metrics: IoU, pointing game, mass accuracy (RQ4)
- Learning curves (extract from training history)

---

## results

| # | model | patients | cv auc | test auc (weighted) | notes |
|---|-------|----------|--------|----------------------|-------|
| 1 | CNN 3-layer | 800 | 0.6465 | 0.7814 | filters=128, lr=1e-4, bs=32, drop=0.25 -- 40 min |
| 2 | CNN 3-layer | 800 | 0.6302 | 0.7886 | same filters, bs=64, epochs=30 -- 90 min |
| 3 | **CNN 3-layer** | **12,784** | n/a* | **0.9265** | filters=[16,32,128], lr=0.001, bs=64, drop=0.25, epochs=14 |
| 4 | ViT | 12,784 | -- | -- | patch=16/32, blocks=2/4, embed=64 -- not run yet |

*CV scorer was broken in earlier runs (needs_proba bug) -- CV score for run 3 is not valid. Best params were selected from a prior run and hardcoded for the full retrain.

**Run 3 per-class AUCs:**

| class | AUC | prevalence |
|-------|-----|-----------|
| epidural | 0.8597 | 0.27% (rarest) |
| intraparenchymal | 0.9442 | 4.3% |
| intraventricular | 0.9647 | 3.2% |
| subarachnoid | 0.9112 | 4.2% |
| subdural | 0.9067 | 5.8% |
| **weighted avg** | **0.9265** | |

Models saved to: `/rds/projects/k/karwatha-karwath-hds-pg-research/axr1222/models/`

---

## scripts (`~/thesis-xai/`)

| file | what it does |
|------|-------------|
| `0_preprocess_rsna.py` | DICOM -> 3-channel float32 .npy cache (mmap-safe, O(1) RAM) |
| `0_preprocess_rsna.sh` | sbatch for above (CPU, bbdefault, 64G, 6h) |
| `1_train_mult_rsna.py` | CNN grid search + full retrain |
| `1_train_mult_rsna.sh` | sbatch for above (GPU, A100, 48h, 490G) |
| `2_train_transformer.py` | ViT grid search + full retrain |
| `2_train_transformer.sh` | sbatch for above (GPU, A100, 48h, 490G) |
| `training_logger.py` | saves model + appends row to training_log.csv |
| `training_log.csv` | all run results |
| `check_log.py` | parse a training log: status, val_auc per epoch, speed, ETA |
| `thesis.ipynb` | main notebook -- EDA, eval, plots |

---

## OOM notes (important for HPC submissions)

The full dataset is 349 GB (x_train) + 76 GB (x_val) + 75 GB (x_test) as float32.
Several failure modes were discovered and fixed:

| problem | cause | fix |
|---------|-------|-----|
| OOM during grid search | sklearn copies fold data; TF held references, accumulating ~28 GB per fold | `DataStreamKerasClassifier` with generator + `clear_session()` in `getModel` |
| OOM during final retrain | `model.fit(numpy)` calls `tf.constant()` on full 349 GB array | Direct `model.fit(tf.data.Dataset)` with sequential generator, bypassing scikeras |
| 2.5h/epoch on NFS | random mmap seeks across 349 GB for each shuffled batch | Sequential reads + TF shuffle buffer (10,000 samples = 7.5 GB) -> ~7 min/epoch |
| CV AUC always 0.5 | `make_scorer` calls `predict()` (binary) not `predict_proba()` | `make_scorer(fn, needs_proba=True)` |
| Node memory cap | `--mem=512G` fails; GPU nodes max at 490G | Use `--mem=490G` |

---

## rules (don't break these)

- always split by `PatientID` -- slice-level split = data leakage
- drop `any` label before training -- it's derived, causes leakage
- HU conversion before anything else -- raw pixels are scanner-dependent
- `set_memory_growth(gpu, True)` always -- otherwise TF grabs all VRAM
- sbatch for training, Jupyter for eval -- kernel dies otherwise
- always use `DataStreamKerasClassifier` for large datasets -- avoids TF materialising 349 GB as a constant
- always add `ModelCheckpoint` to long jobs -- SLURM will kill at walltime with no warning

---

## things to watch

- old 800/200 cache has 6 columns (includes `any` label from notebook bug) -- incompatible with new cache, don't extend
- `InstanceNumber` all NaN in DICOM index -- mention in thesis methodology
- epidural AUC (0.86) is lowest due to extreme rarity (0.27% prevalence) -- expected, discuss in thesis
- CV scorer had `needs_proba` bug in all runs before run 3 -- CV scores in runs 1-2 are from `predict()` not `predict_proba()`, making them unreliable for param selection
- no learning curves saved yet -- add `history_` extraction to eval notebook

---

## next steps

1. `sbatch 2_train_transformer.sh` -- run ViT grid search
2. Update `SKIP_GRID_SEARCH` params in `2_train_transformer.py` with best ViT params, then retrain on full data
3. Compare CNN vs ViT AUC (expect ViT lower on this dataset size -- that's the thesis point)
4. Grad-CAM on best CNN + best ViT
5. PhysioNet NIfTI preprocessing (`nibabel`)
6. U-Net (RQ2)
7. XAI evaluation against PhysioNet masks

---

*updated: 2026-06-05*
