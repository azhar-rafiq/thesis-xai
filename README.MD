# thesis-xai -- ICH detection with XAI

Azhar | MSc Health Data Science, UoB

---

## what am I doing

| RQ | short version | approach | status |
|----|--------------|----------|--------|
| RQ1 | classify ICH subtypes from CT slices | CNN, ViT, Hybrid ensemble | done: CNN 0.9467, ViT 0.9407, Hybrid 0.9541 (weighted AUC, RSNA test) |
| RQ2 | segment haemorrhage at pixel level | supervised U-Net on PhysioNet and Seg-CQ500 pixel masks | done: Dice 0.4009 (PhysioNet), 0.4372 (Seg-CQ500), all test slices |
| RQ3 | do heatmaps actually point at the bleed? | GradCAM from RQ1 models vs pixel masks, against a random-map chance baseline | done |
| RQ4 | how does XAI compare to supervised segmentation? | Dice / IoU / recall / precision / pointing game / AP, plus Holm-corrected significance tests | done: `7_compare_all.py` |
| RQ5 | how well trained models generalise to external dataset? | RSNA-trained classifiers scored zero-shot on PhysioNet and Seg-CQ500 as slice-level "any ICH" | done: best any-ICH AUC 0.8098 (PhysioNet) / 0.7906 (CQ500) |

---

## data

| dataset | format | size | labels | used for |
|---------|--------|------|--------|----------|
| RSNA 2019 | DICOM | 752k slices, 18,938 patients (raw) | 5 subtypes (slice-level only) | RQ1 training, internal test set |
| PhysioNet CT-ICH v1.3.1 | NIfTI | 75 patients (36 with ICH), 2,814 slices | pixel masks (binary) | RQ2 training + RQ3/RQ4/RQ5 eval |
| Seg-CQ500 | NIfTI | 51 patients (all ICH-positive) | pixel masks (binary) | RQ2 training + RQ3/RQ4/RQ5 eval |

PhysioNet path: `/rds/projects/k/karwatha-karwath-hds-pg-research/axr1222/data/physionet.org/files/ct-ich/1.3.1/`
PhysioNet masks are binary (0/1, no per-subtype distinction in mask). Per-subtype labels available in `hemorrhage_diagnosis_raw_ct.csv`.

Seg-CQ500 path: `/rds/projects/k/karwatha-karwath-hds-pg-research/axr1222/data/seg-cq500/Seg-CQ500/data/volumes/`
One directory per patient, each with `CT.nii` and `ICH_mask.nii.gz`. Downloaded by `download_segcq500.py`.

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

Same 3-window preprocessing applied to the PhysioNet and Seg-CQ500 NIfTIs (loaded via nibabel), so a slice from any of the three datasets reaches a model in the same format.

---

## pipeline

```
0_preprocess_rsna.py       DICOM -> HU -> 3-window stack -> memmap (.npy)
1_train_cnn.py             CNN grid search + full retrain on RSNA              (RQ1)
2_train_vit.py             ViT grid search + full retrain on RSNA              (RQ1)
3_train_hybrid.py          ensemble: average CNN + ViT outputs (no retraining) (RQ1)
4_segmentation_xai.py      GradCAM maps from all 3 models + RSNA weakly-sup U-Net  (RQ3)
5_train_physionet_seg.py   supervised U-Net on PhysioNet pixel masks             (RQ2)
6_train_cq500.py           supervised U-Net on Seg-CQ500 pixel masks            (RQ2)
7_compare_all.py           cross-dataset comparison, writes the all_raw bundle   (RQ3/RQ4/RQ5)
thesis.ipynb               inferential statistics from that bundle, CPU only    (RQ3/RQ4/RQ5)
stats_utils.py             DeLong, Wilcoxon, McNemar, chance reference, bootstrap, figures
```

`6_compare_physionet.py` is the earlier PhysioNet-only version of step 7, kept for reference. Step 7 replaces it: same metrics, all three datasets, plus the classification arm and the significance tests.

**Model naming:** `rsna_{method}_{date}-{XX}_{cv_auc}_auc.keras` where `{XX}` is a random 2-letter tag to prevent collisions. e.g. `rsna_vit_20260608-ZU_08740_auc.keras`

---

## model design

### CNN (`1_train_cnn.py`)
- 3-layer Conv2D (filters: 16, 32/64 grid, 128) + BatchNorm + MaxPool + Dropout
- Augmentation baked into model: RandomFlip(h), RandomRotation(+/-5deg), RandomZoom(+/-8%), RandomTranslation(+/-8%), `fill_mode='constant'` (CT-appropriate)
- Loss: **focal, gamma=1.0** (`BinaryFocalCrossentropy`), set at [1_train_cnn.py:248](1_train_cnn.py#L248). The earlier BCE runs are kept in the results table below: BCE reached 0.9243, the focal retrain reached 0.9467
- Optimizer: Adam + ReduceLROnPlateau (grid search and retrain)
- Grid: filters2=[32,64], lr=[0.001, 0.0001]
- **Grid search is disabled.** `SKIP_GRID_SEARCH = True` at [1_train_cnn.py:255](1_train_cnn.py#L255); the selected values (filters1=16, filters2=32, filters3=128, lr=0.001, dropout=0.25, epochs=30, batch_size=64) are hardcoded at [1_train_cnn.py:335](1_train_cnn.py#L335). They come from job 44094050 (2026-06-04), the only full-dataset search that ran to completion. Four earlier attempts were OOM-killed by SLURM and one died on a 27.88 GiB GPU allocation, and 44094050 itself reported a degenerate `Best CV score: 0.5000` from the `make_scorer` multilabel bug. All CNN runs since are single full retrains at these fixed settings, which is why the checkpoints carry `nan_auc`
- Optuna (`1b_train_cnn_optuna.py`) was an exploratory alternative to the grid, not the source of the current model. Job 47067109 hit the SLURM walltime after 41 trials; its best trial (0.9070: filters1=32, filters2=128, filters3=128, lr=0.00054) was never used
- Current model: `rsna_cnn_20260722-BD_nan_auc.keras` (weighted AUC 0.9467, run 17)

### ViT (`2_train_vit.py`)
- Patch embedding (Conv2D stride=patch_size) -> PositionEmbedding -> N transformer blocks -> GAP -> Dense(5, sigmoid)
- Augmentation in dataset pipeline (training only): RandomFlip(h), RandomRotation(+/-5deg), RandomZoom(+/-8%), RandomTranslation(+/-8%)
- Loss: **Focal loss** (`gamma=2.0`) -- AUC 0.8985 -> 0.9470 vs BCE
- Optimizer (grid search): Adam + ReduceLROnPlateau
- Optimizer (retrain): **AdamW** + **cosine decay with 10% linear warmup** -- standard for ViTs
- Grid: embed_dim=[64,128], mlp_dim=[128,256], patch_size=16, num_blocks=4
- **Grid search is also disabled** (`SKIP_GRID_SEARCH = True` at [2_train_vit.py:255](2_train_vit.py#L255), params hardcoded at [2_train_vit.py:329](2_train_vit.py#L329)), but for a different reason than the CNN. ViT searches never hit OOM and returned real CV scores across five jobs: 0.8034, 0.8739, 0.8727, 0.8740, 0.8916. The last of these, job 46809230 (2026-07-12), completed the search and was then walltime-killed during the retrain, so its result was carried forward as `best_score_=0.8916` instead of NaN. Run 16's `best_cv_auc` in `training_log.csv` is a genuine measurement, made in the preceding job rather than in run 16 itself
- Retrain resumes weights from the newest `vit_checkpoint_*.keras` whenever `SKIP_GRID_SEARCH` is set ([2_train_vit.py:369](2_train_vit.py#L369)), so run 16 (job 47003091) continued from the checkpoint left by the walltime-killed 46809230 rather than training from scratch
- XLA note: `TF_XLA_FLAGS="--tf_xla_auto_jit=0"` required in sbatch -- XLA JIT deadlocks on the validation pass for transformer models with large val sets
- Current model: `rsna_vit_20260713-ET_08916_auc.keras` (weighted AUC 0.9407, run 16). The older `rsna_vit_20260608-ZU_08740_auc.keras` scored 0.9473 but is no longer the checkpoint the downstream steps load

### Hybrid (`3_train_hybrid.py`)
- Simple average of CNN and ViT sigmoid outputs, no retraining
- Both components now use focal loss (CNN gamma=1.0, ViT gamma=2.0)
- Models hardcoded by filename to prevent picking up failed/nan runs
- Current model: `rsna_hybrid_20260722-HY_09554_auc.keras` (weighted AUC 0.9541, run 18), built from the run 17 CNN and the run 16 ViT

---

## segmentation map files (step 4, 5 and 6 outputs)

All files saved to `/rds/projects/k/karwatha-karwath-hds-pg-research/axr1222/models/`.
Timestamp in filename = when the Python script started (not when the SLURM job was submitted).

**GradCAM + RSNA U-Net maps:**

| file | generated from |
|------|---------------|
| `cnn_gradcam_maps_{ts}.npy` | `rsna_cnn_20260722-BD_nan_auc.keras` |
| `vit_gradcam_maps_{ts}.npy` | `rsna_vit_20260713-ET_08916_auc.keras` |
| `hybrid_gradcam_maps_{ts}.npy` | avg of CNN + ViT GradCAM |
| `unet_seg_maps_{ts}.npy` | frozen CNN encoder, RSNA labels |
| `seg_maps_test_idx_{ts}.npy` | indices of 2000 RSNA test slices |

These are RSNA-only maps. Step 7 does not reuse them: it regenerates GradCAM on the PhysioNet and CQ500 slices directly, since a map computed on an RSNA slice says nothing about a CQ500 slice.

**PhysioNet U-Net maps** (current timestamp `20260817_105446`):

| file | generated from |
|------|---------------|
| `physionet_seg_maps_20260817_105446.npy` | frozen encoder from `rsna_cnn_20260722-BD_nan_auc.keras`, decoder trained on PhysioNet pixel masks |
| `physionet_test_patients_20260817_105446.npy` | patient IDs for the 15 PhysioNet test patients |

**Seg-CQ500 U-Net maps** (current timestamp `20260817_110225`):

| file | generated from |
|------|---------------|
| `cq500_seg_maps_20260817_110225.npy` | same encoder, decoder trained on Seg-CQ500 pixel masks |
| `cq500_test_patients_20260817_110225.npy` | patient IDs for the 11 Seg-CQ500 test patients |

Step 7 picks up the newest file matching each glob, so an old set left in the directory is ignored rather than silently mixed in.

## step 5 and step 6 design notes

Both segmenters are the same architecture on different pixel-mask datasets.

- encoder initialised from the RSNA CNN checkpoint `rsna_cnn_20260722-BD_nan_auc.keras`, frozen, only the decoder learns
- patient-level 80/20 train/test split at SEED=20260605: PhysioNet 60 train / 15 test, Seg-CQ500 40 train / 11 test
- of the 15 PhysioNet test patients only **6** contribute ICH-positive slices, so the whole localisation arm rests on 6 patients there. All 11 Seg-CQ500 test patients are ICH-positive. 6 and 11 are the effective sample sizes behind every localisation interval and p-value below, not 15 and 11
- loss: BCE + Dice (handles sparse positive pixels better than BCE alone)
- augmentation: horizontal flip only (vertical flip disabled, it is not anatomically plausible for axial CT)
- 50 epochs max, lr 1e-4, early stopping on `val_dice_positive_only`
- test metrics over all test slices: PhysioNet Dice 0.4009, IoU 0.2507, recall 0.3227, precision 0.5290; Seg-CQ500 Dice 0.4372, IoU 0.2797, recall 0.3314, precision 0.6422
- outputs: `{dataset}_seg_maps_*.npy` (float16) + test patient list, both consumed by step 7

---

## step 7 design (`7_compare_all.py`)

It loads each classifier once and scores everything:

**Localisation arm** (PhysioNet and Seg-CQ500, have pixel masks): re-generate GradCAM from CNN, ViT and Hybrid on those slices directly (step 4 maps are on RSNA test data and are not reused), load the step 5 / step 6 U-Net predictions, and compare every method against the ground-truth mask on ICH-positive slices only.

Metrics per method: **Dice**, **IoU**, **pixel recall**, **pixel precision**, **average precision**, **Pointing Game** (Selvaraju et al. 2020, doi:10.1007/s11263-019-01228-7: whether the peak activation pixel falls inside the GT mask, threshold-free).

**Chance baseline**: a uniform random saliency map at the same threshold, entered as just another method so it flows through every table and figure. Its pointing game expectation is the mean lesion area fraction, 0.0040 on PhysioNet and 0.0091 on CQ500, matched by a Monte Carlo estimate over 1000 random peaks per slice (0.0039 and 0.0092). Both go to `all_stats_chance_{ts}.csv` with the expected hit count, 0.16 on PhysioNet and 7.29 on CQ500. A pointing game of 0.00 is only below chance if it is below that value, which is the reason the row exists.

**Classification arm** (all three datasets): ROC and AUC for CNN, ViT, Hybrid and U-Net. RSNA is the internal test set, PhysioNet and CQ500 are zero-shot external validation scored as slice-level "any ICH" (max over the five subtype probabilities). Noisy-OR is run on the same scores as a sensitivity check and written to `all_sensitivity_noisy_or_{ts}.csv`.

**Inferential statistics** (`stats_utils.py`): DeLong for paired AUCs, Wilcoxon signed-rank for the continuous localisation metrics, exact McNemar for the binary pointing game, all Holm-corrected within their own family, plus a patient-clustered bootstrap for every Dice/IoU confidence interval. Wilcoxon runs twice, once per slice and once with each patient collapsed to a single value, and the `unit=patient` rows are the conservative reading.

No omnibus test precedes the pairwise ones: Holm-Bonferroni controls the family-wise rate inside each pairwise family directly. The tables written are `auc`, `delong`, `wilcoxon`, `mcnemar`, `chance` and `seg_descriptive`.

**RSNA qualitative panel**: 3 ICH-positive test slices (`N_QUAL_RSNA`) with CT + CNN / ViT / Hybrid activation maps, titled with true and predicted labels under the subtype codes EDH / IPH / IVH / SAH / SDH. RSNA has no pixel masks, so it gets no Dice.

This is not a direct comparison. As stated in the report's header, the CNN, ViT, and Hybrid models are evaluated in a zero-shot setting, whereas the U-Net results reflect supervised training on the target distributions.

---

## step 7 outputs

All 30 files land in `~/thesis-xai/result/`, prefixed `all_` and stamped with one timestamp per run.

| file | count | content |
|------|------:|---------|
| `all_comparison_results_{ts}.csv` | 1 | localisation metrics per dataset and method |
| `all_comparison_bar_{dataset}_{ts}.png` | 2 | 6-metric bar chart per dataset, mean with a 95% patient-clustered bootstrap CI |
| `all_comparison_soft_{dataset}_p{patient}_s{slice}_{ts}.png` | 4 | unthresholded saliency, one file per sample slice: CT with the mask outlined in cyan, then one panel per method |
| `all_comparison_error_{dataset}_p{patient}_s{slice}_{ts}.png` | 4 | TP/FP/FN/TN error maps, same slices and same layout |
| `all_comparison_dice_combined_{ts}.png` | 1 | Dice, both datasets side by side |
| `all_comparison_rsna_gradcam_{ts}.png` | 1 | RSNA qualitative panel |
| `all_raw_{ts}.npz` | 1 | every per-slice probability and metric; the bundle the notebook analyses |
| `all_auc_results_{ts}.csv` | 1 | AUC, DeLong CI and AP per dataset / target / model |
| `all_seg_descriptive_{ts}.csv` | 1 | localisation means with patient-clustered bootstrap CIs |
| `all_stats_{delong,wilcoxon,mcnemar}_{ts}.csv` | 3 | one file per test family, Holm-adjusted |
| `all_stats_chance_{ts}.csv` | 1 | pointing game under chance, analytic and Monte Carlo, with expected hits |
| `all_sensitivity_noisy_or_{ts}.csv` | 1 | any-ICH AUC under max vs noisy-OR, per dataset and model |
| `all_stats_report_{ts}.txt` | 1 | the whole analysis as a readable report, six sections |
| `all_{roc_rsna,auc_rsna_dots,auc_forest,roc_external,pr_external,dist_dice,dist_iou,pvalues_dice_patient}_{ts}.png` | 8 | statistics figures |

Sample slices per dataset are set by `N_QUAL_IMGS = 2` and the RSNA panel by `N_QUAL_RSNA = 3`. Every figure is written at 300 dpi and sized to a fixed text width (`FIG_W = 6.7in`, `FIG_W_WIDE = 9.6in`), so it drops into the thesis without rescaling.

The p-value heatmap is patient-level by default (`unit='patient'`), and `fig_pvalue_heatmap` raises if the filtered table still holds duplicate method pairs, since a duplicate would silently overwrite a cell.

The statistics pass runs in `thesis.ipynb`, in the **Inferential statistics** section. It reads `all_raw_{ts}.npz` and calls each `stats_utils` test in its own cell, then writes the six CSVs (`auc`, `delong`, `seg_desc`, `wilcoxon`, `mcnemar`, `chance`), the report and the eight statistics figures. It reuses the bundle timestamp, so re-running overwrites its own outputs instead of littering the directory. No GPU, about 3 minutes at `n_boot=2000`.
Step 7 writes the first copy of these files itself, via `stats_utils.run_all_analysis`.

---

## status

**all RQs done**
- HPC setup, GitHub, venv
- DICOM index (873,945 files -> `rsna_dicom_index_2.csv`)
- EDA on all three datasets
- Full RSNA preprocessing (18,253 patients, SEED=20260605, 3-channel CT windowing)
- CNN training: focal gamma=1.0, seed-locked, full dataset (run 17, weighted AUC 0.9467)
- ViT training: focal gamma=2.0 + AdamW + cosine warmup (run 16, weighted AUC 0.9407)
- Hybrid ensemble (run 18, weighted AUC 0.9541), mean of the run 17 CNN and run 16 ViT
- RSNA U-Net classifier head (run 19, weighted AUC 0.8957)
- PhysioNet pixel-supervised U-Net, Dice 0.4009 over all test slices (step 5)
- Seg-CQ500 pixel-supervised U-Net, Dice 0.4372 over all test slices (step 6)
- GradCAM maps generated (step 4)
- `7_compare_all.py` run end to end (job 52036622, run `20260823_083800`), full statistics report in `result/`

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
| 8 | 2026-06-06 | CNN | focal gamma=2.0 | 12,784 | 0.8774 | 0.8386 | 160 min | lower than BCE |
| 9 | 2026-06-06 | CNN | focal gamma=1.0 | 12,784 | 0.8950 | 0.8369 | 230 min | lower than BCE |
| 10 | 2026-06-07 | ViT | focal gamma=2.0 | 12,784 | 0.9470 | 0.8727 | 467 min | job 44333718, ZU model -- **corrupted** |
| 11 | 2026-06-07 | U-Net (RSNA) | BCE | 12,784 | 0.8783 | -- | 552 min | weakly supervised; part of step 4 |
| 12 | 2026-06-07 | Hybrid | focal (ViT) | 12,784 | 0.9513 | -- | -- | job 44361117 -- **corrupted** |
| 13 | 2026-06-08 | ViT | focal gamma=2.0 | 12,784 | 0.9473 | 0.874 | 672 min | job 44366721, retrain of run 10; `rsna_vit_20260608-ZU_08740_auc.keras` |
| 14 | 2026-06-08 | Hybrid | focal (ViT) | 12,784 | 0.9513 | -- | -- | job 44424197, retrain of run 12; `rsna_hybrid_20260608-AR_09512_auc.keras` |
| 15 | 2026-06-08 | U-Net (RSNA) | BCE | 12,784 | 0.8673 | -- | 85 min | `rsna_unet_20260608-SJ_nan_auc.keras`, replaces the corrupted run 11 |
| 16 | 2026-07-13 | ViT | focal gamma=2.0 | 12,784 | **0.9407** | 0.8916 | 834 min | `rsna_vit_20260713-ET_08916_auc.keras`, **current ViT** |
| 17 | 2026-07-22 | CNN | focal gamma=1.0 | 12,784 | **0.9467** | -- | 2,668 min | `rsna_cnn_20260722-BD_nan_auc.keras`, **current CNN** |
| 18 | 2026-07-22 | Hybrid | focal | 12,784 | **0.9541** | 0.9554 | -- | `rsna_hybrid_20260722-HY_09554_auc.keras`, mean of runs 16 + 17, **current Hybrid** |
| 19 | 2026-08-02 | U-Net (RSNA) | BCE | 12,784 | **0.8957** | -- | 120 min | `rsna_unet_20260802-NB_nan_auc.keras`, classifier head on a frozen U-Net encoder, **current U-Net** |

Two ViT checkpoints score close to each other (0.9473 for the June model, 0.9407 for the July one); the July model is what every downstream step loads, so it is the one reported.

**Per-class AUCs on the RSNA test set (101,823 slices, current models):**

| class | CNN (run 17) | ViT (run 16) | Hybrid (run 18) | U-Net (run 19) |
|-------|------------|-------------|----------------|----------------|
| epidural | 0.9305 | 0.9110 | 0.9321 | 0.9172 |
| intraparenchymal | 0.9597 | 0.9567 | 0.9673 | 0.9145 |
| intraventricular | 0.9774 | 0.9711 | 0.9802 | 0.9541 |
| subarachnoid | 0.9308 | 0.9221 | 0.9389 | 0.8867 |
| subdural | 0.9323 | 0.9266 | 0.9417 | 0.8545 |
| **weighted avg** | **0.9467** | **0.9407** | **0.9541** | **0.8957** |

Step 7 recomputes these from the checkpoints and reproduces every value in the table exactly, with DeLong CIs in `all_auc_results_{ts}.csv`.

Models saved to: `/rds/projects/k/karwatha-karwath-hds-pg-research/axr1222/models/`

### external validation, zero-shot any-ICH (step 7, run `20260823_083800`)

Slice-level "any ICH", max over the five subtype probabilities. CNN / ViT / Hybrid never saw these datasets; the U-Net rows were trained on the dataset they are scored on, so they are not a fair comparison, only a ceiling.

| dataset | slices | positive | CNN | ViT | Hybrid | U-Net (in-domain) |
|---------|-------:|---------:|----:|----:|-------:|------------------:|
| PhysioNet | 542 | 40 | 0.8098 | 0.8047 | 0.8015 | 0.8618 |
| Seg-CQ500 | 2,594 | 799 | 0.7234 | 0.7746 | 0.7906 | 0.8467 |

Against their own RSNA-internal weighted AUCs the classifiers lose 0.14 to 0.22 off-domain: CNN 0.137 on PhysioNet and 0.223 on Seg-CQ500, ViT 0.136 and 0.166, Hybrid 0.153 and 0.164. The CNN falls furthest, which is why the ranking inverts. Noisy-OR instead of max gains 0.02 to 0.06 on every row and is written to `all_sensitivity_noisy_or_{ts}.csv` as a sensitivity check.

### localisation quality (ICH-positive slices only, threshold 0.5)

| dataset | method | Dice | IoU | recall | precision | pointing game | AP |
|---------|--------|-----:|----:|-------:|----------:|--------------:|---:|
| PhysioNet (n=40 slices, 6 patients) | CNN | 0.0173 | 0.0092 | 0.0754 | 0.0102 | 0.0000 | 0.0143 |
| | ViT | 0.0187 | 0.0096 | 0.4040 | 0.0096 | 0.0500 | 0.0211 |
| | Hybrid | 0.0307 | 0.0164 | 0.1333 | 0.0182 | 0.0000 | 0.0202 |
| | U-Net | 0.3315 | 0.2482 | 0.2862 | 0.5420 | 0.7000 | 0.4477 |
| | Chance | 0.0080 | 0.0041 | 0.5053 | 0.0041 | 0.0000 | 0.0042 |
| CQ500 (n=799 slices, 11 patients) | CNN | 0.0174 | 0.0094 | 0.0496 | 0.0114 | 0.0125 | 0.0268 |
| | ViT | 0.0589 | 0.0324 | 0.4932 | 0.0340 | 0.1014 | 0.0664 |
| | Hybrid | 0.0501 | 0.0286 | 0.1807 | 0.0339 | 0.0300 | 0.0506 |
| | U-Net | 0.3641 | 0.2882 | 0.3475 | 0.4686 | 0.5807 | 0.4525 |
| | Chance | 0.0176 | 0.0090 | 0.4982 | 0.0091 | 0.0175 | 0.0093 |

Read every row against Chance. A GradCAM is coarse by construction, so a low Dice against a supervised segmenter is expected and the pointing game is the fairer XAI metric. On that metric only ViT clears chance by a useful margin on CQ500 (0.1014 vs 0.0175). Recall is the one column to ignore: a random map thresholded at 0.5 marks about half the image, so its recall is about 0.5 by construction and carries no information about any method.

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
| `6_train_cq500.py` | pixel-supervised U-Net on Seg-CQ500 masks (same encoder as step 5) |
| `6_train_cq500.sh` | sbatch for above (GPU, A100, 4h, 64G) |
| `6_compare_physionet.py` | deprecated |
| `7_compare_all.py` | cross-dataset comparison: GradCAM vs U-Net vs GT masks on PhysioNet and Seg-CQ500, classification AUC on all three datasets, full inferential statistics; writes to `~/thesis-xai/result/` |
| `7_compare_all.sh` | sbatch for above (GPU, A100, 6h, 64G); installs nibabel if missing |
| `stats_utils.py` | statistics and figure layer used by step 7 and the notebook: DeLong, Wilcoxon, McNemar, Holm correction, patient-clustered bootstrap, chance reference, report writer |
| `result/` | every step 7 and notebook statistics output |
| `training_logger.py` | saves model + appends row to `training_log.csv`; `model_status` column tracks ok/corrupted/deleted/retraining |
| `training_log.csv` | all run results (timestamp, method, loss, AUCs, hyperparams, duration, model_status) |
| `thesis.ipynb` | main notebook: EDA, pipeline verification, inferential statistics, results, discussion |

---

## running the notebook

srun --account=karwatha-karwath-hds-pg-research --qos=bbdefault --ntasks=8 --mem=32G --time=3:0:0 --pty /bin/bash

source ~/start_cpu.sh

jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser

open the jupyter notebook, select existing jupyter kernel from terminal, e.g.:
http://bear-pg0201u13a:8888/tree?token=abcde


---

## OOM notes

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

PhysioNet and Seg-CQ500 are ~2 GB and ~4 GB, both fit in RAM with no memmap.

Step 7 touches the full 75 GB RSNA test set as well, so it keeps `x_test` and `y_test` as open memmaps and streams them through each model in batches of 64 (`predict_batched`). 101,823 slices at ~900 slice/s is about 2 minutes per model on an A100. The GradCAM loop stays at batch 16 because the gradient tape holds activations for all five classes.

---

## rules

- always split by `PatientID` -- prevent data leakage
- drop `any` label before training -- it's derived, causes leakage
- use HU conversion -- raw pixels are scanner-dependent
- `set_memory_growth(gpu, True)` always -- otherwise TF grabs all VRAM
- sbatch for training, Jupyter for eval -- kernel dies in Jupyter
- always use `DataStreamKerasClassifier` for large datasets -- avoids TF materialising 349 GB as a constant
- always add `ModelCheckpoint` to long jobs -- SLURM will kill at walltime with no warning
- all experiments use SEED=20260605 -- same seed everywhere for reproducible splits
- augmentation only during training -- `augment=False` for val/test in `_make_dataset`
- hardcode model paths/filename in `3_train_hybrid.py`, `4_segmentation_xai.py`, `5_train_physionet_seg.py`, `6_train_cq500.py`, `7_compare_all.py`
- every localisation number is read against the Chance row

---

## things to watch

- `InstanceNumber` all NaN in DICOM index
- epidural AUC is lowest due to extreme rarity: 351 of 101,823 test slices, 0.34% prevalence, and 1,643 of 475,870 in train, 0.35%
- no learning curves saved yet -- add `history_` extraction to eval notebook
- PhysioNet CSV column order differs from RSNA: Intraventricular, Intraparenchymal, Subarachnoid, Epidural, Subdural (capitalised)
- the slice-level p-values in the statistics report are anti-conservative, slices within a patient are correlated; quote the `unit=patient` rows
- CNN and U-Net checkpoints carry `nan_auc` in their filename because those runs set `SKIP_GRID_SEARCH = True`, so `grid.best_score_` is `float('nan')` and `save_models` writes `nan` into the name. Grid search over the full dataset was repeatedly OOM-killed (jobs 44066393, 44068843, 44072767, 44088354, plus a GPU allocator OOM in 44071888). The one search that completed, 44094050, returned a degenerate `Best CV score: 0.5000` from the `make_scorer` multilabel bug listed in the OOM notes. Hyperparameters were fixed from that run and no `best_cv_auc` has been recorded since. The test AUCs in `training_log.csv` come from complete retrains and are valid
- with only 40 positive slices from 6 patients, the PhysioNet pointing game lacks the statistical power for meaningful evaluation. An expected chance baseline of 0.16 hits renders a 0.00 score indistinguishable from expected random variance

---

*updated: 2026-08-23*