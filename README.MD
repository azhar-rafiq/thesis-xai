# thesis-xai — ICH detection with XAI

Azhar | MSc Health Data Science, UoB

---

## what am I doing

| RQ | short version | approach | status |
|----|--------------|----------|--------|
| RQ1 | classify ICH subtypes from CT slices | CNN -> Transformer | 🟡 running |
| RQ2 | segment haemorrhage at pixel level | U-Net | 🔴 not started |
| RQ3 | do heatmaps actually point at the bleed? | Grad-CAM, SHAP | 🔴 not started |
| RQ4 | how do you measure if XAI is any good? | IoU vs PhysioNet masks | 🔴 not started |

---

## data

| dataset | format | size | labels | used for |
|---------|--------|------|--------|----------|
| RSNA 2019 | DICOM | 752k slices, 18,938 patients | 5 subtypes (slice-level) | classification |
| PhysioNet CT-ICH | NIfTI | 75 patients, 2,814 slices | pixel masks | segmentation + XAI eval |

PhysioNet untouched so far — all current work is RSNA.

---

## status

**done**
- HPC setup, GitHub, venv
- DICOM index (873,945 files -> `rsna_dicom_index_2.csv`)
- EDA on both datasets
- preprocessing pipeline: HU -> brain window [0,80] -> normalise -> resize 256×256
- CNN baseline (800 patients) — weighted test AUC **0.7814**

**running now**
- 3000/750 RSNA cache (`sbatch 0_preprocess_rsna.sh`) — CPU job, bbdefault
- ViT script ready to submit once cache is done

**not started yet**
- PhysioNet NIfTI pipeline
- U-Net (RQ2)
- Grad-CAM / SHAP (RQ3)
- XAI evaluation metrics: IoU, pointing game, mass accuracy (RQ4)
- learning curves (need to extract from `history_`)

---

## results so far

| # | model | patients | cv auc | test auc (weighted) | notes |
|---|-------|----------|--------|----------------------|-------|
| 1 | CNN 3-layer | 800 | 0.6465 | 0.7814 | filters=128, lr=1e-4, bs=32, drop=0.25 — 40 min |
| 2 | CNN 3-layer | 800 | 0.6302 | 0.7886 | same filters, bs=64, epochs=30 — 90 min |
| 3 | ViT | 3000 | — | — | ⏳ patch=16/32, blocks=2/4, embed=64 |

> Run 2 CV AUC is lower (0.63) but test AUC is higher (0.79) — slight overfit signal, worth checking learning curves.

---

## scripts on bluebear (`~/thesis-xai/`)

| file | what it does |
|------|-------------|
| `0_preprocess_rsna.py` | DICOM -> .npz cache |
| `0_preprocess_rsna.sh` | sbatch for above (CPU, bbdefault, 64G, 6h) |
| `1_train_mult_rsna.py` | CNN gridsearch |
| `1_train_mult_rsna.sh` | sbatch for above (GPU, A100) |
| `2_train_transformer.py` | ViT gridsearch |
| `2_train_transformer.sh` | sbatch for above (GPU, A100, 24h) |
| `training_logger.py` | saves model + appends to training_log.csv |
| `training_log.csv` | all run results |
| `thesis3update.ipynb` | main notebook — EDA, eval, plots |

---

## things to fix / watch out for

- old 800/200 cache has 6 columns (includes `any` due to notebook bug) — **incompatible with new cache**, don't try to extend
- patient selection is not stratified — first N by index, epidural only 0.27% so might be missing from small samples
- no learning curves yet — can't tell if models converged
- `InstanceNumber` all NaN in DICOM index — mention in thesis methodology

---

## next steps

1. wait for 3000/750 cache to finish
2. `sbatch 2_train_transformer.sh`
3. compare CNN vs ViT AUC (expect ViT lower — that's the thesis point)
4. Grad-CAM on best CNN + best ViT
5. PhysioNet NIfTI preprocessing (`nibabel`)
6. U-Net
7. XAI evaluation against PhysioNet masks

---

## rules (don't break these)

- always split by `PatientID` — slice-level split = data leakage
- drop `any` label before training — it's derived, causes leakage
- HU conversion before anything else — raw pixels are scanner-dependent
- `set_memory_growth(gpu, True)` always — otherwise TF grabs all VRAM
- sbatch for training, Jupyter for eval — kernel dies otherwise

---

*updated: 2026-05-26 08.50*