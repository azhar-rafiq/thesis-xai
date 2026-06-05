# Preprocessed RSNA Data — Summary

**Path:** `/rds/projects/k/karwatha-karwath-hds-pg-research/axr1222/data/preprocessed/`  
**Created:** 2026-06-01

---

## Dataset Configuration (from `rsna_tr12784_va2731_te2738_256_meta.json`)

| Parameter | Value |
|---|---|
| Image size | 256 × 256 px |
| Channels | 3 (brain / subdural / soft-tissue windows) |
| Split ratios | 70 / 15 / 15 |
| Seed | 20260605 |
| Stratified capping | true |

**Windowing:**

| Window | Centre (WC) | Width (WW) |
|---|---|---|
| Brain | 40 | 80 |
| Subdural | 80 | 200 |
| Soft tissue | 40 | 380 |

**Label columns:** `epidural`, `intraparenchymal`, `intraventricular`, `subarachnoid`, `subdural`

---

## File Sizes

| File | Size | Description |
|---|---|---|
| `rsna_tr12784_va2731_te2738_256_meta.json` | 628 B | dataset metadata |
| `rsna_tr12784_va2731_te2738_256_x_train.npy` | 349 GB | train images |
| `rsna_tr12784_va2731_te2738_256_y_train.npy` | 9.1 MB | train labels |
| `rsna_tr12784_va2731_te2738_256_x_val.npy` | 76 GB | validation images |
| `rsna_tr12784_va2731_te2738_256_y_val.npy` | 2.0 MB | validation labels |
| `rsna_tr12784_va2731_te2738_256_x_test.npy` | 75 GB | test images |
| `rsna_tr12784_va2731_te2738_256_y_test.npy` | 2.0 MB | test labels |
| `rsna_tr13263_va2834_te2841_256_x_train.npy` | 386 GB | alt train images (larger split) |
| `rsna_tr13263_va2834_te2841_256_y_train.npy` | 11 MB | alt train labels (larger split) |
| **Total** | **~578 GB** | |

---

## Array Shapes & Dtypes

| Array | Shape | Dtype |
|---|---|---|
| `x_train` | (475870, 256, 256, 3) | float32 |
| `y_train` | (475870, 5) | float32 |
| `x_val` | (103101, 256, 256, 3) | float32 |
| `y_val` | (103101, 5) | float32 |
| `x_test` | (101823, 256, 256, 3) | float32 |
| `y_test` | (101823, 5) | float32 |

Pixel values are normalised to **[0, 1]**.

---

## Label Distribution

| Split | Total slices | Any hemorrhage | Epidural | Intraparenchymal | Intraventricular | Subarachnoid | Subdural |
|---|---|---|---|---|---|---|---|
| Train | 475,870 | 62,215 (13.1 %) | 1,643 | 20,430 | 15,242 | 19,862 | 27,446 |
| Val | 103,101 | 13,446 (13.0 %) | 476 | 4,705 | 3,302 | 4,673 | 5,633 |
| Test | 101,823 | 13,258 (13.0 %) | 351 | 4,674 | 3,191 | 4,218 | 5,924 |

> Slices can have multiple hemorrhage types simultaneously (multi-label).

---

## 10 Sample Rows from `y_train` (positive / hemorrhage cases)

| Slice idx | Epidural | Intraparenchymal | Intraventricular | Subarachnoid | Subdural |
|---|---|---|---|---|---|
| 17 | 0 | 0 | 0 | 1 | 1 |
| 23 | 0 | 0 | 0 | 0 | 1 |
| 25 | 0 | 0 | 1 | 0 | 0 |
| 30 | 0 | 0 | 0 | 0 | 1 |
| 37 | 0 | 0 | 1 | 1 | 0 |
| 57 | 0 | 0 | 0 | 0 | 1 |
| 61 | 0 | 0 | 0 | 0 | 1 |
| 81 | 0 | 0 | 0 | 0 | 1 |
| 86 | 0 | 0 | 0 | 1 | 0 |
| 102 | 0 | 0 | 0 | 0 | 1 |

*(first 10 positive-label slices in training set order)*
