"""
RSNA Intracranial Hemorrhage - Preprocessing & Cache Script
Writes directly to memory-mapped .npy files - peak RAM is O(1) per slice.
Run via sbatch: sbatch 0_preprocess_rsna.sh
"""

import os
import sys
import json
import datetime
import numpy as np
import pandas as pd
import pydicom
from multiprocessing import Pool
from skimage.transform import resize
from sklearn.model_selection import GroupShuffleSplit
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from pathlib import Path
from dotenv import load_dotenv


#######
# How I preprocess:
# 1. load labels from stage_2_train.csv (long format) and pivot to wide (one row per slice)
# 2. merge with DICOM index to attach PatientID to each slice
# 3. drop patients with conflicting labels across studies (multi_study_conflicts.csv)
# 4. patient-level 70/15/15 train/val/test split via GroupShuffleSplit (never slice-level)
# 5. optionally cap each split with stratified patient sampling (MultilabelStratifiedShuffleSplit)
# 6. for each DICOM: convert raw pixels to HU using RescaleSlope + RescaleIntercept
# 7. apply 3 CT windows (brain, subdural, soft tissue) from Burduja et al. (2020)
# 8. stack the 3 windows as channels -> (IMG_SIZE, IMG_SIZE, 3) float32
# 9. write directly to memory-mapped .npy files via N_WORKERS parallel processes
#    - corrupt DICOMs (truncated pixel data etc.) are skipped with a warning
#    - already-completed splits are skipped on re-run; only the failed split is retried
# 10. save a JSON sidecar with split sizes, windows, and seed for reproducibility
#
# multiprocessing design:
#   N_WORKERS child processes each read and decode one DICOM at a time (I/O + CPU bound).
#   the main process collects results via imap_unordered and writes them sequentially
#   into the pre-allocated memmap. only one decoded image lives in RAM per worker at
#   any moment, so total RAM = N_WORKERS * ~1 MB regardless of dataset size.
#
# why sbatch does not accumulate RAM:
#   old approach built a Python list of all images then called np.array(), which held
#   the entire dataset in RAM twice during conversion (~150 GB for 3750 patients).
#   here, np.lib.format.open_memmap() pre-allocates the output file on disk and returns
#   a view - each write goes directly to the OS page cache and is flushed to disk
#   asynchronously. the job requested 64 GB but real peak usage should be under 10 GB.
#######

# config
SEED     = 20260605
IMG_SIZE = 256

# None = use ALL patients in each split; set an int to cap (e.g. for testing)
N_TRAIN_PATIENTS = None
N_VAL_PATIENTS   = None
N_TEST_PATIENTS  = None

# parallel DICOM workers - 1 CPU reserved for the main writing process
N_WORKERS = max(1, int(os.environ.get('SLURM_CPUS_PER_TASK', 4)) - 1)

label_cols = ['epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural']

# paths
sys.path.insert(0, os.path.expanduser('~/thesis-xai'))
load_dotenv()

_kaggle_cache = os.getenv('KAGGLE_CACHE')
if not _kaggle_cache:
    raise EnvironmentError("KAGGLE_CACHE is not set in .env or environment")
os.environ['KAGGLEHUB_CACHE'] = _kaggle_cache
KAGGLE_CACHE   = Path(_kaggle_cache)
CACHE_DIR      = KAGGLE_CACHE / 'preprocessed'
RSNA_TRAIN_DIR = KAGGLE_CACHE / 'competitions/rsna-intracranial-hemorrhage-detection/rsna-intracranial-hemorrhage-detection/stage_2_train'
LABELS_CSV     = KAGGLE_CACHE / 'competitions/stage_2_train.csv'
INDEX_CSV      = Path.home() / 'rsna_dicom_index_2.csv'

CACHE_DIR.mkdir(parents=True, exist_ok=True)

# windowing params - from Burduja et al. (2020), DOI: 10.3390/s20195611
WINDOWS = {
    'brain':    {'wc': 40,  'ww': 80},   # HU [0,    80]
    'subdural': {'wc': 80,  'ww': 200},  # HU [-20,  180]
    'soft':     {'wc': 40,  'ww': 380},  # HU [-150, 230]
}


def apply_window(hu_array, wc, ww):
    lo = wc - ww / 2.0
    return np.clip((hu_array - lo) / ww, 0.0, 1.0)


def load_dicom_as_array(filepath, img_size):
    ds          = pydicom.dcmread(str(filepath))
    pixel_array = ds.pixel_array.astype(np.float32)
    hu          = pixel_array * ds.RescaleSlope + ds.RescaleIntercept

    brain    = apply_window(hu, **WINDOWS['brain'])
    subdural = apply_window(hu, **WINDOWS['subdural'])
    soft     = apply_window(hu, **WINDOWS['soft'])

    img = np.stack([brain, subdural, soft], axis=-1)  # (H, W, 3)
    return resize(img, (img_size, img_size), anti_aliasing=True).astype(np.float32)


def _process_row(args):
    """worker: read and process one DICOM. returns (image, label) or None if missing/corrupt."""
    filename, label = args
    dcm_path = RSNA_TRAIN_DIR / filename
    if not dcm_path.exists():
        return None
    try:
        return load_dicom_as_array(dcm_path, IMG_SIZE), label
    except Exception as e:
        print(f"  [WARN] skipping corrupt DICOM {filename}: {e}", flush=True)
        return None


def _truncate_memmap(path, actual_count):
    """shrink a .npy memmap to actual_count rows by rewriting the header shape in-place."""
    import re
    with open(path, 'r+b') as f:
        f.read(6)  # magic bytes
        major = ord(f.read(1))
        f.read(1)  # minor version
        hlen = int.from_bytes(f.read(2 if major == 1 else 4), 'little')
        header_offset = f.tell()
        header_bytes = f.read(hlen)
        header_str = header_bytes.decode('latin1')

        # replace the first integer in the shape tuple (axis-0 size)
        new_header_str = re.sub(r'\((\d+),', f'({actual_count},', header_str, count=1)
        new_header_str = new_header_str.rstrip().rstrip('\n').ljust(hlen - 1) + '\n'
        assert len(new_header_str) == hlen

        f.seek(header_offset)
        f.write(new_header_str.encode('latin1'))

        dtype_str = re.search(r"'descr':\s*'([^']+)'", header_str).group(1)
        shape_match = re.search(r"'shape':\s*\(([^)]+)\)", header_str)
        dims = [int(x.strip()) for x in shape_match.group(1).split(',') if x.strip()]
        stride = int(np.prod(dims[1:])) * np.dtype(dtype_str).itemsize
        f.truncate(header_offset + hlen + actual_count * stride)


def write_to_memmap(df, x_path, y_path):
    """
    pre-count valid DICOMs, allocate memmap files at exact size, then read DICOMs
    in parallel (N_WORKERS processes) while the main process writes to the memmap.
    peak RAM per worker: one DICOM image.
    """
    print(f"  scanning for valid DICOMs...", flush=True)
    n = sum((RSNA_TRAIN_DIR / f).exists() for f in df['filename'])
    print(f"  valid slices: {n} / {len(df)} using {N_WORKERS} workers", flush=True)

    x = np.lib.format.open_memmap(x_path, mode='w+', dtype=np.float32,
                                   shape=(n, IMG_SIZE, IMG_SIZE, 3))
    y = np.lib.format.open_memmap(y_path, mode='w+', dtype=np.float32,
                                   shape=(n, len(label_cols)))

    args = [(row['filename'], row[label_cols].values.astype(np.float32))
            for _, row in df.iterrows()]

    idx, missing = 0, 0
    total = len(args)
    with Pool(processes=N_WORKERS) as pool:
        for i, result in enumerate(pool.imap_unordered(_process_row, args, chunksize=64)):
            if i % 5000 == 0:
                ts = datetime.datetime.now().strftime('%H:%M:%S')
                print(f"  [{ts}] {i}/{total} received, {idx} written...", flush=True)
            if result is not None:
                x[idx], y[idx] = result
                idx += 1
            else:
                missing += 1

    x.flush()
    y.flush()
    del x, y  # close memmaps before truncating
    print(f"  written: {idx}, missing: {missing}", flush=True)
    if idx != n:
        print(f"  [WARN] {n - idx} corrupt file(s) skipped - truncating memmaps to {idx} rows", flush=True)
        _truncate_memmap(x_path, idx)
        _truncate_memmap(y_path, idx)
    return idx


def select_patients_stratified(patient_df, n_select, seed):
    """
    from patient_df (one row per patient, with label_cols), return a stratified
    subset of n_select patients. if n_select >= total, returns all.
    """
    total = len(patient_df)
    if n_select is None or n_select >= total:
        return patient_df['PatientID'].values

    test_size = 1 - n_select / total
    msss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=test_size,
                                            random_state=seed)
    sel_idx, _ = next(msss.split(patient_df, patient_df[label_cols]))
    return patient_df.iloc[sel_idx]['PatientID'].values


# main
print(f"\n{'='*60}")
print(f"Preprocessing started at {datetime.datetime.now()}")
print(f"{'='*60}\n", flush=True)

# build label dataframe
print(f"Loading DICOM index: {INDEX_CSV}", flush=True)
dicom_index = pd.read_csv(INDEX_CSV)
dicom_index = dicom_index[dicom_index['split'] == 'train']

print(f"Loading labels: {LABELS_CSV}", flush=True)
labels_rsna_df = pd.read_csv(LABELS_CSV)

labels_rsna_df['SubType'] = labels_rsna_df['ID'].str.rsplit('_', n=1).str[-1]
labels_rsna_df['SliceID'] = labels_rsna_df['ID'].str.rsplit('_', n=1).str[0]
labels_rsna_df = labels_rsna_df.drop_duplicates(subset=['SliceID', 'SubType'], keep='first')

labels_wide = labels_rsna_df.pivot(index='SliceID', columns='SubType', values='Label').reset_index()
labels_wide = labels_wide[['SliceID'] + label_cols + ['any']]
labels_wide = labels_wide.drop(columns=['any'])  # drop 'any' - derived label, causes leakage

labels_wide['filename'] = labels_wide['SliceID'] + '.dcm'
labels_with_patient = labels_wide.merge(
    dicom_index[['filename', 'PatientID']], on='filename', how='left'
)
labels_with_patient = labels_with_patient.dropna(subset=['PatientID'])

# drop patients with conflicting labels across studies (inconsistent ICH annotation)
conflicts_csv = Path(__file__).parent / 'multi_study_conflicts.csv'
conflict_ids  = pd.read_csv(conflicts_csv)['PatientID'].unique()
before = labels_with_patient['PatientID'].nunique()
labels_with_patient = labels_with_patient[~labels_with_patient['PatientID'].isin(conflict_ids)]
after  = labels_with_patient['PatientID'].nunique()
print(f"Dropped {before - after} patients with conflicting multi-study labels ({before} -> {after})", flush=True)

# patient-level split: 70 / 15 / 15 train / val / test, never slice-level
# step 1: carve out 15% test
gss1 = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=SEED)
dev_idx, test_idx = next(gss1.split(labels_with_patient,
                                    groups=labels_with_patient['PatientID']))
dev_df  = labels_with_patient.iloc[dev_idx]
test_df = labels_with_patient.iloc[test_idx]

# step 2: carve 15% val from remaining 85% -> 0.176 of dev = 15% overall
gss2 = GroupShuffleSplit(n_splits=1, test_size=0.176, random_state=SEED)
train_idx2, val_idx2 = next(gss2.split(dev_df, groups=dev_df['PatientID']))
train_df = dev_df.iloc[train_idx2]
val_df   = dev_df.iloc[val_idx2]

train_patients_all = train_df['PatientID'].unique()
val_patients_all   = val_df['PatientID'].unique()
test_patients_all  = test_df['PatientID'].unique()

overlap_tv = set(train_patients_all) & set(val_patients_all)
overlap_tt = set(train_patients_all) & set(test_patients_all)
overlap_vt = set(val_patients_all)   & set(test_patients_all)
print(f"Patient overlap - train/val: {len(overlap_tv)}, train/test: {len(overlap_tt)}, val/test: {len(overlap_vt)} (all should be 0)", flush=True)
print(f"Split sizes - train: {len(train_patients_all)}, val: {len(val_patients_all)}, test: {len(test_patients_all)} patients", flush=True)

# stratified patient capping - preserves label distribution when subsetting
patient_labels = (labels_with_patient
    .groupby('PatientID')[label_cols]
    .max()
    .reset_index())

def cap_df(df, patients_all, n_cap, seed, split_name):
    pat_lab  = patient_labels[patient_labels['PatientID'].isin(patients_all)]
    selected = select_patients_stratified(pat_lab, n_cap, seed)
    print(f"  {split_name}: {len(selected)} / {len(patients_all)} patients selected (stratified)", flush=True)
    return df[df['PatientID'].isin(selected)]

train_df = cap_df(train_df, train_patients_all, N_TRAIN_PATIENTS, SEED, 'train')
val_df   = cap_df(val_df,   val_patients_all,   N_VAL_PATIENTS,   SEED, 'val')
test_df  = cap_df(test_df,  test_patients_all,  N_TEST_PATIENTS,  SEED, 'test')

# use actual patient counts for cache name, not targets (stratification may differ slightly)
n_train_pat = train_df['PatientID'].nunique()
n_val_pat   = val_df['PatientID'].nunique()
n_test_pat  = test_df['PatientID'].nunique()

print(f"Selected slices - train: {len(train_df)}, val: {len(val_df)}, test: {len(test_df)}", flush=True)

cache_base   = CACHE_DIR / f'rsna_tr{n_train_pat}_va{n_val_pat}_te{n_test_pat}_{IMG_SIZE}'
x_train_path = str(cache_base) + '_x_train.npy'
y_train_path = str(cache_base) + '_y_train.npy'
x_val_path   = str(cache_base) + '_x_val.npy'
y_val_path   = str(cache_base) + '_y_val.npy'
x_test_path  = str(cache_base) + '_x_test.npy'
y_test_path  = str(cache_base) + '_y_test.npy'
sidecar_path = str(cache_base) + '_meta.json'

cache_files = [x_train_path, y_train_path, x_val_path, y_val_path, x_test_path, y_test_path]

if all(Path(p).exists() for p in cache_files):
    print(f"\nCache already exists: {cache_base}_*.npy", flush=True)
else:
    written_files = []
    try:
        if not (Path(x_train_path).exists() and Path(y_train_path).exists()):
            written_files += [x_train_path, y_train_path]
            print(f"\nWriting train set to memmap...", flush=True)
            n_train = write_to_memmap(train_df, x_train_path, y_train_path)
        else:
            print(f"\nTrain memmap already exists, skipping.", flush=True)
            n_train = np.load(x_train_path, mmap_mode='r').shape[0]

        if not (Path(x_val_path).exists() and Path(y_val_path).exists()):
            written_files += [x_val_path, y_val_path]
            print(f"\nWriting val set to memmap...", flush=True)
            n_val = write_to_memmap(val_df, x_val_path, y_val_path)
        else:
            print(f"\nVal memmap already exists, skipping.", flush=True)
            n_val = np.load(x_val_path, mmap_mode='r').shape[0]

        if not (Path(x_test_path).exists() and Path(y_test_path).exists()):
            written_files += [x_test_path, y_test_path]
            print(f"\nWriting test set to memmap...", flush=True)
            n_test = write_to_memmap(test_df, x_test_path, y_test_path)
        else:
            print(f"\nTest memmap already exists, skipping.", flush=True)
            n_test = np.load(x_test_path, mmap_mode='r').shape[0]
    except Exception:
        print("\nError during write - cleaning up only partial files...", flush=True)
        for p in written_files[-2:]:  # only the pair that was in-progress
            Path(p).unlink(missing_ok=True)
        raise

    meta = {
        'created': datetime.datetime.now().isoformat(),
        'SEED': SEED,
        'IMG_SIZE': IMG_SIZE,
        'N_TRAIN_PATIENTS': n_train_pat,
        'N_VAL_PATIENTS': n_val_pat,
        'N_TEST_PATIENTS': n_test_pat,
        'n_train_slices': n_train,
        'n_val_slices': n_val,
        'n_test_slices': n_test,
        'label_cols': label_cols,
        'channels': 3,
        'windows': WINDOWS,
        'split_ratios': '70/15/15',
        'stratified_capping': True,
    }
    with open(sidecar_path, 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"\nSidecar written: {sidecar_path}", flush=True)
    print(f"Saved {n_train} train, {n_val} val, {n_test} test slices.", flush=True)

# report shapes via mmap - no data loaded into RAM
x_train = np.load(x_train_path, mmap_mode='r')
y_train = np.load(y_train_path, mmap_mode='r')
x_val   = np.load(x_val_path,   mmap_mode='r')
y_val   = np.load(y_val_path,   mmap_mode='r')
x_test  = np.load(x_test_path,  mmap_mode='r')
y_test  = np.load(y_test_path,  mmap_mode='r')

print(f"\nx_train: {x_train.shape}, y_train: {y_train.shape}")
print(f"x_val:   {x_val.shape},   y_val:   {y_val.shape}")
print(f"x_test:  {x_test.shape},  y_test:  {y_test.shape}")
print(f"\nPreprocessing finished at {datetime.datetime.now()}")
print("Done!")
